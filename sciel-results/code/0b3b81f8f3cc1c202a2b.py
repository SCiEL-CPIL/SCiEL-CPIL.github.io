import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 105  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Mapping functions
    def int_to_word(x):
        w = [0] * n
        for k in range(n):
            w[k] = x % q
            x //= q
        return w

    def word_to_int(w):
        x = 0
        base = 1
        for k in range(n):
            x += w[k] * base
            base *= q
        return x

    if total_words > 50000:
        return [[i // (q ** (n - 1 - k)) % q for k in range(n)] for i in range(total_words)]

    # Precompute ball matrix: ball_matrix[i] is a numpy array of indices within distance radius of word i
    # For efficiency, also precompute as a boolean matrix for vectorized gain computation
    # ball_bool[i, j] = True iff word j is within distance 'radius' of word i
    # This is 2187x2187 = ~4.8M booleans = ~4.8MB, very manageable
    
    # First, generate all words
    all_words_list = [int_to_word(i) for i in range(total_words)]
    
    # Precompute balls as lists of indices
    ball_indices = []
    for i in range(total_words):
        wi = all_words_list[i]
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = wi[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    wi[pos] = new_val
                    ball.append(word_to_int(wi))
                wi[pos] = orig_val
        ball_indices.append(ball)
    
    # Precompute boolean matrix for vectorized operations
    # ball_bool[i, j] = 1 if j in ball_indices[i]
    ball_bool = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        for j in ball_indices[i]:
            ball_bool[i, j] = 1
    
    # Precompute reverse: for each word j, which codewords cover it (i.e., j in ball[i])
    # This is the transpose
    cover_bool = ball_bool.T  # cover_bool[j, i] = 1 if codeword i covers word j
    
    # Also precompute ball as numpy arrays for fast indexing
    ball_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    best_codewords = None
    best_size = float('inf')
    
    num_trials = 30
    
    def vectorized_greedy(selected_set, seed=None):
        """Run greedy set cover using vectorized gain computation."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        selected_list = []
        uncovered = np.ones(total_words, dtype=bool)
        
        while uncovered.any():
            # Compute gain for each candidate: number of uncovered words covered by candidate i
            # gain[i] = sum(uncovered[j] for j in ball_indices[i])
            # = ball_bool[i, :] @ uncovered
            gains = (ball_bool @ uncovered.astype(np.uint8)).astype(np.int32)
            
            # Mask out already selected
            for idx in selected_set:
                gains[idx] = -1
            
            # Find best
            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0:
                break
            
            # Random tie-breaking: find all with max gain
            max_gain = gains[best_idx]
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_idx = int(rng.choice(candidates))
            
            selected_set.add(best_idx)
            selected_list.append(best_idx)
            uncovered[ball_np[best_idx]] = False
        
        return selected_list, selected_set
    
    def remove_redundant_fast(selected_list, selected_set):
        """Remove redundant codewords using coverage counting."""
        if not selected_list:
            return selected_list, selected_set
        
        # Compute coverage count: for each word, how many selected codewords cover it
        # coverage[w] = sum(1 for idx in selected_list if w in ball_indices[idx])
        # = sum over selected of ball_bool[idx, w]
        # = (ball_bool[selected_list, :] @ ones) ... actually:
        # coverage = ball_bool[selected_list, :].sum(axis=0)
        
        sel_arr = np.array(selected_list, dtype=np.int32)
        coverage = ball_bool[sel_arr, :].sum(axis=0).astype(np.int32)
        
        changed = True
        while changed:
            changed = False
            # For each selected codeword, check if it's redundant
            # A codeword i is redundant if for all w in ball[i], coverage[w] > 1
            # i.e., coverage[ball[i]] all > 1
            for i in range(len(selected_list)):
                idx = selected_list[i]
                ball = ball_np[idx]
                if np.all(coverage[ball] > 1):
                    # Remove
                    selected_list.pop(i)
                    selected_set.remove(idx)
                    sel_arr = np.array(selected_list, dtype=np.int32)
                    if len(sel_arr) > 0:
                        coverage = ball_bool[sel_arr, :].sum(axis=0).astype(np.int32)
                    else:
                        coverage = np.zeros(total_words, dtype=np.int32)
                    changed = True
                    break
        
        return selected_list, selected_set
    
    def local_search(selected_list, selected_set, deadline, seed=None):
        """Local search with delete, add, and swap moves using precomputed ball indices."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        if not selected_list:
            return selected_list, selected_set
        
        # Compute coverage
        sel_arr = np.array(selected_list, dtype=np.int32)
        coverage = ball_bool[sel_arr, :].sum(axis=0).astype(np.int32)
        uncovered_count = int((coverage == 0).sum())
        
        best_list = selected_list[:]
        best_set = selected_set.copy()
        best_size = len(selected_list)
        best_uncovered = uncovered_count
        
        all_words = list(range(total_words))
        
        # SA-like local search with kicks
        T_start = 3.0
        T_end = 0.05
        cooling_rate = 0.997
        max_iterations = 3000
        kick_size = 5
        
        T = T_start
        current_list = selected_list[:]
        current_set = selected_set.copy()
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            move_type = rng.random()
            
            if move_type < 0.4 and len(current_list) > 15:
                # Delete move
                remove_idx = rng.choice(current_list)
                ball = ball_np[remove_idx]
                # Words that become uncovered: those with coverage==1 in ball
                newly_uncovered = int((coverage[ball] == 1).sum())
                
                delta_energy = -1 + 100 * newly_uncovered
                
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current_list.remove(remove_idx)
                    current_set.remove(remove_idx)
                    coverage[ball] -= 1
                    uncovered_count += newly_uncovered
                    
                    if len(current_list) + 100 * uncovered_count < best_size + 100 * best_uncovered:
                        best_list = current_list[:]
                        best_set = current_set.copy()
                        best_size = len(current_list)
                        best_uncovered = uncovered_count
                        
            elif move_type < 0.7:
                # Swap move: remove one, add one
                if len(current_list) < 5:
                    continue
                remove_idx = rng.choice(current_list)
                add_idx = rng.choice(all_words)
                
                if add_idx in current_set:
                    continue
                
                ball_remove = ball_np[remove_idx]
                ball_add = ball_np[add_idx]
                
                # Net effect on uncovered:
                # After removing remove_idx: words in ball_remove with coverage==1 become uncovered
                # After adding add_idx: words in ball_add with (coverage-1_if_in_remove)==0 become covered
                # Compute carefully:
                # For words in ball_remove: coverage decreases by 1
                # For words in ball_add: coverage increases by 1
                # Net change in uncovered:
                # Words that were covered and become uncovered: in ball_remove, coverage==1, NOT in ball_add
                # Words that were uncovered and become covered: in ball_add, coverage==0 (before any changes), NOT in ball_remove
                # Words in both: net zero effect on their coverage
                
                # Words in ball_remove but not ball_add with coverage==1 -> become uncovered
                in_remove_not_add = np.setdiff1d(ball_remove, ball_add)
                become_uncovered = int((coverage[in_remove_not_add] == 1).sum()) if len(in_remove_not_add) > 0 else 0
                
                # Words in ball_add but not ball_remove with coverage==0 -> become covered
                in_add_not_remove = np.setdiff1d(ball_add, ball_remove)
                become_covered = int((coverage[in_add_not_remove] == 0).sum()) if len(in_add_not_remove) > 0 else 0
                
                delta_uncovered = become_uncovered - become_covered
                delta_energy = 0 + 100 * delta_uncovered
                
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / T):
                    current_list.remove(remove_idx)
                    current_set.remove(remove_idx)
                    current_list.append(add_idx)
                    current_set.add(add_idx)
                    
                    coverage[ball_remove] -= 1
                    coverage[ball_add] += 1
                    uncovered_count += delta_uncovered
                    
                    if len(current_list) + 100 * uncovered_count < best_size + 100 * best_uncovered:
                        best_list = current_list[:]
                        best_set = current_set.copy()
                        best_size = len(current_list)
                        best_uncovered = uncovered_count
                        
            else:
                # Kick: randomly replace kick_size codewords
                if len(current_list) < kick_size + 5:
                    continue
                
                kick_indices = rng.sample(current_list, kick_size)
                for idx in kick_indices:
                    current_list.remove(idx)
                    current_set.remove(idx)
                    coverage[ball_np[idx]] -= 1
                
                for _ in range(kick_size):
                    new_idx = rng.choice(all_words)
                    while new_idx in current_set:
                        new_idx = rng.choice(all_words)
                    current_list.append(new_idx)
                    current_set.add(new_idx)
                    coverage[ball_np[new_idx]] += 1
                
                uncovered_count = int((coverage == 0).sum())
                
                if len(current_list) + 100 * uncovered_count < best_size + 100 * best_uncovered:
                    best_list = current_list[:]
                    best_set = current_set.copy()
                    best_size = len(current_list)
                    best_uncovered = uncovered_count
            
            T *= cooling_rate
            if T < T_end:
                T = T_end
        
        return best_list, best_set
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Vectorized greedy
        selected_set = set()
        selected_list, selected_set = vectorized_greedy(selected_set, seed=trial)
        
        # Remove redundant
        selected_list, selected_set = remove_redundant_fast(selected_list, selected_set)
        
        # Local search refinement
        if time.time() < deadline - 10:
            selected_list, selected_set = local_search(selected_list, selected_set, deadline, seed=trial)
            selected_list, selected_set = remove_redundant_fast(selected_list, selected_set)
        
        if len(selected_list) < best_size:
            best_size = len(selected_list)
            best_codewords = selected_list[:]
        
        # Adaptive: if we're running out of time, stop
        if time.time() > deadline - 8:
            break
    
    if best_codewords is None:
        return [int_to_word(i) for i in range(total_words)]
        
    return [int_to_word(i) for i in best_codewords]
