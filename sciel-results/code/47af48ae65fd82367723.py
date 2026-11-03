import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety

    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        # Simple greedy for other cases
        N = q ** n
        def word_to_idx(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx
        def idx_to_word(idx):
            word = [0] * n
            for i in range(n-1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word
        
        all_words = [idx_to_word(i) for i in range(N)]
        neighbor_masks = [0] * N
        for c_idx in range(N):
            c_word = all_words[c_idx]
            mask = 1 << c_idx
            if radius == 1:
                for i in range(n):
                    orig = c_word[i]
                    for new_val in range(q):
                        if new_val != orig:
                            new_word = c_word.copy()
                            new_word[i] = new_val
                            w_idx = word_to_idx(new_word)
                            mask |= (1 << w_idx)
            neighbor_masks[c_idx] = mask
            
        uncovered = (1 << N) - 1
        code = []
        while uncovered and time.time() < deadline:
            best_c = -1
            best_count = -1
            for c in range(N):
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            uncovered &= ~neighbor_masks[best_c]
        return [all_words[i] for i in code]

    # Main path: q=3, n=8, radius=1
    N = 3 ** 8  # 6561
    
    # Helper to convert index to word and vice versa
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for i in range(n-1, -1, -1):
            word[i] = temp % q
            temp //= q
        return word
        
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors for all words
    # For q=3, n=8, radius=1: each word has 1 + 8*2 = 17 neighbors
    neighbor_lists = [[] for _ in range(N)]
    word_to_coders = [[] for _ in range(N)]
    
    # Build neighbor lists
    for c_idx in range(N):
        word = idx_to_word(c_idx)
        neighbors = [c_idx]
        for pos in range(n):
            orig = word[pos]
            for delta in [1, 2]:
                new_val = (orig + delta) % q
                new_word = word[:]
                new_word[pos] = new_val
                n_idx = word_to_idx(new_word)
                neighbors.append(n_idx)
        neighbor_lists[c_idx] = neighbors
        for w in neighbors:
            word_to_coders[w].append(c_idx)

    def time_left():
        return deadline - time.time()
    
    def build_cover_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                counts[w] += 1
        return counts

    def is_valid_cover(code):
        counts = build_cover_counts(code)
        return np.all(counts > 0)

    def greedy_criticality():
        """
        Greedy that prioritizes uncovered words with fewest potential coders (criticality).
        """
        code = []
        code_set = set()
        cover_count = np.zeros(N, dtype=np.int32)
        
        while True:
            if time_left() < 5:
                break
            
            # Find uncovered words
            uncovered_indices = np.where(cover_count == 0)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Find the uncovered word with minimum criticality (fewest coders)
            # If all have same criticality (17 for radius 1 in q=3, n=8), 
            # we pick one that allows a codeword to cover the most other uncovered words.
            
            # Optimization: If we pick a random uncovered word, it might be "easy".
            # Let's pick the one with the smallest number of available coders.
            # Since all words in {0,1,2}^8 have exactly 17 coders (1 self + 8*2 neighbors),
            # criticality is uniform initially. 
            # However, as we add codewords, the *available* coders (not in code_set) change.
            
            # Let's pick a random uncovered word to avoid bias, 
            # then pick the best coder for it.
            target_w = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            
            best_c = -1
            best_new_cover = -1
            
            # Iterate through coders of target_w
            for c in word_to_coders[target_w]:
                if c in code_set:
                    continue
                
                # Count how many uncovered words this c covers
                new_cover = np.sum(cover_count[neighbor_lists[c]] == 0)
                
                if new_cover > best_new_cover:
                    best_new_cover = new_cover
                    best_c = c
            
            if best_c == -1:
                break
                
            code.append(best_c)
            code_set.add(best_c)
            for w in neighbor_lists[best_c]:
                cover_count[w] += 1
        
        return code

    def local_search_remove(code):
        """
        Repeatedly remove codewords that do not uniquely cover any word.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        improved = True
        while improved and time_left() > 2:
            improved = False
            # Iterate in reverse to allow popping
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any neighbor has count 1
                # If yes, c is critical, keep it.
                # If no, c is redundant, remove it.
                is_critical = False
                for w in neighbor_lists[c]:
                    if counts[w] == 1:
                        is_critical = True
                        break
                
                if not is_critical:
                    code.pop(i)
                    code_set.discard(c)
                    for w in neighbor_lists[c]:
                        counts[w] -= 1
                    improved = True
                    break # Restart loop to update criticality
        return code

    def two_opt_local_search(code):
        """
        Try to swap one codeword with another to enable removal of the first,
        or to reduce the total number of codewords.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        improved = True
        while improved and time_left() > 5:
            improved = False
            
            # Try single removals first
            for i in range(len(code) - 1, -1, -1):
                if time_left() < 2:
                    break
                c = code[i]
                is_critical = False
                for w in neighbor_lists[c]:
                    if counts[w] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    code.pop(i)
                    code_set.discard(c)
                    for w in neighbor_lists[c]:
                        counts[w] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove c_remove, add c_add
            # We want c_add to cover all words that c_remove uniquely covers.
            # And ideally, c_add covers more uncovered words than c_remove did,
            # or simply allows the structure to be tighter.
            
            for i in range(len(code)):
                if time_left() < 2:
                    break
                
                c_remove = code[i]
                # Words uniquely covered by c_remove
                unique_words = [w for w in neighbor_lists[c_remove] if counts[w] == 1]
                
                if len(unique_words) == 0:
                    # Should have been caught in removal phase, but just in case
                    code.pop(i)
                    code_set.discard(c_remove)
                    for w in neighbor_lists[c_remove]:
                        counts[w] -= 1
                    improved = True
                    break
                
                # Find a candidate c_add that covers all unique_words
                # and is not in code_set
                best_add = -1
                best_gain = -1 # Gain is number of newly covered words (should be 0 if we just want to swap, but we want to improve "tightness")
                
                # Optimization: Only check candidates that cover the first unique word
                # This drastically reduces the search space
                first_unique = unique_words[0]
                candidates = word_to_coders[first_unique]
                
                for c_add in candidates:
                    if c_add in code_set:
                        continue
                    
                    # Check if c_add covers all unique_words
                    covers_all = True
                    for w in unique_words:
                        if counts[w] == 1 and w not in neighbor_lists[c_add]:
                            # Wait, counts[w]==1 means only c_remove covers it.
                            # If c_add covers w, then after swap, w is still covered.
                            # We need to check if w is in neighbor_lists[c_add].
                            if w not in neighbor_lists[c_add]:
                                covers_all = False
                                break
                    
                    if not covers_all:
                        continue
                    
                    # If it covers all unique words, it's a valid swap.
                    # Does it offer any benefit?
                    # Benefit: It might allow other codewords to be removed later.
                    # For now, let's just accept a valid swap that doesn't increase size.
                    # To ensure progress, we can prefer swaps that cover "more" words 
                    # with higher multiplicity, or simply randomize to explore.
                    
                    # Let's calculate the "potential" gain: 
                    # How many words would be covered by c_add that are currently covered by only c_remove?
                    # Actually, any valid swap maintains coverage.
                    # To avoid cycles, we can just accept the first valid swap found,
                    # or one that maximizes the minimum count of its neighbors.
                    
                    # Let's just accept it.
                    best_add = c_add
                    break # Take the first valid one to save time
                    
                if best_add != -1:
                    # Perform swap
                    code[i] = best_add
                    code_set.discard(c_remove)
                    code_set.add(best_add)
                    for w in neighbor_lists[c_remove]:
                        counts[w] -= 1
                    for w in neighbor_lists[best_add]:
                        counts[w] += 1
                    improved = True
                    break
            
        return code

    def simulated_annealing(code, max_iters=500):
        """
        Simulated Annealing to escape local minima.
        Objective: Minimize len(code). 
        Constraint: Full coverage.
        We use a penalty for uncovered words to allow temporary violations.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        uncovered_count = int(np.sum(counts == 0))
        # Penalty should be high enough to discourage uncovered words,
        # but low enough to allow swaps that temporarily increase size?
        # Actually, we only do swaps (size constant) or add/remove (size changes).
        # Let's stick to swaps and add/remove.
        
        penalty = 100
        current_obj = len(code) + penalty * uncovered_count
        best_code = list(code)
        best_obj = current_obj
        
        temp = 10.0
        cooling = 0.99
        
        for it in range(max_iters):
            if time_left() < 5:
                break
            if temp < 0.1:
                break
            
            move_type = random.random()
            
            if move_type < 0.5 and len(code) > 10:
                # Remove a codeword
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                newly_uncovered = int(np.sum(counts[neighbor_lists[c_remove]] == 1))
                delta_obj = -1 + penalty * newly_uncovered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code.pop(idx)
                    code_set.discard(c_remove)
                    for w in neighbor_lists[c_remove]:
                        counts[w] -= 1
                    current_obj = new_obj
                    uncovered_count += newly_uncovered
                    
            elif move_type < 0.8:
                # Add a codeword
                # Pick a random word, pick a random coder for it
                w = random.randint(0, N - 1)
                if not word_to_coders[w]:
                    continue
                c_add = random.choice(word_to_coders[w])
                if c_add in code_set:
                    continue
                
                newly_covered = int(np.sum(counts[neighbor_lists[c_add]] == 0))
                delta_obj = 1 - penalty * newly_covered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code.append(c_add)
                    code_set.add(c_add)
                    for w in neighbor_lists[c_add]:
                        counts[w] += 1
                    current_obj = new_obj
                    uncovered_count -= newly_covered
                    
            else:
                # Swap
                if len(code) == 0:
                    continue
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                
                # Pick a random word covered by c_remove, and a random other coder for it
                w = random.choice(neighbor_lists[c_remove])
                coders = word_to_coders[w]
                if len(coders) < 2:
                    continue
                # Pick a coder different from c_remove
                c_add = random.choice([c for c in coders if c != c_remove])
                
                if c_add in code_set:
                    continue
                
                # Calculate delta
                rem_uncov = int(np.sum(counts[neighbor_lists[c_remove]] == 1))
                add_cov = int(np.sum(counts[neighbor_lists[c_add]] == 0))
                uncov_delta = rem_uncov - add_cov
                delta_obj = penalty * uncov_delta
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code[idx] = c_add
                    code_set.discard(c_remove)
                    code_set.add(c_add)
                    for w in neighbor_lists[c_remove]:
                        counts[w] -= 1
                    for w in neighbor_lists[c_add]:
                        counts[w] += 1
                    current_obj = new_obj
                    uncovered_count += uncov_delta
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_code = list(code)
            
            temp *= cooling
        
        # Ensure full coverage
        counts_best = build_cover_counts(best_code)
        if np.any(counts_best == 0):
            best_code = greedy_criticality() # Re-greedy if broken? Or just complete it.
            # Better: complete it
            code_set = set(best_code)
            counts = build_cover_counts(best_code)
            while True:
                if time_left() < 2:
                    break
                uncovered = np.where(counts == 0)[0]
                if len(uncovered) == 0:
                    break
                w = uncovered[random.randint(0, len(uncovered)-1)]
                for c in word_to_coders[w]:
                    if c not in code_set:
                        best_code.append(c)
                        code_set.add(c)
                        for nw in neighbor_lists[c]:
                            counts[nw] += 1
                        break
                if time_left() < 2:
                    break
        
        return best_code

    # Main Algorithm
    # 1. Criticality-based Greedy Initialization
    code = greedy_criticality()
    
    # 2. Local Search: Removals
    code = local_search_remove(code)
    
    # 3. Local Search: 2-Opt Swaps
    code = two_opt_local_search(code)
    
    # 4. Another pass of removals
    code = local_search_remove(code)
    
    best_code = code
    best_size = len(code)
    
    # 5. Simulated Annealing
    if time_left() > 10:
        sa_code = simulated_annealing(code, max_iters=300)
        sa_code = local_search_remove(sa_code)
        sa_code = two_opt_local_search(sa_code)
        sa_code = local_search_remove(sa_code)
        
        if len(sa_code) < best_size:
            best_code = sa_code
            best_size = len(sa_code)
            
    # 6. Final Polish
    if time_left() > 5:
        best_code = local_search_remove(best_code)
        best_code = two_opt_local_search(best_code)
        best_code = local_search_remove(best_code)
        
    # Verify coverage
    final_counts = build_cover_counts(best_code)
    if np.any(final_counts == 0):
        # Emergency completion
        code_set = set(best_code)
        counts = final_counts
        while True:
            if time_left() < 1:
                break
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            w = uncovered[0]
            for c in word_to_coders[w]:
                if c not in code_set:
                    best_code.append(c)
                    code_set.add(c)
                    for nw in neighbor_lists[c]:
                        counts[nw] += 1
                    break
            if time_left() < 1:
                break

    # Convert indices to words
    result = []
    for c in best_code:
        result.append(idx_to_word(c))
    
    return result
