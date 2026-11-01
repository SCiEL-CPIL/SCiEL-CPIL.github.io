import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120.0
    
    if q != 3 or n != 6 or radius != 1:
        return _greedy_cover(q, n, radius)
        
    N = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls as sets of indices
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                new_w = w.copy()
                new_w[pos] = val
                ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    # Symmetry score
    def symmetry_score(idx):
        w = all_words[idx]
        return tuple(sorted(w))
        
    # Precompute symmetry scores for all words
    sym_scores = [symmetry_score(i) for i in range(N)]
    
    def greedy_init(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        uncovered = set(range(N))
        code_indices = []
        picked = set()
        
        while uncovered:
            best_word = -1
            max_covered = 0
            best_score = None
            
            for i in range(N):
                if i in picked: continue
                inter = len(balls[i] & uncovered)
                if inter > max_covered:
                    max_covered = inter
                    best_word = i
                    best_score = sym_scores[i]
                elif inter == max_covered and inter > 0:
                    curr_score = sym_scores[i]
                    # Add small random noise to break ties differently in different runs
                    noise = random.random() * 0.01
                    if best_score is None or curr_score + noise < best_score:
                        best_word = i
                        best_score = curr_score
                        
            if best_word == -1:
                break
            code_indices.append(best_word)
            picked.add(best_word)
            uncovered -= balls[best_word]
            
        return code_indices

    def local_search(code_indices):
        current_code_set = set(code_indices)
        best_code_set = set(code_indices)
        best_size = len(best_code_set)
        
        counts = np.zeros(N, dtype=int)
        for idx in current_code_set:
            counts[list(balls[idx])] += 1
            
        # Convert balls to sets for fast subset check? 
        # balls is already list of sets.
        
        no_improve_steps = 0
        
        while time.time() - start_time < deadline - 2.0:
            if len(current_code_set) < best_size:
                best_size = len(current_code_set)
                best_code_set = set(current_code_set)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                if no_improve_steps > 1000: # Heuristic limit for stuck
                    break
            
            if not current_code_set:
                break
            
            # Pick a random word to try removing
            idx_remove = random.choice(list(current_code_set))
            
            # Find critical words: those covered ONLY by idx_remove
            critical = []
            for w in balls[idx_remove]:
                if counts[w] == 1:
                    critical.append(w)
            
            if not critical:
                # Safe to remove
                current_code_set.remove(idx_remove)
                for w in balls[idx_remove]:
                    counts[w] -= 1
                continue
            
            # Need to swap. Find w not in code such that critical is subset of Ball(w)
            critical_set = set(critical)
            best_add = -1
            best_add_score = None
            
            # Iterate all possible adds
            for idx_add in range(N):
                if idx_add in current_code_set:
                    continue
                
                # Check if critical is covered by idx_add
                # Optimization: If any critical word is not in ball[idx_add], skip
                # We can check intersection
                if not critical_set.issubset(balls[idx_add]):
                    continue
                
                # Valid candidate. Score it.
                # Score: Prefer high symmetry (small tuple), and maybe high reinforcement?
                # Let's stick to symmetry for consistency with init
                score = sym_scores[idx_add]
                
                if best_add_score is None or score < best_add_score:
                    best_add_score = score
                    best_add = idx_add
                    
            if best_add != -1:
                # Perform swap
                # Remove idx_remove
                current_code_set.remove(idx_remove)
                for w in balls[idx_remove]:
                    counts[w] -= 1
                # Add best_add
                current_code_set.add(best_add)
                for w in balls[best_add]:
                    counts[w] += 1
            else:
                # No valid swap found for this idx_remove.
                # Try another idx_remove?
                # If we loop through all and none work, we are stuck.
                # For now, just break or try a different strategy?
                # Let's just break to save time, as finding a swap for a specific
                # critical set is hard if the set is large.
                # Actually, we should try a few random idx_remove before giving up.
                # But the loop picks random idx_remove each time.
                # If we get here, it means for THIS random pick, no swap worked.
                # We just continue the loop to pick a different random idx_remove.
                pass

        return best_code_set

    # Main Loop: Multiple restarts
    global_best_set = None
    global_best_size = float('inf')
    
    attempts = 0
    while time.time() - start_time < deadline - 10.0:
        attempts += 1
        init_code = greedy_init(random_seed=attempts)
        best_set = local_search(init_code)
        
        if len(best_set) < global_best_size:
            global_best_size = len(best_set)
            global_best_set = best_set
            
        # If we hit the theoretical lower bound (71) or known best (73), we can stop early?
        # 71 is lower bound. 73 is best known.
        if global_best_size <= 73:
            # We might still try to find 71 or 72, but 73 is a very strong result.
            # Given time constraints, 73 is likely the max we can reliably get.
            # Let's run a few more iterations to see if we can beat 73?
            # If we find 73, we can break to ensure we return it before timeout.
            # But maybe we can find 72?
            # Let's allow a few more attempts if time permits.
            if global_best_size == 71:
                break
            elif global_best_size == 72:
                if time.time() - start_time > deadline - 20.0:
                    break
            elif global_best_size == 73:
                # 73 is the target. Break to be safe?
                # Or try to squeeze out 72?
                # Let's try 5 more times if time allows.
                if attempts > 5 or time.time() - start_time > deadline - 30.0:
                    break

    if global_best_set is None:
        # Fallback
        return _greedy_cover(q, n, radius)
        
    return [all_words[i] for i in global_best_set]

def _greedy_cover(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        for delta in itertools.product(range(-radius, radius+1), repeat=n):
            dist = sum(abs(d) for d in delta)
            if dist <= radius:
                new_w = [0]*n
                valid = True
                for i in range(n):
                    val = w[i] + delta[i]
                    if 0 <= val < q:
                        new_w[i] = val
                    else:
                        valid = False
                        break
                if valid:
                    ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    uncovered = set(range(N))
    code_indices = []
    
    while uncovered:
        best_word = -1
        max_covered = 0
        for i in range(N):
            if i in code_indices: continue
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
        if best_word == -1: break
        code_indices.append(best_word)
        uncovered -= balls[best_word]
        
    return [all_words[i] for i in code_indices]

if __name__ == "__main__":
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")
