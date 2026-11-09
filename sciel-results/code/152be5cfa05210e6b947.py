"""Greedy set-cover with Local Search for ternary covering codes K_3(8,1)."""
import numpy as np
import random
import time


def construct(q, n, radius):
    # For this task: q=3, n=8, radius=1
    N = q ** n
    
    # Precompute powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
    
    # Precompute balls (list of indices at distance <= radius)
    # ball[i] = [i] + neighbors
    balls = []
    ball_sets = []
    for i in range(N):
        neighbors = [i]
        for p in range(n):
            cur = (i // powers[p]) % q
            base = i - cur * powers[p]
            for nd in range(q):
                if nd != cur:
                    nb = base + nd * powers[p]
                    neighbors.append(nb)
        balls.append(neighbors)
        ball_sets.append(set(neighbors))
    
    def greedy_random(alpha=5):
        uncovered = np.ones(N, dtype=bool)
        code = []
        
        while np.any(uncovered):
            # Compute coverage for all candidates
            covs = np.empty(N, dtype=np.int32)
            for i in range(N):
                covs[i] = np.sum(uncovered[balls[i]])
            
            # Get top alpha candidates
            if alpha >= N:
                top_indices = np.argsort(covs)[::-1]
            else:
                # Get indices of top alpha
                part_indices = np.argpartition(covs, -alpha)[-alpha:]
                part_covs = covs[part_indices]
                sorted_part_indices = part_indices[np.argsort(part_covs)[::-1]]
                top_indices = sorted_part_indices
            
            # Pick randomly from top_indices where cov > 0
            valid_indices = [idx for idx in top_indices if covs[idx] > 0]
            if not valid_indices:
                break
                
            chosen_idx = random.choice(valid_indices)
            
            code.append(chosen_idx)
            uncovered[balls[chosen_idx]] = False
            
        return code

    def local_search(code):
        if not code:
            return code
        
        # Initialize coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[balls[c]] += 1
            
        code_set = set(code)
        
        # Helper to check if removal of c is valid
        def can_remove(c):
            return np.min(counts[balls[c]]) > 1

        # Helper to get unique words for c
        def get_unique_words(c):
            return [j for j in balls[c] if counts[j] == 1]

        # Iterative improvement
        max_iterations = 100 # Safety cap
        for _ in range(max_iterations):
            improved = False
            
            # Phase 1: Try to remove codewords
            removed_any = True
            while removed_any:
                removed_any = False
                for i in range(len(code)):
                    c = code[i]
                    if can_remove(c):
                        # Remove c
                        counts[balls[c]] -= 1
                        code.pop(i)
                        code_set.remove(c)
                        improved = True
                        removed_any = True
                        break # Restart scan as counts changed
            
            # Phase 2: Try to swap codewords to enable removals
            if not improved:
                swapped = False
                # Randomize order of codewords to avoid bias
                indices = list(range(len(code)))
                random.shuffle(indices)
                
                for i in indices:
                    c = code[i]
                    unique_words = get_unique_words(c)
                    if not unique_words:
                        continue
                    
                    # Try to find a w that covers all unique_words
                    # And w is not already in code
                    found_swap = False
                    
                    # Shuffle w candidates to avoid bias
                    w_candidates = list(range(N))
                    random.shuffle(w_candidates)
                    
                    for w in w_candidates:
                        if w == c:
                            continue
                        if w in code_set:
                            continue
                        
                        # Check if ball(w) covers unique_words
                        is_cover = True
                        for j in unique_words:
                            if j not in ball_sets[w]:
                                is_cover = False
                                break
                        
                        if is_cover:
                            # Perform swap
                            counts[balls[c]] -= 1
                            counts[balls[w]] += 1
                            code[i] = w
                            code_set.remove(c)
                            code_set.add(w)
                            improved = True
                            swapped = True
                            found_swap = True
                            break
                    
                    if swapped:
                        break # Restart whole loop to try removals again
            
        return code

    # Run multiple restarts
    best_code = None
    best_len = float('inf')
    
    # Time limit check
    start_time = time.time()
    time_limit = 110 # Leave 10s for safety
    
    while time.time() - start_time < time_limit:
        code = greedy_random(alpha=5)
        code = local_search(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            # If we hit the known best, we can stop? 
            # Known best is 486.
            if best_len <= 486:
                break
    
    # Convert indices back to words
    result = []
    for idx in best_code:
        word = []
        rem = idx
        for p in range(n):
            digit = rem % q
            word.append(digit)
            rem //= q
        word.reverse()
        result.append(word)
    
    return result
