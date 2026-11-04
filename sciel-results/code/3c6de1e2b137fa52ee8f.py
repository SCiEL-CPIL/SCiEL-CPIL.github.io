import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_points = q ** n  # 6561
    ball_size = 1 + n * (q - 1)  # 17

    # Pre-compute powers for fast index arithmetic
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Pre-compute neighbors (the ball of radius 1) for each index
    neighbors = np.empty((total_points, ball_size), dtype=np.int32)
    for idx in range(total_points):
        temp_idx = idx
        word = [0] * n
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
            
        neighbors[idx, 0] = idx
        neighbor_idx = 1
        for pos in range(n):
            orig_digit = word[pos]
            power = powers[pos]
            
            # +1
            new_digit = (orig_digit + 1) % q
            if new_digit == 0:
                delta = -2 * power
            else:
                delta = power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1
            
            # -1
            new_digit = (orig_digit - 1) % q
            if new_digit == 2:
                delta = 2 * power
            else:
                delta = -power
            n_idx = idx + delta
            neighbors[idx, neighbor_idx] = n_idx
            neighbor_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % q
            temp //= q
        return w

    def greedy_construct(seed):
        rng = np.random.default_rng(seed)
        is_covered = np.zeros(total_points, dtype=bool)
        selected = []
        
        while not np.all(is_covered):
            # Count uncovered points in each ball
            uncovered_counts = np.sum(~is_covered[neighbors], axis=1)
            
            max_count = np.max(uncovered_counts)
            if max_count == 0:
                break
            
            candidates = np.where(uncovered_counts == max_count)[0]
            if len(candidates) > 1:
                best_idx = rng.choice(candidates)
            else:
                best_idx = candidates[0]
                
            selected.append(best_idx)
            is_covered[neighbors[best_idx]] = True
            
        return selected

    def prune(selected_indices):
        if not selected_indices:
            return []
            
        selected_list = list(selected_indices)
        coverage_count = np.zeros(total_points, dtype=np.int32)
        
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            for i, idx in enumerate(selected_list):
                ball = neighbors[idx]
                if np.all(coverage_count[ball] > 1):
                    rem_idx = selected_list.pop(i)
                    coverage_count[neighbors[rem_idx]] -= 1
                    changed = True
                    break # Restart scan
        return selected_list

    def local_search(selected_indices, deadline):
        selected_list = list(selected_indices)
        selected_list = prune(selected_list)
        
        coverage_count = np.zeros(total_points, dtype=np.int32)
        for idx in selected_list:
            coverage_count[neighbors[idx]] += 1
            
        best_len = len(selected_list)
        best_code = selected_list[:]
        
        while time.time() < deadline:
            improved = False
            current_len = len(selected_list)
            
            # Try to remove or swap each element
            # We iterate through a copy to handle modifications
            for i in range(len(selected_list)):
                if time.time() > deadline:
                    break
                
                c = selected_list[i]
                
                # 1. Try removing c
                coverage_count[neighbors[c]] -= 1
                if np.any(coverage_count == 0):
                    # Cannot remove, try swapping
                    # Find best replacement
                    uncovered = (coverage_count == 0)
                    if np.any(uncovered):
                        # Score candidates by how many uncovered points they cover
                        # scores[b] = sum(uncovered[neighbors[b]])
                        scores = np.sum(uncovered[neighbors], axis=1)
                        # Exclude current selected words from candidates? 
                        # If we swap c -> c', c' must not be in selected_list (except c itself which is removed)
                        # Actually, if c' is in selected_list, it's a duplicate, which is bad.
                        # So mask out selected indices.
                        mask = np.ones(total_points, dtype=bool)
                        mask[selected_list] = False
                        mask[c] = True # c is allowed as a candidate? No, c is removed. 
                                       # If we pick c, we just revert. 
                                       # We want c' != c and c' not in S.
                        # But wait, if we pick c, scores[c] is high, but it's a no-op.
                        # Let's just pick max score. If it's c, it means no better swap.
                        
                        best_c_prime = np.argmax(scores)
                        
                        # If best is c, or best is already in selected_list (and not c), it's not a valid new swap
                        # But we masked selected_list. 
                        # Note: c was in selected_list, so mask[c] was False. We set mask[c]=True.
                        # So c is a candidate. If c is best, it means removing c and adding c (revert) is best.
                        # This implies no other word covers the uncovered points better than c did.
                        
                        if best_c_prime != c and not (best_c_prime in selected_list and best_c_prime != c):
                             # Actually, since we masked S, best_c_prime is not in S (except c).
                             # So if best_c_prime != c, it's a valid new word.
                             pass
                        else:
                             # No valid swap or swap is c itself
                             coverage_count[neighbors[c]] += 1
                             continue

                        # Perform swap c -> best_c_prime
                        c_prime = best_c_prime
                        
                        # Add c_prime
                        coverage_count[neighbors[c_prime]] += 1
                        
                        # Check if this allows a prune
                        # We don't know yet. Let's tentatively accept the swap.
                        selected_list[i] = c_prime
                        improved = True # We made a change, even if size is same, structure changed
                        
                        # Now try to prune the whole set? 
                        # That's expensive. 
                        # Let's just check if *any* word is now redundant.
                        # This is O(N * K) per check.
                        
                        # To keep it fast, let's just accept the swap and rely on the next "Try removing" phase
                        # or a global prune every few iterations.
                        
                        # Actually, a swap that doesn't reduce size immediately is only useful
                        # if it enables a removal. 
                        # Let's check if we can remove any word now.
                        # This is a mini-prune.
                        removed_any = False
                        for j in range(len(selected_list)):
                            if j == i: continue # Don't check the one we just moved? Or do?
                            cj = selected_list[j]
                            coverage_count[neighbors[cj]] -= 1
                            if not np.any(coverage_count == 0):
                                # Can remove cj
                                selected_list.pop(j)
                                removed_any = True
                                break
                            else:
                                coverage_count[neighbors[cj]] += 1
                                
                        if removed_any:
                            # We reduced size!
                            if len(selected_list) < best_len:
                                best_len = len(selected_list)
                                best_code = selected_list[:]
                        else:
                            # No removal. Should we revert the swap?
                            # If we keep it, we might explore a different path.
                            # If we revert, we stay on the hill.
                            # Let's revert to avoid drifting into dead ends.
                            coverage_count[neighbors[c_prime]] -= 1
                            coverage_count[neighbors[c]] += 1
                            selected_list[i] = c
                            improved = False
                    else:
                        # No uncovered points? Then c could have been removed.
                        # But we checked np.any(coverage_count == 0) earlier.
                        # If it was 0, we would have removed it in step 1.
                        pass
                else:
                    # Can remove c
                    selected_list.pop(i)
                    improved = True
                    if len(selected_list) < best_len:
                        best_len = len(selected_list)
                        best_code = selected_list[:]
                    break # Restart scan
                    
        return best_code

    start_time = time.time()
    deadline = start_time + 110.0
    
    best_code = None
    best_len = float('inf')
    
    # Run multiple greedy starts
    num_greedy = 5
    for seed in range(num_greedy):
        if time.time() > deadline - 60: # Leave 60s for local search
            break
        selected = greedy_construct(seed)
        selected = prune(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_code = selected
            
    if best_code is None:
        best_code = greedy_construct(0)
        best_code = prune(best_code)
        best_len = len(best_code)
    
    # Local Search on the best greedy result
    # We can also run local search on other greedy results if time permits
    best_code = local_search(best_code, deadline)
    
    # If time remains, try local search on other candidates?
    # The local search is the most expensive part.
    # Let's just run it on the best one found so far.
    
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
