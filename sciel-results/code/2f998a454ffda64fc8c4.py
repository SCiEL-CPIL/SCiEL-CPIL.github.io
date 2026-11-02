import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        # Direct sum with tetracode (simple fallback)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute base-3 weights
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls: balls[w, 0..16] are the 17 words within Hamming distance 1 of w
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute digits for each index (for fast Hamming distance)
    digits_table = np.zeros((total, n), dtype=np.int8)
    for w in range(total):
        temp = w
        for i in range(n - 1, -1, -1):
            digits_table[w, i] = temp % 3
            temp //= 3

    def hamming_dist_fast(a, b):
        return int(np.sum(digits_table[a] != digits_table[b]))

    best_size = total
    best_set = None
    
    # Helper to convert index to digits for output
    def idx_to_digits(idx):
        return digits_table[idx].tolist()

    # We will run multiple iterations with different random seeds
    # Each iteration: Init -> Dual Greedy -> Local Search
    num_iterations = 200 # We will break early if time is up
    
    for iter in range(num_iterations):
        if time.time() > deadline - 10.0:
            break
            
        np.random.seed(iter * 13 + 7)
        random.seed(iter * 13 + 7)

        # --- Phase 1: Initialization ---
        # Use a diversity-based initialization to get a good spread.
        m_init = 80 
        selected = []
        if m_init > 0:
            # First few random
            for _ in range(min(10, m_init)):
                selected.append(np.random.randint(total))
            
            # Diversity expansion using precomputed digits
            while len(selected) < m_init:
                if time.time() > deadline - 10.0:
                    break
                # Sample candidates, pick one with max min-distance to selected
                best_cand = -1
                best_min_dist = -1
                # Sample 15 candidates
                candidates = np.random.randint(0, total, 15)
                for cand in candidates:
                    if cand in selected:
                        continue
                    min_d = n + 1
                    for s in selected:
                        d = hamming_dist_fast(cand, s)
                        if d < min_d:
                            min_d = d
                            if min_d <= best_min_dist:
                                break
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_cand = cand
                if best_cand != -1:
                    selected.append(best_cand)
                else:
                    selected.append(np.random.randint(total))

        # --- Phase 2: Dual-Guided Greedy Set Cover ---
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        uncovered_mask = (cover_count == 0)
        
        if np.any(uncovered_mask):
            while np.any(uncovered_mask):
                if time.time() > deadline - 10.0:
                    break
                
                # Smooth hardness: use 1/sqrt(count) for smoother gradient
                hardness = np.zeros(total, dtype=np.float64)
                nonzero = cover_count > 0
                hardness[nonzero] = 1.0 / np.sqrt(cover_count[nonzero].astype(np.float64))
                # Uncovered words get high priority
                max_hardness = np.max(hardness[nonzero]) if np.any(nonzero) else 1.0
                hardness[~nonzero] = max_hardness * 2.0 + 1.0
                
                # Score for each candidate c: sum of hardness over balls[c]
                scores = hardness[balls].sum(axis=1)
                
                # Pick the candidate with max score
                # Tie-breaking: random among max
                max_score = np.max(scores)
                candidates = np.where(scores == max_score)[0]
                best_idx = candidates[np.random.randint(len(candidates))]
                
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                uncovered_mask = (cover_count == 0)

        # --- Phase 3: Local Search ---
        in_selected = np.zeros(total, dtype=bool)
        in_selected[selected] = True
        
        changed = True
        local_iter = 0
        while changed and time.time() < deadline - 5.0:
            changed = False
            local_iter += 1
            
            # Pass 1: Removal
            # A word i is removable if min(cover_count[neighbors[i]]) > 1
            min_counts = np.min(cover_count[balls], axis=1)
            removable_mask = in_selected & (min_counts > 1)
            
            if np.any(removable_mask):
                removable_indices = np.where(removable_mask)[0]
                # Pick one randomly to explore
                idx_to_remove = removable_indices[np.random.randint(len(removable_indices))]
                
                selected.remove(idx_to_remove)
                in_selected[idx_to_remove] = False
                cover_count[balls[idx_to_remove]] -= 1
                changed = True
                continue # Restart loop to re-evaluate removals first
            
            # Pass 2: 1-Swap with optimized coverage check
            if len(selected) > 0:
                idx_to_remove = np.random.randint(len(selected))
                cw_removed = selected[idx_to_remove]
                
                # Temporarily remove
                selected.pop(idx_to_remove)
                in_selected[cw_removed] = False
                cover_count[balls[cw_removed]] -= 1
                
                # Find uncovered words
                U = np.where(cover_count == 0)[0]
                
                if len(U) > 0:
                    # Vectorized: for each candidate c, count how many of U it covers
                    # A candidate c covers word u iff u is in balls[c], i.e., c is in balls[u]
                    # So for each u in U, increment count for all c in balls[u]
                    covers_U = np.zeros(total, dtype=np.int32)
                    # Vectorized update
                    for u in U:
                        covers_U[balls[u]] += 1
                    
                    valid_mask = (covers_U == len(U)) & (~in_selected)
                    valid_candidates = np.where(valid_mask)[0]
                    
                    if len(valid_candidates) > 0:
                        # Pick one randomly
                        cw2 = valid_candidates[np.random.randint(len(valid_candidates))]
                        selected.append(cw2)
                        in_selected[cw2] = True
                        cover_count[balls[cw2]] += 1
                        changed = True
                    else:
                        # Revert
                        selected.insert(idx_to_remove, cw_removed)
                        in_selected[cw_removed] = True
                        cover_count[balls[cw_removed]] += 1
                else:
                    # U is empty, removal was valid
                    changed = True

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # If we hit the theoretical lower bound (403), we can stop
            if best_size <= 403:
                break

    if best_set is None:
        # Fallback: return full space (should not happen)
        return [idx_to_digits(w) for w in range(total)]
        
    result = []
    for w in best_set:
        result.append(idx_to_digits(w))
        
    return result
