import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Fallback algebraic construction
    def get_fallback():
        if n == 4:
            return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n > 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    if q != 3 or n != 8 or radius != 1:
        return get_fallback()

    total_words = q ** n
    num_neighbors = 1 + n * (q - 1)

    # Helper to convert int to base-q digits
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    # Precompute neighbors array
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    # Precompute ball_membership: ball_membership[c, w] = 1 if w is in the ball of radius 1 around c
    # This is equivalent to neighbors but transposed for faster lookup
    # ball_membership[c] gives all words covered by codeword c
    ball_membership = np.zeros((total_words, total_words), dtype=np.uint8)
    for c in range(total_words):
        ball_membership[c, neighbors[c]] = 1

    # Greedy Set Cover with vectorized NumPy
    uncovered = np.ones(total_words, dtype=np.uint8)
    selected_indices = []

    while uncovered.sum() > 0:
        if time.time() > deadline:
            return get_fallback()

        # Calculate gain for each candidate using np.take (avoids boolean-indexing overhead)
        gains = np.take(uncovered, neighbors).sum(axis=1)
        best_idx = np.argmax(gains)

        if gains[best_idx] == 0:
            break

        selected_indices.append(int(best_idx))
        uncovered[neighbors[best_idx]] = 0

    # Convert selected indices back to codewords
    codewords = [int_to_digits(idx, n, q) for idx in selected_indices]

    # Post-processing: Local Search / Redundancy Removal
    if time.time() < deadline - 5.0:
        # Rebuild covered array for the current code
        covered_final = np.zeros(total_words, dtype=bool)
        for idx in selected_indices:
            covered_final[neighbors[idx]] = True

        # Try to remove redundant codewords
        changed = True
        while changed:
            if time.time() > deadline - 5.0:
                break
            changed = False
            for i in range(len(selected_indices)):
                if time.time() > deadline - 5.0:
                    break
                idx = selected_indices[i]
                # Temporarily remove this codeword
                was_covered = covered_final[neighbors[idx]].copy()
                covered_final[neighbors[idx]] = False

                # Check if any point becomes uncovered
                if not covered_final[neighbors[idx]].all():
                    # This codeword is essential, restore it
                    covered_final[neighbors[idx]] = was_covered
                else:
                    # This codeword is redundant, remove it
                    selected_indices.pop(i)
                    changed = True
                    break  # Restart the loop since indices changed

        # Update codewords list
        codewords = [int_to_digits(idx, n, q) for idx in selected_indices]

    # Swap-based local search to reduce code size further
    if time.time() < deadline - 5.0 and len(selected_indices) > 0:
        random.seed(42)
    covered_arr = np.zeros(total_words, dtype=np.int32)
    for idx in selected_indices:
        covered_arr[neighbors[idx]] += 1
        
    best_size = len(selected_indices)
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Shuffle order for diversification
        order = list(range(len(selected_indices)))
        random.shuffle(order)
            
        for i in order:
            if time.time() > deadline - 5.0:
                break
            if len(selected_indices) >= best_size:
                continue
                    
            idx = selected_indices[i]
            # Try to remove this codeword
            neighbors_i = neighbors[idx]
            # Check if all neighbors are still covered by other codewords
            if np.all(covered_arr[neighbors_i] > 1):
                # Safe to remove
                covered_arr[neighbors_i] -= 1
                selected_indices.pop(i)
                best_size = len(selected_indices)
                improved = True
                break # Restart loop with new state
                
            # Try swap: replace idx with a better candidate
            only_covered_by_idx = (covered_arr[neighbors_i] == 1)
            critical_points = neighbors_i[only_covered_by_idx]
                
            if len(critical_points) > 0:
                # Find candidates c that cover all critical_points
                covers_critical = ball_membership[:, critical_points].all(axis=1)
                    
                # Exclude current codeword and already selected codewords
                covers_critical[idx] = False
                for sel_idx in selected_indices:
                    if sel_idx != idx:
                        covers_critical[sel_idx] = False
                    
                valid_candidates = np.where(covers_critical)[0]
                    
                if len(valid_candidates) > 0:
                    # Score: prefer candidates whose neighbors are already well-covered
                    # (i.e., after adding c, many of c's neighbors have count > 1)
                    # This makes it more likely that c or others become removable.
                    cand_neigh = neighbors[valid_candidates]  # shape (nvalid, num_neighbors)
                    cand_cov = covered_arr[cand_neigh]  # coverage counts for each candidate's neighbors
                    # For each candidate, count how many of its neighbors have coverage >= 1
                    # (i.e., already covered by some codeword in the current set)
                    candidate_scores = (cand_cov >= 1).sum(axis=1)
                        
                    best_local = int(np.argmax(candidate_scores))
                    best_candidate = int(valid_candidates[best_local])
                        
                    # Perform the swap
                    covered_arr[neighbors_i] -= 1
                    covered_arr[neighbors[best_candidate]] += 1
                    selected_indices[i] = best_candidate
                        
                    # Try to remove redundant codewords after swap (multiple passes)
                    removal_done = False
                    for j in range(len(selected_indices)):
                        if time.time() > deadline - 5.0:
                            break
                        j_idx = selected_indices[j]
                        j_neighbors = neighbors[j_idx]
                        if np.all(covered_arr[j_neighbors] > 1):
                            covered_arr[j_neighbors] -= 1
                            selected_indices.pop(j)
                            best_size = len(selected_indices)
                            improved = True
                            removal_done = True
                            break
                        
                    if removal_done:
                        break # Restart outer loop with new state

    return codewords
