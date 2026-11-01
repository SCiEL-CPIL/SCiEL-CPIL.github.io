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

        # Calculate gain for each candidate
        gains = uncovered[neighbors].sum(axis=1)
        best_idx = np.argmax(gains)

        if gains[best_idx] == 0:
            break

        selected_indices.append(best_idx)
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
    if time.time() < deadline - 10.0 and len(selected_indices) > 0:
        random.seed(42)
        covered_arr = np.zeros(total_words, dtype=np.int32)
        for idx in selected_indices:
            covered_arr[neighbors[idx]] += 1
        
        best_size = len(selected_indices)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Shuffle order for diversification
            order = list(range(len(selected_indices)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 10.0:
                    break
                if len(selected_indices) >= best_size:
                    continue
                    
                idx = selected_indices[i]
                # Try to remove this codeword
                neighbors_i = neighbors[idx]
                # Check if all neighbors are still covered by other codewords
                # covered_arr[neighbors_i] > 1 means covered by at least 2 codewords (including current)
                # So we need covered_arr[neighbors_i] > 1 for all neighbors to safely remove
                if np.all(covered_arr[neighbors_i] > 1):
                    # Safe to remove
                    covered_arr[neighbors_i] -= 1
                    selected_indices.pop(i)
                    best_size = len(selected_indices)
                    improved = True
                    break # Restart loop with new state
                
                # Try swap: replace idx with a better candidate
                # For non-redundant codeword idx, find points only covered by idx
                only_covered_by_idx = (covered_arr[neighbors_i] == 1)
                critical_points = neighbors_i[only_covered_by_idx]
                
                if len(critical_points) > 0:
                    # Find candidates c that cover all critical_points
                    # c is valid if for all p in critical_points, ball_membership[c, p] == 1
                    # Also c should not already be in selected_indices
                    # And we should prefer c that covers many points already covered by idx (to allow future removals)
                    
                    # Vectorized check: for each candidate c, check if it covers all critical_points
                    # critical_points has shape (k,), we need ball_membership[:, critical_points].all(axis=1)
                    covers_critical = ball_membership[:, critical_points].all(axis=1)
                    
                    # Exclude current codeword and already selected codewords
                    covers_critical[idx] = False
                    for sel_idx in selected_indices:
                        if sel_idx != idx:
                            covers_critical[sel_idx] = False
                    
                    # Find valid candidates
                    valid_candidates = np.where(covers_critical)[0]
                    
                    if len(valid_candidates) > 0:
                        # Score candidates by how many of idx's neighbors they cover
                        # This helps find swaps that allow future removals
                        # candidate_score[c] = number of neighbors of idx that c also covers
                        # = sum of ball_membership[c, neighbors_i]
                        candidate_scores = ball_membership[valid_candidates, neighbors_i].sum(axis=1)
                        
                        # Pick the best candidate (highest score)
                        if len(candidate_scores) > 0:
                            best_local = np.argmax(candidate_scores)
                            best_candidate = valid_candidates[best_local]
                            
                            # Perform the swap
                            # Remove idx from coverage
                            covered_arr[neighbors_i] -= 1
                            # Add best_candidate to coverage
                            covered_arr[neighbors[best_candidate]] += 1
                            
                            # Update selected_indices
                            selected_indices[i] = best_candidate
                            
                            # Try to remove redundant codewords after swap
                            removal_done = False
                            for j in range(len(selected_indices)):
                                if time.time() > deadline - 10.0:
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
