import numpy as np
import time
import itertools

def construct(q, n, radius):
    # Fallback for non-target parameters
    if not (q == 3 and n == 8 and radius == 1):
        # Use the seed logic for other cases
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Handle small n generically if needed, though task is fixed
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 5.0
    
    # Precompute balls (coverage sets)
    # balls[i] is a list of indices j covered by codeword i
    balls = []
    # M is the incidence matrix
    M = np.zeros((N, N), dtype=np.uint8)
    
    # Helper to convert int to base-q digits
    def int_to_digits(num, length):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % q
            num //= q
        return digits

    # Helper to convert base-q digits to int
    def digits_to_int(digits):
        val = 0
        for i in range(len(digits) - 1, -1, -1):
            val = val * q + digits[i]
        return val

    for i in range(N):
        digits = int_to_digits(i, n)
        ball = [i]
        M[i, i] = 1
        
        for k in range(n):
            orig_val = digits[k]
            for v in range(q):
                if v == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[k] = v
                j = digits_to_int(new_digits)
                if j not in ball: # Avoid duplicates if any (though logic prevents it)
                    ball.append(j)
                    M[i, j] = 1
        balls.append(ball)

    # Phase 1: Greedy Construction
    uncovered = np.ones(N, dtype=np.uint8)
    codewords = []
    
    while uncovered.any():
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        # Calculate how many uncovered words each candidate covers
        # M is N x N, uncovered is N x 1 (implicitly)
        scores = M.dot(uncovered)
        
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            # Should not happen if uncovered is not empty
            break
            
        codewords.append(best_idx)
        # Update uncovered
        # M[best_idx] is a row of 0s and 1s
        # We want to set uncovered[j] = 0 where M[best_idx, j] = 1
        uncovered[M[best_idx] == 1] = 0

    # Phase 2: Greedy Removal
    if time.time() - start_time > time_limit - safety_margin:
        pass

    # Initialize counts
    counts = np.zeros(N, dtype=np.int32)
    for c in codewords:
        counts[balls[c]] += 1

    changed = True
    while changed and (time.time() - start_time < time_limit - safety_margin):
        changed = False
        new_codewords = []
        for c in codewords:
            mask = balls[c]
            counts[mask] -= 1
            if np.min(counts) >= 1:
                # We can remove c
                changed = True
            else:
                # We must keep c
                counts[mask] += 1
                new_codewords.append(c)
        codewords = new_codewords

    # Phase 3: Local Search (Swap-based optimization)
    # Try to swap one codeword for another to maintain coverage while potentially enabling further removals
    if time.time() - start_time < time_limit - safety_margin:
        # Rebuild counts after removal
        counts = np.zeros(N, dtype=np.int32)
        for c in codewords:
            counts[balls[c]] += 1
        
        # Set of current codewords for quick lookup
        cw_set = set(codewords)
        
        improved = True
        while improved and (time.time() - start_time < time_limit - safety_margin):
            improved = False
            new_codewords = list(codewords)
            
            for i, c in enumerate(codewords):
                if time.time() - start_time > time_limit - safety_margin:
                    break
                
                # Try removing c temporarily
                counts[balls[c]] -= 1
                min_count = np.min(counts)
                
                if min_count >= 1:
                    # c is redundant, remove it
                    new_codewords.pop(i)
                    cw_set.discard(c)
                    improved = True
                    continue
                else:
                    # c is not redundant, try to find a swap
                    # Find positions where count == 1 (critical positions)
                    critical_mask = (counts == 1)
                    critical_count = np.sum(critical_mask)
                    
                    # For each candidate replacement d not in cw_set,
                    # check if d covers all critical positions
                    # Score: how many critical positions d covers
                    # We need d to cover ALL critical positions for a valid swap
                    
                    # Get indices of critical positions
                    critical_indices = np.where(critical_mask)[0]
                    
                    if len(critical_indices) > 0:
                        # For each candidate d, check coverage of critical positions
                        # Use M to check: M[d, critical_indices] should all be 1
                        # This is expensive, so only try a subset of candidates
                        
                        # Get candidates not in current set
                        # Instead of iterating all N, focus on promising ones
                        # Candidates that cover many critical positions
                        
                        # Compute coverage score for all candidates on critical positions
                        # M[:, critical_indices].sum(axis=1) gives number of critical positions covered
                        crit_coverage = M[:, critical_indices].sum(axis=1)
                        
                        # Find candidates that cover ALL critical positions
                        all_covered = crit_coverage == len(critical_indices)
                        
                        # Among those, pick the one with highest overall coverage (to enable future removals)
                        if np.any(all_covered):
                            # Get indices of candidates that cover all critical positions
                            candidate_indices = np.where(all_covered)[0]
                            
                            # Exclude current codewords
                            valid_candidates = [d for d in candidate_indices if d not in cw_set]
                            
                            if valid_candidates:
                                # Pick the one with highest total coverage
                                best_d = max(valid_candidates, key=lambda d: M[d].sum())
                                
                                # Perform swap
                                new_codewords[i] = best_d
                                cw_set.discard(c)
                                cw_set.add(best_d)
                                counts[balls[c]] += 1  # Restore c's coverage
                                counts[balls[best_d]] += 1  # Add best_d's coverage
                                improved = True
                                continue
                    else:
                        # No critical positions, c can be removed
                        new_codewords.pop(i)
                        cw_set.discard(c)
                        improved = True
                        continue
                
                # If we get here, c stays
                counts[balls[c]] += 1
            
            codewords = new_codewords
    
    # Convert back to lists of integers
    result = []
    for c in codewords:
        digits = int_to_digits(c, n)
        result.append(digits[::-1])

    return result
