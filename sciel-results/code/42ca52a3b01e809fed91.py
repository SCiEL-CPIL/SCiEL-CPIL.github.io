import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    N = q ** n
    ball_size = 1 + n * (q - 1)

    # Build the ball table: for each index, the indices of all words within Hamming distance radius
    # For radius 1, this is the word itself plus all single-position changes.
    # We use a vectorized approach for speed.
    
    # Precompute all indices as a 2D array of digits
    all_indices = np.arange(N, dtype=np.int64)
    
    # Compute digits for all indices at once using base-q expansion
    # digits[i][k] = digit k of index i
    digits_all = np.zeros((N, n), dtype=np.int64)
    powers = q ** np.arange(n, dtype=np.int64)
    digits_all = (all_indices[:, None] // powers[None, :]) % q
    
    # Build ball table
    ball_table = np.zeros((N, ball_size), dtype=np.int64)
    ball_table[:, 0] = all_indices
    
    pos = 1
    for k in range(n):
        for new_digit in range(q):
            # Skip if same as original digit
            mask = digits_all[:, k] != new_digit
            if not np.any(mask):
                pos += 1
                continue
            # For each index, if digit k is not new_digit, the neighbor is obtained by
            # changing digit k to new_digit
            # new_val = old_val - old_digit * q^k + new_digit * q^k
            old_digits = digits_all[:, k]
            delta = (new_digit - old_digits) * powers[k]
            new_vals = all_indices + delta
            # Only valid where mask is True
            ball_table[mask, pos] = new_vals[mask]
            pos += 1
    
    # Verify ball_size matches
    assert pos == ball_size, f"Ball table construction error: {pos} vs {ball_size}"

    # Greedy set cover with pruning
    def greedy_cover():
        covered = np.zeros(N, dtype=bool)
        code = []
        
        while not covered.all():
            # Compute gain for each candidate: number of currently uncovered points in its ball
            uncovered_in_ball = (~covered[ball_table]).sum(axis=1)
            
            best_idx = np.argmax(uncovered_in_ball)
            best_gain = uncovered_in_ball[best_idx]
            
            if best_gain == 0:
                break
            
            code.append(int(best_idx))
            covered[ball_table[best_idx]] = True
            
            if time.time() > deadline - 20:
                break
        
        return code

    def try_remove(code, ball_table):
        """Try to remove codewords that are redundant."""
        if not code:
            return code
        
        coverage_counts = np.zeros(N, dtype=np.int64)
        for c in code:
            coverage_counts[ball_table[c]] += 1
        
        changed = True
        new_code = list(code)
        while changed:
            changed = False
            # Try removing each codeword (in order)
            i = 0
            while i < len(new_code):
                c = new_code[i]
                coverage_counts[ball_table[c]] -= 1
                if coverage_counts.min() > 0:
                    # Safe to remove
                    new_code.pop(i)
                    changed = True
                else:
                    coverage_counts[ball_table[c]] += 1
                    i += 1
                if time.time() > deadline - 10:
                    return new_code
        
        return new_code

    # Local search: swap one codeword for another to reduce size
    def local_search_swap(code, ball_table):
        """Try to find a swap that reduces the code size."""
        if not code:
            return code
        
        code_set = set(code)
        
        # Compute coverage counts
        coverage_counts = np.zeros(N, dtype=np.int64)
        for c in code:
            coverage_counts[ball_table[c]] += 1
        
        improved = True
        while improved:
            improved = False
            new_code = list(code)
            
            # Try removing each codeword and see if we can cover the gaps with fewer additions
            for i in range(len(new_code)):
                if time.time() > deadline - 15:
                    break
                
                c = new_code[i]
                # Temporarily remove c
                coverage_counts[ball_table[c]] -= 1
                uncovered = coverage_counts == 0
                
                if not np.any(uncovered):
                    # c was redundant, remove it
                    new_code.pop(i)
                    improved = True
                    continue
                
                # Find best single codeword to add that covers most uncovered
                # Only consider codewords not already in the set
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) == 0:
                    coverage_counts[ball_table[c]] += 1
                    continue
                
                # Score: for each candidate not in code_set, how many uncovered does it cover
                # To be fast, only check a subset of candidates
                candidates = np.array([x for x in range(N) if x not in code_set and x not in new_code])
                if len(candidates) > 500:
                    # Sample
                    candidates = np.random.choice(candidates, 500, replace=False)
                
                gains = np.array([np.isin(ball_table[cand], uncovered_indices).sum() for cand in candidates])
                
                if len(gains) > 0:
                    best_cand_idx = np.argmax(gains)
                    best_gain = gains[best_cand_idx]
                    
                    if best_gain >= len(uncovered_indices):
                        # This candidate covers all uncovered points
                        best_cand = int(candidates[best_cand_idx])
                        coverage_counts[ball_table[best_cand]] += 1
                        new_code.pop(i)
                        new_code.append(best_cand)
                        code_set.discard(c)
                        code_set.add(best_cand)
                        improved = True
                        break
                    else:
                        coverage_counts[ball_table[c]] += 1
                else:
                    coverage_counts[ball_table[c]] += 1
            
            code = new_code
        
        return code

    # Run greedy
    np.random.seed(42)
    best_code = greedy_cover()
    best_code = try_remove(best_code, ball_table)
    best_len = len(best_code)
    
    # Try local search swaps
    if time.time() < deadline - 20:
        best_code = local_search_swap(best_code, ball_table)
        best_len = len(best_code)
        best_code = try_remove(best_code, ball_table)
        best_len = len(best_code)
    
    # Try a few random restarts with different tie-breaking
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Slight perturbation: randomize initial choice among top candidates
        covered = np.zeros(N, dtype=bool)
        code = []
        
        while not covered.all():
            uncovered_in_ball = (~covered[ball_table]).sum(axis=1)
            
            # Get top few candidates
            top_k = min(10, N)
            top_indices = np.argpartition(uncovered_in_ball, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(uncovered_in_ball[top_indices])[::-1]]
            
            # Randomly pick from top candidates with slight preference for higher gain
            weights = uncovered_in_ball[top_indices]
            if weights.max() > 0:
                probs = weights / weights.sum()
            else:
                break
            best_idx = int(np.random.choice(top_indices, p=probs))
            
            if uncovered_in_ball[best_idx] == 0:
                break
            
            code.append(best_idx)
            covered[ball_table[best_idx]] = True
            
            if time.time() > deadline - 20:
                break
        
        code = try_remove(code, ball_table)
        
        if time.time() < deadline - 15:
            code = local_search_swap(code, ball_table)
            code = try_remove(code, ball_table)
        
        if len(code) < best_len:
            best_code = code
            best_len = len(code)
    
    # Final polish: try to remove more
    if time.time() < deadline - 5:
        best_code = try_remove(best_code, ball_table)
        best_len = len(best_code)

    # Convert to list of digit lists
    result = []
    for idx in best_code:
        digits = []
        val = int(idx)
        for _ in range(n):
            digits.append(val % q)
            val //= q
        result.append(digits)
    
    return result
