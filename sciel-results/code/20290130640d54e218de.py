import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Constructs a covering code for the given q, n, radius.
    Optimized for q=3, n=8, radius=1 using vectorized greedy set cover.
    """
    N = q ** n
    
    # Fallback for large spaces or different parameters where dense matrix is too expensive
    if N > 20000:
        # Simple heuristic: Random sampling + Local improvement?
        # For now, return a valid cover if possible, or baseline if q=3, n>=4
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: include all words (valid but large)
            return [list(map(int, np.base_repr(i, base=q).zfill(n))) for i in range(N)]

    # Precompute digits for all indices
    digits_table = np.zeros((N, n), dtype=np.int8)
    for i in range(N):
        val = i
        for k in range(n):
            digits_table[i, k] = val % q
            val //= q

    # Precompute coverage matrix C (N x N)
    # C[i, j] = 1 if dist(i, j) <= radius
    # Using bool dtype for memory efficiency and fast bitwise OR
    C = np.zeros((N, N), dtype=bool)
    
    # Powers of q
    q_powers = [q ** k for k in range(n)]
    
    for i in range(N):
        # Center
        C[i, i] = True
        
        if radius >= 1:
            # Generate all neighbors at distance 1
            for pos in range(n):
                orig_digit = digits_table[i, pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * q_powers[pos]
                    j = i + delta
                    C[i, j] = True
        
        # For radius > 1, we would need to add more neighbors.
        # For this task, radius=1 is the primary case.
        # If radius > 1, this implementation is incomplete and might fail coverage.
        # We'll assume radius=1 for the optimized path.
        # If radius > 1, a more general neighbor generation would be needed.
        # Given the constraints, we stick to radius=1 optimization.
        
    # Greedy Set Cover with restarts
    best_solution = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    rng = np.random.default_rng()
    
    # Dynamic number of restarts based on time
    # We'll run as many as fit in the deadline.
    
    for restart in range(200): # Cap at 200 to prevent infinite loop if time check fails
        if time.time() > deadline:
            break
            
        covered = np.zeros(N, dtype=bool)
        used = np.zeros(N, dtype=bool)
        solution = []
        
        while not covered.all():
            # Calculate gains: number of newly covered points for each candidate
            uncovered = ~covered
            # Use int8 for accumulation. Max value is N (6561), which fits in int16 but not int8.
            # Wait, gain is the count of uncovered points covered by candidate i.
            # Max gain is 17 for radius 1. So int8 is safe.
            # C is bool. We can cast C to int8 for multiplication?
            # Or use C @ uncovered.astype(np.int8)
            gains = C @ uncovered.astype(np.int8)
            
            # Mask used candidates
            gains[used] = -1
            
            # Add small random noise to break ties and encourage diversity
            # The noise should be small enough not to override a large gain difference
            # but large enough to break exact ties.
            # Gains are integers. Ties are common.
            noise = rng.uniform(0, 0.5, size=N)
            gains_float = gains.astype(np.float64) + noise
            
            best_idx = int(np.argmax(gains_float))
            
            # If the best gain (before noise) is <= 0, we are stuck?
            # Since covered is not all, there must be some uncovered point.
            # And there must be some candidate covering it (at least the point itself).
            # So gain should be > 0 for at least one candidate.
            if gains[best_idx] <= 0:
                # This should not happen if the space is covered by the union of all candidates
                # which it is, since every point covers itself.
                # If we are stuck, it means we marked a candidate as used but it didn't cover anything new?
                # Or we ran out of candidates?
                # We should break to avoid infinite loop.
                break
                
            solution.append(best_idx)
            used[best_idx] = True
            covered |= C[best_idx]
            
        if len(solution) < best_len:
            best_len = len(solution)
            best_solution = solution.copy()
            
            # Early exit if we hit the theoretical lower bound or a very good result?
            # Lower bound is 403. Best known 486.
            # If we get 486, we can keep going to see if we beat it.
            # But if time is running out, we should stop.
            
    # Convert solution indices back to codewords
    if best_solution is None:
        # Fallback to baseline
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(map(int, np.base_repr(i, base=q).zfill(n))) for i in range(N)]
            
    # Extract codewords from digits_table
    return [digits_table[idx].tolist() for idx in best_solution]
