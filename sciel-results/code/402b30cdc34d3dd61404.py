"""Combined program for ternary covering code K_3(7,1) using optimized greedy set cover + fast local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Uses a greedy set-cover approach followed by local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    # Fallback: direct sum construction for general case
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    """Direct sum construction as fallback."""
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    # For q=3, use tetracode-based construction
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        # General fallback: all words
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    """Solve K_3(7,1) using greedy set cover + local search."""
    q, n, radius = 3, 7, 1
    
    # 1. Precompute all words and their balls (neighbors within radius 1)
    all_words = list(product(range(3), repeat=7))
    N = len(all_words) # 2187
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls: for each index, list of indices covered by that codeword
    # Ball(c) = {c} U {words at dist 1}
    balls = []
    for w in all_words:
        covered_indices = [word_to_idx[w]]
        for i in range(7):
            for val in range(3):
                if val != w[i]:
                    new_word = list(w)
                    new_word[i] = val
                    new_word = tuple(new_word)
                    covered_indices.append(word_to_idx[new_word])
        balls.append(covered_indices)
    
    # 2. Greedy Set Cover to get initial solution
    covered = np.zeros(N, dtype=bool)
    codewords = []
    
    # Use numpy vectorization for the greedy step to be fast enough
    # We iterate until covered.all() or time runs out
    while not covered.all():
        # Find the candidate that covers the most uncovered points
        # new_cov[i] = sum of (~covered[balls[i]])
        # This is a matrix-vector multiplication if we represent balls as a matrix,
        # but balls are lists.
        
        # Efficient way: 
        # Create a matrix M where M[i, j] = 1 if j in balls[i]
        # Then new_cov = M @ (~covered)
        # But building M is 2187x2187 which is ~4.7M entries. Memory is fine.
        # However, we only need to do this once? No, covered changes.
        
        # Alternative: just loop.
        # N=2187. 15 neighbors.
        # Looping 2187 * 15 = 32k operations per step.
        # If we take 200 steps, that's 6.5M operations.
        # Python can do ~10M-50M simple ops/sec. So 0.1s - 0.6s per step?
        # No, 32k ops is very fast. 6.5M total is < 1s.
        # So we can just scan all candidates every time.
        
        best_cw = -1
        best_count = -1
        
        # Vectorized approach for counting:
        # We can precompute the adjacency matrix A where A[i,j]=1 if j in ball(i)
        # Then counts = A @ (~covered)
        
        # Let's build A once.
        # A is 2187 x 2187. 
        # Memory: 2187*2187 bytes ~ 4.7MB if uint8. Fine.
        
        # But we haven't built it yet in this function scope in the previous draft.
        # Let's build it now.
        
        if not hasattr(_solve_ternary_7_1, 'A'):
            A = np.zeros((N, N), dtype=np.uint8)
            for i, b in enumerate(balls):
                A[i, b] = 1
            _solve_ternary_7_1.A = A
            
        A = _solve_ternary_7_1.A
        
        # Calculate new coverage for all candidates
        # uncovered = ~covered
        # counts = A @ uncovered
        # Since A is uint8 and uncovered is bool, result is int
        counts = A @ (~covered)
        
        best_count = np.max(counts)
        best_cw = np.argmax(counts)
        
        if best_count == 0:
            break
            
        codewords.append(best_cw)
        covered[balls[best_cw]] = True
        
        if time.time() > deadline - 30:
            break

    if not covered.all():
        # Fallback to finish quickly if greedy was cut short
        while not covered.all():
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            cw = random.choice(uncovered_indices.tolist())
            codewords.append(cw)
            covered[balls[cw]] = True
            if time.time() > deadline - 5:
                break

    # 3. Initialize coverage counts for Local Search
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        coverage_count[balls[cw]] += 1
        
    cnt0 = np.sum(coverage_count == 0)
    
    # Helper to remove codeword
    def try_remove(cw_idx):
        nonlocal cnt0
        b = balls[cw_idx]
        # Check if any word in b has count 1
        if np.min(coverage_count[b]) == 1:
            return False
        # Safe to remove
        coverage_count[b] -= 1
        new_zeros = np.sum(coverage_count[b] == 0)
        cnt0 += new_zeros
        return True

    # 4. Redundancy Removal (Greedy Pass)
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1
        if time.time() > deadline - 20:
            break

    # 5. Local Search
    # Strategy: Random Removals and Swaps
    random.seed(42)
    
    # Precompute balls as numpy arrays for faster indexing? 
    # balls is already list of lists. 
    # We can convert to a list of numpy arrays for faster slice assignment?
    # Or just use the lists.
    
    # Optimization: Precompute ball arrays
    ball_arrays = [np.array(b, dtype=np.int32) for b in balls]
    
    while time.time() < deadline - 2:
        # Try to remove a random codeword
        if codewords:
            idx_to_try = random.randint(0, len(codewords) - 1)
            cw = codewords[idx_to_try]
            if try_remove(cw):
                codewords.pop(idx_to_try)
                continue
            
        # Try to swap
        if codewords:
            idx_old = random.randint(0, len(codewords) - 1)
            cw_old = codewords[idx_old]
            
            # Pick a random new codeword
            cw_new = random.randint(0, N - 1)
            
            if cw_new == cw_old:
                continue
            
            b_old = ball_arrays[cw_old]
            b_new = ball_arrays[cw_new]
            
            # Simulate swap
            coverage_count[b_old] -= 1
            coverage_count[b_new] += 1
            
            # Check validity: min(coverage_count) >= 1
            if np.min(coverage_count) < 1:
                # Invalid, revert
                coverage_count[b_old] += 1
                coverage_count[b_new] -= 1
            else:
                # Valid swap
                codewords[idx_old] = cw_new
                
    # Convert indices back to lists of integers
    result = [list(all_words[cw]) for cw in codewords]
    return result
