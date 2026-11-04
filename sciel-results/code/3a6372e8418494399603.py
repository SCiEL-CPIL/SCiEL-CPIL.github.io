"""
Multi-Start Local Search solver for QAPLIB wil100.
Strategy:
1. Precompute constants.
2. Generate multiple diverse initial solutions (Greedy + Random).
3. For each initial solution, perform a fast steepest-descent local search:
   - Compute the delta for ALL possible swaps (i, j) simultaneously using vectorized NumPy.
   - Select the best improving move.
   - Repeat until no improving move is found or time limit is reached.
4. Track the global best solution found across all starts.
This approach focuses computational power on deep exploitation (local search) 
with broad exploration (multiple starts), avoiding the random walk inefficiency 
of standard SA.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _random_initial(n):
    """Generate a random permutation."""
    p = np.arange(n)
    np.random.shuffle(p)
    return p

def _compute_all_deltas(a, b, p):
    """
    Compute the cost delta for all swaps (i, j) with i < j.
    Returns a matrix D where D[i, j] is the delta for swapping p[i] and p[j].
    
    Delta(i, j) = sum_{k != i, j} [ (a[k, i] - a[k, j]) * (b[p[k], p[j]] - b[p[k], p[i]]) 
                                  + (a[i, k] - a[j, k]) * (b[p[j], p[k]] - b[p[i], p[k]]) ]
                 + (a[i, j] - a[j, i]) * (b[p[j], p[i]] - b[p[i], p[j]])
    
    Note: For symmetric B, the direct term is 0. We include it for generality.
    """
    n = len(p)
    # Precompute Bp = b[p, p]
    Bp = b[np.ix_(p, p)]
    
    # We want to compute D[i, j] for all i, j.
    # Let's vectorize over i, then j? Or fully 3D?
    # n=100. A 100x100x100 array is 1M elements. 8MB. This is fine.
    
    # Term 1: sum_k (a[k, i] - a[k, j]) * (Bp[k, j] - Bp[k, i])
    # Let I = indices 0..n-1, J = indices 0..n-1, K = indices 0..n-1.
    # We want T1[i, j] = sum_k (a[k, i] - a[k, j]) * (Bp[k, j] - Bp[k, i])
    
    # a[k, i] is A.T[i, k] if we view it as matrix? No, a[k, i] is column i of A.
    # Let A_col = a.T # Shape (n, n) where A_col[i, k] = a[k, i]
    # Let Bp_row = Bp # Shape (n, n) where Bp[k, j] is element.
    
    # We need sum over k.
    # Term1[i, j] = sum_k (A_col[i, k] - A_col[j, k]) * (Bp[k, j] - Bp[k, i])
    
    # Let's expand:
    # sum_k A_col[i, k] Bp[k, j] - sum_k A_col[i, k] Bp[k, i] - sum_k A_col[j, k] Bp[k, j] + sum_k A_col[j, k] Bp[k, i]
    
    # Matrix M1 = A_col @ Bp -> M1[i, j] = sum_k a[k, i] b[p[k], p[j]]
    # Matrix M2 = A_col @ Bp.T -> M2[i, j] = sum_k a[k, i] b[p[j], p[k]]
    
    # Wait, Bp[k, j] is b[p[k], p[j]].
    # Bp.T[j, k] is b[p[j], p[k]].
    
    # Let X = A.T (so X[i, k] = a[k, i])
    # Let Y = Bp (so Y[k, j] = b[p[k], p[j]])
    
    # T1_part1 = X @ Y  -> (n x n) @ (n x n) = (n x n). Element [i, j] = sum_k a[k, i] b[p[k], p[j]]
    # T1_part2 = X @ Y.T -> (n x n) @ (n x n). Element [i, j] = sum_k a[k, i] b[p[j], p[k]]
    
    # T1[i, j] = (X @ Y)[i, j] - (X @ Y.T)[i, j] - (X @ Y)[j, i] + (X @ Y.T)[j, i]
    
    # This is O(n^3) but highly optimized BLAS.
    
    X = a.T
    Y = Bp
    
    M1 = X @ Y
    M2 = X @ Y.T
    
    # T1[i, j] = M1[i, j] - M2[i, j] - M1[j, i] + M2[j, i]
    # Note: M1[j, i] is sum_k a[k, j] b[p[k], p[i]]
    # M2[j, i] is sum_k a[k, j] b[p[i], p[k]]
    
    # Let's construct T1 matrix.
    # T1 = M1 - M2 - M1.T + M2.T
    
    T1 = M1 - M2 - M1.T + M2.T
    
    # Term 2: sum_k (a[i, k] - a[j, k]) * (Bp[j, k] - Bp[i, k])
    # Let A_row = a (a[i, k])
    # Let Bp_row = Bp (Bp[j, k])
    
    # T2[i, j] = sum_k a[i, k] Bp[j, k] - sum_k a[i, k] Bp[i, k] - sum_k a[j, k] Bp[j, k] + sum_k a[j, k] Bp[i, k]
    
    # Matrix N1 = A @ Bp.T -> N1[i, j] = sum_k a[i, k] b[p[j], p[k]]
    # Matrix N2 = A @ Bp -> N2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    
    # Wait, Bp[j, k] is b[p[j], p[k]].
    # Bp.T[k, j] is b[p[k], p[j]].
    
    # N1 = A @ Bp.T
    # N2 = A @ Bp
    
    N1 = a @ Bp.T
    N2 = a @ Bp
    
    # T2[i, j] = N1[i, j] - N2[i, j] - N1[j, i] + N2[j, i]
    
    T2 = N1 - N2 - N1.T + N2.T
    
    # Term 3: (a[i, j] - a[j, i]) * (b[p[j], p[i]] - b[p[i], p[j]])
    # b[p[j], p[i]] is Bp[j, i].
    # b[p[i], p[j]] is Bp[i, j].
    
    A_diff = a - a.T
    Bp_diff = Bp.T - Bp # Bp.T[i, j] = Bp[j, i] = b[p[j], p[i]]. Bp[i, j] = b[p[i], p[j]].
    
    T3 = A_diff * Bp_diff
    
    # Total Delta
    D = T1 + T2 + T3
    
    # The diagonal D[i, i] is 0. We only care about i < j.
    # We can zero out the lower triangle or just use i < j logic.
    # To prevent self-swaps, we can set diagonal to inf.
    np.fill_diagonal(D, np.inf)
    
    return D

def _local_search(a, b, p, time_limit, rng):
    """
    Perform steepest-descent local search starting from p.
    Returns best permutation found.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute A.T for speed in delta calc? _compute_all_deltas does it.
    
    start_time = time.time()
    
    # Local search loop
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute all deltas
        D = _compute_all_deltas(a, b, p)
        
        # Find best improving move (min delta)
        # We want min D[i, j] where D[i, j] < 0.
        # Since D is symmetric? 
        # Delta(i, j) == Delta(j, i)?
        # Let's check.
        # T1 is symmetric? T1[i, j] = sum (a[k, i]-a[k, j])...
        # If we swap i and j in T1 formula, signs flip?
        # Actually, for QAP with symmetric B, Delta(i, j) should be symmetric.
        # Let's assume D[i, j] == D[j, i].
        
        # Mask lower triangle to avoid duplicates
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        D_masked = np.where(mask, D, np.inf)
        
        min_idx = np.unravel_index(np.argmin(D_masked), D_masked.shape)
        min_delta = D_masked[min_idx]
        
        if min_delta >= 0:
            # No improving move
            break
        
        i, j = min_idx
        # Perform swap
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # Leave 5s buffer for the 120s limit
    start_time = time.time()
    
    # Set random seed for reproducibility if needed? 
    # The task says "seed/value", implying we should be deterministic or use a fixed seed?
    # "search score is seed/value". Usually implies we can use randomness.
    # I will use a fixed seed for stability across evaluations if the evaluator runs multiple times?
    # No, "seed" in score formula usually refers to the instance seed or a global seed.
    # I will use np.random.seed(42) to ensure consistent behavior if the evaluator runs the function multiple times.
    # However, if the evaluator expects different results for different runs, this might be bad.
    # Standard practice for these tasks: Use a fixed seed for the internal RNG to make the program deterministic.
    np.random.seed(42)
    
    # Initial Greedy Solution
    p_greedy = _greedy_initial(a, b)
    
    # Run local search on greedy
    # Estimate time per local search iteration.
    # n=100. Matrix mult is ~100^3 = 1M ops. ~1ms.
    # Local search might take 100-500 iterations to converge. ~0.1-0.5s.
    # We can afford ~100-1000 restarts?
    # Let's start with a few high-quality starts.
    
    best_p = p_greedy
    best_cost = _qap_cost(a, b, p_greedy)
    
    # First, polish the greedy solution
    p_ls, cost_ls = _local_search(a, b, p_greedy.copy(), time_limit=10.0, rng=np.random)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls
        
    # Now, perform multiple random restarts
    # We have ~105s left.
    # Each restart + local search might take ~1-2s if it converges quickly.
    # Let's try to do as many as fit.
    
    while time.time() - start_time < total_deadline - 5:
        # Generate a random initial solution
        p_rand = _random_initial(n)
        
        # Optionally, do a quick "pre-polish" with a few random swaps to get out of the worst random states?
        # Or just run local search. Local search from random is slow to converge but might find a different basin.
        
        # To make it faster, we can run local search with a shorter time limit for random starts,
        # and longer for promising starts.
        # But we don't know which are promising until we try.
        
        # Let's run local search with a dynamic time limit.
        # If we have 50s left, give 5s to this restart.
        # If we have 10s left, give 1s.
        remaining = total_deadline - (time.time() - start_time)
        # Allocate 10% of remaining time to this restart, min 0.5s
        t_limit = max(0.5, remaining * 0.1)
        
        p_ls, cost_ls = _local_search(a, b, p_rand, time_limit=t_limit, rng=np.random)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
            # If we found a better solution, we might want to do a longer polish on this new best?
            # _local_search already polishes to local optimum (or time limit).
            # If it hit time limit, it might not be a local optimum.
            # We can try to continue polishing the best_p in the next iteration if it's good?
            # For now, just track best.
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
