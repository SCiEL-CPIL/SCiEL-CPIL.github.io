"""Search-based solver for QAPLIB wil100: Fully Vectorized Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """Create an initial permutation using rank-matching heuristic."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow strength
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance centrality (lower sum is better)
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _vectorized_sa(a, b, p, time_limit, seed=42):
    """
    Perform fully vectorized Simulated Annealing.
    At each step, compute the cost delta for swapping ANY pair (i, j) in the current permutation.
    This allows evaluating O(n^2) moves per iteration, vastly increasing the effective step count.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    # Precompute B matrix indexed by p for fast access
    # B_p[i, j] = b[p[i], p[j]]
    B_p = b[np.ix_(p, p)]
    
    # Precompute A matrix (static)
    # A[i, j] = a[i, j]
    
    # We need to compute delta for swapping positions i and j.
    # Let x = p[i], y = p[j].
    # New cost = Old cost - (terms involving i or j with old positions) + (terms with new positions)
    # Delta = sum_{k} a[i,k] (B_p[j,k] - B_p[i,k]) + a[k,i] (B_p[k,j] - B_p[k,i])
    #        + sum_{k} a[j,k] (B_p[i,k] - B_p[j,k]) + a[k,j] (B_p[k,i] - B_p[k,j])
    #        + a[i,j] (B_p[i,j] - B_p[j,i]) + a[j,i] (B_p[j,i] - B_p[i,j])
    #        + a[i,i] (B_p[j,j] - B_p[i,i]) + a[j,j] (B_p[i,i] - B_p[j,j])
    
    # Vectorized calculation of all pairwise deltas:
    # Let D[i, j] be the delta for swapping i and j.
    # We can compute this using matrix multiplications.
    
    # Term 1: sum_k a[i,k] (B_p[j,k] - B_p[i,k])
    # This is (A @ (B_p.T - B_p.T))[i, j]? No.
    # Let's look at it as:
    # Delta[i, j] = sum_k (a[i,k] - a[j,k]) (B_p[j,k] - B_p[i,k]) 
    #             + sum_k (a[k,i] - a[k,j]) (B_p[k,j] - B_p[k,i])
    #             + a[i,j] (B_p[i,j] - B_p[j,i]) + a[j,i] (B_p[j,i] - B_p[i,j])
    #             + a[i,i] (B_p[j,j] - B_p[i,i]) + a[j,j] (B_p[i,i] - B_p[j,j])
    
    # Note: The first two sums cover all k.
    # Let's define matrices to facilitate this.
    
    # For a fixed swap (i, j):
    # Row part: sum_k (a[i,k] - a[j,k]) * (B_p[j,k] - B_p[i,k])
    # Col part: sum_k (a[k,i] - a[k,j]) * (B_p[k,j] - B_p[k,i])
    
    # We can compute the full matrix of deltas for all i, j at once.
    # Let R[i, k] = a[i, k]
    # Let C[k, i] = a[k, i]
    
    # This is computationally heavy (O(n^3) per step if done naively, but we can use BLAS).
    # Actually, let's simplify.
    # Delta[i, j] = 
    #   a[i, :] @ (B_p[j, :] - B_p[i, :]) 
    # - a[j, :] @ (B_p[j, :] - B_p[i, :]) 
    # + a[:, i] @ (B_p[:, j] - B_p[:, i])
    # - a[:, j] @ (B_p[:, j] - B_p[:, i])
    # + a[i, j] * (B_p[i, j] - B_p[j, i]) + a[j, i] * (B_p[j, i] - B_p[i, j])
    # + a[i, i] * (B_p[j, j] - B_p[i, i]) + a[j, j] * (B_p[i, i] - B_p[j, j])
    
    # This is still O(n^3) if we do it for all pairs.
    # However, n=100. n^3 = 1,000,000 operations.
    # NumPy can do 1M ops in ~1ms.
    # So we can do ~1000 steps per second? No, 1M ops is very fast.
    # Let's try to vectorize the "all pairs" calculation.
    
    # Let's compute Delta matrix of shape (n, n).
    # Delta[i, j] for i != j.
    
    # Part 1: sum_k (a[i,k] - a[j,k]) * (B_p[j,k] - B_p[i,k])
    # Let's expand:
    # sum_k a[i,k]B_p[j,k] - sum_k a[i,k]B_p[i,k] - sum_k a[j,k]B_p[j,k] + sum_k a[j,k]B_p[i,k]
    
    # Let M1 = A @ B_p.T  (Shape n x n). M1[i, j] = sum_k a[i,k] B_p[j,k]
    # Let M2 = A @ B_p    (Shape n x n). M2[i, j] = sum_k a[i,k] B_p[k,j] -- Wait, B_p is symmetric? No.
    # B_p[i, k] = b[p[i], p[k]].
    # sum_k a[i,k] B_p[i,k] is the i-th diagonal element of A @ B_p? No.
    # (A @ B_p)[i, k] = sum_m a[i,m] B_p[m,k].
    # sum_k a[i,k] B_p[i,k] is not a simple matrix product element unless we are careful.
    
    # Let's use the explicit loop for the O(n^2) pair selection but vectorize the inner O(n) sum?
    # Or just compute the Delta matrix using broadcasting?
    # n=100.
    # Delta[i, j] = ...
    
    # Let's stick to a simpler vectorized approach:
    # Compute the cost change for swapping i and j using precomputed vectors.
    # For each i, precompute:
    # v_i_row = a[i, :]
    # v_i_col = a[:, i]
    # w_i_row = B_p[i, :]
    # w_i_col = B_p[:, i]
    
    # Delta[i, j] = 
    #   v_i_row @ (w_j_row - w_i_row) 
    # - v_j_row @ (w_j_row - w_i_row) 
    # + v_i_col @ (w_j_col - w_i_col)
    # - v_j_col @ (w_j_col - w_i_col)
    # + a[i, j] * (B_p[i, j] - B_p[j, i]) + a[j, i] * (B_p[j, i] - B_p[i, j])
    # + a[i, i] * (B_p[j, j] - B_p[i, i]) + a[j, j] * (B_p[i, i] - B_p[j, j])
    
    # This can be computed for all i, j using matrix operations.
    # Let's define:
    # D1[i, j] = v_i_row @ (w_j_row - w_i_row)
    #            = v_i_row @ w_j_row - v_i_row @ w_i_row
    #            = (A @ B_p.T)[i, j] - (A @ B_p)[i, i] -- Wait.
    #            v_i_row @ w_j_row = sum_k a[i,k] B_p[j,k] = (A @ B_p.T)[i, j]
    #            v_i_row @ w_i_row = sum_k a[i,k] B_p[i,k] = (A * B_p)[i, :] sum? No.
    #            It is the i-th element of the row-wise dot product of A and B_p.
    #            Let S_A_B = np.sum(A * B_p, axis=1)  # Shape n
    #            Then v_i_row @ w_i_row = S_A_B[i]
            
    # So D1[i, j] = (A @ B_p.T)[i, j] - S_A_B[i]
    
    # Similarly for the second term: - v_j_row @ (w_j_row - w_i_row)
    #            = - (v_j_row @ w_j_row - v_j_row @ w_i_row)
    #            = - (S_A_B[j] - (A @ B_p.T)[j, i])
    #            = (A @ B_p.T)[j, i] - S_A_B[j]
    
    # So Row Part = D1[i, j] + (A @ B_p.T)[j, i] - S_A_B[i] - S_A_B[j]
    
    # Now Col Part:
    # v_i_col @ (w_j_col - w_i_col)
    # v_i_col is column i of A. w_j_col is column j of B_p.
    # sum_k a[k,i] B_p[k,j] = (A.T @ B_p)[i, j]
    # sum_k a[k,i] B_p[k,i] = S_A_T_B[i] where S_A_T_B = np.sum(A.T * B_p, axis=1)?
    # Let's check: (A.T * B_p)[k, i] = a[i,k] B_p[k,i]. Sum over k gives sum_k a[i,k] B_p[k,i].
    # So S_A_T_B[i] = np.sum(A * B_p, axis=0)[i]? No.
    # sum_k a[k,i] B_p[k,i] is the i-th element of the column-wise dot product of A and B_p.
    # Let S_A_B_col = np.sum(A * B_p, axis=0)
    # Then sum_k a[k,i] B_p[k,i] = S_A_B_col[i]
    
    # So Col Part = (A.T @ B_p)[i, j] - S_A_B_col[i] - (S_A_B_col[j] - (A.T @ B_p)[j, i])
    #             = (A.T @ B_p)[i, j] + (A.T @ B_p)[j, i] - S_A_B_col[i] - S_A_B_col[j]
    
    # Diagonal Terms:
    # T_diag = a[i, j] * (B_p[i, j] - B_p[j, i]) + a[j, i] * (B_p[j, i] - B_p[i, j])
    #        + a[i, i] * (B_p[j, j] - B_p[i, i]) + a[j, j] * (B_p[i, i] - B_p[j, j])
    
    # We can compute this matrix T of shape (n, n) using broadcasting.
    
    # Total Delta[i, j] = Row Part + Col Part + T_diag[i, j]
    
    # Let's precompute the constant matrices.
    M_AB_T = A @ B_p.T
    M_AT_B = A.T @ B_p
    S_AB_row = np.sum(A * B_p, axis=1)
    S_AB_col = np.sum(A * B_p, axis=0)
    
    # Row Part Matrix R[i, j] = M_AB_T[i, j] + M_AB_T[j, i] - S_AB_row[i] - S_AB_row[j]
    # Note: M_AB_T[j, i] is the transpose.
    R = M_AB_T + M_AB_T.T - S_AB_row[:, None] - S_AB_row[None, :]
    
    # Col Part Matrix C[i, j] = M_AT_B[i, j] + M_AT_B[j, i] - S_AB_col[i] - S_AB_col[j]
    C = M_AT_B + M_AT_B.T - S_AB_col[:, None] - S_AB_col[None, :]
    
    # Diagonal Part T
    # T[i, j] = a[i, j] * (B_p[i, j] - B_p[j, i]) + a[j, i] * (B_p[j, i] - B_p[i, j])
    #        + a[i, i] * (B_p[j, j] - B_p[i, i]) + a[j, j] * (B_p[i, i] - B_p[j, j])
    
    # Let's build this.
    # Term 1: a * (B_p - B_p.T)
    # Term 2: a.T * (B_p.T - B_p) = -a.T * (B_p - B_p.T)
    # So T12 = (a - a.T) * (B_p - B_p.T)
    
    # Term 3: a[i, i] * (B_p[j, j] - B_p[i, i])
    # Let diag_A = np.diag(a)
    # Let diag_Bp = np.diag(B_p)
    # T34[i, j] = diag_A[i] * (diag_Bp[j] - diag_Bp[i]) + diag_A[j] * (diag_Bp[i] - diag_Bp[j])
    #           = (diag_A[i] - diag_A[j]) * (diag_Bp[j] - diag_Bp[i])
    #           = -(diag_A[i] - diag_A[j]) * (diag_Bp[i] - diag_Bp[j])
    # Let's use broadcasting:
    # T34 = (diag_A[:, None] - diag_A[None, :]) * (diag_Bp[None, :] - diag_Bp[:, None])
    # Wait:
    # diag_A[i] * diag_Bp[j] - diag_A[i] * diag_Bp[i] + diag_A[j] * diag_Bp[i] - diag_A[j] * diag_Bp[j]
    # = (diag_A[i] - diag_A[j]) * diag_Bp[j] - (diag_A[i] - diag_A[j]) * diag_Bp[i]
    # = (diag_A[i] - diag_A[j]) * (diag_Bp[j] - diag_Bp[i])
    
    T12 = (a - a.T) * (B_p - B_p.T)
    diag_A = np.diag(a)
    diag_Bp = np.diag(B_p)
    T34 = (diag_A[:, None] - diag_A[None, :]) * (diag_Bp[None, :] - diag_Bp[:, None])
    
    T = T12 + T34
    
    Delta = R + C + T
    
    # Zero out diagonal (no swap) and set to inf to ignore
    np.fill_diagonal(Delta, np.inf)
    
    # Now we have the Delta matrix.
    # We perform SA:
    # 1. Pick a random pair (i, j) from the upper triangle? Or just any i, j?
    #    Usually SA picks a random move. Here, "move" is a swap.
    #    We can pick a random i, j.
    #    Or, to be more efficient, we can pick the best move? That's Greedy.
    #    SA: Accept with probability exp(-Delta/T).
    
    # To maximize speed, we can evaluate ALL moves and pick one?
    # If we pick the best move, it's Greedy Hill Climbing.
    # If we pick a random move, we need to compute Delta for that move.
    # But we already computed ALL Deltas!
    # So we can use this information.
    
    # Strategy: 
    # 1. Compute Delta matrix.
    # 2. Find the best move (min Delta). If Delta < 0, accept it.
    # 3. If no improving move, pick a random move and accept with prob exp(-Delta/T).
    #    Since we have all Deltas, we can sample from the distribution?
    #    Or just pick a random pair (i, j) and check its Delta.
    
    # Let's do:
    # 1. Find best_delta, best_i, best_j.
    # 2. If best_delta < 0, accept.
    # 3. Else, pick random i, j. If accept, swap.
    
    # This is a hybrid.
    # Actually, pure vectorized SA usually means:
    # - Compute Delta for a specific random move? No, that's O(n).
    # - Compute Delta for ALL moves? O(n^3) per step.
    # With n=100, O(n^3) is 1e6 ops. NumPy does this in ~1-2ms.
    # So we can do ~500-1000 steps per second.
    # In 100s, we can do 50,000 - 100,000 steps.
    # This is much more than the sequential version which does ~10-100 steps/sec.
    
    # Let's implement this loop.
    
    t_start = time.time()
    t_end = t_start + time_limit
    
    # Cooling schedule
    t0 = 1000.0
    t_final = 0.1
    
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute constant parts of Delta that don't depend on p?
    # No, B_p changes every step. So we must recompute M_AB_T, M_AT_B, etc. every step.
    # This is the bottleneck.
    
    # Optimization:
    # Only a few rows/cols of B_p change when we swap i, j.
    # But updating the matrix products incrementally is complex.
    # Given n=100, full recomputation is likely fast enough.
    
    while time.time() < t_end:
        # Compute Delta
        B_p = b[np.ix_(p, p)]
        
        M_AB_T = a @ B_p.T
        M_AT_B = a.T @ B_p
        S_AB_row = np.sum(a * B_p, axis=1)
        S_AB_col = np.sum(a * B_p, axis=0)
        
        R = M_AB_T + M_AB_T.T - S_AB_row[:, None] - S_AB_row[None, :]
        C = M_AT_B + M_AT_B.T - S_AB_col[:, None] - S_AB_col[None, :]
        
        T12 = (a - a.T) * (B_p - B_p.T)
        diag_A = np.diag(a)
        diag_Bp = np.diag(B_p)
        T34 = (diag_A[:, None] - diag_A[None, :]) * (diag_Bp[None, :] - diag_Bp[:, None])
        
        Delta = R + C + T12 + T34
        np.fill_diagonal(Delta, np.inf)
        
        # Find best move
        min_idx = np.unravel_index(np.argmin(Delta), Delta.shape)
        best_delta = Delta[min_idx]
        
        # Temperature
        elapsed = time.time() - t_start
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac
        
        if best_delta < 0:
            # Accept best move
            i, j = min_idx
        else:
            # Pick a random move
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = Delta[i, j]
            
            if np.random.random() < np.exp(-delta / max(temp, 1e-10)):
                # Accept
                pass
            else:
                continue
        
        # Perform swap
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta if best_delta < 0 else Delta[i, j]
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 110.0  # Safety margin
    start = time.time()
    
    np.random.seed(42)
    
    # Initial solution
    p = _greedy_initial(a, b)
    
    # Run vectorized SA
    # Split time into chunks to allow for multiple restarts if time permits?
    # Or just one long run?
    # Vectorized SA is fast, so we can do many steps.
    # Let's do one long run.
    
    time_remaining = deadline - (time.time() - start)
    p_best, cost_best = _vectorized_sa(a, b, p, time_remaining)
    
    # Check if we have time for a second run with different seed?
    time_elapsed = time.time() - start
    if time_elapsed < deadline - 10:
        # Restart with perturbed solution
        p2 = p_best.copy()
        # Perturb
        for _ in range(10):
            i, j = np.random.choice(n, 2, replace=False)
            p2[i], p2[j] = p2[j], p2[i]
        
        p_best2, cost_best2 = _vectorized_sa(a, b, p2, deadline - time_elapsed - 5)
        if cost_best2 < cost_best:
            p_best = p_best2
            cost_best = cost_best2
            
    return p_best.tolist()

# EVOLVE-BLOCK-END
