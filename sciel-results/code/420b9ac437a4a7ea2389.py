"""
Multi-start Iterated Local Search (ILS) for QAPLIB wil100.
Strategy:
1. Vectorized O(n^2) computation of cost deltas for ALL possible 2-opt swaps.
   This allows a very fast steepest-descent local search (finding the best swap 
   in O(n^2) time instead of O(n^3)).
2. Perturbation: Randomly swap a small number of pairs to escape local optima.
3. Multi-start: Run multiple ILS iterations with different random seeds/perturbations.
4. Keep the best solution found across all starts.

Key structural change from parent:
- Parent used Simulated Annealing (random moves, Metropolis acceptance).
- This uses Iterated Local Search (steepest-descent local search + perturbation).
- ILS is often more effective for QAP because it systematically improves the solution
  before perturbing, rather than accepting random uphill moves.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_all_swap_deltas(a, b, p):
    """
    Compute the cost delta for ALL possible swaps (i, j) with i < j.
    Returns a matrix D of shape (n, n) where D[i, j] is the delta for swapping positions i and j.
    For i == j, D[i, i] = 0.
    For i > j, D[i, j] = D[j, i] (symmetric).
    
    The delta for swapping positions i and j (letting x = p[i], y = p[j]) is:
    delta = sum_{k != i, j} (a[i,k] - a[j,k]) * (b[y, p[k]] - b[x, p[k]])
          + sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], y] - b[p[k], x])
          + (a[i,j] - a[j,i]) * (b[y,x] - b[x,y])
    
    We can vectorize this computation.
    """
    n = len(p)
    D = np.zeros((n, n), dtype=np.int64)
    
    # Precompute Bp = b[np.ix_(p, p)] for faster access
    Bp = b[np.ix_(p, p)]
    
    # For each pair (i, j), we need:
    # Term 1: sum_{k != i, j} (a[i,k] - a[j,k]) * (Bp[j, k] - Bp[i, k])
    # Term 2: sum_{k != i, j} (a[k,i] - a[k,j]) * (Bp[k, j] - Bp[k, i])
    # Term 3: (a[i,j] - a[j,i]) * (Bp[j, i] - Bp[i, j])
    
    # Let's compute this efficiently using broadcasting.
    # For fixed i, we want to compute deltas for all j > i.
    
    for i in range(n):
        # a[i, :] - a[j, :] for all j -> shape (n, n) but we only need j > i
        # Let's compute for all j, then mask.
        
        # Term 1 vectorized over j:
        # For each j, sum_k (a[i,k] - a[j,k]) * (Bp[j,k] - Bp[i,k])
        # = sum_k a[i,k]*Bp[j,k] - sum_k a[i,k]*Bp[i,k] - sum_k a[j,k]*Bp[j,k] + sum_k a[j,k]*Bp[i,k]
        # This doesn't simplify nicely. Let's just do the direct computation.
        
        # For fixed i, compute delta for all j in one go.
        # We need to exclude k=i and k=j from the sums.
        
        # Let's compute the full sum over all k, then subtract the k=i and k=j terms.
        
        # Term 1 full sum over k:
        # S1[j] = sum_k (a[i,k] - a[j,k]) * (Bp[j,k] - Bp[i,k])
        #       = sum_k (a[i,k] - a[j,k]) * Bp[j,k] - sum_k (a[i,k] - a[j,k]) * Bp[i,k]
        #       = (a[i,:] @ Bp[j,:]) - (a[j,:] @ Bp[j,:]) - (a[i,:] @ Bp[i,:]) + (a[j,:] @ Bp[i,:])
        
        # Let A = a, B = Bp.
        # S1[j] = A[i,:] @ B[j,:] - A[j,:] @ B[j,:] - A[i,:] @ B[i,:] + A[j,:] @ B[i,:]
        
        # Precompute:
        # A_i_B_i = A[i,:] @ B[i,:] (scalar)
        # For each j:
        #   A_i_B_j = A[i,:] @ B[j,:]
        #   A_j_B_j = A[j,:] @ B[j,:]
        #   A_j_B_i = A[j,:] @ B[i,:]
        # S1[j] = A_i_B_j - A_j_B_j - A_i_B_i + A_j_B_i
        
        # Similarly for Term 2:
        # S2[j] = sum_k (A[k,i] - A[k,j]) * (B[k,j] - B[k,i])
        #       = sum_k A[k,i]*B[k,j] - sum_k A[k,i]*B[k,i] - sum_k A[k,j]*B[k,j] + sum_k A[k,j]*B[k,i]
        #       = (A[:,i] @ B[:,j]) - (A[:,i] @ B[:,i]) - (A[:,j] @ B[:,j]) + (A[:,j] @ B[:,i])
        
        # Let's precompute the necessary dot products.
        
        # For Term 1:
        # A_i_B = A[i,:] @ B  -> shape (n,) where element j is A[i,:] @ B[j,:]
        # A_B_i = A @ B[i,:]  -> shape (n,) where element j is A[j,:] @ B[i,:]
        # A_B_j_diag = A[j,:] @ B[j,:] for each j -> this is the diagonal of A @ B
        
        A_i_B = a[i, :] @ Bp  # shape (n,)
        A_B_i = a @ Bp[i, :]  # shape (n,)
        
        # A[j,:] @ B[j,:] for all j is the diagonal of (a @ Bp)
        # But we only need it for j != i.
        # Let's compute diag_A_B = np.diag(a @ Bp)
        # This is expensive. Instead, note that for a fixed i, we can compute:
        # A_j_B_j[j] = a[j,:] @ Bp[j,:]
        
        # Actually, let's just compute the full matrices once outside the loop?
        # No, Bp changes with p. So we need to recompute.
        
        # Let's compute:
        # M1 = a @ Bp  (shape n x n)
        # M1[i, j] = a[i,:] @ Bp[j,:]
        # M1[j, j] = a[j,:] @ Bp[j,:]
        
        # This is O(n^3) if done naively. But n=100, so 10^6 ops. Fast.
        
        # Let's compute M1 = a @ Bp once per local search iteration?
        # That's O(n^3) = 10^6. Then the delta computation is O(n^2).
        # Total per local search step: O(n^3) for M1 + O(n^2) for deltas.
        # But we only need M1 once per local search step, then we can find the best swap in O(n^2).
        # Then we update p, and need to recompute Bp and M1.
        # So each local search step is O(n^3).
        # Number of local search steps until convergence? Maybe 10-50.
        # So total per ILS iteration: O(n^3 * 50) = 5 * 10^7.
        # With 120s budget, we can do maybe 10-20 ILS iterations.
        
        # Let's implement this approach.
        
        # Compute M1 = a @ Bp
        M1 = a @ Bp  # shape (n, n)
        # M1[i, j] = sum_k a[i,k] * Bp[k, j] = sum_k a[i,k] * b[p[k], p[j]]
        
        # Wait, Bp[k, j] = b[p[k], p[j]].
        # So M1[i, j] = sum_k a[i,k] * b[p[k], p[j]].
        
        # Similarly, M2 = a.T @ Bp
        # M2[i, j] = sum_k a[k, i] * Bp[k, j] = sum_k a[k, i] * b[p[k], p[j]]
        
        M2 = a.T @ Bp  # shape (n, n)
        
        # Now for each pair (i, j):
        # Term 1 full sum: S1[j] = M1[i, j] - M1[j, j] - M1[i, i] + M1[j, i]
        # Term 2 full sum: S2[j] = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
        
        # But we need to exclude k=i and k=j.
        # The full sum includes k=i and k=j.
        
        # Let's derive the correction terms.
        
        # For Term 1, the k=i term is:
        # (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
        # The k=j term is:
        # (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
        
        # For Term 2, the k=i term is:
        # (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
        # The k=j term is:
        # (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
        
        # So the corrected delta is:
        # delta[i,j] = (S1[j] - corr1_i - corr1_j) + (S2[j] - corr2_i - corr2_j) + term3
        
        # Let's compute this for all j at once for fixed i.
        
        # S1[j] for all j:
        S1_all = M1[i, :] - M1[:, i] - M1[i, i] + M1[np.arange(n), np.arange(n)]
        # Wait, M1[j, j] is the diagonal. Let's extract it.
        diag_M1 = np.diag(M1)
        S1_all = M1[i, :] - diag_M1 - M1[i, i] + M1[:, i]
        
        # S2[j] for all j:
        S2_all = M2[i, :] - diag_M2 - M2[i, i] + M2[:, i] if False else None # Need diag_M2
        
        # Let's recompute more carefully.
        
        # S1[j] = M1[i, j] - M1[j, j] - M1[i, i] + M1[j, i]
        # Vectorized:
        # M1[i, :] is shape (n,)
        # diag_M1 is shape (n,)
        # M1[i, i] is scalar
        # M1[:, i] is shape (n,)
        
        S1_vec = M1[i, :] - diag_M1 - M1[i, i] + M1[:, i]
        
        # S2[j] = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
        # Wait, M2[i, j] = sum_k a[k,i] Bp[k,j].
        # S2[j] = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
        # Let's check indices.
        # S2[j] = sum_k (a[k,i] - a[k,j]) * (Bp[k,j] - Bp[k,i])
        #       = sum_k a[k,i]Bp[k,j] - sum_k a[k,i]Bp[k,i] - sum_k a[k,j]Bp[k,j] + sum_k a[k,j]Bp[k,i]
        #       = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
        
        S2_vec = M2[i, :] - M2[i, i] - diag_M2 + M2[:, i]
        
        # Correction terms for k=i and k=j:
        # For each j, we need to subtract:
        # corr1_i[j] = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
        # corr1_j[j] = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
        # corr2_i[j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
        # corr2_j[j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
        
        # Let's vectorize these.
        
        # corr1_i[j] = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
        # a[j,i] is a[:, i]
        # Bp[j,i] is Bp[:, i]
        
        corr1_i_vec = (a[i, i] - a[:, i]) * (Bp[:, i] - Bp[i, i])
        
        # corr1_j[j] = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
        # a[i,j] is a[i, :]
        # a[j,j] is diag_a = np.diag(a)
        # Bp[j,j] is diag_Bp = np.diag(Bp)
        # Bp[i,j] is Bp[i, :]
        
        diag_a = np.diag(a)
        diag_Bp = np.diag(Bp)
        
        corr1_j_vec = (a[i, :] - diag_a) * (diag_Bp - Bp[i, :])
        
        # corr2_i[j] = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
        corr2_i_vec = (a[i, i] - a[i, :]) * (Bp[i, :] - Bp[i, i])
        
        # corr2_j[j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
        corr2_j_vec = (a[:, i] - diag_a) * (diag_Bp - Bp[:, i])
        
        # Term 3: (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
        term3_vec = (a[i, :] - a[:, i]) * (Bp[:, i] - Bp[i, :])
        
        # Total delta for all j:
        delta_vec = (S1_vec - corr1_i_vec - corr1_j_vec) + \
                    (S2_vec - corr2_i_vec - corr2_j_vec) + \
                    term3_vec
        
        # Set delta[i, i] = 0
        delta_vec[i] = 0
        
        # Store in D
        D[i, :] = delta_vec
        D[:, i] = delta_vec  # Symmetric
    
    return D

def _local_search(a, b, p, max_iters=100, time_limit=None, start_time=None):
    """
    Steepest-descent local search using vectorized delta computation.
    Finds the best swap (i, j) that reduces cost, applies it, and repeats.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    if start_time is None:
        start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Compute all swap deltas
        D = _compute_all_swap_deltas(a, b, p)
        
        # Find the best swap (most negative delta)
        # Only consider i < j to avoid duplicates
        np.fill_diagonal(D, 0)  # Don't swap with self
        # We want the minimum delta
        min_delta = np.min(D)
        
        if min_delta >= 0:
            break  # Local optimum
        
        # Find the indices of the best swap
        i, j = np.unravel_index(np.argmin(D), D.shape)
        
        if i == j:
            break
        
        # Apply the swap
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost

def _perturb(p, num_swaps=3):
    """
    Perturb the permutation by performing a few random swaps.
    """
    n = len(p)
    p_perturbed = p.copy()
    for _ in range(num_swaps):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
    return p_perturbed

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Leave 5s buffer
    start_time = time.time()
    
    # 1. Initial Solution
    p = _greedy_initial(a, b)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # 2. Iterated Local Search
    num_ilts = 0
    while time.time() - start_time < total_deadline:
        # Local search
        p_ls, cost_ls = _local_search(a, b, p, max_iters=200, 
                                       time_limit=total_deadline - (time.time() - start_time) - 2.0,
                                       start_time=start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Perturb
        p = _perturb(p_ls, num_swaps=3)
        
        num_ilts += 1
        
        # Check time
        if time.time() - start_time > total_deadline - 5.0:
            break
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
