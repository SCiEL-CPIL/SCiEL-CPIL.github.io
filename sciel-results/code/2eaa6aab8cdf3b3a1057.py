"""
Strategy: Vector Alignment / Soft-Assignment QAP Solver.

Instead of pure combinatorial search (2-opt), we:
1. Embed facilities (A) and locations (B) into a latent space using PCA (leading eigenvectors of centered matrices).
2. Use an iterative alignment procedure (similar to Procrustes or gradient descent on the soft assignment matrix)
   to minimize the mismatch between the embedded flow structure and distance structure.
3. Hard-assign using the Hungarian algorithm on the resulting cost matrix derived from the embeddings.
4. Polish with a fast, time-limited 2-opt local search to fix any discrete inconsistencies.

This approach exploits the low-rank structure often present in QAP instances (like tai256c) where
the optimal assignment aligns the principal axes of the flow and distance matrices.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))

def _fast_swap_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping facilities at positions i and j.
    Optimized for speed.
    """
    u = p[i]
    v = p[j]
    
    # Diagonal and cross terms for i, j
    old_ii = a[i, i] * b[u, u] + a[j, j] * b[v, v] + a[i, j] * b[u, v] + a[j, i] * b[v, u]
    new_ii = a[i, i] * b[v, v] + a[j, j] * b[u, u] + a[i, j] * b[v, u] + a[j, i] * b[u, v]
    
    # Interaction with all other k
    # Sum over k != i, j of (a[i,k]*b[u,p[k]] + a[j,k]*b[v,p[k]] + a[k,i]*b[p[k],u] + a[k,j]*b[p[k],v])
    # vs new
    # We can vectorize this.
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    k_indices = np.where(mask)[0]
    
    if len(k_indices) > 0:
        pk = p[k_indices]
        
        # a[i, k_indices] is shape (n-2,)
        # b[u, pk] is shape (n-2,)
        old_cross = np.sum(a[i, k_indices] * b[u, pk] + 
                           a[j, k_indices] * b[v, pk] + 
                           a[k_indices, i] * b[pk, u] + 
                           a[k_indices, j] * b[pk, v])
        
        new_cross = np.sum(a[i, k_indices] * b[v, pk] + 
                           a[j, k_indices] * b[u, pk] + 
                           a[k_indices, i] * b[pk, v] + 
                           a[k_indices, j] * b[pk, u])
    else:
        old_cross = 0
        new_cross = 0
    
    delta = (new_ii + new_cross) - (old_ii + old_cross)
    return int(delta)

def _local_search(a, b, p, time_limit, start_time):
    """
    Fast 2-opt local search.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        # Randomize the order of pairs to avoid bias, or just iterate.
        # For speed, we iterate i, j but break early if time is tight.
        for i in range(n):
            if time.time() - start_time > time_limit - 0.5:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit - 0.1:
                    break
                
                delta = _fast_swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    
    return best_p, best_cost

def _vector_alignment_init(a, b, n_components=64):
    """
    Initialize permutation using vector alignment.
    
    1. Center A and B.
    2. Compute leading eigenvectors (PCA) of A and B.
    3. Align the embeddings.
    4. Compute a cost matrix based on the aligned embeddings and solve the assignment problem.
    """
    n = a.shape[0]
    
    # Center the matrices to remove trivial offsets
    A_centered = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
    B_centered = b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + b.mean()
    
    # Symmetrize just in case, though QAP matrices are usually symmetric
    A_sym = (A_centered + A_centered.T) / 2.0
    B_sym = (B_centered + B_centered.T) / 2.0
    
    # PCA: Get leading eigenvectors
    # We use np.linalg.eigh for symmetric matrices
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(A_sym)
        eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
        
        # Sort in descending order of eigenvalue magnitude (largest components first)
        # eigh returns in ascending order
        idx_a = np.argsort(eigvals_a)[::-1]
        idx_b = np.argsort(eigvals_b)[::-1]
        
        # Take top k components
        k = min(n_components, n)
        V_a = eigvecs_a[:, idx_a[:k]]
        V_b = eigvecs_b[:, idx_b[:k]]
        
        # The embeddings are rows of V_a and V_b
        # Embedding of facility i is V_a[i, :]
        # Embedding of location j is V_b[j, :]
        
        # We want to assign facility i to location j such that the "flow structure" matches "distance structure".
        # In the spectral embedding, the inner product of embeddings approximates the matrix entry (for low-rank approx).
        # Specifically, A[i,j] ~ V_a[i] dot V_a[j].
        # We want B[p[i], p[j]] to be similar to A[i,j].
        # This is equivalent to minimizing sum (A[i,j] - B[p[i], p[j]])^2.
        # Expanding, this involves matching the outer products.
        
        # A simpler heuristic: 
        # Compute the "cost" of assigning i to j based on the mismatch of their neighborhood structures in the embedding space.
        # Or, use the fact that if we align the eigenvectors, the assignment should map similar vectors to similar vectors.
        
        # Let's construct a cost matrix C where C[i, j] is the "distance" between facility i and location j in the embedding space.
        # If we assume the optimal permutation aligns the principal axes, then:
        # V_a @ P @ V_b^T should be close to some diagonal or identity-like structure?
        # Actually, a common heuristic is to minimize || V_a - V_b @ P ||_F^2.
        # This is a linear assignment problem if we expand:
        # sum_i || V_a[i] - V_b[P[i]] ||^2 = sum_i (||V_a[i]||^2 + ||V_b[P[i]]||^2 - 2 V_a[i] dot V_b[P[i]])
        # The first two terms are constant for a fixed permutation P (sum of norms is invariant to permutation).
        # So we just need to maximize sum_i V_a[i] dot V_b[P[i]].
        # This is equivalent to maximizing the trace of V_a.T @ V_b @ P_matrix, or simply finding the best match in the embedding space.
        
        # Let's compute the dot product matrix D where D[i, j] = V_a[i] dot V_b[j]
        D = V_a @ V_b.T
        
        # We want to maximize sum D[i, P[i]].
        # This is a standard assignment problem.
        # scipy's linear_sum_assignment minimizes the cost. So we pass -D.
        row_ind, col_ind = linear_sum_assignment(-D)
        
        # col_ind is the permutation: facility row_ind[k] goes to location col_ind[k]
        # But we need p such that p[facility] = location.
        # Here, facility i is assigned to location j if i in row_ind and j in col_ind at same index.
        p = np.empty(n, dtype=np.int64)
        for i, j in zip(row_ind, col_ind):
            p[i] = j
            
    except Exception:
        # Fallback to simple rank matching if PCA fails
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        
    return p

def _soft_assignment_iteration(a, b, p, n_iter=10, time_limit=20, start_time=0):
    """
    Iterate on the soft assignment using the cost matrix derived from current permutation.
    This is a simplified version of the "Frank-Wolfe" or "Sinkhorn" style relaxation.
    Here, we just re-embed based on the current best permutation's induced structure and re-assign.
    """
    current_p = p.copy()
    current_cost = _qap_cost(a, b, current_p)
    best_p = current_p.copy()
    best_cost = current_cost
    
    for _ in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Create the permuted distance matrix
        bp = b[np.ix_(current_p, current_p)]
        
        # The "effective" flow matrix in location space is A_eff[i, j] = A[inv_p[i], inv_p[j]]
        # Actually, we want to align A and B.
        # Let's try to find a better permutation by looking at the correlation between rows of A and rows of B.
        
        # Compute row-wise correlations or dot products
        # Normalize rows of A and B to compare directions
        A_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        B_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        
        # Cost matrix based on negative dot product (we want high similarity)
        # C[i, j] = - A_norm[i] dot B_norm[j]
        # This is O(n^3) if done naively, but matrix multiplication is O(n^3) with high constant.
        # For n=256, 256x256 matmul is fast.
        C = -A_norm @ B_norm.T
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        
        new_p = np.empty(len(p), dtype=np.int64)
        for i, j in zip(row_ind, col_ind):
            new_p[i] = j
            
        new_cost = _qap_cost(a, b, new_p)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_p = new_p.copy()
            current_p = new_p.copy()
        else:
            # If no improvement, break or try a small perturbation?
            # For now, break to save time for local search.
            break
            
    return best_p, best_cost

def solve(a, b):
    """
    Solve the QAP instance tai256c.
    Returns a permutation p of 0..n-1 as a list of integers.
    """
    start_time = time.time()
    n = len(a)
    
    # Time budget: 120s limit, keep 10s margin.
    time_budget = 110
    
    # 1. Vector Alignment Initialization
    p_init = _vector_alignment_init(a, b, n_components=64)
    current_cost = _qap_cost(a, b, p_init)
    
    best_p = p_init.copy()
    best_cost = current_cost
    
    # 2. Soft Assignment Iteration (Refinement)
    # Allocate 20s for this phase
    soft_time_limit = 20
    p_soft, cost_soft = _soft_assignment_iteration(a, b, p_init, n_iter=5, time_limit=soft_time_limit, start_time=start_time)
    if cost_soft < best_cost:
        best_cost = cost_soft
        best_p = p_soft.copy()
        
    # 3. Local Search Polish
    # Allocate remaining time for 2-opt
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time > 5:
        p_opt, cost_opt = _local_search(a, b, best_p, time_limit=remaining_time - 2, start_time=start_time)
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
            
    # 4. If time remains, try a few random restarts of the vector alignment?
    # Or just more local search from different starts?
    rng = np.random.default_rng(42)
    while time.time() - start_time < time_budget - 10:
        # Generate a random perturbation of the best solution
        p_rand = best_p.copy()
        # Swap a few random pairs
        for _ in range(10):
            i, j = rng.integers(0, n, 2)
            p_rand[i], p_rand[j] = p_rand[j], p_rand[i]
            
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        per_restart_time = min(remaining_time - 2, 10)
        if per_restart_time < 2:
            break
            
        p_opt, cost_opt = _local_search(a, b, p_rand, time_limit=per_restart_time, start_time=time.time())
        
        if cost_opt < best_cost:
            best_cost = cost_opt
            best_p = p_opt.copy()
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
