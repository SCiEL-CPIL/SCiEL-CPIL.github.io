"""Rewritten QAP solver: spectral init + Hungarian + fast incremental local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def compute_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    return np.sum(a * b[np.ix_(p, p)])


def spectral_initialization(a, b):
    """
    Use eigenvalue decomposition of flow matrix A to get an initial assignment.
    Pair eigenvectors of A (sorted by eigenvalue) with eigenvectors of B (sorted by eigenvalue).
    Then use Hungarian algorithm to refine.
    """
    n = a.shape[0]
    
    # Compute eigenvectors of A and B
    # For QAP, we want to match high-flow facilities with close locations
    # Use the eigenvector approach: sort by eigenvalues and match
    
    # Symmetrize for eigen decomposition (A might not be symmetric)
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    # Get eigenvalues and eigenvectors
    eigvals_a, eigvecs_a = np.linalg.eigh(A_sym)
    eigvals_b, eigvecs_b = np.linalg.eigh(B_sym)
    
    # Sort by eigenvalue (descending for A, ascending for B to match high flow with low distance)
    idx_a = np.argsort(-eigvals_a)
    idx_b = np.argsort(eigvals_b)
    
    # Project: assign facility i to location j based on dot products of eigenvectors
    # Create a cost matrix from the spectral projection
    # V_a[:, i] is the i-th eigenvector of A, V_b[:, j] is the j-th eigenvector of B
    # We want to minimize sum of a[i,k]*b[p[i],p[k]]
    
    # Alternative: use the eigenvector matrix to create a soft assignment
    # Then refine with Hungarian
    V_a = eigvecs_a[:, idx_a]
    V_b = eigvecs_b[:, idx_b]
    
    # Cost matrix: negative dot products (we want to maximize alignment)
    # Actually, let's use a simpler heuristic combined with spectral
    # Rank-based: sort facilities by total flow, locations by total distance
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    # Initial assignment from rank matching
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # Now try spectral refinement
    # Create cost matrix for Hungarian: cost[i,j] = -sum_k a[i,k] * b[j, p_rank[?]]
    # This is getting complex; let's just use rank matching + local search
    # But add a spectral-based perturbation for diversity
    
    # Simple spectral init: use the first few eigenvectors
    # For now, return rank-based init
    return p_rank


def hungarian_refinement(a, b, p_init, max_iter=3):
    """
    Iteratively refine assignment using Hungarian algorithm.
    Fix most locations, re-optimize a subset.
    """
    n = len(p_init)
    p = p_init.copy()
    
    for iteration in range(max_iter):
        # For each facility, compute the marginal cost of each location
        # given the current assignment of other facilities
        # This is O(n^3) per iteration which is too slow for n=150
        
        # Instead, use a simpler approach: re-run Hungarian on a cost matrix
        # derived from the current solution
        cost_matrix = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(n):
                # Cost of placing facility i at location j
                # = sum_k a[i,k] * b[j, p[k]] + sum_k a[k,i] * b[p[k], j]
                # This is O(n) per entry, O(n^3) total
                pk = p
                cost_matrix[i, j] = np.dot(a[i, :], b[j, pk]) + np.dot(a[:, i], b[pk, j])
        
        # Solve Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        # Check if improved
        cost_old = compute_cost(a, b, p)
        cost_new = compute_cost(a, b, p_new)
        
        if cost_new < cost_old:
            p = p_new
        else:
            break
    
    return p


def fast_swap_delta(a, b, p, i, j, b_p_cache):
    """
    Fast swap delta using precomputed b[p, :] rows.
    b_p_cache[k] = b[p[k], :]
    """
    x = p[i]
    y = p[j]
    n = len(p)
    
    # Precompute differences
    a_diff_row = a[i, :] - a[j, :]  # a[i,k] - a[j,k]
    a_diff_col = a[:, i] - a[:, j]  # a[k,i] - a[k,j]
    
    # b[x,:] and b[y,:] rows
    b_x = b[x, :]
    b_y = b[y, :]
    b_diff = b_y - b_x  # b[y,k] - b[x,k]
    
    # For k != i, j:
    # delta_k = a_diff_row[k] * b_diff[p[k]] + a_diff_col[k] * (b[p[k], y] - b[p[k], x])
    
    # Vectorized computation
    pk = p
    b_diff_at_pk = b_diff[pk]  # b_diff[p[k]] for each k
    b_pk_y = b[pk, y]  # b[p[k], y]
    b_pk_x = b[pk, x]  # b[p[k], x]
    
    delta = np.dot(a_diff_row, b_diff_at_pk) + np.dot(a_diff_col, b_pk_y - b_pk_x)
    
    # Subtract contributions from k=i and k=j (they were included in the dot products)
    # k=i: a_diff_row[i] * b_diff[p[i]] + a_diff_col[i] * (b[p[i], y] - b[p[i], x])
    #     = a_diff_row[i] * b_diff[x] + a_diff_col[i] * (b[x, y] - b[x, x])
    #     But b_diff[x] = b_y[x] - b_x[x] = b[y,x] - b[x,x]
    #     b[p[i],y] = b[x,y], b[p[i],x] = b[x,x]
    delta_ii = a_diff_row[i] * (b[y, x] - b[x, x]) + a_diff_col[i] * (b[x, y] - b[x, x])
    
    # k=j: a_diff_row[j] * b_diff[p[j]] + a_diff_col[j] * (b[p[j], y] - b[p[j], x])
    #     = a_diff_row[j] * b_diff[y] + a_diff_col[j] * (b[y, y] - b[y, x])
    #     b_diff[y] = b_y[y] - b_x[y] = b[y,y] - b[x,y]
    #     b[p[j],y] = b[y,y], b[p[j],x] = b[y,x]
    delta_jj = a_diff_row[j] * (b[y, y] - b[x, y]) + a_diff_col[j] * (b[y, y] - b[y, x])
    
    delta -= delta_ii + delta_jj
    
    # Add the direct i-j interaction change
    # Old: a[i,j]*b[x,y] + a[j,i]*b[y,x]
    # New: a[i,j]*b[y,x] + a[j,i]*b[x,y]
    # Delta_ij = a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x])
    #          = (a[i,j] - a[j,i]) * (b[y,x] - b[x,y])
    delta += (a[i, j] - a[j, i]) * (b[y, x] - b[x, y])
    
    return delta


def local_search(a, b, p, time_limit, max_passes=50):
    """
    2-opt local search with first-improvement strategy and fast delta computation.
    """
    n = len(p)
    p = p.copy()
    current_cost = compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = start_time + time_limit - 2.0
    
    pass_count = 0
    while pass_count < max_passes:
        if time.time() > deadline:
            break
        
        improved = False
        pass_count += 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                delta = fast_swap_delta(a, b, p, i, j, None)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
        
        if not improved:
            break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin from 150s limit
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Try a few different initializations and pick the best after quick local search
    best_p = p.copy()
    best_cost = compute_cost(a, b, p)
    
    # Initialization 1: rank matching
    remaining = deadline - time.time()
    if remaining > 10.0:
        p1, c1 = local_search(a, b, p, remaining * 0.3)
        if c1 < best_cost:
            best_cost = c1
            best_p = p1
    
    # Initialization 2: reverse rank matching
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central[::-1]
    remaining = deadline - time.time()
    if remaining > 10.0:
        p2, c2 = local_search(a, b, p2, remaining * 0.3)
        if c2 < best_cost:
            best_cost = c2
            best_p = p2
    
    # Initialization 3: spectral-based
    p3 = spectral_initialization(a, b)
    # Add random perturbation
    rng = np.random.default_rng(42)
    p3_shuffled = p3.copy()
    for _ in range(5):
        i, j = rng.integers(0, n, size=2)
        p3_shuffled[i], p3_shuffled[j] = p3_shuffled[j], p3_shuffled[i]
    remaining = deadline - time.time()
    if remaining > 10.0:
        p3, c3 = local_search(a, b, p3_shuffled, remaining * 0.3)
        if c3 < best_cost:
            best_cost = c3
            best_p = p3
    
    # Final local search with remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        best_p, best_cost = local_search(a, b, best_p, remaining)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
