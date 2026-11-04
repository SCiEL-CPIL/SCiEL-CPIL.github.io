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
                    break
        
        if not improved:
            break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # --- Initialization: Spectral Embedding + Rank Matching ---
    # Center the matrices
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    
    # Get eigenvectors
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
    except np.linalg.LinAlgError:
        eigvals_a, eigvecs_a = np.linalg.eig(a_centered)
        eigvals_b, eigvecs_b = np.linalg.eig(b_centered)
        idx_a = np.argsort(-np.abs(eigvals_a))
        idx_b = np.argsort(-np.abs(eigvals_b))
        eigvecs_a = eigvecs_a[:, idx_a]
        eigvecs_b = eigvecs_b[:, idx_b]
    else:
        idx_a = np.argsort(-np.abs(eigvals_a))
        idx_b = np.argsort(-np.abs(eigvals_b))
        eigvecs_a = eigvecs_a[:, idx_a]
        eigvecs_b = eigvecs_b[:, idx_b]
        
    # Use spectral info combined with rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by flow strength (descending)
    heavy_facilities = np.argsort(-a_sums, kind='stable')
    # Sort locations by distance centrality (ascending)
    central_locations = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = compute_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for fast delta calculation
    # We'll maintain the permutation p and use vectorized operations for delta
    
    def get_delta(i, j, p, a, b):
        x = p[i]
        y = p[j]
        
        # Term 1: (A_ii - A_jj)(B_yy - B_xx)
        term1 = (a[i, i] - a[j, j]) * (b[y, y] - b[x, x])
        
        # Term 2: (A_ij - A_ji)(B_yx - B_xy)
        term2 = (a[i, j] - a[j, i]) * (b[y, x] - b[x, y])
        
        # Term 3: Sum_{k != i, j} (A_ik - A_jk)(B_y, p[k] - B_x, p[k])
        diff_a = a[i, :] - a[j, :]
        diff_b = b[y, p] - b[x, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract k=i term: (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
        term_ik = (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
        
        # Subtract k=j term: (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
        term_jk = (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
        
        term3 = total_dot - term_ik - term_jk
        
        return term1 + term2 + term3
    
    # Linearized assignment step using scipy's Hungarian algorithm
    def linearized_assignment(p, a, b, n):
        # Build the linearized cost matrix
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # This is a matrix product: a @ b[:, p]
        # But we need C[i, j] for facility i to location j
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # Let B_p = b[:, p] (columns of b permuted by p)
        # Then C = a @ B_p.T? No.
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # Let's define M where M[k, l] = a[i, k] * b[l, p[k]]? No, that's not right.
        # Actually, C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # This can be written as: C = a @ (b[:, p]).T? Let's check.
        # (a @ B_p.T)[i, j] = sum_k a[i, k] * B_p.T[k, j] = sum_k a[i, k] * B_p[j, k] = sum_k a[i, k] * b[j, p[k]]
        # Yes! So C = a @ b[:, p].T
        
        B_p = b[:, p]  # shape (n, n), column k is b[:, p[k]]
        C = a @ B_p.T  # shape (n, n), C[i, j] = sum_k a[i, k] * b[j, p[k]]
        
        # Solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(C)
        return col_ind
    
    # Main loop: alternate between linearized assignment and 2-opt local search
    while time.time() < deadline:
        # Step 1: Linearized assignment
        if time.time() < deadline - 5:
            p_new = linearized_assignment(p, a, b, n)
            cost_new = compute_cost(a, b, p_new)
            
            if cost_new < current_cost:
                p = p_new
                current_cost = cost_new
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        # Step 2: 2-opt local search
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                delta = get_delta(i, j, p, a, b)
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    
                    break
        
        if not improved:
            # Local Minimum Reached. Perturb.
            if time.time() < deadline - 10:
                # Random Reversal Perturbation
                len_seg = np.random.randint(3, 15)
                if len_seg < n:
                    start = np.random.randint(0, n - len_seg)
                    end = start + len_seg
                    p[start:end+1] = p[start:end+1][::-1]
                    current_cost = compute_cost(a, b, p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                else:
                    for _ in range(10):
                        i = np.random.randint(n)
                        j = np.random.randint(n)
                        if i != j:
                            p[i], p[j] = p[j], p[i]
                    current_cost = compute_cost(a, b, p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
