"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Iterate i from 0 to n-2
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            # diff_A = A_sym[i] - A_sym[js]  (shape: (n-i-1, n))
            # diff_D = D[js] - D[i]         (shape: (n-i-1, n))
            # S = sum(diff_A * diff_D, axis=1)
            
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction terms for the diagonal and i-j interactions
            # The formula S includes k=i and k=j terms which are incorrect for the swap delta.
            # We need to subtract the contributions from k=i and k=j in the sum, 
            # and add the correct terms for the swap.
            
            # Term for k=i in sum:
            # (A_sym[i,i] - A_sym[j,i]) * (D[j,i] - D[i,i])
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            # Term for k=j in sum:
            # (A_sym[i,j] - A_sym[j,j]) * (D[j,j] - D[i,j])
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            # Direct terms for the swap of i and j:
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            # cross_term: (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    The surrogate cost C[i, j] approximates the contribution of placing facility i at location j.
    C[i, j] = sum_k a[i, k] * b[j, k] + sum_k a[k, i] * b[k, j]
    This is equivalent to (A @ B)[i, j] + (A.T @ B.T)[i, j] = (A @ B + A.T @ B.T)[i, j]
    Since we want to minimize sum a[i,k] b[p[i], p[k]], this surrogate captures the "average" interaction.
    """
    n = len(a)
    # Calculate surrogate cost matrix
    # Cost[i, j] = sum_k a[i,k] b[j,k] + sum_k a[k,i] b[k,j]
    # Note: b is usually symmetric in QAP, but let's be general.
    # If b is symmetric, b[j,k] = b[k,j].
    # Term 1: (A @ B)[i, j] = sum_k a[i,k] b[k,j]
    # Term 2: (A.T @ B.T)[i, j] = sum_k a[k,i] b[j,k]
    
    cost_matrix = a @ b + a.T @ b.T
    
    # Solve assignment problem
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    
    return p

def _eigen_init(a, b, n, k=10):
    """
    Generate a permutation using eigenvalue-based initialization.
    Projects facilities and locations onto the principal eigenvectors of A and B respectively,
    then aligns them by sorting.
    """
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    try:
        # Use scipy for efficiency if available, otherwise numpy
        # We just need the top k eigenvectors. 
        # For n=256, full eigh is fast enough (~0.1s)
        w_a, v_a = np.linalg.eigh(A_sym)
        w_b, v_b = np.linalg.eigh(B_sym)
    except Exception:
        return np.random.permutation(n)
        
    # Take top k eigenvectors (largest eigenvalues)
    # eigh returns in ascending order, so take last k
    v_a_top = v_a[:, -k:]
    v_b_top = v_b[:, -k:]
    
    # Compute a "score" for each facility and location based on these components
    # We want to align high flow facilities with high distance locations? 
    # Actually, we want to minimize sum a[i,j] b[p[i], p[j]].
    # If A and B are both positive definite-ish, we want to align the principal axes.
    # Let's project onto the first principal component for simplicity and speed.
    # Or sum of projections weighted by eigenvalues?
    
    # Simple heuristic: Sort by the first principal component
    # If we want to minimize, we might want to align the "strongest" directions.
    # Let's try sorting by the coordinate along the first eigenvector.
    
    # Align signs so that the dot product is positive (correlation)
    if np.dot(v_a_top[:, 0], v_b_top[:, 0]) < 0:
        v_b_top[:, 0] = -v_b_top[:, 0]
        
    # Use the first eigenvector for the primary sort
    # Or use a combination? Let's stick to the first one for now.
    # Actually, using all k might be better. Let's compute a distance in the subspace.
    # But sorting requires a 1D projection.
    # Let's use the first eigenvector.
    
    coords_a = v_a_top[:, 0]
    coords_b = v_b_top[:, 0]
    
    # Sort facilities by their coordinate
    order_a = np.argsort(coords_a)
    # Sort locations by their coordinate
    order_b = np.argsort(coords_b)
    
    # Assign facility order_a[i] to location order_b[i]
    p = np.empty(n, dtype=int)
    p[order_a] = order_b
    
    return p

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    
    # Initial phase: Use Hungarian init and Eigen init and random restarts
    
    # 1. Hungarian Initialization
    p_hung = _hungarian_init(a, b)
    # Refine with local search
    ls_limit = time_budget * 0.2
    p_ls, cost_ls = _local_search(a, b, p_hung, ls_limit, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 2. Eigen Initialization
    if time.time() - start_time < time_budget * 0.4:
        p_eig = _eigen_init(a, b, n, k=10)
        ls_limit_eig = time_budget * 0.1
        p_ls_e, cost_ls_e = _local_search(a, b, p_eig, ls_limit_eig, start_time)
        if cost_ls_e < best_cost:
            best_cost = cost_ls_e
            best_p = p_ls_e.copy()
            
    # 3. Random Initials
    if time.time() - start_time < time_budget * 0.5:
        # Try 2 random starts
        time_per_random = (time_budget * 0.1) / 2
        for _ in range(2):
            if time.time() - start_time > time_budget * 0.5:
                break
            p_init = np.random.permutation(n)
            p_ls_r, cost_ls_r = _local_search(a, b, p_init, time_per_random, start_time)
            if cost_ls_r < best_cost:
                best_cost = cost_ls_r
                best_p = p_ls_r.copy()
    
    # ILS phase
    # Use remaining time for Iterated Local Search
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        # Check time
        if time.time() - start_time > time_budget - 2:
            break
            
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = _perturb(best_p, perturb_k)
        
        # Allocate time for this LS run
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
             break
             
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
