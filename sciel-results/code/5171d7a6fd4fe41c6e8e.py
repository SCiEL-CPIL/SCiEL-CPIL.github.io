# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum a[i,j] * b[p[i], p[j]]"""
    return np.sum(a * b[np.ix_(p, p)])

def _mds_embed(b, n_components=2):
    """
    Embed locations in 2D space such that Euclidean distances approximate b.
    Uses classical MDS.
    """
    n = b.shape[0]
    # Symmetrize b for distance interpretation
    b_sym = (b + b.T) / 2.0
    
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ b_sym @ J
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    
    # Clamp negative eigenvalues to 0
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # Compute embedding
    U = eigenvectors * np.sqrt(eigenvalues)
    
    return U

def _procrustes_align(a, b, p_init):
    """
    Use Procrustes alignment to find a better permutation.
    Embed facilities (from a) and locations (from b), then find optimal rotation/reflection.
    """
    n = a.shape[0]
    
    # Embed facilities based on flow matrix a
    a_sym = (a + a.T) / 2.0
    fac_coords = _mds_embed(a_sym, n_components=2)
    
    # Embed locations based on distance matrix b
    b_sym = (b + b.T) / 2.0
    loc_coords = _mds_embed(b_sym, n_components=2)
    
    # Center both
    fac_coords = fac_coords - fac_coords.mean(axis=0)
    loc_coords = loc_coords - loc_coords.mean(axis=0)
    
    # Procrustes: find orthogonal transformation T such that fac_coords @ T ≈ loc_coords
    # SVD of cross-covariance matrix
    H = fac_coords.T @ loc_coords
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag(np.concatenate([np.ones(2), [d]]))
    T = Vt.T @ D @ U.T
    
    # Transform facility coordinates
    fac_transformed = fac_coords @ T
    
    # Find optimal assignment: minimize sum ||fac_transformed[i] - loc_coords[p[i]]||^2
    # This is equivalent to maximizing sum fac_transformed[i] . loc_coords[p[i]]
    # Cost matrix for LAP: -fac_transformed[i] . loc_coords[j]
    
    # Compute similarity matrix
    sim = fac_transformed @ loc_coords.T
    cost_lap = -sim
    
    # Solve LAP
    row_ind, col_ind = linear_sum_assignment(cost_lap)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _two_opt_search(a, b, p, time_limit=50.0):
    """
    Efficient 2-swap local search with vectorized delta computation.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    # Precompute for faster delta calculation
    # For swap (i, j) with u=p[i], v=p[j]:
    # Delta = sum_k (a[i,k] - a[j,k]) * (b[v,p[k]] - b[u,p[k]]) 
    #        + sum_k (a[k,i] - a[k,j]) * (b[p[k],v] - b[p[k],u])
    #        + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
    #        - corrections for k=i and k=j
    
    improved = True
    passes = 0
    max_passes = 200
    
    while improved and passes < max_passes:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        passes += 1
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            u = p[i]
            A_i = a[i, :]
            A_iT = a[:, i]
            
            for j in range(i + 1, n):
                v = p[j]
                A_j = a[j, :]
                A_jT = a[:, j]
                
                D1 = A_i - A_j
                D2 = A_iT - A_jT
                
                diff_b_vu = b[v, p] - b[u, p]
                diff_b_pv_u = b[p, v] - b[p, u]
                
                Sum1_all = D1 @ diff_b_vu
                Sum2_all = D2 @ diff_b_pv_u
                
                # Subtract k=i terms
                Sub_i_1 = D1[i] * (b[v, u] - b[u, u])
                Sub_i_2 = D2[i] * (b[u, v] - b[u, u])
                
                # Subtract k=j terms
                Sub_j_1 = D1[j] * (b[v, v] - b[u, v])
                Sub_j_2 = D2[j] * (b[v, v] - b[v, u])
                
                Sum1 = Sum1_all - Sub_i_1 - Sub_j_1
                Sum2 = Sum2_all - Sub_i_2 - Sub_j_2
                
                Cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
                
                delta = Cross + Sum1 + Sum2
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            break
    
    return best_p, best_cost

def _simulated_annealing(a, b, p, time_limit=30.0):
    """
    Simulated Annealing to escape local minima.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    T = 1000.0
    cooling = 0.995
    min_T = 1.0
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
        
        # Random 2-swap
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        
        u = p[i]
        v = p[j]
        
        # Compute delta for swap (i, j)
        A_i = a[i, :]
        A_iT = a[:, i]
        A_j = a[j, :]
        A_jT = a[:, j]
        
        D1 = A_i - A_j
        D2 = A_iT - A_jT
        
        diff_b_vu = b[v, p] - b[u, p]
        diff_b_pv_u = b[p, v] - b[p, u]
        
        Sum1_all = D1 @ diff_b_vu
        Sum2_all = D2 @ diff_b_pv_u
        
        Sub_i_1 = D1[i] * (b[v, u] - b[u, u])
        Sub_i_2 = D2[i] * (b[u, v] - b[u, u])
        Sub_j_1 = D1[j] * (b[v, v] - b[u, v])
        Sub_j_2 = D2[j] * (b[v, v] - b[v, u])
        
        Sum1 = Sum1_all - Sub_i_1 - Sub_j_1
        Sum2 = Sum2_all - Sub_i_2 - Sub_j_2
        
        Cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        delta = Cross + Sum1 + Sum2
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Phase 1: Multiple initializations
    candidates = []
    
    # 1a. Rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[fac_order] = loc_order
    candidates.append(p_rank)
    
    # 1b. Procrustes alignment
    if time.time() < deadline - 100.0:
        p_proc = _procrustes_align(a, b, p_rank)
        candidates.append(p_proc)
    
    # 1c. Random initialization
    if time.time() < deadline - 100.0:
        p_rand = np.random.default_rng(123).permutation(n)
        candidates.append(p_rand)
    
    # 1d. LAP-based initialization
    if time.time() < deadline - 100.0:
        # Use row sums to create a cost matrix
        a_row = a.sum(axis=1)
        b_col = b.sum(axis=0)
        cost_lap = np.outer(a_row, b_col)
        row_ind, col_ind = linear_sum_assignment(cost_lap)
        p_lap = np.empty(n, dtype=np.int64)
        p_lap[row_ind] = col_ind
        candidates.append(p_lap)
    
    # Phase 2: Local search from each candidate
    best_p = None
    best_cost = np.inf
    
    for p_init in candidates:
        if time.time() > deadline - 60.0:
            break
        
        p_ls, cost_ls = _two_opt_search(a, b, p_init, time_limit=15.0)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 3: Simulated Annealing from best solution
    if time.time() < deadline - 20.0:
        p_sa, cost_sa = _simulated_annealing(a, b, best_p, time_limit=20.0)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5.0:
        p_final, cost_final = _two_opt_search(a, b, best_p, time_limit=10.0)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
