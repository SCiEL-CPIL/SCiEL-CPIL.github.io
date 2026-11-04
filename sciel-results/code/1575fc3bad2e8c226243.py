<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _hungarian_init(a, b):
    """
    Generate initial assignment using Hungarian algorithm on flow-distance correlation.
    Facility i centrality = sum of flows involving i
    Location j centrality = sum of distances involving j
    Minimize sum of centrality products.
    """
    n = len(a)
    # Flow centrality: total flow for each facility
    flow_centrality = a.sum(axis=1) + a.sum(axis=0)
    # Distance centrality: total distance for each location
    dist_centrality = b.sum(axis=1) + b.sum(axis=0)
    
    # Create cost matrix: we want to assign high flow centrality to low distance centrality
    # Cost[i, j] = flow_centrality[i] * dist_centrality[j]
    cost_matrix = np.outer(flow_centrality, dist_centrality)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # p[i] = location assigned to facility i
    p = np.zeros(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _local_search_fast(a, b, p, time_limit, start_time):
    """
    Fast local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    
    # Precompute symmetric a
    a_sym = a + a.T
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            best_j = -1
            best_delta = 0.0
            
            for j in range(i + 1, n):
                # Compute delta for swapping positions i and j
                S = np.dot(a_sym[i] - a_sym[j], D[j] - D[i])
                
                ci = 2*a[i,i] - a[j,i] - a[i,j]
                cj = a[i,j] + a[j,i] - 2*a[j,j]
                
                Ti = ci * (D[j,i] - D[i,i])
                Tj = cj * (D[j,j] - D[i,j])
                
                diag = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
                cross = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
                
                delta = S - Ti - Tj + diag + cross
                
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j >= 0 and best_delta < -1e-9:
                j = best_j
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += best_delta
                improved = True
    
    return p, current_cost

def _perturb(p, k=5, rng=None):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115
    
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # First: Hungarian initialization (globally consistent starting point)
    p_hung = _hungarian_init(a, b)
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5:
        p_ls, cost_ls = _local_search_fast(a, b, p_hung, remaining - 5, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Then: Multiple random initializations
    num_inits = 3
    remaining = time_budget - (time.time() - start_time)
    init_budget = max(1.0, (remaining - 10) / num_inits)
    
    for init_idx in range(num_inits):
        if (time.time() - start_time) > time_budget - 5:
            break
        
        p_init = rng.permutation(n)
        remaining = time_budget - (time.time() - start_time)
        ls_budget = min(init_budget, remaining - 2)
        
        if ls_budget < 1:
            break
        
        p_ls, cost_ls = _local_search_fast(a, b, p_init, ls_budget, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search
    while (time.time() - start_time) < time_budget - 3:
        k_pert = 3 + int(2 * rng.random())
        p_pert = _perturb(best_p, k=k_pert, rng=rng)
        
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search_fast(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return float(np.sum(a * b[np.ix_(p, p)]))

def _spectral_init(a, b, rng):
    """
    Initialize using spectral decomposition of flow matrix.
    Use eigenvectors to create structured assignment that captures
    global flow structure, then refine with Hungarian on residual.
    """
    n = len(a)
    a_sym = (a + a.T) / 2.0
    k = min(8, n)
    try:
        U, s, Vt = np.linalg.svd(a_sym, full_matrices=False)
        fac_embed = Vt[:k, :]
    except:
        fac_embed = np.random.randn(k, n)
    b_sym = (b + b.T) / 2.0
    try:
        U_b, s_b, Vt_b = np.linalg.svd(b_sym, full_matrices=False)
        loc_embed = Vt_b[:k, :]
    except:
        loc_embed = np.random.randn(k, n)
    cost_matrix = -fac_embed.T @ loc_embed
    cost_matrix += 0.01 * rng.standard_normal((n, n))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.zeros(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _hungarian_init(a, b, rng):
    """
    Generate initial assignment using Hungarian algorithm on flow-distance correlation.
    """
    n = len(a)
    flow_centrality = a.sum(axis=1) + a.sum(axis=0)
    dist_centrality = b.sum(axis=1) + b.sum(axis=0)
    cost_matrix = np.outer(flow_centrality, dist_centrality)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.zeros(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _local_search_tabu(a, b, p, time_limit, start_time, rng, tabu_len=50):
    """
    Tabu Search with 2-opt moves and aspiration criteria.
    Returns (p, cost).
    """
    n = len(p)
    p = np.array(p, dtype=np.int64).copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = float(np.sum(a * D))
    a_sym = a + a.T
    tabu_list = []
    best_p = p.copy()
    best_cost = current_cost
    no_improve_count = 0

    while (time.time() - start_time) < time_limit:
        if len(tabu_list) > tabu_len:
            tabu_list.pop(0)
        
        best_delta = 0.0
        best_move = None
        
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n):
                if (i, j) in tabu_list or (j, i) in tabu_list:
                    continue
                S = np.dot(a_sym[i] - a_sym[j], D[j] - D[i])
                ci = 2*a[i,i] - a[j,i] - a[i,j]
                cj = a[i,j] + a[j,i] - 2*a[j,j]
                Ti = ci * (D[j,i] - D[i,i])
                Tj = cj * (D[j,j] - D[i,j])
                diag = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
                cross = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
                delta = S - Ti - Tj + diag + cross
                is_tabu = (i, j) in tabu_list or (j, i) in tabu_list
                if not is_tabu or delta < best_cost:
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
        
        if best_move is not None and best_delta < -1e-9:
            i, j = best_move
            p[i], p[j] = p[j], p[i]
            D[i], D[j] = D[j].copy(), D[i].copy()
            D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
            current_cost += best_delta
            tabu_list.append((i, j))
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            if no_improve_count > 5:
                i, j = rng.choice(n, size=2, replace=False)
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                current_cost = float(np.sum(a * D))
                no_improve_count = 0
    
    return best_p, best_cost

def _perturb_diversified(p, k, rng, freq=None, n=None):
    """Perturb biased toward under-explored assignments."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        if freq is not None:
            row_sums = freq.sum(axis=1)
            probs = 1.0 / (row_sums + 1.0)
            probs /= probs.sum()
            i = rng.choice(n, p=probs)
        else:
            i = rng.choice(n)
        j = rng.choice(n)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145
    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng(42)
    
    inits = []
    if time_budget - (time.time() - start_time) > 10:
        inits.append(_spectral_init(a, b, rng))
    if time_budget - (time.time() - start_time) > 10:
        inits.append(_hungarian_init(a, b, rng))
    inits.append(rng.permutation(n).astype(np.int64))
    
    for idx, p_init in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed - 5
        if remaining < 5:
            break
        ls_budget = min(remaining, 40)
        p_ls, cost_ls = _local_search_tabu(a, b, p_init, ls_budget, start_time, rng, tabu_len=60)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    freq = np.zeros((n, n), dtype=np.int32)
    while (time.time() - start_time) < time_budget - 3:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5:
            break
        k_pert = 2 + int(3 * rng.random())
        p_pert = _perturb_diversified(best_p, k_pert, rng, freq=freq, n=n)
        ls_budget = min(remaining - 2, 25)
        p_ls, cost_ls = _local_search_tabu(a, b, p_pert, ls_budget, start_time, rng, tabu_len=50)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
            for i in range(n):
                freq[i, best_p[i]] += 1
    
    if best_p is None:
        best_p = np.arange(n)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
