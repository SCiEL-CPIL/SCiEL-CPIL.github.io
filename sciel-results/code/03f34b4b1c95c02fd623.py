"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _vectorized_deltas(a, a_sum, D, i, n, diag_a, diag_D):
    """
    Compute delta cost for swapping position i with all other positions j.
    Returns array of deltas for j in range(n), excluding i.
    """
    # For each j, delta = S - T_i - T_j + diag_term + cross_term
    # S_j = (a_sum[i] - a_sum[j]) dot (D[j] - D[i])
    # = a_sum[i] dot D[j] - a_sum[i] dot D[i] - a_sum[j] dot D[j] + a_sum[j] dot D[i]
    
    # Precompute for fixed i:
    ai_dot_Di = np.dot(a_sum[i], D[i])
    
    # a_sum[i] dot D[j] for all j: a_sum[i] @ D
    ai_dot_D_all = a_sum[i] @ D  # shape (n,)
    
    # a_sum[j] dot D[j] for all j: sum of a_sum * D along axis 1
    aj_dot_Dj = np.sum(a_sum * D, axis=1)  # shape (n,)
    
    # a_sum[j] dot D[i] for all j: a_sum @ D[i]
    aj_dot_Di = a_sum @ D[i]  # shape (n,)
    
    # S_j = ai_dot_D_all[j] - ai_dot_Di - aj_dot_Dj[j] + aj_dot_Di[j]
    S_all = ai_dot_D_all - ai_dot_Di - aj_dot_Dj + aj_dot_Di
    
    # T_i_j = (a_sum[i,i] - a_sum[j,i]) * (D[j,i] - D[i,i])
    coeff_i_all = a_sum[i, i] - a_sum[:, i]
    term_i_all = coeff_i_all * (D[:, i] - D[i, i])
    
    # T_j_i = (a_sum[i,j] - a_sum[j,j]) * (D[j,j] - D[i,j])
    coeff_j_all = a_sum[i, :] - diag_a
    term_j_all = coeff_j_all * (diag_D - D[i, :])
    
    # diag_term_j = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
    diag_term_all = (a[i, i] - diag_a) * (diag_D - D[i, i])
    
    # cross_term_j = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
    cross_term_all = (a[i, :] - a[:, i]) * (D[:, i] - D[i, :])
    
    deltas = S_all - term_i_all - term_j_all + diag_term_all + cross_term_all
    deltas[i] = 0  # Exclude self-swap
    return deltas

def _local_search_vectorized(a, b, p, time_limit, start_time):
    """
    Local search with vectorized delta computation for each position.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    a_sum = a + a.T
    diag_a = np.diag(a).copy()
    diag_D = np.diag(D).copy()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            # Compute all deltas for swapping position i with any j
            deltas = _vectorized_deltas(a, a_sum, D, i, n, diag_a, diag_D)
            # Find best improving move
            j = np.argmin(deltas)
            if deltas[j] < 0:
                # Perform swap
                p[i], p[j] = p[j], p[i]
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                diag_D[i], diag_D[j] = diag_D[j], diag_D[i]
                current_cost += deltas[j]
                improved = True
    
    return p, current_cost

def _rank_match_init(a, b):
    """Initialize by matching facility ranks to location ranks based on row sums."""
    n = len(a)
    # Sort facilities by their total flow (row sum of a)
    fac_order = np.argsort(a.sum(axis=1))
    # Sort locations by their total distance (row sum of b)
    loc_order = np.argsort(b.sum(axis=1))
    # Assign facility at rank k to location at rank k
    p = np.empty(n, dtype=int)
    p[fac_order] = loc_order
    return p

def _hungarian_init(a, b):
    """
    Initialize using Hungarian algorithm on a reduced cost matrix.
    Uses row sums of a and b to construct a linear assignment approximation.
    """
    n = len(a)
    # Construct cost matrix: cost[i, j] = sum_k a[i, k] * b[j, k] / n
    # This is an approximation that captures the "average" distance each location sees
    row_sums_a = a.sum(axis=1)
    col_sums_b = b.sum(axis=1)
    
    # Better: use the actual interaction
    # cost[i, j] = sum_k a[i,k] * b[j,k]
    # This is (a @ b.T)[i, j]
    cost_matrix = a @ b.T
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # p[facility] = location
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    return p

def _eigen_init(a, b):
    """
    Initialize using eigendecomposition of the flow matrix.
    Match eigenvectors of a and b.
    """
    n = len(a)
    # For symmetric matrices, use eigendecomposition
    # a is symmetric (flow matrix in QAP)
    # b is symmetric (distance matrix)
    
    # Get top few eigenvectors of a
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a)
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
        
        # Match by ordering of eigenvalues (largest to smallest)
        # Reverse to get largest first
        eigvals_a = eigvals_a[::-1]
        eigvecs_a = eigvecs_a[:, ::-1]
        eigvals_b = eigvals_b[::-1]
        eigvecs_b = eigvecs_b[:, ::-1]
        
        # Use top k eigenvectors for matching
        k = min(10, n)
        
        # Project onto top eigenvectors
        proj_a = eigvecs_a[:, :k]  # (n, k)
        proj_b = eigvecs_b[:, :k]  # (n, k)
        
        # Match facilities to locations by nearest projection
        # For each facility i, find location j that minimizes ||proj_a[i] - proj_b[j]||
        cost_matrix = np.zeros((n, n))
        for i in range(n):
            cost_matrix[i, :] = np.sum((proj_b - proj_a[i, :])**2, axis=1)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=int)
        p[row_ind] = col_ind
        return p
    except:
        return _rank_match_init(a, b)

def _perturb(p, k=5, rng=None):
    """Randomly swap k pairs of positions."""
    if rng is None:
        rng = np.random.default_rng()
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def _large_neighbor_perturb(p, k=10, rng=None):
    """
    Large neighborhood perturbation: randomly swap k pairs.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115  # leave 5s buffer
    
    best_p = None
    best_cost = np.inf
    rng = np.random.default_rng(42)
    
    # Try multiple initialization strategies
    inits = []
    
    # 1. Rank-matching initialization
    p_rank = _rank_match_init(a, b)
    inits.append(p_rank)
    
    # 2. Hungarian initialization
    p_hung = _hungarian_init(a, b)
    inits.append(p_hung)
    
    # 3. Eigen initialization
    p_eig = _eigen_init(a, b)
    inits.append(p_eig)
    
    # 4. Random initialization
    p_rand = rng.permutation(n)
    inits.append(p_rand)
    
    # Run local search on each initialization
    for p_init in inits:
        if time.time() - start_time > time_budget - 5:
            break
        p_ls, cost_ls = _local_search_vectorized(a, b, p_init, time_budget - 5, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search with adaptive perturbation strength
    perturb_k = 5
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() - start_time < time_budget - 5:
        # Perturb with adaptive strength
        if no_improve_count > max_no_improve:
            # Increase perturbation strength
            perturb_k = min(perturb_k + 2, 20)
            no_improve_count = 0
        elif no_improve_count > max_no_improve // 2:
            # Moderate increase
            perturb_k = min(perturb_k + 1, 15)
        
        p_pert = _large_neighbor_perturb(best_p, k=perturb_k, rng=rng)
        
        # Local search
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        p_ls, cost_ls = _local_search_vectorized(a, b, p_pert, remaining - 1, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            no_improve_count = 0
            # Decrease perturbation strength on improvement
            perturb_k = max(3, perturb_k - 1)
        else:
            no_improve_count += 1
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
