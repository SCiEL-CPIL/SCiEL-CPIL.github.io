"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _compute_cost(a, b, p):
    """Compute total QAP cost for permutation p."""
    return np.sum(a * b[np.ix_(p, p)])

def _eigen_init(a, b, n):
    """
    Initialize by aligning top eigenvectors of A and B.
    """
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    k = min(32, n)
    try:
        wA, vA = np.linalg.eigh(A_sym)
        wB, vB = np.linalg.eigh(B_sym)
    except:
        return np.arange(n, dtype=np.int64)
    
    idxA = np.argsort(wA)[::-1][:k]
    idxB = np.argsort(wB)[::-1][:k]
    
    V_A = vA[:, idxA]
    V_B = vB[:, idxB]
    
    cost_mat = -V_A @ V_B.T
    
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    p = np.zeros(n, dtype=np.int64)
    for i in range(n):
        p[row_ind[i]] = col_ind[i]
    
    return p

def _hungarian_init(a, b, n):
    """
    Hungarian-based initialization using flow-weighted distance.
    """
    flow_avg = np.sum(a, axis=1)
    dist_avg = np.sum(b, axis=1)
    
    cost_mat = np.outer(flow_avg, dist_avg)
    
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    p = np.zeros(n, dtype=np.int64)
    for i in range(n):
        p[row_ind[i]] = col_ind[i]
    
    return p

def _fast_delta(a_sym, D, i, j):
    """
    Compute delta cost for swapping positions i and j.
    a_sym = (a + a.T) / 2
    D = b[p, p] (symmetric)
    delta = (a_sym[i] - a_sym[j]) dot (D[j] - D[i])
    """
    diff_a = a_sym[i] - a_sym[j]
    diff_D = D[j] - D[i]
    return np.dot(diff_a, diff_D)

def _local_search(a, a_sym, b, p, time_limit, start_time, max_passes=100):
    """
    First-improvement local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    p = p.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = np.sum(a * D)
    
    improved = True
    passes = 0
    while improved and passes < max_passes and time.time() - start_time < time_limit:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _fast_delta(a_sym, D, i, j)
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    D[i], D[j] = D[j].copy(), D[i].copy()
                    D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                    current_cost += delta
                    improved = True
    
    return p, current_cost

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
    time_budget = 145
    
    a_sym = (a + a.T) / 2.0
    
    best_p = None
    best_cost = np.inf
    
    inits = []
    
    p_eig = _eigen_init(a, b, n)
    inits.append(p_eig)
    
    p_hun = _hungarian_init(a, b, n)
    inits.append(p_hun)
    
    rng = random.Random(42)
    p_rand = np.arange(n, dtype=np.int64)
    rng.shuffle(p_rand.tolist())
    p_rand = np.array(p_rand, dtype=np.int64)
    inits.append(p_rand)
    
    rng2 = random.Random(123)
    p_rand2 = np.arange(n, dtype=np.int64)
    rng2.shuffle(p_rand2.tolist())
    p_rand2 = np.array(p_rand2, dtype=np.int64)
    inits.append(p_rand2)
    
    for idx, p_init in enumerate(inits):
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        num_remaining = len(inits) - idx
        time_per_init = max(2, (remaining - 5) / num_remaining)
        
        p_ls, cost_ls = _local_search(a, a_sym, b, p_init, time_per_init, start_time, max_passes=50)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    perturb_size = 5
    while time.time() - start_time < time_budget - 3:
        remaining = time_budget - (time.time() - start_time)
        if remaining < 3:
            break
        
        p_pert = _perturb(best_p, perturb_size)
        
        time_for_ls = remaining - 2
        if time_for_ls < 1:
            break
        
        p_ls, cost_ls = _local_search(a, a_sym, b, p_pert, time_for_ls, start_time, max_passes=30)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
            perturb_size = 5
        else:
            perturb_size = min(20, perturb_size + 1)
    
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2 and best_p is not None:
        p_final, cost_final = _local_search(a, a_sym, b, best_p, remaining - 1, start_time, max_passes=20)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
