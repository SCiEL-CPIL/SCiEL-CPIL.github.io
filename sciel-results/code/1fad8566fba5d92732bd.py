"""QAP tai256c solver with ILS and vectorized local search."""
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.dot(a.ravel(), bp.ravel()))


def _swap_delta(a, b, p, i, j):
    """Compute cost change when swapping p[i] and p[j].
    
    Returns the delta (new_cost - old_cost).
    """
    pi = p[i]
    pj = p[j]
    
    # For all k != i, j:
    # Terms involving i: a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
    # Terms involving j: a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
    
    # Vectorized computation over all k
    # delta = sum_k [a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
    #            + a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])]
    #         + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    
    # Note: a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    #       = (a[i,k] - a[j,k]) * (b[pj,p[k]] - b[pi,p[k]])
    # Similarly for the other terms.
    
    # Let's use a direct but efficient approach:
    # Create mask for k != i, j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    ks = np.where(mask)[0]
    
    pk = p[ks]
    
    # Term 1: sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
    t1 = np.dot(a[i, ks], b[pj, pk] - b[pi, pk])
    # Term 2: sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi])
    t2 = np.dot(a[ks, i], b[pk, pj] - b[pk, pi])
    # Term 3: sum_k a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
    t3 = np.dot(a[j, ks], b[pi, pk] - b[pj, pk])
    # Term 4: sum_k a[k,j]*(b[p[k],pi] - b[p[k],pj])
    t4 = np.dot(a[ks, j], b[pk, pi] - b[pk, pj])
    
    # Diagonal terms
    t5 = a[i, j] * (b[pj, pi] - b[pi, pj])
    t6 = a[j, i] * (b[pi, pj] - b[pj, pi])
    
    return t1 + t2 + t3 + t4 + t5 + t6


def _local_search(a, b, p, max_iter=1000, time_limit=None, start_time=None):
    """First-improvement local search using 2-opt swaps."""
    n = len(p)
    cost = _qap_cost(a, b, p)
    p = p.copy()
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Try all pairs (i, j) in a random order for diversity
        # For speed, use a limited number of random pairs per iteration
        # or try all pairs (n=256, so 32640 pairs - manageable with vectorization)
        
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Sample a subset of pairs for speed, or do full scan
        # With n=256, full scan is 32640 pairs. Let's do full scan but optimize.
        
        # Actually, let's try random sampling for speed:
        num_trials = min(200, n * (n - 1) // 2)
        for _ in range(num_trials):
            i, j = random.sample(range(n), 2)
            d = _swap_delta(a, b, p, i, j)
            if d < best_delta:
                best_delta = d
                best_i, best_j = i, j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            improved = True
    
    return p, cost


def _perturb(p, strength=2):
    """Randomly swap 'strength' pairs in the permutation."""
    n = len(p)
    p = p.copy()
    for _ in range(strength):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    start_time = time.time()
    deadline = start_time + 115.0  # 35s margin before 150s kill
    
    best_p = None
    best_cost = np.iinfo(np.int64).max
    
    def check_time():
        return time.time() < deadline
    
    # Generate multiple initial solutions
    # 1. Rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # 2. Random initializations
    num_random = 3
    initial_solutions = [p_rank]
    for _ in range(num_random):
        p_rand = np.random.permutation(n).astype(np.int64)
        initial_solutions.append(p_rand)
    
    # Local search on each initial solution
    for p0 in initial_solutions:
        if not check_time():
            break
        
        p_ls, c_ls = _local_search(a, b, p0, max_iter=200, time_limit=deadline - time.time(), start_time=start_time)
        
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()
    
    # Iterated Local Search from the best solution
    if best_p is not None:
        p_ils = best_p.copy()
        
        # ILS with increasing perturbation
        perturb_strength = 2
        ils_iter = 0
        
        while check_time() and ils_iter < 50:
            ils_iter += 1
            
            # Perturb
            p_pert = _perturb(p_ils, strength=perturb_strength)
            
            # Local search
            p_ls, c_ls = _local_search(a, b, p_pert, max_iter=300, 
                                       time_limit=deadline - time.time(), start_time=start_time)
            
            if c_ls < best_cost:
                best_cost = c_ls
                best_p = p_ls.copy()
                p_ils = p_ls.copy()
                perturb_strength = max(2, perturb_strength - 1)  # Reduce perturbation on improvement
            else:
                # Occasionally restart from best with larger perturbation
                if ils_iter % 10 == 0:
                    p_ils = best_p.copy()
                    perturb_strength = min(5, perturb_strength + 1)
    
    if best_p is None:
        # Fallback: return the rank-matching solution
        return p_rank.tolist()
    
    return best_p.tolist()
