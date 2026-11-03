"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return int(np.sum(a * b[np.ix_(p, p)]))

    def compute_swap_delta(p, i, j):
        """
        Compute delta in cost when swapping p[i] and p[j].
        Returns cost(p_swapped) - cost(p).
        """
        u = p[i]
        v = p[j]
        
        # Terms involving i or j in the original
        # Original contribution from pairs (i,k), (k,i), (j,k), (k,j) for all k
        # We need: sum_k [a[i,k]*b[u,p[k]] + a[k,i]*b[p[k],u] + a[j,k]*b[v,p[k]] + a[k,j]*b[p[k],v]]
        # minus the k=i and k=j terms, plus the diagonal terms
        
        # Vectorized computation
        bu_p = b[u, p]  # b[u, p[k]] for all k
        bv_p = b[v, p]  # b[v, p[k]] for all k
        bp_u = b[p, u]  # b[p[k], u] for all k
        bp_v = b[p, v]  # b[p[k], v] for all k
        
        a_i = a[i, :]
        a_j = a[j, :]
        a_col_i = a[:, i]
        a_col_j = a[:, j]
        
        # Full sums over all k
        s_i_u = a_i @ bu_p   # sum_k a[i,k]*b[u,p[k]]
        s_i_v = a_i @ bv_p   # sum_k a[i,k]*b[v,p[k]]
        s_j_u = a_j @ bu_p   # sum_k a[j,k]*b[u,p[k]]
        s_j_v = a_j @ bv_p   # sum_k a[j,k]*b[v,p[k]]
        s_u_i = a_col_i @ bp_u  # sum_k a[k,i]*b[p[k],u]
        s_v_i = a_col_i @ bp_v  # sum_k a[k,i]*b[p[k],v]
        s_u_j = a_col_j @ bp_u  # sum_k a[k,j]*b[p[k],u]
        s_v_j = a_col_j @ bp_v  # sum_k a[k,j]*b[p[k],v]
        
        # Extract k=i and k=j terms
        aii = a[i, i]
        aij = a[i, j]
        aji = a[j, i]
        ajj = a[j, j]
        buu = b[u, u]
        buv = b[u, v]
        bvu = b[v, u]
        bvv = b[v, v]
        
        # Subtract k=i and k=j from each sum
        s_i_u_c = s_i_u - aii * buu - aij * buv
        s_i_v_c = s_i_v - aii * bvu - aij * bvv
        s_j_u_c = s_j_u - aji * buu - ajj * buv
        s_j_v_c = s_j_v - aji * bvu - ajj * bvv
        s_u_i_c = s_u_i - aii * buu - aji * bvu
        s_v_i_c = s_v_i - aii * buv - aji * bvv
        s_u_j_c = s_u_j - aij * buu - ajj * bvu
        s_v_j_c = s_v_j - aij * buv - ajj * bvv
        
        # Delta = new - old
        # New: a[i,k]*b[v,p[k]] + a[k,i]*b[p[k],v] + a[j,k]*b[u,p[k]] + a[k,j]*b[p[k],u] (for k!=i,j)
        #      + diagonal: aii*bvv + aij*bvu + aji*buv + ajj*buu
        # Old: a[i,k]*b[u,p[k]] + a[k,i]*b[p[k],u] + a[j,k]*b[v,p[k]] + a[k,j]*b[p[k],v] (for k!=i,j)
        #      + diagonal: aii*buu + aij*buv + aji*bvu + ajj*bvv
        
        delta = (s_i_v_c + s_v_i_c + s_j_u_c + s_u_j_c) - \
                (s_i_u_c + s_u_i_c + s_j_v_c + s_v_j_c) + \
                (aii * bvv + aij * bvu + aji * buv + ajj * buu) - \
                (aii * buu + aij * buv + aji * bvu + ajj * bvv)
        
        return int(delta)

    def local_search(p, time_limit):
        """
        2-swap local search with best-improvement strategy.
        Returns (p, cost).
        """
        current_cost = compute_cost(p)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Precompute for faster access
            b_p = b[np.ix_(p, p)]  # b_p[i,j] = b[p[i], p[j]]
            
            for i in range(n):
                if time.time() > time_limit:
                    break
                u = p[i]
                
                # Precompute vectors that don't depend on j
                a_i = a[i, :]
                a_col_i = a[:, i]
                bu_p = b[u, p]
                bp_u = b[p, u]
                
                for j in range(i + 1, n):
                    v = p[j]
                    a_j = a[j, :]
                    a_col_j = a[:, j]
                    bv_p = b[v, p]
                    bp_v = b[p, v]
                    
                    # Compute delta
                    aii = a[i, i]
                    aij = a[i, j]
                    aji = a[j, i]
                    ajj = a[j, j]
                    buu = b[u, u]
                    buv = b[u, v]
                    bvu = b[v, u]
                    bvv = b[v, v]
                    
                    s_i_u = a_i @ bu_p
                    s_i_v = a_i @ bv_p
                    s_j_u = a_j @ bu_p
                    s_j_v = a_j @ bv_p
                    s_u_i = a_col_i @ bp_u
                    s_v_i = a_col_i @ bp_v
                    s_u_j = a_col_j @ bp_u
                    s_v_j = a_col_j @ bp_v
                    
                    s_i_u_c = s_i_u - aii * buu - aij * buv
                    s_i_v_c = s_i_v - aii * bvu - aij * bvv
                    s_j_u_c = s_j_u - aji * buu - ajj * buv
                    s_j_v_c = s_j_v - aji * bvu - ajj * bvv
                    s_u_i_c = s_u_i - aii * buu - aji * bvu
                    s_v_i_c = s_v_i - aii * buv - aji * bvv
                    s_u_j_c = s_u_j - aij * buu - ajj * bvu
                    s_v_j_c = s_v_j - aij * buv - ajj * bvv
                    
                    delta = (s_i_v_c + s_v_i_c + s_j_u_c + s_u_j_c) - \
                            (s_i_u_c + s_u_i_c + s_j_v_c + s_v_j_c) + \
                            (aii * bvv + aij * bvu + aji * buv + ajj * buu) - \
                            (aii * buu + aij * buv + aji * bvu + ajj * bvv)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
            else:
                break
        
        return p, current_cost

    def simulated_annealing(p, time_limit, t_initial=1000.0, t_final=1.0, alpha=0.9999):
        """
        Simulated annealing with 2-swap moves.
        Returns (p, cost).
        """
        current_cost = compute_cost(p)
        best_p = p.copy()
        best_cost = current_cost
        
        t = t_initial
        n_swaps_per_iter = 10
        
        while time.time() < time_limit and t > t_final:
            for _ in range(n_swaps_per_iter):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                delta = compute_swap_delta(p, i, j)
                
                if delta < 0 or random.random() < np.exp(-delta / t):
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
            
            t *= alpha
        
        return best_p, best_cost

    def rank_matching_init():
        """Initialize by matching facility ranks to location ranks."""
        # Sort facilities by total flow (row sums of a)
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        # Sort locations by total distance (row sums of b)
        central = np.argsort(b.sum(axis=1), kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def random_init():
        """Random permutation initialization."""
        return np.random.permutation(n).astype(np.int64)

    # Main search loop with multiple restarts
    best_p = None
    best_cost = np.inf
    num_restarts = 8
    time_per_restart = (deadline - start_time) / num_restarts

    for restart in range(num_restarts):
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Allocate time for each phase
        t_ls_limit = time.time() + min(time_per_restart * 0.5, remaining * 0.4)
        t_sa_limit = time.time() + min(time_per_restart * 0.3, remaining * 0.25)
        t_polish_limit = deadline - 1.0
        
        # Choose initialization
        if restart == 0:
            p = rank_matching_init()
        elif restart == 1:
            # Alternative: match column sums
            heavy = np.argsort(-a.sum(axis=0), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif restart == 2:
            # Reverse rank matching
            heavy = np.argsort(a.sum(axis=1), kind="stable")
            central = np.argsort(-b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        else:
            p = random_init()
            # Add some perturbation
            for _ in range(5):
                i, j = random.sample(range(n), 2)
                p[i], p[j] = p[j], p[i]
        
        # Phase 1: Local search
        p, cost = local_search(p, t_ls_limit)
        
        # Phase 2: Simulated annealing for escape
        if time.time() < t_sa_limit:
            p, cost = simulated_annealing(p, t_sa_limit)
        
        # Phase 3: Final local search polish
        if time.time() < t_polish_limit:
            p, cost = local_search(p, t_polish_limit)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    if best_p is None:
        best_p = random_init()

    return best_p.tolist()
# EVOLVE-BLOCK-END
