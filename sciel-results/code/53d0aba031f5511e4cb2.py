"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    # Safety margin: 120s limit, aim to finish by 118s
    deadline = time.time() + 118.0

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Sums over all k
        # Si = sum_k a[i,k] * (b[v, p[k]] - b[u, p[k]])
        # SiT = sum_k a[k,i] * (b[p[k], v] - b[p[k], u])
        # Sj = sum_k a[j,k] * (b[u, p[k]] - b[v, p[k]])
        # SjT = sum_k a[k,j] * (b[p[k], u] - b[p[k], v])
        
        diff_b_vu = b[v, p] - b[u, p]
        diff_b_uv = b[u, p] - b[v, p]
        
        Si = np.dot(a[i, :], diff_b_vu)
        Sj = np.dot(a[j, :], diff_b_uv)
        
        diff_b_pv = b[p, v] - b[p, u]
        diff_b_pu = b[p, u] - b[p, v]
        
        SiT = np.dot(a[:, i], diff_b_pv)
        SjT = np.dot(a[:, j], diff_b_pu)
        
        # Subtract k=i and k=j terms from Si, SiT, Sj, SjT
        # p[i] = u, p[j] = v
        
        # Si[i] = a[i,i] * (b[v, u] - b[u, u])
        # Si[j] = a[i,j] * (b[v, v] - b[u, v])
        Si_sub = a[i, i] * (b[v, u] - b[u, u]) + a[i, j] * (b[v, v] - b[u, v])
        
        # SiT[i] = a[i,i] * (b[u, v] - b[u, u])
        # SiT[j] = a[j,i] * (b[v, v] - b[v, u])
        SiT_sub = a[i, i] * (b[u, v] - b[u, u]) + a[j, i] * (b[v, v] - b[v, u])
        
        # Sj[i] = a[j,i] * (b[u, u] - b[v, u])
        # Sj[j] = a[j,j] * (b[u, v] - b[v, v])
        Sj_sub = a[j, i] * (b[u, u] - b[v, u]) + a[j, j] * (b[u, v] - b[v, v])
        
        # SjT[i] = a[i,j] * (b[u, u] - b[u, v])
        # SjT[j] = a[j,j] * (b[v, u] - b[v, v])
        SjT_sub = a[i, j] * (b[u, u] - b[u, v]) + a[j, j] * (b[v, u] - b[v, v])
        
        # Direct terms for the swap of i and j
        # Old: a[i,j]b[u,v] + a[j,i]b[v,u] + a[i,i]b[u,u] + a[j,j]b[v,v]
        # New: a[i,j]b[v,u] + a[j,i]b[u,v] + a[i,i]b[v,v] + a[j,j]b[u,u]
        DT = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u]) + a[i, i] * (b[v, v] - b[u, u]) + a[j, j] * (b[u, u] - b[v, v])
        
        return (Si - Si_sub) + (SiT - SiT_sub) + (Sj - Sj_sub) + (SjT - SjT_sub) + DT

    def local_search(p, current_cost, deadline):
        improved = True
        max_iter = 1000  # Increase max iterations as delta is now correct and likely allows more steps
        iter_count = 0
        
        while improved and time.time() < deadline - 2.0 and iter_count < max_iter:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Scan all pairs to find the steepest descent move
            for i in range(n - 1):
                if time.time() >= deadline - 2.0:
                    return p, current_cost
                for j in range(i + 1, n):
                    if time.time() >= deadline - 2.0:
                        return p, current_cost
                    d = swap_delta(i, j, p)
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                iter_count += 1
            else:
                break
                
        return p, current_cost

    # --- Initialization Strategies ---
    inits = []

    # 1. Rank-based initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy] = central[::-1]
    
    inits.extend([p_rank, p_rank_rev])

    # 2. Hungarian Algorithm Initialization (Linear Relaxation)
    # Construct cost matrix C where C[i, j] approximates the contribution of placing i at j.
    # C[i, j] = sum_k a[i, k] * b[j, k] is a good linear proxy if we assume k is distributed uniformly or similar.
    # More accurately, the linear relaxation often uses C[i, j] = sum_k a[i, k] * (b[j, k] - mean_b) or similar.
    # Let's use C[i, j] = - sum_k a[i, k] * b[j, k] to maximize affinity.
    # Note: This is O(n^3) to compute, which is 150^3 = 3.3M ops, very fast.
    
    try:
        if time.time() < deadline - 5.0:
            # Compute the linearized cost matrix
            # We want to minimize sum a[i,k] b[p[i], p[k]].
            # A common linearization is to fix the "average" location for each facility index k.
            # Let's try C[i, j] = sum_k a[i, k] * b[j, k]
            # We want to minimize this, so we pass C to linear_sum_assignment.
            C = a @ b
            
            # Solve the assignment problem
            # row_ind, col_ind = linear_sum_assignment(C)
            # p = np.empty(n, dtype=np.int64)
            # p[row_ind] = col_ind
            
            # However, linear_sum_assignment minimizes the cost.
            # So we pass C directly.
            row_ind, col_ind = linear_sum_assignment(C)
            p_hung = np.empty(n, dtype=np.int64)
            p_hung[row_ind] = col_ind
            inits.append(p_hung)
            
            # Also try the negative (maximization) just in case sign conventions vary in interpretation, 
            # though for minimization of sum a*b, minimizing sum a[i,k]b[j,k] is the direct linear relaxation.
    except Exception:
        pass

    # 3. Spectral initialization
    try:
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        
        a_scale = np.sqrt(np.mean(a_centered ** 2)) + 1e-10
        b_scale = np.sqrt(np.mean(b_centered ** 2)) + 1e-10
        
        a_norm = a_centered / a_scale
        b_norm = b_centered / b_scale
        
        # Symmetrized combined matrix
        C = 0.5 * (a_norm + a_norm.T) + 0.5 * (b_norm + b_norm.T)
        C = 0.5 * (C + C.T)
        
        # Get top k eigenvectors
        k = 5
        if time.time() < deadline - 30.0:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Take the last k eigenvectors (largest eigenvalues)
            eigvecs = eigenvectors[:, n-k:]
            
            for idx in range(k):
                eigvec = eigvecs[:, idx]
                facility_order = np.argsort(eigvec, kind="stable")
                location_order = np.argsort(-eigvec, kind="stable")
                
                p = np.empty(n, dtype=np.int64)
                p[facility_order] = location_order
                inits.append(p)
    except Exception:
        pass

    # 4. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(3):
        inits.append(rng.permutation(n))

    # --- Main Search Loop ---
    
    best_cost = np.inf
    best_p = None
    
    # Run local search from each initialization
    for p_init in inits:
        if time.time() > deadline - 15.0:
            break
        
        p_work = p_init.copy()
        cost = compute_cost(p_work)
        
        p_work, cost = local_search(p_work, cost, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_work.copy()
    
    if best_p is None:
        best_p = inits[0].copy()
        best_cost = compute_cost(best_p)

    # --- Iterated Local Search (ILS) with Adaptive Perturbation ---
    
    no_improve_count = 0
    base_perturb = 3
    max_perturb = 15
    
    while time.time() < deadline - 10.0:
        p = best_p.copy()
        
        # Adaptive perturbation size
        perturb_size = base_perturb + min(no_improve_count, 10)
        perturb_size = min(perturb_size, max_perturb)
        
        # Occasionally do a large jump (20% chance)
        if rng.rand() < 0.2:
            perturb_size = max_perturb + rng.randint(0, 10)
        
        num_swaps = max(2, perturb_size)
        for _ in range(num_swaps):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                p[i], p[j] = p[j], p[i]
        
        current_cost = compute_cost(p)
        p, current_cost = local_search(p, current_cost, deadline)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Reset counter if it gets too high to allow exploration
        if no_improve_count > 20:
            no_improve_count = 10

    # Final verification
    if best_p is not None:
        final_cost = compute_cost(best_p)
        if final_cost < best_cost:
            best_cost = final_cost
        return best_p.tolist()
    
    return list(range(n))

# EVOLVE-BLOCK-END
