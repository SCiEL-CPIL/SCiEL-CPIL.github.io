"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    def compute_cost(p):
        p_arr = np.asarray(p, dtype=np.int64)
        return int(np.sum(a * b[np.ix_(p_arr, p_arr)]))

    def delta_swap(i, j, p, B_p):
        """
        Compute cost change from swapping facilities i and j.
        B_p[i] = b[p[i], :]; B_p[j] = b[p[j], :]
        """
        pi = p[i]
        pj = p[j]
        if pi == pj:
            return 0

        # Delta = sum_{k != i,j} [ (a[i,k]-a[j,k])*(B_p[j][p[k]] - B_p[i][p[k]])
        #                       + (a[k,i]-a[k,j])*(B_p[p[k]][pj] - B_p[p[k]][pi]) ]
        # Note: for symmetric a and b, this simplifies but let's keep general.
        # Actually for symmetric a: a[k,i]=a[i,k], a[k,j]=a[j,k]
        # So Delta = sum_{k != i,j} (a[i,k]-a[j,k]) * (B_p[j][p[k]] - B_p[i][p[k]])
        #           + sum_{k != i,j} (a[i,k]-a[j,k]) * (b[p[k],pj] - b[p[k],pi])
        # But B_p[j][p[k]] = b[p[j], p[k]] and b[p[k],pi] = b[pi, p[k]] if symmetric b
        # Let's just compute it properly.
        
        # For general case:
        # C = sum a[i,j] b[p[i],p[j]]
        # Swap i<->j:
        # New cost terms involving i or j change.
        # Terms with k not i,j:
        #   a[i,k] b[p[i],p[k]] -> a[i,k] b[p[j],p[k]]
        #   a[j,k] b[p[j],p[k]] -> a[j,k] b[p[i],p[k]]
        #   a[k,i] b[p[k],p[i]] -> a[k,i] b[p[k],p[j]]
        #   a[k,j] b[p[k],p[j]] -> a[k,j] b[p[k],p[i]]
        # Terms i,j themselves:
        #   a[i,j] b[p[i],p[j]] -> a[i,j] b[p[j],p[i]] (same if b symmetric)
        #   a[j,i] b[p[j],p[i]] -> a[j,i] b[p[i],p[j]] (same if a symmetric)
        
        # So delta = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[pj,pk]-b[pi,pk])
        #             + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[pk,pj]-b[pk,pi])
        
        # Using B_p: B_p[j][pk] = b[pj, pk], B_p[i][pk] = b[pi, pk]
        # b[pk, pj] = B_p[pk][pj] ... this is harder to vectorize simply.
        # Let's just do a loop for n=100, it's fast enough per swap.
        
        ai = a[i]
        aj = a[j]
        b_pi = b[pi]
        b_pj = b[pj]
        
        sum1 = 0.0
        sum2 = 0.0
        for k in range(n):
            if k == i or k == j:
                continue
            pk = p[k]
            sum1 += (ai[k] - aj[k]) * (b_pj[pk] - b_pi[pk])
            sum2 += (a[k, i] - a[k, j]) * (b[pk, pj] - b[pk, pi])
        
        return sum1 + sum2

    # Quadratic Linearization (QL) initialization
    def ql_initial():
        # Eigendecomposition of flow matrix
        eigenvalues, eigenvectors = np.linalg.eigh(a)
        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Use top r eigenvectors
        r = min(20, n)  # number of rank-1 terms
        
        # Build surrogate cost matrix
        C = np.zeros((n, n), dtype=np.float64)
        for k in range(r):
            if eigenvalues[k] == 0:
                break
            v = eigenvectors[:, k]
            # Rank-1 outer product: v * v^T
            # Surrogate: for facility i at location l, cost contribution from this term
            # is a[i,i'] * b[l,l'] ~ (eigenval_k * v[i] * v[i']) * b[l,l']
            # For linear assignment, we need a matrix M where M[i, l] is the cost
            # of assigning facility i to location l.
            # From the rank-1 decomposition, the term is eigenval_k * v[i] * v[j] * b[p[i], p[j]]
            # This is quadratic in the assignment. The linear surrogate for each term:
            # If we fix other assignments, the marginal cost of assigning i to l is:
            # eigenval_k * v[i] * sum_j v[j] * b[l, p[j]]
            # But for QL, we typically solve: min sum_i sum_j c_k[i] * c_k[j] * b[p[i],p[j]]
            # Linearized: for each k, minimize sum_i d_k[i] * b_row_sum[p[i]]
            # where d_k[i] = eigenval_k * v[i]
            # and b_row_sum[l] = sum_j ... 
            # Actually, the standard QL linearization:
            # min sum_k eigenval_k * (sum_i v_k[i] * x_{i,l_i}) * (sum_j v_k[j] * x_{j,l_j})
            # This is not directly linear. 
            # Alternative: treat each eigenvector as defining a "priority" for facilities
            # and locations, then do assignment.
            
            # Let's use a different approach: for each eigenvector, compute
            # facility scores: s_i = eigenval_k * v_k[i]
            # location scores: t_l = sum of b[l, :] weighted by something
            # Then do linear assignment minimizing sum_i s_i * t_{p[i]}
            
            # Actually, let's just use the eigenvector to define an ordering
            # and do a greedy assignment.
            
            # For now, let's use a simpler heuristic: 
            # Sort facilities by |v[i]| * eigenval descending
            # Sort locations by sum of b[l,:] ascending (central locations)
            # Assign in that order
            
            scores = eigenvalues[k] * eigenvectors[:, k]
            fac_order = np.argsort(-np.abs(scores))
            loc_order = np.argsort(b.sum(axis=1))
            
            # Assign
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p.tolist()
        
        # Fallback: rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p.tolist()

    # Local search
    def local_search(p, max_passes=50):
        p = p[:]
        cur_cost = compute_cost(p)
        best_cost = cur_cost
        best_p = p[:]
        
        for _ in range(max_passes):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, n):
                    d = delta_swap(i, j, p, b)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        cur_cost += d
                        improved = True
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best_p = p[:]
            if not improved:
                break
        
        return best_p, best_cost

    # Simulated Annealing
    def simulated_annealing(p, start_temp=50000.0, end_temp=1.0):
        p = p[:]
        cur_cost = compute_cost(p)
        best_cost = cur_cost
        best_p = p[:]
        t = start_temp
        cool_factor = 0.9995
        no_improve = 0
        restart_threshold = 30000
        
        while time.time() < deadline - 5.0:
            if time.time() > deadline:
                break
            
            if no_improve > restart_threshold:
                p = best_p[:]
                cur_cost = best_cost
                t = start_temp * 0.5
                no_improve = 0
            
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            d = delta_swap(i, j, p, b)
            
            if d < 0 or (t > 1e-9 and random.random() < np.exp(-d / t)):
                p[i], p[j] = p[j], p[i]
                cur_cost += d
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            t = max(end_temp, t * cool_factor)
        
        return best_p, best_cost

    # Main
    # Step 1: QL initialization
    p = ql_initial()
    cur_cost = compute_cost(p)
    best_p = p[:]
    best_cost = cur_cost
    
    # Step 2: Local search
    p, cur_cost = local_search(p, max_passes=30)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p[:]
    
    # Step 3: Simulated Annealing for remaining time
    p, cur_cost = simulated_annealing(p, start_temp=30000.0, end_temp=1.0)
    if cur_cost < best_cost:
        best_cost = cur_cost
        best_p = p[:]
    
    # Step 4: Final local search polish
    if time.time() < deadline - 5.0:
        p, cur_cost = local_search(best_p, max_passes=20)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p[:]
    
    return best_p

# EVOLVE-BLOCK-END
