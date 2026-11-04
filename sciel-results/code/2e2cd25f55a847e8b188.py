"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    def get_delta(i, j, p):
        """Compute delta for swapping facilities i and j.
        For symmetric a and b: delta = 2 * sum_k (a[i,k]-a[j,k])*(b[p[i],p[k]]-b[p[j],p[k]])
        minus the i,j self terms."""
        u = p[i]
        v = p[j]
        
        # diff_a[k] = a[i,k] - a[j,k]
        diff_a = a[i] - a[j]
        # diff_b[k] = b[u, p[k]] - b[v, p[k]]
        diff_b = b[u, p] - b[v, p]
        
        # Full sum over all k
        s = np.dot(diff_a, diff_b)
        
        # Correct for k=i and k=j (these are double-counted or wrong)
        # When k=i: p[k]=u, so diff_b[i] = b[u,u]-b[v,u], diff_a[i] = a[i,i]-a[j,i]
        # When k=j: p[k]=v, so diff_b[j] = b[u,v]-b[v,v], diff_a[j] = a[i,j]-a[j,j]
        # The correct delta for QAP with symmetric matrices:
        # delta = 2*(a[i,i]-a[j,i])*(b[u,u]-b[v,u]) + 2*(a[i,j]-a[j,j])*(b[u,v]-b[v,v])
        #         + 2*sum_{k!=i,j} (a[i,k]-a[j,k])*(b[u,p[k]]-b[v,p[k]])
        # = 2*(s - diff_a[i]*diff_b[i] - diff_a[j]*diff_b[j])
        #   + 2*(a[i,i]-a[j,i])*(b[u,u]-b[v,u]) + 2*(a[i,j]-a[j,j])*(b[u,v]-b[v,v])
        # Note: diff_a[i]*diff_b[i] = (a[i,i]-a[j,i])*(b[u,u]-b[v,u])
        #       diff_a[j]*diff_b[j] = (a[i,j]-a[j,j])*(b[u,v]-b[v,v])
        # So: delta = 2*s
        # Wait, that's just 2*s. Let me re-derive.
        
        # Actually for QAP: cost = sum_{i,j} a[i,j]*b[p[i],p[j]]
        # Swap i,j: new cost - old cost
        # Terms involving i or j change.
        # delta = sum_k [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[j,k]*(b[u,p[k]]-b[v,p[k]])]
        #        + sum_k [(a[i,k]-a[j,k])*0]  # no, let me be more careful
        # 
        # Let me just compute: delta = 2*sum_k (a[i,k]-a[j,k])*(b[u,p[k]]-b[v,p[k]])
        # This is the standard formula for symmetric QAP.
        
        delta = 2 * s
        return int(delta)
    
    def local_search(p, current_cost, max_iters=50):
        """First-improvement local search with vectorized delta for each i."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(n):
                if time.time() > 110:
                    break
                u = p[i]
                # Compute delta for swapping i with each j > i
                # delta(i,j) = 2 * sum_k (a[i,k]-a[j,k])*(b[u,p[k]]-b[p[j],p[k]])
                # = 2 * (sum_k a[i,k]*(b[u,p[k]]-b[p[j],p[k]]) - sum_k a[j,k]*(b[u,p[k]]-b[p[j],p[k]]))
                # This is still O(n) per j, so O(n^2) total per pass.
                # Let's vectorize: for fixed i, compute delta for all j at once.
                # delta(i,j) = 2 * dot(a[i]-a[j], b[u,p]-b[p[j],p])
                # = 2 * (dot(a[i], b[u,p]-b[p[j],p]) - dot(a[j], b[u,p]-b[p[j],p]))
                # = 2 * (a[i] @ (b[u,p] - b[p[j],p]) - a[j] @ (b[u,p] - b[p[j],p]))
                # Let X = b[u,p] (length n), Y_j = b[p[j],p] (length n)
                # delta(i,j) = 2 * (a[i] @ X - a[i] @ Y_j - a[j] @ X + a[j] @ Y_j)
                # = 2 * ((a[i]-a[j]) @ X - (a[i]-a[j]) @ Y_j)
                # = 2 * ((a[i]-a[j]) @ (X - Y_j))
                # Still O(n) per j. Let me just do the inner loop but break early.
                for j in range(i + 1, n):
                    v = p[j]
                    diff_a = a[i] - a[j]
                    diff_b = b[u, p] - b[v, p]
                    delta = 2 * int(np.dot(diff_a, diff_b))
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        break  # restart scan from i+1
        return p, current_cost
    
    # Generate multiple initial solutions and pick the best
    def hungarian_init():
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        cost_matrix = np.outer(flow_sums, dist_sums)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    
    def random_init():
        return np.random.permutation(n).astype(np.int64)
    
    def spectral_init():
        # Use eigenvectors of row sums
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        # Normalize and use as cost
        fs_norm = flow_sums / (np.max(flow_sums) + 1e-10)
        ds_norm = dist_sums / (np.max(dist_sums) + 1e-10)
        cost_matrix = np.outer(fs_norm, ds_norm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    
    # Try multiple initializations
    candidates = [hungarian_init(), spectral_init(), random_init()]
    best_p = None
    best_cost = float('inf')
    for cand in candidates:
        c = calc_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()
    
    p = best_p.copy()
    current_cost = best_cost
    
    start_time = time.time()
    timeout = 115.0
    
    # ILS with simulated annealing perturbation
    rng = np.random.RandomState(42)
    T0 = 1000.0
    Tmin = 1.0
    cooling = 0.995
    
    T = T0
    while time.time() - start_time < timeout:
        # Local search
        p, current_cost = local_search(p, current_cost, max_iters=50)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        if time.time() - start_time > timeout - 10:
            break
        
        # Perturbation: number of swaps based on temperature
        num_swaps = max(1, int(3 + 10 * T / T0))
        for _ in range(num_swaps):
            i = rng.randint(n)
            j = rng.randint(n)
            if i == j:
                continue
            delta = get_delta(i, j, p)
            if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
        
        T *= cooling
        if T < Tmin:
            T = T0  # restart temperature
    
    # Final polish: run local search a few more times on best solution
    if time.time() - start_time < timeout - 5:
        p_final = best_p.copy()
        c_final = best_cost
        for _ in range(5):
            if time.time() - start_time > timeout - 3:
                break
            p_final, c_final = local_search(p_final, c_final, max_iters=100)
            if c_final < best_cost:
                best_cost = c_final
                best_p = p_final.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
