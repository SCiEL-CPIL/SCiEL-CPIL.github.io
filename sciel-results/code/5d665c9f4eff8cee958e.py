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
        Assumes symmetric a and b."""
        u = p[i]
        v = p[j]
        
        ai = a[i]
        aj = a[j]
        bv = b[v]
        bu = b[u]
        
        diff_a = ai - aj
        
        # Sum over all k
        sum_all = np.sum(diff_a * (bv[p] - bu[p]))
        
        # Subtract k=i term: (a[i,i]-a[j,i])*(b[v,p[i]]-b[u,p[i]]) = (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
        term_i = (ai[i] - aj[i]) * (bv[u] - bu[u])
        
        # Subtract k=j term: (a[i,j]-a[j,j])*(b[v,p[j]]-b[u,p[j]]) = (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        term_j = (ai[j] - aj[j]) * (bv[v] - bu[v])
        
        # Self-term correction: 2*(a[i,i]-a[j,j])*(b[v,v]-b[u,u])
        self_term = 2 * (ai[i] - aj[j]) * (bv[v] - bu[u])
        
        delta = 2 * (sum_all - term_i - term_j) + self_term
        return int(delta)
    
    def local_search(p, current_cost, max_iters=50):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(n):
                for j in range(i + 1, n):
                    delta = get_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
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
    
    while time.time() - start_time < timeout:
        # Local search
        p, current_cost = local_search(p, current_cost, max_iters=30)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        if time.time() - start_time > timeout - 5:
            break
        
        # Perturbation: random swaps with cooling
        num_swaps = 3 + rng.randint(7)
        for _ in range(num_swaps):
            i = rng.randint(n)
            j = rng.randint(n)
            if i == j:
                continue
            delta = get_delta(i, j, p)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
