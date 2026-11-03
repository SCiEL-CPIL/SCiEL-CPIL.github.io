"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Initial Heuristic: Hungarian Matching
    # Use row sums of A and B to create a cost matrix for assignment
    # We want to match high flow facilities to low distance locations.
    # Cost matrix C[i, j] = flow_sum[i] * dist_sum[j]
    # Minimizing sum C[i, p[i]] approximates the QAP.
    
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Create cost matrix for Hungarian algorithm
    # We want to minimize sum flow_sums[i] * dist_sums[p[i]]
    # This is a linear assignment problem.
    cost_matrix = np.outer(flow_sums, dist_sums)
    
    # Solve linear assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Helper to compute delta for swapping facilities i and j
    # p[i] = u, p[j] = v
    # Delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    # Note: We assume a and b are symmetric. If not, the formula is more complex.
    # QAPLIB instances typically have symmetric matrices.
    def get_delta(i, j, p, current_cost):
        u = p[i]
        v = p[j]
        
        # Extract rows
        ai = a[i]
        aj = a[j]
        bv = b[v]
        bu = b[u]
        
        diff_a = ai - aj
        
        # Full sum
        sum_all = np.sum(diff_a * (bv[p] - bu[p]))
        
        # Subtract k=i
        term_i = (ai[i] - aj[i]) * (bv[u] - bu[u])
        
        # Subtract k=j
        term_j = (ai[j] - aj[j]) * (bv[v] - bu[v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return int(delta)

    def local_search(p, current_cost, max_iters=100):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Iterate over all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    delta = get_delta(i, j, p, current_cost)
                    if delta < 0:
                        # Accept
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # ILS Loop
    while time.time() - start_time < timeout:
        # Local Search
        p, current_cost = local_search(p, current_cost, max_iters=50)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Check time
        if time.time() - start_time > timeout - 5:
            break
            
        # Perturbation
        # Perform a few random swaps to escape local optimum
        # Scale perturbation with n
        num_swaps = 5 + np.random.randint(10)
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            # Calculate delta to update cost correctly
            delta = get_delta(i, j, p, current_cost)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
