"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # Compute row sums of a (flow intensity) and column sums of b (distance intensity)
    # We want to match high-flow facilities with low-distance locations.
    # Use a greedy rank-based initialization, then improve with 2-opt swaps.
    
    # Initial assignment: sort facilities by descending row-sum of a,
    # sort locations by ascending row-sum of b, and match them.
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Heavy flows should go to central (low-distance) locations
    facility_order = np.argsort(-a_row_sums, kind='stable')
    location_order = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order
    
    # Compute initial cost
    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This is equivalent to sum over all pairs
        # For efficiency, we can use matrix operations
        # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i (a[i,:] @ b[:,p[i]]) but that's not quite right
        # Actually: cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Let's compute it directly using broadcasting for small n, 
        # but for n=150, O(n^2) per evaluation is fine if we do it smartly.
        
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # We can write this as: for each i, sum_j a[i,j] * b[p[i], p[j]]
        # = sum_i (a[i,:] dot b[p[i], :])  -- no, that's not right either
        # b[p[i], p[j]] means we need b at row p[i] and col p[j]
        
        # Let's use: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # Create a matrix B where B[i,j] = b[p[i], p[j]]
        # Then cost = sum(a * B)
        
        B = b[np.ix_(p, p)]
        return int(np.sum(a * B))
    
    current_cost = compute_cost(p)
    
    # 2-opt style local search: try swapping two facilities' locations
    # and accept if it improves the cost.
    # To make this fast, we can compute the delta cost of a swap (i, j) efficiently.
    
    # For a swap of locations p[i] and p[j], the change in cost is:
    # Only terms involving i or j change.
    # delta = sum_k [a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])]
    #       + sum_k [a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])]
    #       - 2*a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])  -- wait, need to be careful
    
    # Actually, let's just recompute the full cost for candidate swaps in a batched way,
    # or use a simpler approach: random restarts with greedy construction.
    
    # Given the 120s time limit and n=150, let's do a limited number of swap iterations.
    # We'll do multiple passes of random swaps, accepting improvements.
    
    import random
    random.seed(42)
    
    max_iterations = 50000
    for it in range(max_iterations):
        # Pick two random distinct facilities
        i, j = random.sample(range(n), 2)
        
        # Try swapping
        p[i], p[j] = p[j], p[i]
        new_cost = compute_cost(p)
        
        if new_cost < current_cost:
            current_cost = new_cost
        else:
            # Revert
            p[i], p[j] = p[j], p[i]
        
        # Early termination if no improvement in a long time
        # (simple heuristic)
    
    return p.tolist()
# EVOLVE-BLOCK-END
