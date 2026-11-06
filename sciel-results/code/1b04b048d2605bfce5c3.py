import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin against 60s limit
    
    # Step 1: Solve for optimal antichain in [2, 128]
    small_nodes = list(range(2, 129))
    n_small = len(small_nodes)
    node_to_idx = {v: i for i, v in enumerate(small_nodes)}
    
    # Weights
    weights = np.array([1.0 / v for v in small_nodes])
    c = -weights  # Minimize negative sum
    
    # Constraints: x_i + x_j <= 1 if i divides j or j divides i
    row_indices = []
    col_indices = []
    data = []
    constraint_idx = 0
    
    for i in range(n_small):
        a = small_nodes[i]
        for j in range(i + 1, n_small):
            b = small_nodes[j]
            # Check if a divides b or b divides a
            if b % a == 0 or a % b == 0:
                row_indices.append(constraint_idx)
                col_indices.append(i)
                data.append(1)
                row_indices.append(constraint_idx)
                col_indices.append(j)
                data.append(1)
                constraint_idx += 1
                
    num_constraints = constraint_idx
    
    if num_constraints > 0:
        import scipy.sparse as sp
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, n_small))
        lb = np.zeros(num_constraints)
        ub = np.ones(num_constraints)
        constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(lb=np.zeros(n_small), ub=np.ones(n_small))
        integrality = np.ones(n_small)
        
        try:
            res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds,
                       options={"time_limit": 5.0, "mip_gap": 0.0})
            
            if res.x is not None:
                A_small = [small_nodes[i] for i in range(n_small) if res.x[i] > 0.5]
            else:
                A_small = []
        except Exception:
            A_small = []
    else:
        A_small = small_nodes[:]

    # Fallback if MILP fails (should not happen for this size)
    if not A_small:
        # Greedy fallback
        A_small = []
        used = [False] * 129
        for v in sorted(small_nodes, key=lambda x: 1.0/x, reverse=True):
            if used[v]: continue
            # Check if compatible with current A_small
            compatible = True
            for u in A_small:
                if v % u == 0 or u % v == 0:
                    compatible = False
                    break
            if compatible:
                A_small.append(v)
                used[v] = True
                for k in range(2, 129):
                    if v * k <= 128:
                        used[v * k] = True

    # Step 2: Add compatible large numbers [129, 256]
    large_nodes = list(range(129, 257))
    C_large = []
    for x in large_nodes:
        compatible = True
        for a in A_small:
            if x % a == 0:
                compatible = False
                break
        if compatible:
            C_large.append(x)
            
    result = A_small + C_large
    
    if not result:
        return [256]
        
    return result
