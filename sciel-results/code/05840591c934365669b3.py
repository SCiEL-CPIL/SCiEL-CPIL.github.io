import time
import math
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    nodes = list(range(2, 257))
    num_nodes = len(nodes)
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_val = nodes

    # Precompute conflict masks for fallback
    conflict_mask = [0] * num_nodes
    for i in range(num_nodes):
        vi = idx_to_val[i]
        mask = 0
        for j in range(num_nodes):
            if i == j:
                continue
            vj = idx_to_val[j]
            if math.lcm(vi, vj) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Try MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Build A matrix
        # We only need to add constraints for i < j to avoid redundancy
        rows = []
        cols = []
        vals = []
        
        for i in range(num_nodes):
            vi = idx_to_val[i]
            for j in range(i + 1, num_nodes):
                vj = idx_to_val[j]
                if math.lcm(vi, vj) <= 256:
                    r = len(rows)
                    rows.append(r)
                    cols.append(i)
                    vals.append(1)
                    rows.append(r)
                    cols.append(j)
                    vals.append(1)
        
        if not rows:
            return nodes

        A = np.zeros((len(rows), num_nodes))
        for r, c, v in zip(rows, cols, vals):
            A[r, c] = v
        
        c_obj = np.array([-1.0 / idx_to_val[i] for i in range(num_nodes)])
        
        bounds = Bounds(lb=np.zeros(num_nodes), ub=np.ones(num_nodes))
        constraints = LinearConstraint(A, ub=np.ones(len(rows)))
        integrality = np.ones(num_nodes)
        
        res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": deadline - time.time() - 2.0})
        
        if res.status == 0:
            x = res.x
            result = []
            for i in range(num_nodes):
                if x[i] > 0.5:
                    result.append(idx_to_val[i])
            if result:
                return result
    except Exception:
        pass

    # Fallback: Local Search
    # ... (insert previous local search code here) ...
    
    def local_search(initial_mask, time_limit=5.0):
        # ... (same as before) ...
        pass

    # Run fallback
    # Greedy init
    mask = 0
    for v in range(2, 257):
        idx = val_to_idx[v]
        if (conflict_mask[idx] & mask) == 0:
            mask |= (1 << idx)
            
    # Simple polish
    improved = True
    start_ls = time.time()
    while improved and time.time() < deadline - 2.0:
        improved = False
        # 1-swap
        in_mask = [i for i in range(num_nodes) if mask & (1 << i)]
        not_in_mask = [i for i in range(num_nodes) if not (mask & (1 << i))]
        
        for u in in_mask:
            for v in not_in_mask:
                if 1.0/idx_to_val[v] > 1.0/idx_to_val[u]:
                    new_mask = (mask & ~(1 << u)) | (1 << v)
                    if (conflict_mask[v] & (new_mask & ~(1 << v))) == 0:
                        mask = new_mask
                        improved = True
                        break
            if improved:
                break
                
    result = []
    m = mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx_to_val[idx])
        m ^= lsb
        
    return result
