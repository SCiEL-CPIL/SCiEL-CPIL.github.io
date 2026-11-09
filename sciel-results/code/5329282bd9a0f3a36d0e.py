# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def _get_support_and_probs(distribution_spec):
    """
    Return support (list of (v1,v2) tuples) and probabilities.
    Values in {0,1,2,3} for each good.
    distribution_spec can be:
      - dict {(v1,v2): prob}
      - list of probs in row-major order (v1 outer, v2 inner) of length 16
      - 2D array 4x4
      - None (default uniform)
    """
    V = 4  # values 0..3
    support = [(v1, v2) for v1 in range(V) for v2 in range(V)]
    
    if distribution_spec is None:
        probs = np.ones((V, V), dtype=float) / (V * V)
    elif isinstance(distribution_spec, dict):
        probs = np.zeros((V, V), dtype=float)
        for (v1, v2), p in distribution_spec.items():
            probs[v1, v2] = float(p)
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == V * V:
            probs = np.array(distribution_spec, dtype=float).reshape(V, V)
        else:
            # Maybe it's a dict-like or something else; default to uniform
            probs = np.ones((V, V), dtype=float) / (V * V)
    elif hasattr(distribution_spec, 'shape'):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (V, V):
            probs = arr
        elif arr.size == V * V:
            probs = arr.reshape(V, V)
        else:
            probs = np.ones((V, V), dtype=float) / (V * V)
    else:
        probs = np.ones((V, V), dtype=float) / (V * V)
    
    # Normalize to sum to 1
    total = probs.sum()
    if total > 1e-15:
        probs = probs / total
    else:
        probs = np.ones((V, V), dtype=float) / (V * V)
    
    return support, probs

def _index(v1, v2):
    """Flatten (v1, v2) to index in row-major order."""
    return v1 * 4 + v2

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 90.0 - 5.0  # 5s safety margin
    
    support_list, probs = _get_support_and_probs(distribution_spec)
    # probs is 4x4, probs[v1, v2]
    
    # Variables:
    # x1[v1, v2] for v1 in 0..3, v2 in 0..3  -> 16 vars
    # x2[v1, v2] for v1 in 0..3, v2 in 0..3  -> 16 vars
    # Total 32 vars
    # Var indices: 0..15 for x1, 16..31 for x2
    def x1_idx(v1, v2):
        return _index(v1, v2)
    def x2_idx(v1, v2):
        return 16 + _index(v1, v2)
    
    # Payment t(v1, v2) = v1*x1[v1,v2] - sum_{k=0}^{v1-1} x1[k,v2] + v2*x2[v1,v2] - sum_{k=0}^{v2-1} x2[v1,k]
    # We maximize sum_{v1,v2} probs[v1,v2] * t(v1,v2)
    # = sum_{v1,v2} probs[v1,v2] * [v1*x1[v1,v2] - sum_{k<v1} x1[k,v2] + v2*x2[v1,v2] - sum_{k<v2} x2[v1,k]]
    
    # Objective coefficients for each variable
    c = np.zeros(32, dtype=float)
    
    # Coefficients for x1[v1, v2]:
    # From term v1*x1[v1,v2]: contributes probs[v1,v2] * v1
    # From term -sum_{k<v1'} x1[k,v2'] for all (v1',v2') where v2'=v2 and v1'<v1: contributes -probs[v1',v2]
    for v1 in range(4):
        for v2 in range(4):
            idx = x1_idx(v1, v2)
            # Direct term: v1 * probs[v1, v2]
            c[idx] += v1 * probs[v1, v2]
            # Negative terms: for all v1' > v1, same v2, x1[v1,v2] appears in -sum_{k<v1'} x1[k,v2'] with k=v1
            for v1p in range(v1 + 1, 4):
                c[idx] -= probs[v1p, v2]
    
    # Coefficients for x2[v1, v2]:
    for v1 in range(4):
        for v2 in range(4):
            idx = x2_idx(v1, v2)
            # Direct term: v2 * probs[v1, v2]
            c[idx] += v2 * probs[v1, v2]
            # Negative terms: for all v2' > v2, same v1, x2[v1,v2] appears in -sum_{k<v2'} x2[v1,k] with k=v2
            for v2p in range(v2 + 1, 4):
                c[idx] -= probs[v1, v2p]
    
    # Maximize c^T x  =>  Minimize -c^T x
    c_min = -c
    
    # Constraints:
    # 1. 0 <= x <= 1 for all variables -> handled by bounds
    # 2. Monotonicity: x1[v1, v2] <= x1[v1+1, v2] for v1 in 0..2, v2 in 0..3
    #    => x1[v1, v2] - x1[v1+1, v2] <= 0
    # 3. Monotonicity: x2[v1, v2] <= x2[v1, v2+1] for v1 in 0..3, v2 in 0..2
    #    => x2[v1, v2] - x2[v1, v2+1] <= 0
    # 4. Non-negative payments: t(v1, v2) >= 0 for all v1, v2
    #    => -t(v1, v2) <= 0
    #    t(v1,v2) = v1*x1[v1,v2] - sum_{k<v1} x1[k,v2] + v2*x2[v1,v2] - sum_{k<v2} x2[v1,k]
    #    So constraint: -v1*x1[v1,v2] + sum_{k<v1} x1[k,v2] - v2*x2[v1,v2] + sum_{k<v2} x2[v1,k] <= 0
    
    A_ub_list = []
    b_ub_list = []
    
    # Monotonicity constraints for x1
    for v1 in range(3):  # 0,1,2
        for v2 in range(4):
            row = np.zeros(32, dtype=float)
            row[x1_idx(v1, v2)] = 1.0
            row[x1_idx(v1+1, v2)] = -1.0
            A_ub_list.append(row)
            b_ub_list.append(0.0)
    
    # Monotonicity constraints for x2
    for v1 in range(4):
        for v2 in range(3):  # 0,1,2
            row = np.zeros(32, dtype=float)
            row[x2_idx(v1, v2)] = 1.0
            row[x2_idx(v1, v2+1)] = -1.0
            A_ub_list.append(row)
            b_ub_list.append(0.0)
    
    # Non-negative payment constraints: -t(v1,v2) <= 0
    for v1 in range(4):
        for v2 in range(4):
            row = np.zeros(32, dtype=float)
            # -v1*x1[v1,v2]
            row[x1_idx(v1, v2)] -= v1
            # +sum_{k<v1} x1[k,v2]
            for k in range(v1):
                row[x1_idx(k, v2)] += 1.0
            # -v2*x2[v1,v2]
            row[x2_idx(v1, v2)] -= v2
            # +sum_{k<v2} x2[v1,k]
            for k in range(v2):
                row[x2_idx(v1, k)] += 1.0
            A_ub_list.append(row)
            b_ub_list.append(0.0)
    
    A_ub = np.array(A_ub_list, dtype=float)
    b_ub = np.array(b_ub_list, dtype=float)
    
    # Bounds: 0 <= x <= 1
    bounds = [(0.0, 1.0)] * 32
    
    # Solve LP
    result = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation with zero payment (feasible but low revenue)
        allocs = np.ones((16, 2), dtype=float)
        pays = np.zeros(16, dtype=float)
        return {'allocations': allocs, 'payments': pays}
    
    x_opt = result.x
    
    # Construct allocations and payments
    allocations = np.zeros((16, 2), dtype=float)
    payments = np.zeros(16, dtype=float)
    
    for v1 in range(4):
        for v2 in range(4):
            idx = _index(v1, v2)
            a1 = x_opt[x1_idx(v1, v2)]
            a2 = x_opt[x2_idx(v1, v2)]
            allocations[idx, 0] = a1
            allocations[idx, 1] = a2
            
            # Compute payment
            t = 0.0
            # Good 1 term
            t += v1 * a1
            for k in range(v1):
                t -= x_opt[x1_idx(k, v2)]
            # Good 2 term
            t += v2 * a2
            for k in range(v2):
                t -= x_opt[x2_idx(v1, k)]
            
            # Clamp tiny negative values to 0 for numerical stability
            if t < 0 and t > -1e-10:
                t = 0.0
            payments[idx] = t
    
    # Clip allocations to [0,1] for safety
    allocations = np.clip(allocations, 0.0, 1.0)
    # Clip payments to non-negative
    payments = np.clip(payments, 0.0, None)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
