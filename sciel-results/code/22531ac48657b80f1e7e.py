"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Generate all 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    type_values = np.array(types, dtype=float)  # shape (16, 2)
    
    # Parse distribution_spec to get probabilities for each type
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == 16:
            probs = probs / probs.sum()
        else:
            probs = np.ones(16) / 16.0
    else:
        # Default: uniform distribution
        probs = np.ones(16) / 16.0
    
    # LP variables: x1[0..15], x2[0..15], p[0..15] -> 48 variables
    # Index: var i = x1[i], var 16+i = x2[i], var 32+i = p[i]
    
    # Objective: maximize E[v1*x1 + v2*x2 - p] = -minimize E[-v1*x1 - v2*x2 + p]
    c = np.zeros(48)
    for i in range(16):
        v1, v2 = types[i]
        c[i] = -probs[i] * v1       # -v1 * x1
        c[16 + i] = -probs[i] * v2  # -v2 * x2
        c[32 + i] = probs[i]        # +p
    
    # Inequality constraints: A_ub @ x <= b_ub
    # IC: for all (i, j): v_i·x_i - p_i >= v_i·x_j - p_j
    #   => v_i·x_i - v_i·x_j - p_i + p_j >= 0
    #   => -v_i·x_i + v_i·x_j + p_i - p_j <= 0
    # IR: for all i: p_i <= v_i·x_i => p_i - v_i·x_i <= 0
    # p_i >= 0 => -p_i <= 0
    
    A_rows = []
    b_rows = []
    
    # IC constraints (256)
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            v1, v2 = types[i]
            row = np.zeros(48)
            row[i] = -v1          # -v1 * x1[i]
            row[16 + i] = -v2     # -v2 * x2[i]
            row[j] = v1           # +v1 * x1[j]
            row[16 + j] = v2      # +v2 * x2[j]
            row[32 + i] = 1       # +p[i]
            row[32 + j] = -1      # -p[j]
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints (16): p_i - v1*x1_i - v2*x2_i <= 0
    for i in range(16):
        v1, v2 = types[i]
        row = np.zeros(48)
        row[i] = -v1
        row[16 + i] = -v2
        row[32 + i] = 1
        A_rows.append(row)
        b_rows.append(0.0)
    
    # p_i >= 0 => -p_i <= 0
    for i in range(16):
        row = np.zeros(48)
        row[32 + i] = -1
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x1, x2, p <= 1 (for x), p >= 0 (no upper bound needed but set to large)
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))  # p unbounded above
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status != 0:
            # Fallback: uniform allocation with no payment
            allocations = np.ones((16, 2))
            payments = np.zeros(16)
            return {'allocations': allocations, 'payments': payments}
        
        x1 = res.x[0:16]
        x2 = res.x[16:32]
        p = res.x[32:48]
        
        # Clip to [0,1] for safety
        x1 = np.clip(x1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        p = np.clip(p, 0, None)
        
        allocations = np.column_stack((x1, x2))
        
    except Exception:
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if time.time() > deadline:
        return {'allocations': allocations, 'payments': p}
    
    return {'allocations': allocations, 'payments': p}
# EVOLVE-BLOCK-END
