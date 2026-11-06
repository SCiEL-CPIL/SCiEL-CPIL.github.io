# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 80.0
    
    # Values in {0,1,2,3} for each of 2 goods -> 16 types
    values = list(range(4))
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    
    # Build probability distribution from spec
    # distribution_spec likely contains probabilities or a method to get them
    # Try to extract probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'])
        elif 'pmf' in distribution_spec:
            probs = np.array(distribution_spec['pmf'])
        elif 'values' in distribution_spec and 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'])
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum()
    
    # Variables: x1[0..15], x2[0..15], p[0..15] -> 48 variables
    # Objective: maximize sum_i probs[i] * (v1_i*x1_i + v2_i*x2_i - p_i)
    # = maximize sum_i probs[i] * (v1_i*x1_i + v2_i*x2_i) - sum_i probs[i] * p_i
    
    # c = -objective coefficients for linprog (minimizes)
    c = np.zeros(3 * n_types)
    for i, (v1, v2) in enumerate(types):
        c[i] = -probs[i] * v1          # x1
        c[n_types + i] = -probs[i] * v2 # x2
        c[2 * n_types + i] = probs[i]   # p (negative of -probs[i]*p_i)
    
    # Constraints:
    # 1. 0 <= x1 <= 1, 0 <= x2 <= 1, p >= 0
    # 2. Monotonicity of x1 in v1: x1(v1,v2) <= x1(v1',v2) for v1' > v1
    # 3. Monotonicity of x2 in v2: x1(v1,v2) <= x1(v1,v2') for v2' > v2
    #    Wait, x2 should be monotone in v2, x1 in v1
    # 4. IC: v·x(v) - p(v) >= v'·x(v') - p(v') for all v, v'
    # 5. IR: p(v) <= v1*x1(v) + v2*x2(v) -> v1*x1 + v2*x2 - p >= 0
    
    A_ub = []
    b_ub = []
    
    # Monotonicity constraints for x1 (in v1):
    # For each v2, and each pair v1 < v1': x1(v1,v2) - x1(v1',v2) <= 0
    for v2 in values:
        for v1 in values:
            for v1p in values:
                if v1p > v1:
                    idx_v1 = v1 * 4 + v2
                    idx_v1p = v1p * 4 + v2
                    row = np.zeros(3 * n_types)
                    row[idx_v1] = 1
                    row[n_types + idx_v1p] = 0  # x2 not involved
                    row[idx_v1p] = -1  # x1(v1p) 
                    # Wait, x1 is in first n_types, x2 in next n_types
                    # x1(v1,v2) - x1(v1p,v2) <= 0
                    row = np.zeros(3 * n_types)
                    row[idx_v1] = 1
                    row[idx_v1p] = -1
                    A_ub.append(row)
                    b_ub.append(0)
    
    # Monotonicity constraints for x2 (in v2):
    for v1 in values:
        for v2 in values:
            for v2p in values:
                if v2p > v2:
                    idx_v2 = v1 * 4 + v2
                    idx_v2p = v1 * 4 + v2p
                    row = np.zeros(3 * n_types)
                    row[n_types + idx_v2] = 1
                    row[n_types + idx_v2p] = -1
                    A_ub.append(row)
                    b_ub.append(0)
    
    # IC constraints: v·x(v) - p(v) >= v'·x(v') - p(v')
    # v1*x1(v) + v2*x2(v) - p(v) - v1'*x1(v') - v2'*x2(v') + p(v') >= 0
    # => -(v1*x1(v) + v2*x2(v) - p(v) - v1'*x1(v') - v2'*x2(v') + p(v')) <= 0
    for i, (v1, v2) in enumerate(types):
        for j, (v1p, v2p) in enumerate(types):
            if i == j:
                continue
            row = np.zeros(3 * n_types)
            # -v1*x1(v) - v2*x2(v) + p(v) + v1p*x1(v') + v2p*x2(v') - p(v') <= 0
            row[i] = -v1
            row[n_types + i] = -v2
            row[2 * n_types + i] = 1
            row[j] = v1p
            row[n_types + j] = v2p
            row[2 * n_types + j] = -1
            A_ub.append(row)
            b_ub.append(0)
    
    # IR constraints: v1*x1(v) + v2*x2(v) - p(v) >= 0
    # => -(v1*x1(v) + v2*x2(v) - p(v)) <= 0
    for i, (v1, v2) in enumerate(types):
        row = np.zeros(3 * n_types)
        row[i] = -v1
        row[n_types + i] = -v2
        row[2 * n_types + i] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = np.array([])
        b_ub = np.array([])
    
    # Bounds: 0 <= x1 <= 1, 0 <= x2 <= 1, p >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # x1
    for i in range(n_types):
        bounds.append((0, 1))  # x2
    for i in range(n_types):
        bounds.append((0, None))  # p
    
    # Solve LP
    from scipy.optimize import linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    x1 = x[:n_types]
    x2 = x[n_types:2*n_types]
    p = x[2*n_types:]
    
    allocations = np.column_stack([x1, x2])
    payments = p
    
    # Verify and clean up
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
