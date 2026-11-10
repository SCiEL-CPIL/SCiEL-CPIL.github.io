import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 45.0 # 45s guard for 60s limit (or 90s limit mentioned in directive)
    
    # Parse distribution_spec
    # Assuming distribution_spec is a dict or list of probabilities for each type
    # Types are (i, j) where i, j in {0, 1, 2, 3}
    # Support is {0, 1, 2, 3} for each good
    
    values = [0, 1, 2, 3]
    n_vals = len(values)
    n_types = n_vals * n_vals
    
    # Map (i, j) to index
    def type_index(i, j):
        return i * n_vals + j
        
    # Get probabilities
    # distribution_spec might be a list of length 16 or a dict
    if isinstance(distribution_spec, dict):
        pi = np.zeros(n_types)
        for key, prob in distribution_spec.items():
            i, j = key
            pi[type_index(i, j)] = prob
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        pi = np.array(distribution_spec)
        if len(pi) == n_types:
            pass
        else:
            # Try to reshape or handle differently
            pi = np.array(distribution_spec).flatten()
            if len(pi) < n_types:
                pi = np.pad(pi, (0, n_types - len(pi)))
            else:
                pi = pi[:n_types]
    else:
        # Default uniform
        pi = np.ones(n_types) / n_types
        
    pi = pi / np.sum(pi)
    
    # LP Variables: q_0..q_15, p_0..p_15
    # Total 32 variables
    n_vars = 2 * n_types
    
    # Objective: Maximize sum(pi * p) -> Minimize -sum(pi * p)
    c = np.zeros(n_vars)
    for idx in range(n_types):
        c[n_types + idx] = -pi[idx]
        
    # Constraints: A_ub x <= b_ub
    A_ub = []
    b_ub = []
    
    # 1. q <= 1
    # q_i <= 1 -> 1*q_i + 0*... <= 1
    for idx in range(n_types):
        row = np.zeros(n_vars)
        row[idx] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)
        
    # 2. p >= 0 is handled by bounds, but let's add -p <= 0 if needed? 
    # Bounds are better.
    
    # 3. IR: p_ij - i*q_i1 - j*q_i2 <= 0
    # Note: q_i1 is the probability of allocating good 1 to type i?
    # Wait, the problem says "two goods". 
    # Allocation is usually a vector (alloc_1, alloc_2) or probabilities for each good.
    # The return format is allocations:(16,2). So allocations[idx] = [q_{i, good1}, q_{i, good2}].
    # In my LP, I have q_0..q_15. I need to split them into q_good1 and q_good2?
    # No, the allocation for a type is a pair of probabilities.
    # Let's redefine variables:
    # q1_0..q1_15 (prob of good 1)
    # q2_0..q2_15 (prob of good 2)
    # p_0..p_15
    # Total 48 variables.
    
    n_vars = 3 * n_types
    # Indices:
    # q1_idx = idx
    # q2_idx = n_types + idx
    # p_idx = 2 * n_types + idx
    
    c = np.zeros(n_vars)
    for idx in range(n_types):
        c[2 * n_types + idx] = -pi[idx]
        
    A_ub = []
    b_ub = []
    
    # Bounds for q1, q2: [0, 1]
    # Bounds for p: [0, inf]
    bounds = []
    for _ in range(n_types):
        bounds.append((0, 1)) # q1
    for _ in range(n_types):
        bounds.append((0, 1)) # q2
    for _ in range(n_types):
        bounds.append((0, None)) # p
        
    # IR Constraints:
    # p_ij - i*q1_ij - j*q2_ij <= 0
    for i in range(n_vals):
        for j in range(n_vals):
            idx = type_index(i, j)
            row = np.zeros(n_vars)
            row[2 * n_types + idx] = 1.0 # p
            row[idx] = -i # -i * q1
            row[n_types + idx] = -j # -j * q2
            A_ub.append(row)
            b_ub.append(0.0)
            
    # IC Constraints:
    # p_ij - p_kl - i*(q1_ij - q1_kl) - j*(q2_ij - q2_kl) <= 0
    # p_ij - i*q1_ij - j*q2_ij + p_kl + i*q1_kl + j*q2_kl <= 0 ?
    # Let's re-verify:
    # u_ij >= u_kl + i*(q1_kl - q1_ij) + j*(q2_kl - q2_ij) ? No.
    # u_ij = i*q1_ij + j*q2_ij - p_ij
    # u_ij >= i*q1_kl + j*q2_kl - p_kl
    # i*q1_ij + j*q2_ij - p_ij >= i*q1_kl + j*q2_kl - p_kl
    # p_ij <= p_kl + i*(q1_ij - q1_kl) + j*(q2_ij - q2_kl)
    # p_ij - p_kl - i*q1_ij + i*q1_kl - j*q2_ij + j*q2_kl <= 0
    
    for i in range(n_vals):
        for j in range(n_vals):
            idx_ij = type_index(i, j)
            for k in range(n_vals):
                for l in range(n_vals):
                    idx_kl = type_index(k, l)
                    if idx_ij == idx_kl:
                        continue
                    row = np.zeros(n_vars)
                    # p_ij
                    row[2 * n_types + idx_ij] = 1.0
                    # - p_kl
                    row[2 * n_types + idx_kl] = -1.0
                    # - i * q1_ij
                    row[idx_ij] = -i
                    # + i * q1_kl
                    row[idx_kl] = i
                    # - j * q2_ij
                    row[n_types + idx_ij] = -j
                    # + j * q2_kl
                    row[n_types + idx_kl] = j
                    A_ub.append(row)
                    b_ub.append(0.0)
                    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback to simple heuristic
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for i in range(n_vals):
            for j in range(n_vals):
                idx = type_index(i, j)
                if i > 0:
                    allocations[idx, 0] = 1.0
                    payments[idx] = i
                if j > 0:
                    allocations[idx, 1] = 1.0
                    payments[idx] += j
        return {"allocations": allocations, "payments": payments}
        
    x = res.x
    q1 = x[0:n_types]
    q2 = x[n_types:2*n_types]
    p = x[2*n_types:3*n_types]
    
    # Clip to ensure bounds
    q1 = np.clip(q1, 0, 1)
    q2 = np.clip(q2, 0, 1)
    p = np.clip(p, 0, None)
    
    allocations = np.column_stack((q1, q2))
    payments = p
    
    return {"allocations": allocations, "payments": payments}
