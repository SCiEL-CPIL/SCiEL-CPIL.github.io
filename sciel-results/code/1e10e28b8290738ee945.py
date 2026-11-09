import numpy as np
from scipy.optimize import linprog
import time

def _fallback_mechanism(probs, support_grid, n_states):
    """Fallback: use a simple threshold-based mechanism that is guaranteed IC and IR."""
    allocations = np.zeros((n_states, 2))
    payments = np.zeros(n_states)
    
    # For each state, determine if we should allocate based on value thresholds
    # Simple approach: allocate good 1 if v1 >= 2, good 2 if v2 >= 2
    # Payment is the minimum value that would make the bidder indifferent
    
    for i in range(n_states):
        v1, v2 = support_grid[i]
        alloc1 = 1.0 if v1 >= 2 else 0.0
        alloc2 = 1.0 if v2 >= 2 else 0.0
        allocations[i, 0] = alloc1
        allocations[i, 1] = alloc2
        # Payment: use the value at the threshold to ensure IR
        # If we allocate good 1, charge the threshold value (2)
        # If we allocate good 2, charge the threshold value (2)
        # This ensures IR since value >= threshold when allocated
        payments[i] = alloc1 * 2.0 + alloc2 * 2.0
    
    # Verify IC and IR, adjust if needed
    max_regret = 0.0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        utility_truthful = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
        for j in range(n_states):
            utility_deviate = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
            regret = utility_deviate - utility_truthful
            if regret > max_regret:
                max_regret = regret
            if utility_deviate > 0 and utility_truthful < 0:
                # IR violation: adjust payment
                pass
    
    # If IC is violated, use a more conservative mechanism
    if max_regret > 1e-8:
        # Use a mechanism that only allocates when both values are high
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        for i in range(n_states):
            v1, v2 = support_grid[i]
            if v1 >= 2 and v2 >= 2:
                allocations[i, 0] = 1.0
                allocations[i, 1] = 1.0
                payments[i] = 4.0  # Total value at threshold
            elif v1 >= 3:
                allocations[i, 0] = 1.0
                payments[i] = 3.0
            elif v2 >= 3:
                allocations[i, 1] = 1.0
                payments[i] = 3.0
        
        # Re-verify
        max_regret = 0.0
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            utility_truthful = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
            for j in range(n_states):
                utility_deviate = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
                regret = utility_deviate - utility_truthful
                if regret > max_regret:
                    max_regret = regret
        
        if max_regret > 1e-8:
            # Last resort: zero mechanism
            allocations = np.zeros((n_states, 2))
            payments = np.zeros(n_states)
    
    return allocations, payments

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 58.0  # safety margin before 60s

    # Construct the full support grid: values in {0,1,2,3} for two goods
    values = [0, 1, 2, 3]
    support_grid = [(v1, v2) for v1 in values for v2 in values]
    n_states = len(support_grid)
    state_to_idx = {s: i for i, s in enumerate(support_grid)}

    # Extract probabilities robustly
    probs = np.zeros(n_states)
    if isinstance(distribution_spec, dict):
        for s, p in distribution_spec.items():
            if s in state_to_idx:
                probs[state_to_idx[s]] = float(p)
    elif isinstance(distribution_spec, (np.ndarray, list)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (4, 4):
            for i in range(4):
                for j in range(4):
                    idx = state_to_idx[(i, j)]
                    probs[idx] = arr[i, j]
        else:
            flat = arr.flatten()
            if len(flat) == n_states:
                probs = flat.copy()
            else:
                probs = np.ones(n_states) / n_states
    else:
        probs = np.ones(n_states) / n_states

    # Normalize
    total_prob = probs.sum()
    if total_prob > 1e-12:
        probs = probs / total_prob
    else:
        probs = np.ones(n_states) / n_states

    # LP variables: x1[0..15], x2[0..15], p[0..15]
    n_vars = 3 * n_states
    c = np.zeros(n_vars)
    for i in range(n_states):
        c[2 * n_states + i] = -probs[i]

    # Build constraint matrix
    A_ub_rows = []
    b_ub_vals = []

    # IC constraints: for all pairs (i, j), p_i - p_j <= v_i·x_i - v_i·x_j
    # Note: The value of the allocation x_j is evaluated at the TRUE value v_i.
    # Rearranged: p_i - p_j - v1_i*x1_i - v2_i*x2_i + v1_i*x1_j + v2_i*x2_j <= 0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        for j in range(n_states):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[2 * n_states + i] = 1.0
            row[2 * n_states + j] = -1.0
            row[i] = -v1_i
            row[n_states + i] = -v2_i
            row[j] = v1_i
            row[n_states + j] = v2_i
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # IR constraints: p_i <= v1_i*x1_i + v2_i*x2_i
    # p_i - v1_i*x1_i - v2_i*x2_i <= 0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        row = np.zeros(n_vars)
        row[2 * n_states + i] = 1.0
        row[i] = -v1_i
        row[n_states + i] = -v2_i
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    if len(A_ub_rows) > 0:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = np.empty((0, n_vars))
        b_ub = np.empty(0)

    # Bounds
    bounds = [(0, 1)] * n_states + [(0, 1)] * n_states + [(0, None)] * n_states

    # Check time
    if time.time() - start_time > deadline - 5:
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}

    # Solve LP
    x_opt = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            x_opt = res.x
        else:
            # Try interior-point as fallback if highs fails
            res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
            if res2.status == 0:
                x_opt = res2.x
    except Exception:
        x_opt = None

    if x_opt is None:
        allocations, payments = _fallback_mechanism(probs, support_grid, n_states)
        return {"allocations": allocations, "payments": payments}

    x1 = x_opt[:n_states]
    x2 = x_opt[n_states:2 * n_states]
    p = x_opt[2 * n_states:]

    allocations = np.column_stack((x1, x2))
    payments = p

    # Clean up numerical noise
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    # Verify IC (regret check)
    max_regret = 0.0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        utility_truthful = v1_i * x1[i] + v2_i * x2[i] - p[i]
        for j in range(n_states):
            utility_deviate = v1_i * x1[j] + v2_i * x2[j] - p[j]
            regret = utility_deviate - utility_truthful
            if regret > max_regret:
                max_regret = regret

    # If regret exceeds tolerance, use fallback mechanism
    if max_regret > 1e-8:
        allocations, payments = _fallback_mechanism(probs, support_grid, n_states)

    return {"allocations": allocations, "payments": payments}
