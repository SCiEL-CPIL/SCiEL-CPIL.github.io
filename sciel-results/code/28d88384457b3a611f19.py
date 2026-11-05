import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin out of 60s

    # --- Parse Distribution ---
    # Target: 4x4 matrix pi where pi[i,j] is prob of (i,j)
    pi = np.zeros((4, 4))
    
    try:
        if isinstance(distribution_spec, dict):
            # Check if keys are tuples (joint) or ints (marginal)
            keys = list(distribution_spec.keys())
            if keys and isinstance(keys[0], tuple):
                for (i, j), p in distribution_spec.items():
                    if 0 <= i < 4 and 0 <= j < 4:
                        pi[i, j] = p
            else:
                # Assume marginal
                marg = np.zeros(4)
                for i, p in distribution_spec.items():
                    if 0 <= i < 4:
                        marg[i] = p
                pi = np.outer(marg, marg)
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.array(distribution_spec, dtype=float)
            if arr.size == 4:
                marg = arr / arr.sum()
                pi = np.outer(marg, marg)
            elif arr.size == 16:
                pi = arr.reshape(4, 4)
            else:
                # Fallback: uniform
                pi = np.ones((4, 4)) / 16.0
        else:
            pi = np.ones((4, 4)) / 16.0
    except Exception:
        pi = np.ones((4, 4)) / 16.0

    # Normalize and clean up
    pi = pi / pi.sum()
    pi[pi < 0] = 0
    if pi.sum() == 0:
        pi = np.ones((4, 4)) / 16.0

    # --- Define Type Space ---
    # Types are (i, j) for i, j in 0..3
    types = [(i, j) for i in range(4) for j in range(4)]
    type_index = {t: idx for idx, t in enumerate(types)}
    n_types = 16

    # Variables:
    # x_{t,0}, x_{t,1} for t in 0..15 (32 vars)
    # p_t for t in 0..15 (16 vars)
    # Total 48 vars
    # Index mapping:
    # var_idx(t, g) = t * 2 + g
    # var_idx_pay(t) = 32 + t

    c = np.zeros(48)
    for t in range(16):
        i, j = types[t]
        # Maximize sum pi_t * p_t => Minimize -sum pi_t * p_t
        c[32 + t] = -pi[i, j]

    A_ub = []
    b_ub = []

    # 1. IR Constraints: p_t <= v1*x_t1 + v2*x_t2
    # p_t - v1*x_t1 - v2*x_t2 <= 0
    for t in range(16):
        i, j = types[t]
        row = np.zeros(48)
        row[32 + t] = 1.0
        row[t * 2 + 0] = -i
        row[t * 2 + 1] = -j
        A_ub.append(row)
        b_ub.append(0.0)

    # 2. IC Constraints: U(t,t) >= U(t,s)
    # v1*x_t1 + v2*x_t2 - p_t >= v1*x_s1 + v2*x_s2 - p_s
    # p_t - p_s + v1*(x_s1 - x_t1) + v2*(x_s2 - x_t2) <= 0
    # p_t - p_s - v1*x_t1 + v1*x_s1 - v2*x_t2 + v2*x_s2 <= 0
    for t in range(16):
        i, j = types[t]
        for s in range(16):
            # If t == s, constraint is 0 <= 0, skip to save time
            if t == s:
                continue
            row = np.zeros(48)
            row[32 + t] = 1.0
            row[32 + s] = -1.0
            row[t * 2 + 0] = -i
            row[s * 2 + 0] = i
            row[t * 2 + 1] = -j
            row[s * 2 + 1] = j
            A_ub.append(row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = []
    for _ in range(32):
        bounds.append((0, 1))
    for _ in range(16):
        bounds.append((0, None))

    # Solve LP
    alloc = np.zeros((16, 2))
    pay = np.zeros(16)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            alloc = res.x[:32].reshape(16, 2)
            pay = res.x[32:]
            
            # Clean up small numerical errors
            alloc = np.clip(alloc, 0, 1)
            pay = np.maximum(pay, 0)
            
            # Optional: Verify constraints and adjust if necessary
            # Given the small size, we can check regret
            # But usually HiGHS is precise enough.
            
        else:
            # Fallback: Simple valid mechanism
            # Allocate if value > 0, pay 0? No, must be IR.
            # If we pay 0, IR is satisfied if alloc * v >= 0.
            # IC: If we allocate based on threshold, is it IC?
            # Let's use a safe fallback: No allocation, no payment. Revenue 0.
            # Or better: A simple deterministic mechanism that is IC and IR.
            # e.g. Second price auction logic adapted for single bidder?
            # For single bidder, the optimal mechanism is often a threshold rule.
            # Let's just return zeros to be safe if LP fails, though it shouldn't.
            alloc = np.zeros((16, 2))
            pay = np.zeros(16)
            
    except Exception:
        alloc = np.zeros((16, 2))
        pay = np.zeros(16)

    # Ensure types are float for consistency
    alloc = alloc.astype(float)
    pay = pay.astype(float)

    return {"allocations": alloc, "payments": pay}
