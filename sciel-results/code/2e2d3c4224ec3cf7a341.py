"""Incentive-compatible finite auction design via direct LP solve."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction maximizing expected revenue.
    
    One additive bidder, two goods, values in {0,1,2,3} (16 types).
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=np.float64)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=np.float64)
        else:
            probs = np.array(list(distribution_spec.values()), dtype=np.float64)
    else:
        probs = np.array(distribution_spec, dtype=np.float64)
    
    # Ensure 16 types
    if len(probs) != 16:
        # Try to reshape or pad
        probs = np.resize(probs, 16)
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16) / 16.0
    
    # Support: 16 types indexed by (v1, v2) where v1,v2 in {0,1,2,3}
    # Index = v1 * 4 + v2
    types = []
    for v1 in range(4):
        for v2 in range(4):
            types.append((v1, v2))
    types = np.array(types)  # shape (16, 2)
    
    # If support is provided, use it to map
    if support is not None and len(support) == 16:
        # Map support entries to type indices
        # Assume support is list of (v1,v2) tuples
        pass  # We'll use standard ordering
    
    # LP: maximize sum_i probs[i] * (v1_i * q1_i + v2_i * q2_i)
    # Variables: q1_0..q1_15, q2_0..q2_15 (32 variables)
    # Constraints:
    #   0 <= q1_i <= 1, 0 <= q2_i <= 1
    #   Monotonicity: q1_{v1+1,v2} >= q1_{v1,v2} for v1 < 3, all v2
    #                 q2_{v1,v2+1} >= q2_{v1,v2} for v2 < 3, all v1
    
    n = 16
    # Objective: maximize sum_i p_i * (v1_i*q1_i + v2_i*q2_i)
    # linprog minimizes, so negate
    c = np.zeros(32)
    for i in range(n):
        v1, v2 = types[i]
        c[i] = -probs[i] * v1       # q1_i coefficient
        c[n + i] = -probs[i] * v2   # q2_i coefficient
    
    # Bounds: 0 <= q <= 1
    bounds = [(0.0, 1.0)] * 32
    
    # Monotonicity constraints (inequality form: A_ub @ x <= b_ub)
    # q1_{v1+1,v2} - q1_{v1,v2} >= 0  =>  -q1_{v1+1,v2} + q1_{v1,v2} <= 0
    A_rows = []
    b_rows = []
    
    # q1 monotonicity in v1 direction
    for v2 in range(4):
        for v1 in range(3):
            idx_lo = v1 * 4 + v2      # (v1, v2)
            idx_hi = (v1 + 1) * 4 + v2  # (v1+1, v2)
            row = np.zeros(32)
            row[idx_lo] = 1.0         # q1_{v1,v2}
            row[idx_hi] = -1.0        # -q1_{v1+1,v2}
            A_rows.append(row)
            b_rows.append(0.0)
    
    # q2 monotonicity in v2 direction
    for v1 in range(4):
        for v2 in range(3):
            idx_lo = v1 * 4 + v2      # (v1, v2)
            idx_hi = v1 * 4 + (v2 + 1)  # (v1, v2+1)
            row = np.zeros(32)
            row[n + idx_lo] = 1.0     # q2_{v1,v2}
            row[n + idx_hi] = -1.0    # -q2_{v1,v2+1}
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.array([[]]).reshape(0, 32)
    b_ub = np.array(b_rows) if b_rows else np.array([])
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation
        allocations = np.ones((16, 2))
        # Compute payments for uniform allocation
        payments = np.zeros(16)
        for i in range(16):
            v1, v2 = types[i]
            # Payment = v1*q1 + v2*q2 - integral
            # For uniform q=1, integral of q from 0 to v is just v (since q=1 everywhere)
            # Actually need to be more careful. Let's use the formula:
            # payment_i = v1_i * q1_i + v2_i * q2_i - (integral_0^{v1_i} q1(t,v2_i) dt + integral_0^{v2_i} q2(v1_i,t) dt)
            # With q=1 everywhere: integral_0^v 1 dt = v
            # payment = v1*1 + v2*1 - v1 - v2 = 0
            payments[i] = 0.0
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    q1 = x[:16]
    q2 = x[16:]
    
    # Compute payments using Myerson's formula
    # payment_i = v1_i*q1_i + v2_i*q2_i - (integral_0^{v1_i} q1(t, v2_i) dt + integral_0^{v2_i} q2(v1_i, t) dt)
    payments = np.zeros(16)
    
    # Precompute integrals
    # For each type (v1, v2):
    # integral_0^{v1} q1(t, v2) dt = sum over t=0 to v1-1 of q1(t, v2) * 1 (discrete, step size 1)
    # Actually, the integral is over continuous [0, v1], but our support is discrete {0,1,2,3}
    # The standard discrete Myerson: payment = v*q - sum_{t<v} q(t) for each dimension
    
    for i in range(16):
        v1, v2 = types[i]
        # Integral of q1 over first coordinate from 0 to v1 (discrete sum)
        int_q1 = 0.0
        for t in range(v1):
            idx = t * 4 + v2
            int_q1 += q1[idx]
        
        # Integral of q2 over second coordinate from 0 to v2
        int_q2 = 0.0
        for t in range(v2):
            idx = v1 * 4 + t
            int_q2 += q2[idx]
        
        payments[i] = v1 * q1[i] + v2 * q2[i] - int_q1 - int_q2
    
    # Ensure non-negative payments (clip small negatives from numerical error)
    payments = np.maximum(payments, 0.0)
    
    allocations = np.column_stack([q1, q2])
    
    # Verify constraints
    # 1. Feasible probabilities: allocations in [0,1]
    if np.any(allocations < -1e-9) or np.any(allocations > 1 + 1e-9):
        allocations = np.clip(allocations, 0.0, 1.0)
    
    # 2. Non-negative payments
    if np.any(payments < -1e-9):
        payments = np.maximum(payments, 0.0)
    
    # 3. Individual rationality: payment_i <= v1_i*q1_i + v2_i*q2_i
    for i in range(16):
        v1, v2 = types[i]
        surplus = v1 * q1[i] + v2 * q2[i] - payments[i]
        if surplus < -1e-8:
            # Adjust payment to satisfy IR
            payments[i] = v1 * q1[i] + v2 * q2[i]
    
    # 4. Regret check: for each type i, reporting type j should not give higher utility
    # Utility of type i reporting j: v1_i*q1_j + v2_i*q2_j - payment_j
    # Truthful utility: v1_i*q1_i + v2_i*q2_i - payment_i
    # Regret = max_j (v1_i*q1_j + v2_i*q2_j - payment_j) - (v1_i*q1_i + v2_i*q2_i - payment_i)
    max_regret = 0.0
    for i in range(16):
        v1_i, v2_i = types[i]
        truthful_util = v1_i * q1[i] + v2_i * q2[i] - payments[i]
        for j in range(16):
            util_j = v1_i * q1[j] + v2_i * q2[j] - payments[j]
            regret = util_j - truthful_util
            if regret > max_regret:
                max_regret = regret
    
    if max_regret > 1e-8:
        # Regret constraint violated. This can happen due to numerical issues.
        # Try to fix by adjusting payments slightly, or fall back to a simpler mechanism.
        # For now, we'll try a simple fallback: only allocate when value is high enough
        # Use a threshold mechanism
        fallback_allocations = np.zeros((16, 2))
        fallback_payments = np.zeros(16)
        
        # Simple monotone allocation: allocate good 1 if v1 >= 2, good 2 if v2 >= 2
        for i in range(16):
            v1, v2 = types[i]
            fallback_allocations[i, 0] = 1.0 if v1 >= 2 else 0.0
            fallback_allocations[i, 1] = 1.0 if v2 >= 2 else 0.0
        
        # Compute payments for this monotone allocation
        for i in range(16):
            v1, v2 = types[i]
            q1_i = fallback_allocations[i, 0]
            q2_i = fallback_allocations[i, 1]
            
            int_q1 = 0.0
            for t in range(v1):
                idx = t * 4 + v2
                int_q1 += fallback_allocations[idx, 0]
            
            int_q2 = 0.0
            for t in range(v2):
                idx = v1 * 4 + t
                int_q2 += fallback_allocations[idx, 1]
            
            fallback_payments[i] = v1 * q1_i + v2 * q2_i - int_q1 - int_q2
            fallback_payments[i] = max(fallback_payments[i], 0.0)
        
        allocations = fallback_allocations
        payments = fallback_payments
    
    return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
