"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16, 2) array of probabilities, 'payments': (16,) array}
    """
    t0 = time.time()
    T_LIMIT = 85.0  # 5s safety margin from 90s limit
    
    # Generate all 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    N = 16
    V = np.array(types, dtype=np.float64)  # shape (16, 2)
    
    def _elapsed():
        return time.time() - t0
    
    def _parse_probs():
        if isinstance(distribution_spec, dict):
            if 'probs' in distribution_spec:
                arr = np.asarray(distribution_spec['probs'], dtype=np.float64)
                if arr.shape == (N,):
                    arr = np.maximum(arr, 0.0)
                    s = arr.sum()
                    return arr / s if s > 0 else np.ones(N) / N
            if 'distribution' in distribution_spec:
                d = distribution_spec['distribution']
                if d == 'uniform':
                    return np.ones(N) / N
                if d == 'geometric':
                    base = 0.7
                    arr = np.array([base**types[i][0] * base**types[i][1] for i in range(N)])
                    return arr / arr.sum()
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.asarray(distribution_spec, dtype=np.float64)
            if arr.shape == (N,):
                arr = np.maximum(arr, 0.0)
                s = arr.sum()
                if s > 0:
                    return arr / s
        return np.ones(N) / N
    
    probs = _parse_probs()
    
    if _elapsed() > T_LIMIT - 5:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
    
    # LP Variables: [q1_0..q1_15, q2_0..q2_15, p_0..p_15]
    # Total: 3*N = 48 variables
    n_vars = 3 * N
    
    # Objective: maximize sum(probs[i] * p_i) => minimize -sum(probs[i] * p_i)
    c_obj = np.zeros(n_vars)
    c_obj[2*N : 3*N] = -probs  # p variables are at indices 2N to 3N-1
    
    # Build constraints
    A_rows = []
    b_vals = []
    
    # IR constraints: p_i <= v1_i * q1_i + v2_i * q2_i
    # => -v1_i * q1_i - v2_i * q2_i + p_i <= 0
    for i in range(N):
        row = np.zeros(n_vars)
        row[i] = -V[i, 0]           # q1_i
        row[N + i] = -V[i, 1]       # q2_i
        row[2*N + i] = 1.0          # p_i
        A_rows.append(row)
        b_vals.append(0.0)
    
    # IC constraints: for all i != j:
    # u(i, j) <= u(i, i)
    # v1_i*q1_j + v2_i*q2_j - p_j <= v1_i*q1_i + v2_i*q2_i - p_i
    # v1_i*(q1_j - q1_i) + v2_i*(q2_j - q2_i) - p_j + p_i <= 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[j] = V[i, 0]        # v1_i * q1_j
            row[i] = -V[i, 0]       # -v1_i * q1_i
            row[N + j] = V[i, 1]    # v2_i * q2_j
            row[N + i] = -V[i, 1]   # -v2_i * q2_i
            row[2*N + j] = -1.0     # -p_j
            row[2*N + i] = 1.0      # +p_i
            A_rows.append(row)
            b_vals.append(0.0)
    
    A_ub = np.array(A_rows, dtype=np.float64)
    b_ub = np.array(b_vals, dtype=np.float64)
    
    # Bounds: q1, q2 in [0, 1], p in [0, 6] (max value = 3+3=6)
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 6.0)] * N
    
    # Solve LP
    remaining_time = T_LIMIT - _elapsed() - 5.0
    if remaining_time < 1.0:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
    
    try:
        res = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={
                'presolve': True,
                'dual_feasibility_tolerance': 1e-12,
                'primal_feasibility_tolerance': 1e-12,
                'time_limit': remaining_time
            }
        )
    except Exception:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
    
    if not res.success or res.x is None:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
    
    x = res.x
    q1 = np.clip(x[:N], 0.0, 1.0)
    q2 = np.clip(x[N:2*N], 0.0, 1.0)
    p = np.maximum(x[2*N:3*N], 0.0)
    
    # Post-processing: Fix numerical violations
    v1 = V[:, 0]
    v2 = V[:, 1]
    
    # Check and fix IR: p_i <= v1_i*q1_i + v2_i*q2_i
    util_true = v1 * q1 + v2 * q2
    ir_violation = p - util_true
    max_ir_viol = np.max(ir_violation)
    if max_ir_viol > 1e-10:
        p = p - max_ir_viol
        p = np.maximum(p, 0.0)
    
    # Verify IC: p_i - p_j <= v1_i*(q1_i - q1_j) + v2_i*(q2_i - q2_j)
    ic_ok = True
    max_ic_viol = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            lhs = p[i] - p[j]
            rhs = v1[i]*(q1[i] - q1[j]) + v2[i]*(q2[i] - q2[j])
            viol = lhs - rhs
            if viol > max_ic_viol:
                max_ic_viol = viol
            if viol > 1e-8:
                ic_ok = False
    
    if not ic_ok:
        # Try to fix by reducing payments uniformly
        # Uniform reduction doesn't change IC slack, so we need a different approach
        # Reduce all payments by the max IC violation
        if max_ic_viol > 0:
            p = p - max_ic_viol
            p = np.maximum(p, 0.0)
            
            # Re-check
            ic_ok2 = True
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    lhs = p[i] - p[j]
                    rhs = v1[i]*(q1[i] - q1[j]) + v2[i]*(q2[i] - q2[j])
                    if lhs - rhs > 1e-8:
                        ic_ok2 = False
                        break
                if not ic_ok2:
                    break
            
            if not ic_ok2:
                # Fallback: conservative mechanism
                return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}
    
    allocations = np.column_stack((q1, q2))
    
    return {'allocations': allocations, 'payments': p}
# EVOLVE-BLOCK-END
