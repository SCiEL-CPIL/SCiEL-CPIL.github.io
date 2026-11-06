"""Improved program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction with one additive bidder, two goods,
    values in {0,1,2,3}, using the optimal mechanism derived from an LP.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    start_time = time.time()
    DEADLINE = 80.0  # seconds, safety margin before 90s kill

    # 1. Define Types and Bundles
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    n_types = 16
    n_bundles = 4
    bundles = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # 2. Parse Distribution
    probs = np.zeros(16)
    try:
        if isinstance(distribution_spec, dict):
            for i, t in enumerate(types):
                if t in distribution_spec:
                    probs[i] = distribution_spec[t]
                elif str(t) in distribution_spec:
                    probs[i] = distribution_spec[str(t)]
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(16) / 16.0
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == 16:
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = np.ones(16) / 16.0
            else:
                probs = np.ones(16) / 16.0
        else:
            probs = np.ones(16) / 16.0
    except Exception:
        probs = np.ones(16) / 16.0

    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0

    # 3. Setup Value Matrix
    val = np.zeros((n_types, n_bundles))
    for i, (v1, v2) in enumerate(types):
        for j, (x1, x2) in enumerate(bundles):
            val[i, j] = v1 * x1 + v2 * x2

    # 4. Define LP Variables
    # x[i, j] for i in 0..15, j in 0..3 (64 vars)
    # p[i] for i in 0..15 (16 vars)
    # Total vars: 80

    n_vars = 80

    # Objective: Maximize sum_i probs[i] * p[i]
    # Minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[64 + i] = -probs[i]

    # 5. Equality Constraints
    # Sum_j x[i, j] = 1 for all i
    A_eq = np.zeros((n_types, n_vars))
    b_eq = np.ones(n_types)
    for i in range(n_types):
        for j in range(n_bundles):
            A_eq[i, i * n_bundles + j] = 1

    # 6. Inequality Constraints
    # A_ub * vars <= b_ub

    # IR: p[i] <= val[i] . x[i]  =>  p[i] - sum_j val[i,j]*x[i,j] <= 0
    # IC: u[i] >= u[j]  =>  val[i].x[i] - p[i] >= val[i].x[j] - p[j]
    #     => p[i] - p[j] + val[i].x[j] - val[i].x[i] <= 0

    n_ineq = n_types + n_types * (n_types - 1)
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    row = 0
    # IR Constraints
    for i in range(n_types):
        A_ub[row, 64 + i] = 1  # p[i]
        for j in range(n_bundles):
            A_ub[row, i * n_bundles + j] = -val[i, j]
        b_ub[row] = 0
        row += 1

    # IC Constraints
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            A_ub[row, 64 + i] = 1
            A_ub[row, 64 + j] = -1
            for k in range(n_bundles):
                A_ub[row, j * n_bundles + k] += val[i, k]
                A_ub[row, i * n_bundles + k] -= val[i, k]
            b_ub[row] = 0
            row += 1

    bounds = [(0, None)] * n_vars

    x_opt = None
    p_opt = None

    # 7. Solve LP with tight tolerances
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12})

        if result.status == 0:
            x_opt = result.x[:64].reshape(n_types, n_bundles)
            p_opt = result.x[64:80]
        else:
            pass
    except Exception:
        pass

    if x_opt is not None:
        # Clean up numerical noise
        x_opt = np.clip(x_opt, 0, 1)
        
        # Renormalize allocations to ensure sum_j x[i,j] = 1
        for i in range(n_types):
            s = x_opt[i].sum()
            if s > 0:
                x_opt[i] = x_opt[i] / s
            else:
                # Fallback for zero row (should not happen if LP is correct)
                x_opt[i] = np.ones(n_bundles) / n_bundles
        
        p_opt = np.clip(p_opt, 0, None)

        # 8. Re-optimize payments for the fixed x to ensure strict feasibility
        # This handles any drift in x caused by clipping/renormalization
        try:
            # Aux LP: Max sum probs[i] p[i]
            # s.t. p[i] - sum_j val[i,j] x[i,j] <= 0 (IR)
            #      p[i] - p[j] - val[i].x[i] + val[i].x[j] <= 0 (IC)
            #      p[i] >= 0
            
            n_p = n_types
            c_p = -probs
            
            A_ub_p = np.zeros((n_types + n_types*(n_types-1), n_p))
            b_ub_p = np.zeros(n_types + n_types*(n_types-1))
            
            row_p = 0
            # IR
            for i in range(n_types):
                A_ub_p[row_p, i] = 1
                b_ub_p[row_p] = val[i] @ x_opt[i]
                row_p += 1
                
            # IC
            for i in range(n_types):
                for j in range(n_types):
                    if i == j: continue
                    A_ub_p[row_p, i] = 1
                    A_ub_p[row_p, j] = -1
                    b_ub_p[row_p] = val[i] @ x_opt[i] - val[i] @ x_opt[j]
                    row_p += 1
                    
            bounds_p = [(0, None)] * n_p
            
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                            options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                     'primal_feasibility_tolerance': 1e-12})
            
            if res_p.status == 0:
                p_opt = res_p.x
            else:
                # If aux LP fails, use original p_opt (might be infeasible, but better than 0?)
                # Or fall back to 0?
                # Let's try to use p_opt from main LP, but verify
                pass
                
        except Exception:
            pass

        # Final Verification
        max_violation = 0.0
        for i in range(n_types):
            u_i = val[i] @ x_opt[i] - p_opt[i]
            if u_i < -max_violation:
                max_violation = -u_i
            for j in range(n_types):
                if i == j: continue
                u_i_xi = val[i] @ x_opt[i] - p_opt[i]
                u_i_xj = val[i] @ x_opt[j] - p_opt[j]
                if u_i_xj - u_i_xi > max_violation:
                    max_violation = u_i_xj - u_i_xi
        
        # Check allocation feasibility
        for i in range(n_types):
            if x_opt[i].min() < -1e-8:
                max_violation = max(max_violation, -x_opt[i].min())
            s = x_opt[i].sum()
            if abs(s - 1) > 1e-8:
                max_violation = max(max_violation, abs(s - 1))

        if max_violation <= 1e-8:
            allocations = np.zeros((n_types, 2))
            for i in range(n_types):
                for j, (x1, x2) in enumerate(bundles):
                    allocations[i, 0] += x1 * x_opt[i, j]
                    allocations[i, 1] += x2 * x_opt[i, j]

            return {
                'allocations': allocations,
                'payments': p_opt
            }
        else:
            # If still infeasible, return 0 mechanism
            pass

    # 9. Fallback: Safe zero mechanism (always IC and IR)
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)

    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
