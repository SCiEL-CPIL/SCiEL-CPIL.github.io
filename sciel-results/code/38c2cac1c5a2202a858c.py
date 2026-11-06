"""Incentive-compatible finite auction design via direct LP solve with polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=np.float64)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=np.float64)
        else:
            probs = None
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 16:
                    probs = np.array(val, dtype=np.float64)
                    break
            if probs is None:
                probs = np.ones(16, dtype=np.float64) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=np.float64)
        if len(probs) != 16:
            probs = np.ones(16, dtype=np.float64) / 16.0
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(16, dtype=np.float64) / 16.0
    
    # Parse support
    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i, j) for i in range(4) for j in range(4)]
    
    n_reports = len(support_list)
    if n_reports != 16:
        if n_reports < 16:
            support_list = support_list + [(3, 3)] * (16 - n_reports)
            probs = np.concatenate([probs, np.zeros(16 - len(probs))])
        else:
            support_list = support_list[:16]
            probs = probs[:16]
    
    # Ensure support values are floats
    support_list = [(float(s[0]), float(s[1])) for s in support_list]
    
    # Variables: x[i][j] for i in 0..15, j in 0..1 (32 vars)
    #           u[i] for i in 0..15 (utility, 16 vars)
    # Total: 48 vars
    #
    # Objective: maximize sum_i probs[i] * (v_i·x(i) - u(i))
    # This is equivalent to maximizing expected revenue since revenue = sum p_i * (v_i·x_i - u_i)
    # where u_i is the utility (v_i·x_i - p_i), so p_i = v_i·x_i - u_i
    # Revenue = sum p_i * (v_i·x_i - u_i) ... wait, let's re-derive.
    #
    # Actually, the standard formulation:
    # Maximize: sum_i pi_i * p_i
    # Subject to:
    #   p_i <= v_i . x_i  (IR)
    #   p_i - v_i . x_i <= p_j - v_i . x_j  for all i, j (IC)
    #   0 <= x_i <= 1
    #   p_i >= 0
    #
    # Alternative with utility variables u_i = v_i . x_i - p_i:
    #   p_i = v_i . x_i - u_i
    #   IR: u_i >= 0
    #   IC: u_i - u_j >= v_i . x_j - v_j . x_j  for all i, j
    #       => -u_i + u_j + v_i . x_j - v_j . x_j <= 0
    #   Objective: maximize sum_i pi_i * (v_i . x_i - u_i)
    #
    # This is what the current program uses. Let's stick with this.
    
    n_x = 32
    n_u = 16
    n_vars = n_x + n_u
    
    # Objective: maximize sum_i probs[i] * (v_i·x(i) - u(i))
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for i in range(16):
        v1, v2 = support_list[i]
        c[i * 2 + 0] = -probs[i] * v1
        c[i * 2 + 1] = -probs[i] * v2
        c[n_x + i] = probs[i]  # because we're negating -u(i)
    
    # Build dense constraint matrix
    A_ub_rows = []
    b_ub_vals = []
    
    # Constraint 1: x[i][j] <= 1 for each i, j
    for i in range(16):
        for j in range(2):
            row = np.zeros(n_vars)
            row[i * 2 + j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
    
    # Constraint 2: -u[i] <= 0 (i.e., u[i] >= 0) [IR]
    for i in range(16):
        row = np.zeros(n_vars)
        row[n_x + i] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # Constraint 3: IC: u(i) - u(j) >= v_i·x(j) - v_j·x(j) for all i != j
    # i.e., -u(i) + u(j) + v_i·x(j) - v_j·x(j) <= 0
    for i in range(16):
        v1_i, v2_i = support_list[i]
        for j in range(16):
            if i == j:
                continue
            v1_j, v2_j = support_list[j]
            row = np.zeros(n_vars)
            # -u(i)
            row[n_x + i] = -1.0
            # +u(j)
            row[n_x + j] = 1.0
            # +v_i·x(j)
            row[j * 2 + 0] = v1_i
            row[j * 2 + 1] = v2_i
            # -v_j·x(j)
            row[j * 2 + 0] -= v1_j
            row[j * 2 + 1] -= v2_j
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: x[i][j] in [0,1], u[i] >= 0 (unbounded above)
    bounds = []
    for i in range(16):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, None))
    
    # Solve LP with tight tolerances
    result = None
    # Try progressively tighter tolerances if time permits
    tolerances = [1e-12, 1e-13, 1e-14, 1e-15]
    
    for tol in tolerances:
        if time.time() > deadline - 10.0:
            break
        try:
            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                options={'presolve': True, 'dual_feasibility_tolerance': tol,
                         'primal_feasibility_tolerance': tol,
                         'integrality_tolerance': tol}
            )
            if result is not None and result.success:
                if tol <= 1e-14:
                    break
        except Exception:
            result = None
            break

    # Check time
    if time.time() > deadline:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if result is None or not result.success:
        # Fallback: deterministic allocation
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            v1, v2 = support_list[i]
            allocations[i] = [1.0 if v1 >= 1 else 0.0, 1.0 if v2 >= 1 else 0.0]
            payments[i] = 0.0
        return {'allocations': allocations, 'payments': payments}
    
    # Extract solution
    x_vals = result.x[:n_x].reshape(16, 2)
    u_vals = result.x[n_x:]
    
    # Compute payments: p(i) = v_i·x(i) - u(i)
    p_vals = np.zeros(16)
    for i in range(16):
        v1, v2 = support_list[i]
        p_vals[i] = v1 * x_vals[i, 0] + v2 * x_vals[i, 1] - u_vals[i]
    
    # Clip allocations to [0, 1]
    x_vals = np.clip(x_vals, 0.0, 1.0)
    
    # Ensure payments are non-negative
    p_vals = np.clip(p_vals, 0.0, None)
    
    # Verification: check IC and IR
    def check_violation(x, p):
        max_viol = 0.0
        for i in range(16):
            v1_i, v2_i = support_list[i]
            util_i = v1_i * x[i, 0] + v2_i * x[i, 1] - p[i]
            if util_i < -1e-9:
                max_viol = max(max_viol, -util_i)
            for j in range(16):
                if i == j:
                    continue
                v1_j, v2_j = support_list[j]
                # Utility of type i when reporting j: v_i dot x_j - p_j
                util_i_report_j = v1_i * x[j, 0] + v2_i * x[j, 1] - p[j]
                if util_i < util_i_report_j - 1e-9:
                    max_viol = max(max_viol, util_i_report_j - util_i)
        return max_viol
    
    max_violation = check_violation(x_vals, p_vals)
    
    if max_violation > 1e-8:
        # Polish pass 1: tighter linprog
        if time.time() < deadline - 5.0:
            try:
                result3 = linprog(
                    c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                    options={'presolve': True, 'dual_feasibility_tolerance': 1e-15,
                             'primal_feasibility_tolerance': 1e-15,
                             'integrality_tolerance': 1e-15,
                             'time_limit': max(1.0, deadline - 3.0 - time.time())}
                )
                if result3 is not None and result3.success:
                    x_vals3 = result3.x[:n_x].reshape(16, 2)
                    u_vals3 = result3.x[n_x:]
                    p_vals3 = np.zeros(16)
                    for i in range(16):
                        v1, v2 = support_list[i]
                        p_vals3[i] = v1 * x_vals3[i, 0] + v2 * x_vals3[i, 1] - u_vals3[i]
                    x_vals3 = np.clip(x_vals3, 0.0, 1.0)
                    p_vals3 = np.clip(p_vals3, 0.0, None)
                    max_violation3 = check_violation(x_vals3, p_vals3)
                    if max_violation3 <= 1e-8:
                        x_vals = x_vals3
                        p_vals = p_vals3
                        max_violation = max_violation3
            except Exception:
                pass
        
        # Polish pass 2: local repair via utility re-optimization
        if max_violation > 1e-8 and time.time() < deadline - 3.0:
            try:
                # Fix allocations, re-solve for utilities only (smaller LP)
                # Variables: u[0..15], minimize -sum p_i*(v_i.x_i - u_i) = -sum p_i*v_i.x_i + sum p_i*u_i
                # Since v_i.x_i is fixed, maximize sum p_i*u_i subject to IC+IR
                # Actually we want to minimize sum p_i*u_i (since revenue = sum p_i*(v_i.x_i - u_i))
                c_u = np.zeros(16)
                for i in range(16):
                    c_u[i] = probs[i]
                
                A_ub_u = []
                b_ub_u = []
                
                # IR: -u[i] <= 0
                for i in range(16):
                    row = np.zeros(16)
                    row[i] = -1.0
                    A_ub_u.append(row)
                    b_ub_u.append(0.0)
                
                # IC: -u(i) + u(j) + v_i.x(j) - v_j.x(j) <= 0
                for i in range(16):
                    v1_i, v2_i = support_list[i]
                    for j in range(16):
                        if i == j:
                            continue
                        v1_j, v2_j = support_list[j]
                        row = np.zeros(16)
                        row[i] = -1.0
                        row[j] = 1.0
                        # v_i.x(j) - v_j.x(j) is constant
                        const = v1_i * x_vals[j, 0] + v2_i * x_vals[j, 1] - \
                                v1_j * x_vals[j, 0] - v2_j * x_vals[j, 1]
                        A_ub_u.append(row)
                        b_ub_u.append(-const)
                
                A_ub_u = np.array(A_ub_u)
                b_ub_u = np.array(b_ub_u)
                
                bounds_u = [(0.0, None)] * 16
                
                result_u = linprog(c_u, A_ub=A_ub_u, b_ub=b_ub_u, bounds=bounds_u,
                                   method='highs',
                                   options={'presolve': True,
                                            'dual_feasibility_tolerance': 1e-15,
                                            'primal_feasibility_tolerance': 1e-15})
                
                if result_u is not None and result_u.success:
                    u_repaired = result_u.x
                    p_repaired = np.zeros(16)
                    for i in range(16):
                        v1, v2 = support_list[i]
                        p_repaired[i] = v1 * x_vals[i, 0] + v2 * x_vals[i, 1] - u_repaired[i]
                    p_repaired = np.clip(p_repaired, 0.0, None)
                    
                    # Re-verify
                    max_viol_r = check_violation(x_vals, p_repaired)
                    if max_viol_r <= 1e-8:
                        p_vals = p_repaired
                        max_violation = max_viol_r
            except Exception:
                pass
        
        # Polish pass 3: if still violating, try a second allocation repair
        if max_violation > 1e-8 and time.time() < deadline - 1.0:
            try:
                # Small perturbation: nudge x slightly toward feasibility
                # Try solving again with slightly different presolve
                result4 = linprog(
                    c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                    options={'presolve': False,
                             'dual_feasibility_tolerance': 1e-15,
                             'primal_feasibility_tolerance': 1e-15}
                )
                if result4 is not None and result4.success:
                    x_vals4 = result4.x[:n_x].reshape(16, 2)
                    u_vals4 = result4.x[n_x:]
                    p_vals4 = np.zeros(16)
                    for i in range(16):
                        v1, v2 = support_list[i]
                        p_vals4[i] = v1 * x_vals4[i, 0] + v2 * x_vals4[i, 1] - u_vals4[i]
                    x_vals4 = np.clip(x_vals4, 0.0, 1.0)
                    p_vals4 = np.clip(p_vals4, 0.0, None)
                    max_violation4 = check_violation(x_vals4, p_vals4)
                    if max_violation4 <= 1e-8:
                        x_vals = x_vals4
                        p_vals = p_vals4
                        max_violation = max_violation4
            except Exception:
                pass
        
        if max_violation > 1e-8:
            # Significant violation: conservative fallback
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': x_vals, 'payments': p_vals}
# EVOLVE-BLOCK-END
