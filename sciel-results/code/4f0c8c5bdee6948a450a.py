import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    """
    start_time = time.time()
    DEADLINE = start_time + 80.0

    # Parse the problem structure
    values = [0, 1, 2, 3]

    # Get distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None

    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = np.maximum(probs, 0)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(16) / 16.0

    # Determine types
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]

    n_types = len(types)
    assert n_types == 16, f"Expected 16 types, got {n_types}"

    # type_to_vals mapping
    type_to_vals = {}
    for idx, (v1, v2) in enumerate(types):
        type_to_vals[idx] = (v1, v2)

    # Precompute arrays
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]

    # LP Formulation
    # Variables: x1[0..15], x2[16..31], p[32..47]
    n_vars = 48

    A_rows = []
    b_rows = []

    # IC constraints: p(t) - p(t') <= v1(t)*(x1(t) - x1(t')) + v2(t)*(x2(t) - x2(t'))
    # => -v1(t)*x1(t) + v1(t)*x1(t') - v2(t)*x2(t) + v2(t)*x2(t') + p(t) - p(t') <= 0
    for t in range(16):
        for t2 in range(16):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]
            row[t2] = v1_all[t]
            row[16 + t] = -v2_all[t]
            row[16 + t2] = v2_all[t]
            row[32 + t] = 1.0
            row[32 + t2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)

    # IR constraints: p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    # => -v1(t)*x1(t) - v2(t)*x2(t) + p(t) <= 0
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Objective: maximize sum p(t) * probs[t]
    c = np.zeros(n_vars)
    c[32:48] = -probs

    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16

    # Solve LP
    result = None
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-14,
            'primal_feasibility_tolerance': 1e-14,
            'dual_objective_tolerance': 1e-14,
            'primal_objective_tolerance': 1e-14,
            'time_limit': 60
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        pass

    if result is not None and result.success:
        x = result.x
        x1 = np.clip(x[0:16], 0, 1)
        x2 = np.clip(x[16:32], 0, 1)
        p = np.maximum(x[32:48], 0)
        allocations = np.column_stack((x1, x2))
        payments = p
    else:
        # Fallback: Zero mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)

    # Verification and High-Precision Polish
    def check_violations(allocs, pays):
        """Returns max IC violation and max IR violation."""
        max_ic_viol = 0.0
        max_ir_viol = 0.0
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            # IR check
            util_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - pays[t]
            if util_t < 0:
                max_ir_viol = max(max_ir_viol, -util_t)
            
            # IC check: u(t) >= u(t') for all t'
            u_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - pays[t]
            for t2 in range(16):
                if t == t2:
                    continue
                u_t2_as_t = v1_t * allocs[t2, 0] + v2_t * allocs[t2, 1] - pays[t2]
                if u_t < u_t2_as_t:
                    viol = u_t2_as_t - u_t
                    if viol > max_ic_viol:
                        max_ic_viol = viol
        return max_ic_viol, max_ir_viol

    def polish_solution(allocs, pays, max_iter=200):
        """
        Minimally adjust payments to ensure strict feasibility (IC and IR).
        Strategy:
        1. Enforce IR: p(t) <= v(t) . x(t)
        2. Enforce IC: For each pair (t, t2) where u(t) < u(t2 as t),
           increase p(t) minimally to close the gap, respecting IR bound.
           If IR bound is hit, we might need to decrease p(t2) or accept violation.
           Given the LP solution should be close, increasing p(t) is usually sufficient.
        """
        allocs = allocs.copy()
        pays = pays.copy()
        
        for _ in range(max_iter):
            changed = False
            
            # Pass 1: Fix IR
            for t in range(16):
                v1_t, v2_t = type_to_vals[t]
                max_pay = v1_t * allocs[t, 0] + v2_t * allocs[t, 1]
                if pays[t] > max_pay + 1e-15:
                    pays[t] = max(0.0, max_pay)
                    changed = True
            
            # Pass 2: Fix IC
            for t in range(16):
                v1_t, v2_t = type_to_vals[t]
                u_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - pays[t]
                
                # Find the maximum utility among all other reports
                max_u_other = -np.inf
                for t2 in range(16):
                    if t == t2:
                        continue
                    u_t2_as_t = v1_t * allocs[t2, 0] + v2_t * allocs[t2, 1] - pays[t2]
                    if u_t2_as_t > max_u_other:
                        max_u_other = u_t2_as_t
                
                # If u_t < max_u_other, we have an IC violation
                if u_t < max_u_other - 1e-15:
                    # We need to increase pays[t] such that:
                    # v1_t*x1_t + v2_t*x2_t - new_pays[t] >= max_u_other
                    # new_pays[t] <= v1_t*x1_t + v2_t*x2_t - max_u_other
                    # Wait, IC requires u_t >= u_other.
                    # u_t = val_t - p_t.
                    # val_t - p_t >= u_other  =>  p_t <= val_t - u_other.
                    # This is an UPPER bound on p_t.
                    # If current p_t is already low, u_t is high.
                    # If u_t < u_other, it means p_t is TOO LOW? No.
                    # u_t = val_t - p_t. If u_t is small, p_t is large.
                    # We want u_t to be larger. So we must DECREASE p_t.
                    
                    # Let's re-verify the logic.
                    # IC: val_t - p_t >= val_t_as_t2 - p_t2
                    # If violated: val_t - p_t < val_t_as_t2 - p_t2
                    # => p_t > val_t - val_t_as_t2 + p_t2
                    # We need to reduce p_t to satisfy this.
                    
                    required_max_pay = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - max_u_other
                    # We need p_t <= required_max_pay.
                    # Also p_t >= 0.
                    # If p_t > required_max_pay, we set p_t = required_max_pay.
                    
                    if pays[t] > required_max_pay + 1e-15:
                        new_pay = max(0.0, required_max_pay)
                        if new_pay < pays[t]:
                            pays[t] = new_pay
                            changed = True
                            # Note: decreasing p_t increases u_t, which helps IC for t,
                            # but might hurt IC for other types t' that compare against t.
                            # Specifically, if t' compares to t, u_t' >= u_t_as_t'.
                            # u_t_as_t' = val_t' . x_t - p_t.
                            # Decreasing p_t increases u_t_as_t', making it harder for t' to satisfy IC.
                            # So this simple projection might oscillate or fail.
                            # However, for the optimal LP solution, the payments are usually at the boundary.
                            # Let's see if we need a more robust method.
            
            if not changed:
                break
        
        return allocs, pays

    # Apply polish
    allocations, payments = polish_solution(allocations, payments)

    # Final Verification
    max_ic_viol, max_ir_viol = check_violations(allocations, payments)
    
    if max_ic_viol > 1e-8 or max_ir_viol > 1e-8:
        # If polish failed, try a more aggressive fix:
        # Reduce all payments to 0? No, that reduces revenue.
        # Try solving again with slightly tighter tolerances or different method?
        # Or just accept the LP solution if it's close?
        # The task says "require ... regret <= 1e-8".
        # If we can't meet it, we should probably return the zero mechanism to be safe,
        # or try to find a feasible point.
        
        # Let's try one more time with a different approach:
        # Compute the "minimal" payments for a given allocation that satisfy IC and IR.
        # For a fixed allocation X, the minimal payments P that satisfy IC and IR are:
        # P(t) = max(0, max_{t'} [v(t) . X(t') - v(t') . X(t') + P(t')]) ... this is recursive.
        
        # Actually, for a fixed allocation, the IC constraints define a system.
        # Let u(t) be the utility.
        # u(t) >= v(t) . X(t') - P(t') for all t'.
        # P(t) = v(t) . X(t) - u(t).
        # So u(t) >= v(t) . X(t') - (v(t') . X(t') - u(t')) = v(t) . X(t') - v(t') . X(t') + u(t').
        # This is a set of difference constraints: u(t) - u(t') >= v(t) . X(t') - v(t') . X(t').
        # We want to MAXIMIZE revenue, which means MINIMIZING P(t), which means MAXIMIZING u(t).
        # Wait, P(t) = v(t).X(t) - u(t). To max sum P(t), we min sum u(t).
        # So we want the MINIMAL feasible utilities u(t).
        # The minimal u(t) satisfying u(t) >= u(t') + C(t, t') is given by the shortest path (or longest path in negated graph) in the constraint graph.
        # Let's compute the minimal u(t) for the given allocation.
        
        # Graph nodes: 0..15.
        # Edge t' -> t with weight w = v(t) . X(t') - v(t') . X(t').
        # Constraint: u(t) >= u(t') + w.
        # This is a longest path problem if no negative cycles (which correspond to IC infeasibility).
        # Since X is from an LP, it should be feasible.
        
        # We can solve this by Bellman-Ford or simply by iterating relaxation.
        u_min = np.zeros(16)
        # Initialize u_min to 0? No, we need the lower bounds.
        # IR: P(t) <= v(t).X(t) => u(t) >= 0.
        # So u(t) >= 0 is a base constraint.
        
        # Let's relax:
        for _ in range(32):
            changed = False
            for t in range(16):
                for t2 in range(16):
                    if t == t2:
                        continue
                    # u(t) >= u(t2) + v(t).X(t2) - v(t2).X(t2)
                    rhs = u_min[t2] + v1_all[t] * allocations[t2, 0] + v2_all[t] * allocations[t2, 1] - (v1_all[t2] * allocations[t2, 0] + v2_all[t2] * allocations[t2, 1])
                    if u_min[t] < rhs - 1e-15:
                        u_min[t] = rhs
                        changed = True
            # Also enforce u(t) >= 0
            for t in range(16):
                if u_min[t] < 0:
                    u_min[t] = 0
                    changed = True
            if not changed:
                break
        
        # Now compute payments from u_min
        new_payments = np.zeros(16)
        for t in range(16):
            new_payments[t] = v1_all[t] * allocations[t, 0] + v2_all[t] * allocations[t, 1] - u_min[t]
            if new_payments[t] < 0:
                new_payments[t] = 0
        
        # Check if this new payment vector is feasible
        max_ic_viol2, max_ir_viol2 = check_violations(allocations, new_payments)
        
        if max_ic_viol2 < 1e-8 and max_ir_viol2 < 1e-8:
            payments = new_payments
        else:
            # Last resort: zero mechanism
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)

    # Deadline guard
    if time.time() > DEADLINE:
        pass

    return {'allocations': allocations, 'payments': payments}
