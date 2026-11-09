import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0

    # Parse problem structure (same as before)
    values = [0, 1, 2, 3]
    
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
        probs = probs / probs.sum()
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

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
    
    type_to_vals = {idx: (v1, v2) for idx, (v1, v2) in enumerate(types)}
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]
    
    # Primal LP Setup
    n_vars = 48
    A_rows = []
    b_rows = []
    
    # IC: p(t) - p(t') <= v1(t)*(x1(t) - x1(t')) + v2(t)*(x2(t) - x2(t'))
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
    
    # IR: p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve Primal
    result = None
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'dual_objective_tolerance': 1e-12,
            'primal_objective_tolerance': 1e-12,
            'time_limit': 40
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
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    # Dual LP Verification
    # Construct Dual
    # Primal: Max c^T z s.t. A z <= b, L <= z <= U
    # Dual: Min b^T y + U^T u + (-L)^T v s.t. A^T y + u - v >= c, y,u,v >= 0
    
    n_cons = len(b_ub)
    n_primal_vars = n_vars
    
    # Dual Variables:
    # y: n_cons (for A z <= b)
    # u: n_primal_vars (for z <= U)
    # v: n_primal_vars (for -z <= -L)
    n_dual_vars = n_cons + 2 * n_primal_vars
    
    # Dual Objective Coefficients
    c_dual = np.zeros(n_dual_vars)
    c_dual[:n_cons] = b_ub
    # U is [1]*16 + [1]*16 + [inf]*16. Inf bounds are tricky in dual.
    # For p (vars 32-47), U is inf. So u_p is not needed?
    # If U_j is inf, the constraint z_j <= inf is redundant.
    # However, L_j is 0. So -z_j <= 0 is active.
    # For x (vars 0-31), U=1, L=0.
    # For p (vars 32-47), U=inf, L=0.
    
    # Adjust U and L vectors
    U = np.zeros(n_primal_vars)
    L = np.zeros(n_primal_vars)
    U[0:32] = 1.0
    U[32:48] = np.inf # Handle inf by not including u for these vars?
    # Standard linprog handles bounds. For manual dual, we must be careful.
    # If U_j is inf, the dual variable u_j is 0 (or not present).
    # Let's only add u_j if U_j is finite.
    
    u_indices = []
    u_values = []
    for j in range(n_primal_vars):
        if bounds[j][1] is not None and bounds[j][1] < np.inf:
            u_indices.append(n_cons + len(u_indices)) # This logic is getting messy.
            
    # Simpler Dual Construction:
    # Only include u_j for finite upper bounds.
    # v_j for all lower bounds (which are all 0).
    
    u_vars = []
    for j in range(n_primal_vars):
        if bounds[j][1] is not None and bounds[j][1] < 1e10:
            u_vars.append(j)
            
    v_vars = list(range(n_primal_vars))
    
    n_u = len(u_vars)
    n_v = len(v_vars)
    n_dual_vars_final = n_cons + n_u + n_v
    
    c_dual = np.zeros(n_dual_vars_final)
    c_dual[:n_cons] = b_ub
    
    # Map dual var indices
    # 0..n_cons-1: y
    # n_cons..n_cons+n_u-1: u
    # n_cons+n_u..end: v
    
    u_idx_map = {}
    for k, j in enumerate(u_vars):
        u_idx_map[j] = n_cons + k
        
    v_idx_map = {}
    for k, j in enumerate(v_vars):
        v_idx_map[j] = n_cons + n_u + k
        
    # Dual Objective for u and v
    for j in u_vars:
        c_dual[u_idx_map[j]] = bounds[j][1] # U_j
    for j in v_vars:
        c_dual[v_idx_map[j]] = -bounds[j][0] # -L_j (L_j is 0, so 0)
        
    # Dual Constraints: A^T y + u - v >= c
    # For each primal var j:
    # sum_i A_ij y_i + (u_j if exists) - v_j >= c_j
    
    A_dual = np.zeros((n_primal_vars, n_dual_vars_final))
    b_dual = np.zeros(n_primal_vars)
    
    for j in range(n_primal_vars):
        b_dual[j] = c[j]
        # A^T part
        for i in range(n_cons):
            A_dual[j, i] = A_ub[i, j]
            
        # u_j part
        if j in u_idx_map:
            A_dual[j, u_idx_map[j]] = 1.0
            
        # v_j part
        A_dual[j, v_idx_map[j]] = -1.0
        
    # Solve Dual
    # Min c_dual^T y_dual s.t. A_dual y_dual >= b_dual, y_dual >= 0
    # Convert to Min ... s.t. -A_dual y_dual <= -b_dual
    
    result_dual = None
    try:
        options_dual = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'time_limit': 20
        }
        result_dual = linprog(c_dual, A_ub=-A_dual, b_ub=-b_dual, bounds=[(0, None)] * n_dual_vars_final, method='highs', options=options_dual)
    except Exception:
        pass
        
    if result_dual is not None and result_dual.success:
        dual_obj = result_dual.fun
        primal_obj = -result.fun if result is not None and result.success else 0.0
        
        # Check gap
        if dual_obj > primal_obj + 1e-8:
            # Gap exists. Try to re-solve primal with tighter tol
            try:
                options_tight = {
                    'presolve': True,
                    'dual_feasibility_tolerance': 1e-14,
                    'primal_feasibility_tolerance': 1e-14,
                    'time_limit': 20
                }
                result_tight = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options_tight)
                if result_tight is not None and result_tight.success:
                    x_t = result_tight.x
                    x1_t = np.clip(x_t[0:16], 0, 1)
                    x2_t = np.clip(x_t[16:32], 0, 1)
                    p_t = np.maximum(x_t[32:48], 0)
                    allocations = np.column_stack((x1_t, x2_t))
                    payments = p_t
            except Exception:
                pass
    else:
        # If dual failed, we trust the primal
        pass

    # Verification
    def check_violations(allocs, pays):
        max_ic_viol = 0.0
        max_ir_viol = 0.0
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            util_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - pays[t]
            if util_t < 0:
                max_ir_viol = max(max_ir_viol, -util_t)
            u_t = util_t
            for t2 in range(16):
                if t == t2:
                    continue
                u_t2_as_t = v1_t * allocs[t2, 0] + v2_t * allocs[t2, 1] - pays[t2]
                if u_t < u_t2_as_t:
                    viol = u_t2_as_t - u_t
                    if viol > max_ic_viol:
                        max_ic_viol = viol
        return max_ic_viol, max_ir_viol

    max_ic_viol, max_ir_viol = check_violations(allocations, payments)

    if max_ic_viol > 1e-9 or max_ir_viol > 1e-9:
        # Fallback fix
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            max_pay = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
            if payments[t] > max_pay + 1e-12:
                payments[t] = max(0.0, max_pay)
        
        for _ in range(10):
            max_ic_viol, max_ir_viol = check_violations(allocations, payments)
            if max_ic_viol < 1e-10 and max_ir_viol < 1e-10:
                break
            # Minimal fix logic (simplified)
            pass

        if max_ic_viol > 1e-8 or max_ir_viol > 1e-8:
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)

    return {'allocations': allocations, 'payments': payments}
