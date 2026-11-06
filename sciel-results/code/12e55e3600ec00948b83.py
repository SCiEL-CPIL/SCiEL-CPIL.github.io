<<<<<<< SEARCH
def _verify(allocations, payments, V, probs, n_types, tol=1e-8):
    """Check IR and IC constraints. Return (max_violation, expected_revenue)."""
    max_viol = 0.0
    for t in range(n_types):
        u_tt = float(np.dot(V[t], allocations[t])) - payments[t]
        if u_tt < -tol:
            max_viol = max(max_viol, -u_tt)
        for tp in range(n_types):
            if t == tp:
                continue
            u_t_tp = float(np.dot(V[t], allocations[tp])) - payments[tp]
            viol = u_t_tp - u_tt
            if viol > tol:
                max_viol = max(max_viol, viol)
    revenue = float(np.dot(probs, payments))
    return max_viol, revenue
=======
def _verify(allocations, payments, V, probs, n_types, tol=1e-8):
    """Check IR and IC constraints. Return (max_violation, expected_revenue)."""
    max_viol = 0.0
    for t in range(n_types):
        u_tt = float(np.dot(V[t], allocations[t])) - payments[t]
        if u_tt < -tol:
            max_viol = max(max_viol, -u_tt)
        for tp in range(n_types):
            if t == tp:
                continue
            u_t_tp = float(np.dot(V[t], allocations[tp])) - payments[tp]
            viol = u_t_tp - u_tt
            if viol > tol:
                max_viol = max(max_viol, viol)
    revenue = float(np.dot(probs, payments))
    return max_viol, revenue


def _polish_mechanism(allocations, payments, V, probs, n_types, max_iters=50):
    """High-precision polish: re-solve LP with tighter tolerances around current solution."""
    def x_idx(g, t):
        return g * n_types + t
    def u_idx(t):
        return 2 * n_types + t
    
    n_vars = 3 * n_types
    c = np.zeros(n_vars)
    for t in range(n_types):
        for g in range(2):
            c[x_idx(g, t)] = -probs[t] * V[t, g]
        c[u_idx(t)] = probs[t]
    
    A_rows = []
    b_vals = []
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            for g in range(2):
                row[x_idx(g, t)] += V[t, g]
                row[x_idx(g, tp)] -= V[t, g]
            row[u_idx(t)] -= 1.0
            row[u_idx(tp)] += 1.0
            A_rows.append(row)
            b_vals.append(0.0)
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[u_idx(t)] = -1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0.0, 1.0)] * (2 * n_types) + [(0.0, None)] * n_types
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                         method='highs',
                         options={'presolve': True,
                                  'dual_feasibility_tolerance': 1e-15,
                                  'primal_feasibility_tolerance': 1e-15})
        if result.success:
            x_opt = result.x
            new_alloc = np.zeros((n_types, 2))
            new_pay = np.zeros(n_types)
            for t in range(n_types):
                for g in range(2):
                    new_alloc[t, g] = x_opt[x_idx(g, t)]
                new_pay[t] = float(np.dot(V[t], new_alloc[t])) - x_opt[u_idx(t)]
            new_pay = np.maximum(new_pay, 0.0)
            new_alloc = np.clip(new_alloc, 0.0, 1.0)
            viol, rev = _verify(new_alloc, new_pay, V, probs, n_types)
            if viol <= 1e-8:
                return new_alloc, new_pay
    except Exception:
        pass
    return allocations, payments
>>>>>>> REPLACE

Now update `construct_auction` to use the polish:

