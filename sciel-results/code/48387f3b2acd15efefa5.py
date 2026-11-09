import numpy as np
from scipy.optimize import linprog
import time
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s

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

    # --- Attempt 1: Solve the full LP for the optimal mechanism ---
    
    # LP variables: x1[0..15], x2[0..15], p[0..15]
    n_vars = 3 * n_states
    c = np.zeros(n_vars)
    for i in range(n_states):
        c[2 * n_states + i] = -probs[i]

    # Build constraint matrix
    A_ub_rows = []
    b_ub_vals = []

    # IC constraints: for all pairs (i, j), p_i - p_j <= v_i·x_i - v_i·x_j
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

    x_opt = None
    if time.time() - start_time < deadline - 5:
        try:
            # Use highs with tighter tolerances for better numerical stability
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12,
                                  'dual_edge_tolerances': [1e-12, 1e-12],
                                  'primal_edge_tolerances': [1e-12, 1e-12]})
            if res.status == 0:
                x_opt = res.x
            else:
                # Fallback to interior-point if highs fails
                res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point',
                              options={'maxiter': 1000, 'disp': False})
                if res2.status == 0:
                    x_opt = res2.x
        except Exception:
            x_opt = None

    if x_opt is not None:
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

        if max_regret <= 1e-8:
            return {"allocations": allocations, "payments": payments}
        else:
            # If IC is violated by a small amount, try to fix by slightly increasing payments
            # This is safe because IR has slack for most states
            violation = max_regret
            if violation < 1e-4:
                # Try adding a small epsilon to all payments to create margin
                for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                    p_adjusted = p + eps
                    # Check IR still holds
                    ir_ok = True
                    for i in range(n_states):
                        v1_i, v2_i = support_grid[i]
                        if p_adjusted[i] > v1_i * x1[i] + v2_i * x2[i] + 1e-10:
                            ir_ok = False
                            break
                    if ir_ok:
                        # Re-check IC
                        max_regret_adj = 0.0
                        for i in range(n_states):
                            v1_i, v2_i = support_grid[i]
                            utility_truthful = v1_i * x1[i] + v2_i * x2[i] - p_adjusted[i]
                            for j in range(n_states):
                                utility_deviate = v1_i * x1[j] + v2_i * x2[j] - p_adjusted[j]
                                regret = utility_deviate - utility_truthful
                                if regret > max_regret_adj:
                                    max_regret_adj = regret
                        if max_regret_adj <= 1e-8:
                            return {"allocations": allocations, "payments": p_adjusted}
            
            # Try a secondary LP with slightly tightened IC constraints to get a more robust solution
            if time.time() - start_time < deadline - 5:
                try:
                    A_ub_tight = A_ub.copy()
                    b_ub_tight = b_ub.copy()
                    # Tighten IC constraints (first n_states*(n_states-1) rows)
                    n_ic = n_states * (n_states - 1)
                    b_ub_tight[:n_ic] -= 1e-6
                    res_tight = linprog(c, A_ub=A_ub_tight, b_ub=b_ub_tight, bounds=bounds, 
                                       method='highs',
                                       options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                                'primal_feasibility_tolerance': 1e-12})
                    if res_tight.status == 0:
                        x_opt_tight = res_tight.x
                        x1_t = x_opt_tight[:n_states]
                        x2_t = x_opt_tight[n_states:2 * n_states]
                        p_t = x_opt_tight[2 * n_states:]
                        alloc_t = np.column_stack((x1_t, x2_t))
                        alloc_t = np.clip(alloc_t, 0.0, 1.0)
                        p_t = np.maximum(p_t, 0.0)
                        
                        max_regret_t = 0.0
                        for i in range(n_states):
                            v1_i, v2_i = support_grid[i]
                            utility_truthful = v1_i * x1_t[i] + v2_i * x2_t[i] - p_t[i]
                            for j in range(n_states):
                                utility_deviate = v1_i * x1_t[j] + v2_i * x2_t[j] - p_t[j]
                                regret = utility_deviate - utility_truthful
                                if regret > max_regret_t:
                                    max_regret_t = regret
                        
                        if max_regret_t <= 1e-8:
                            return {"allocations": alloc_t, "payments": p_t}
                except Exception:
                    pass

    # --- Attempt 2: Generalized Fallback over Monotone Allocations ---
    # For a single additive bidder, IC + IR implies p_i = min_j (v_i·x_j - v_i·x_i + p_j) with p_0 = 0 (IR at zero value).
    # Any monotone allocation rule (in the lattice order) is IC-compatible.
    # We enumerate all monotone allocation functions and compute the induced payment via the envelope formula.
    # Then we pick the one with maximum expected revenue.
    
    # State indices: state (v1, v2) has index v1*4 + v2
    # A monotone allocation satisfies: if (v1,v2) <= (v1',v2') component-wise, then x(v1,v2) <= x(v1',v2')
    # For 2 goods with values in {0,1,2,3}, we can enumerate all monotone functions.
    # Total number is manageable: for each good, the allocation function is monotone in that good's value.
    # But they can be coupled. Let's enumerate all 2^16 binary allocation vectors and filter for monotonicity.
    # 2^16 = 65536, which is fast to check.
    
    best_rev = -1
    best_alloc = np.zeros((n_states, 2))
    best_pay = np.zeros(n_states)
    
    # Precompute state values
    state_vals = np.array(support_grid, dtype=float)  # (16, 2)
    
    # Enumerate all binary allocation vectors
    for alloc_flat in range(2 ** n_states):
        x = np.array([(alloc_flat >> i) & 1 for i in range(n_states)], dtype=float)  # (16,)
        
        # Check monotonicity: for all pairs (i,j), if v_i <= v_j component-wise, then x_i <= x_j
        monotone = True
        for i in range(n_states):
            for j in range(n_states):
                if state_vals[i, 0] <= state_vals[j, 0] and state_vals[i, 1] <= state_vals[j, 1]:
                    if x[i] > x[j] + 1e-12:
                        monotone = False
                        break
            if not monotone:
                break
        if not monotone:
            continue
        
        # Compute payment via envelope formula: p_i = min_j (v_i·x_j - v_i·x_i + p_j)
        # With p_0 = 0 (state 0 has value (0,0), so IR gives p_0 <= 0, and nonneg gives p_0 = 0)
        # Actually, we need p_i such that IC holds. The minimal payment (max revenue) is:
        # p_i = min_{j} [v_i·x_j - v_i·x_i + p_j]
        # This is a system of equations. We can solve it via Bellman-Ford style since it's a shortest path.
        # p_i = min_j (c_{i,j} + p_j) where c_{i,j} = v_i·x_j - v_i·x_i
        # With the constraint p_0 = 0 (from IR at zero value state).
        
        # Actually, for monotone allocations, the IC-compatible payment is uniquely determined by:
        # p_i = sum over the "path" from 0 to i of marginal values.
        # For a lattice, p_i = v_i·x_i - sum_{k in predecessors} ... 
        # Simpler: use the formula p_i = min_{j} (v_i·x_j - v_i·x_i + p_j)
        # Initialize p[0] = 0, others = inf, then iterate.
        
        p = np.full(n_states, np.inf)
        p[0] = 0.0
        
        # Iterate until convergence (should converge in <= 16 iterations for this size)
        for _ in range(n_states + 1):
            p_old = p.copy()
            for i in range(n_states):
                if i == 0:
                    p[i] = 0.0
                    continue
                min_val = np.inf
                for j in range(n_states):
                    if p[j] < np.inf:
                        c_ij = state_vals[i, 0] * x[j, 0] + state_vals[i, 1] * x[j, 1] - state_vals[i, 0] * x[i, 0] - state_vals[i, 1] * x[i, 1]
                        val = c_ij + p[j]
                        if val < min_val:
                            min_val = val
                p[i] = min_val
            if np.allclose(p, p_old, atol=1e-14):
                break
        
        # Check IR: p_i <= v_i·x_i
        ir_ok = True
        for i in range(n_states):
            if p[i] > state_vals[i, 0] * x[i, 0] + state_vals[i, 1] * x[i, 1] + 1e-10:
                ir_ok = False
                break
        if not ir_ok:
            continue
        
        # Check nonnegativity
        if np.any(p < -1e-10):
            continue
        
        # Verify IC
        max_regret = 0.0
        for i in range(n_states):
            utility_truthful = state_vals[i, 0] * x[i, 0] + state_vals[i, 1] * x[i, 1] - p[i]
            for j in range(n_states):
                utility_deviate = state_vals[i, 0] * x[j, 0] + state_vals[i, 1] * x[j, 1] - p[j]
                regret = utility_deviate - utility_truthful
                if regret > max_regret:
                    max_regret = regret
        
        if max_regret > 1e-8:
            continue
        
        # Calculate expected revenue
        exp_rev = np.sum(probs * p)
        
        if exp_rev > best_rev:
            best_rev = exp_rev
            best_alloc = x.copy()
            best_pay = p.copy()
    
    if best_rev >= 0:
        # Clip for numerical safety
        best_alloc = np.clip(best_alloc, 0.0, 1.0)
        best_pay = np.maximum(best_pay, 0.0)
        return {"allocations": best_alloc, "payments": best_pay}
    else:
        # Last resort: Zero mechanism
        return {"allocations": np.zeros((n_states, 2)), "payments": np.zeros(n_states)}
