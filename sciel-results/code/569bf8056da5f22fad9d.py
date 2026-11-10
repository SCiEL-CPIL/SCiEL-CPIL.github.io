"""Direct Myerson auction with per-good optimal reserves from marginal revenue analysis.
Avoids LP solvers and ensures IC by construction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: dict with 'type' key indicating which distribution
        support: list of value tuples (v1, v2) where each vi in {0,1,2,3}
        constraints: dict with constraint specifications
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    t_start = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    # Parse inputs
    values = sorted(set(v for tup in support for v in tup))
    # Ensure we have exactly the 4 values
    if len(values) != 4:
        values = [0, 1, 2, 3]
    
    # Build the 16-type grid
    types = [(v1, v2) for v1 in values for v2 in values]
    type_index = {t: i for i, t in enumerate(types)}
    n_types = len(types)
    
    # Get distribution
    if isinstance(distribution_spec, dict):
        dist_type = distribution_spec.get('type', 'uniform')
    else:
        dist_type = str(distribution_spec)
    
    # Compute probabilities for each type
    probs = np.zeros(n_types)
    if 'uniform' in dist_type:
        for i, t in enumerate(types):
            probs[i] = 1.0 / n_types
    elif 'product' in dist_type or 'independent' in dist_type:
        # Parse marginals
        if isinstance(distribution_spec, dict):
            marg1 = distribution_spec.get('marginal_1', distribution_spec.get('marginals', [0.25, 0.25, 0.25, 0.25]))
            marg2 = distribution_spec.get('marginal_2', distribution_spec.get('marginals', [0.25, 0.25, 0.25, 0.25]))
            if isinstance(marg1, list) and isinstance(marg2, list):
                for i, t in enumerate(types):
                    idx1 = values.index(t[0])
                    idx2 = values.index(t[1])
                    probs[i] = marg1[idx1] * marg2[idx2]
            else:
                for i in range(n_types):
                    probs[i] = 1.0 / n_types
        else:
            for i in range(n_types):
                probs[i] = 1.0 / n_types
    elif 'correlated' in dist_type:
        if isinstance(distribution_spec, dict):
            joint = distribution_spec.get('joint', None)
            if joint is not None:
                for i, t in enumerate(types):
                    idx1 = values.index(t[0])
                    idx2 = values.index(t[1])
                    probs[i] = joint[idx1][idx2]
            else:
                for i in range(n_types):
                    probs[i] = 1.0 / n_types
        else:
            for i in range(n_types):
                probs[i] = 1.0 / n_types
    else:
        # Default: uniform
        for i in range(n_types):
            probs[i] = 1.0 / n_types
    
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    
    # Direct Myerson approach: compute optimal reserve for each good independently
    # For a single good with values v in {0,1,2,3} and distribution p(v):
    # Myerson's optimal reserve r* maximizes E[max(v - r, 0) * p(v) / (1-F(v))]
    # where F is the CDF. The virtual value is phi(v) = v - (1-F(v))/f(v)
    # Optimal reserve is where virtual value crosses zero.
    
    def compute_optimal_reserve(marginal_probs, value_grid):
        """Compute optimal Myerson reserve for a single good."""
        n_vals = len(value_grid)
        # CDF and PDF
        cdf = np.zeros(n_vals)
        for i in range(n_vals):
            cdf[i] = sum(marginal_probs[:i+1])
        
        # Virtual values: phi(v) = v - (1 - F(v)) / f(v)
        # For discrete case, use the discrete virtual value
        # phi_i = v_i - (1 - F(v_i)) / f(v_i) where f(v_i) = P(V = v_i)
        virtual_values = np.zeros(n_vals)
        for i in range(n_vals):
            v = value_grid[i]
            f = marginal_probs[i]
            if f > 1e-15:
                # Discrete virtual value
                one_minus_cdf = 1.0 - cdf[i]
                virtual_values[i] = v - one_minus_cdf / f
            else:
                virtual_values[i] = v
        
        # Optimal reserve: allocate if virtual value >= 0
        # The reserve is the smallest v where virtual value >= 0
        # For discrete, we find the threshold
        # Allocate to type v if phi(v) >= 0
        # The optimal mechanism: x(v) = 1 if phi(v) >= 0, else 0
        # For payment: p(v) = v * x(v) - integral of x
        
        # Find allocation rule
        alloc = np.zeros(n_vals)
        for i in range(n_vals):
            if virtual_values[i] >= -1e-10:
                alloc[i] = 1.0
        
        # Compute expected revenue under this rule
        # Revenue = E[v * x(v)] - E[integral of x]
        # For discrete: payment at v_i = v_i * x(v_i) - sum_{j<i} x(v_j) * (v_i - v_j) * ... 
        # Actually for discrete Myerson: p(v_i) = v_i * x(v_i) - sum_{j=0}^{i-1} x(v_j) * (v_i - v_j)
        # Wait, the standard formula: p(v) = v*x(v) - integral_0^v x(t) dt
        # For discrete: p(v_i) = v_i * x(v_i) - sum_{j=0}^{i-1} x(v_j) * (v_i - v_j)
        # Hmm, that's not quite right either. Let me use the LP approach for payments.
        
        return alloc, virtual_values
    
    # Compute marginal distributions for each good
    marg1 = np.zeros(len(values))
    marg2 = np.zeros(len(values))
    for i, t in enumerate(types):
        idx1 = values.index(t[0])
        idx2 = values.index(t[1])
        marg1[idx1] += probs[i]
        marg2[idx2] += probs[i]
    
    # Normalize marginals
    marg1 = marg1 / marg1.sum()
    marg2 = marg2 / marg2.sum()
    
    # Compute optimal allocation rules for each good
    alloc1_rule, vval1 = compute_optimal_reserve(marg1, values)
    alloc2_rule, vval2 = compute_optimal_reserve(marg2, values)
    
    # Build full allocation matrix (16 x 2)
    allocations = np.zeros((n_types, 2))
    for i, t in enumerate(types):
        idx1 = values.index(t[0])
        idx2 = values.index(t[1])
        allocations[i, 0] = alloc1_rule[idx1]
        allocations[i, 1] = alloc2_rule[idx2]
    
    # Compute payments using the Myerson payment formula
    # For each good independently, p_g(v_g) = v_g * x_g(v_g) - sum_{w < v_g} x_g(w) * (v_g - w)
    # Wait, the correct discrete payment formula for a single-parameter mechanism:
    # p(v_i) = v_i * x(v_i) - sum_{j=0}^{i-1} x(v_j) * (v_i - v_j)
    # This ensures IC: u(v_i) = v_i * x(v_i) - p(v_i) = sum_{j=0}^{i-1} x(v_j) * (v_i - v_j)
    # And u is increasing in v_i, so IC holds.
    
    def compute_payments_for_good(alloc_rule, value_grid, marginal_probs):
        """Compute payments for a single good given allocation rule."""
        n_vals = len(value_grid)
        payments = np.zeros(n_vals)
        for i in range(n_vals):
            v_i = value_grid[i]
            x_i = alloc_rule[i]
            # p(v_i) = v_i * x(v_i) - sum_{j=0}^{i-1} x(v_j) * (v_i - v_j)
            integral_term = 0.0
            for j in range(i):
                integral_term += alloc_rule[j] * (v_i - value_grid[j])
            payments[i] = v_i * x_i - integral_term
        return payments
    
    pay1 = compute_payments_for_good(alloc1_rule, values, marg1)
    pay2 = compute_payments_for_good(alloc2_rule, values, marg2)
    
    # Combine payments
    payments = np.zeros(n_types)
    for i, t in enumerate(types):
        idx1 = values.index(t[0])
        idx2 = values.index(t[1])
        payments[i] = pay1[idx1] + pay2[idx2]
    
    # Verify IC (incentive compatibility)
    # For each type t_i, check that reporting t_i is optimal
    # Utility of type t_i reporting t_j: u(t_i, t_j) = v_i1 * x(t_j,1) + v_i2 * x(t_j,2) - p(t_j)
    # Need: u(t_i, t_i) >= u(t_i, t_j) for all j
    
    ic_violation = 0.0
    for i, t_i in enumerate(types):
        u_ii = t_i[0] * allocations[i, 0] + t_i[1] * allocations[i, 1] - payments[i]
        for j, t_j in enumerate(types):
            u_ij = t_i[0] * allocations[j, 0] + t_i[1] * allocations[j, 1] - payments[j]
            if u_ij > u_ii + 1e-8:
                ic_violation = max(ic_violation, u_ij - u_ii)
    
    # Verify IR (individual rationality)
    ir_violation = 0.0
    for i, t_i in enumerate(types):
        u_ii = t_i[0] * allocations[i, 0] + t_i[1] * allocations[i, 1] - payments[i]
        if u_ii < -1e-8:
            ir_violation = max(ir_violation, -u_ii)
    
    # Verify feasibility
    feasibility_ok = True
    for i in range(n_types):
        for g in range(2):
            if allocations[i, g] < -1e-8 or allocations[i, g] > 1 + 1e-8:
                feasibility_ok = False
        if payments[i] < -1e-8:
            # Allow small negative payments? No, nonnegative required.
            feasibility_ok = False
    
    if ic_violation > 1e-8 or ir_violation > 1e-8 or not feasibility_ok:
        # Fallback: use LP to find a valid mechanism
        if time.time() - t_start > DEADLINE:
            # Return trivial mechanism
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        # Use LP to maximize revenue subject to IC, IR, feasibility
        # Variables: x_{i,g} for i in 0..15, g in 0,1 (32 vars)
        #            p_i for i in 0..15 (16 vars)
        # Total: 48 variables
        # 
        # Maximize: sum_i p_i * prob_i
        # Subject to:
        #   0 <= x_{i,g} <= 1
        #   p_i >= 0
        #   IR: v_i1 * x_{i,1} + v_i2 * x_{i,2} - p_i >= 0 for all i
        #   IC: v_i1 * x_{j,1} + v_i2 * x_{j,2} - p_j <= v_i1 * x_{i,1} + v_i2 * x_{i,2} - p_i for all i,j
        #       i.e., p_j - p_i >= v_i1 * (x_{j,1} - x_{i,1}) + v_i2 * (x_{j,2} - x_{i,2})
        #       and also the reverse isn't needed since we're maximizing revenue
        #       Actually IC means: for all i, j: u(i,i) >= u(i,j)
        #       v_i1*x_{i,1} + v_i2*x_{i,2} - p_i >= v_i1*x_{j,1} + v_i2*x_{j,2} - p_j
        #       => p_j - p_i >= v_i1*(x_{j,1}-x_{i,1}) + v_i2*(x_{j,2}-x_{i,2})
        #       This must hold for all pairs (i,j).
        #       But this is a lot of constraints. Let's use the standard IC linearization.
        
        # Actually, for a finite type space, IC is equivalent to:
        # For each i: u(i,i) >= u(i,j) for all j
        # This gives n_types^2 constraints, but many are redundant.
        # The binding constraints are along the "adjacent" types in the value ordering.
        
        # Let's use a simpler approach: since we have only 16 types, we can enumerate
        # all IC constraints.
        
        n_vars = 2 * n_types + n_types  # 32 + 16 = 48
        # x_{i,0} at index i, x_{i,1} at index n_types + i, p_i at index 2*n_types + i
        
        # Objective: maximize sum_i p_i * prob_i
        # linprog minimizes, so minimize -sum_i p_i * prob_i
        c = np.zeros(n_vars)
        for i in range(n_types):
            c[2 * n_types + i] = -probs[i]
        
        # Inequality constraints: A_ub @ vars <= b_ub
        A_ub_rows = []
        b_ub_vals = []
        
        # x_{i,g} >= 0: -x_{i,g} <= 0
        for i in range(n_types):
            for g in range(2):
                row = np.zeros(n_vars)
                row[i + g * n_types] = -1.0
                A_ub_rows.append(row)
                b_ub_vals.append(0.0)
        
        # x_{i,g} <= 1: x_{i,g} <= 1
        for i in range(n_types):
            for g in range(2):
                row = np.zeros(n_vars)
                row[i + g * n_types] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(1.0)
        
        # p_i >= 0: -p_i <= 0
        for i in range(n_types):
            row = np.zeros(n_vars)
            row[2 * n_types + i] = -1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
        
        # IR: v_i1*x_{i,1} + v_i2*x_{i,2} - p_i >= 0
        # => -v_i1*x_{i,1} - v_i2*x_{i,2} + p_i <= 0
        for i in range(n_types):
            row = np.zeros(n_vars)
            row[i] = -types[i][0]
            row[n_types + i] = -types[i][1]
            row[2 * n_types + i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
        
        # IC: for all i, j: p_j - p_i >= v_i1*(x_{j,1}-x_{i,1}) + v_i2*(x_{j,2}-x_{i,2})
        # => v_i1*(x_{j,1}-x_{i,1}) + v_i2*(x_{j,2}-x_{i,2}) - p_j + p_i <= 0
        for i in range(n_types):
            for j in range(n_types):
                if i == j:
                    continue
                row = np.zeros(n_vars)
                row[j] = types[i][0]
                row[i] = -types[i][0]
                row[n_types + j] = types[i][1]
                row[n_types + i] = -types[i][1]
                row[2 * n_types + j] = -1.0
                row[2 * n_types + i] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(0.0)
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
        # Solve LP
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            if result.success:
                x_vals = result.x[:2 * n_types]
                p_vals = result.x[2 * n_types:]
                allocations = np.column_stack([x_vals[:n_types], x_vals[n_types:2*n_types]])
                payments = p_vals
                # Clip small negatives
                payments = np.maximum(payments, 0.0)
                allocations = np.clip(allocations, 0.0, 1.0)
            else:
                # Fallback to trivial
                allocations = np.zeros((n_types, 2))
                payments = np.zeros(n_types)
        except Exception:
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
    
    # Final clipping for safety
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
