import numpy as np
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    # Choose a small K to keep the LP problem small and fast
    # The metric a = -sum f(k) log(k) / k benefits from large negative f(k) for k>1.
    # We can set f(1) to absorb the constraint slack.
    
    K = 20
    
    # Domain is 1..K
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    # Coefficients for the objective function: maximize -sum (ln k / k) * v_k
    # linprog minimizes c^T x. So we set c = - (ln k / k).
    # For k=1, ln(1)=0, so coefficient is 0.
    c = np.zeros(K)
    for i, k in enumerate(domain):
        if k > 1:
            c[i] = -np.log(k) / k
        else:
            c[i] = 0.0
            
    # Constraints: M(x) <= 1.0001 for x in 1..10K
    # M(x) = (v_1 - sum(v_j/j)) * x + sum_{k>1} v_k * floor(x/k)
    # Let S = sum(v_j/j). Then v_1 - S = v_1 - sum(v_j/j).
    # M(x) = v_1 * x - x * sum(v_j/j) + sum_{k>1} v_k * floor(x/k)
    # M(x) = v_1 * x + sum_{j=1}^K v_j * (-x/j + (1 if j>1 else 0) * floor(x/j))
    # Note: for j=1, floor(x/1) = x. So term is v_1 * (-x/1 + x) = 0.
    # So v_1 only appears in the term v_1 * x.
    # Wait, let's re-verify.
    # v'_1 = v_1 - S.
    # M(x) = v'_1 * floor(x/1) + sum_{k>1} v'_k * floor(x/k)
    # M(x) = (v_1 - S) * x + sum_{k>1} v_k * floor(x/k)
    # M(x) = v_1 * x - S * x + sum_{k>1} v_k * floor(x/k)
    # S = v_1/1 + sum_{k>1} v_k/k
    # M(x) = v_1 * x - x * (v_1 + sum_{k>1} v_k/k) + sum_{k>1} v_k * floor(x/k)
    # M(x) = v_1 * x - v_1 * x - x * sum_{k>1} v_k/k + sum_{k>1} v_k * floor(x/k)
    # M(x) = sum_{k>1} v_k * (floor(x/k) - x/k)
    
    # This is a crucial simplification! v_1 cancels out completely from M(x).
    # Therefore, v_1 does not affect the feasibility of M(x).
    # And v_1 does not affect the metric a (since ln 1 = 0).
    # So we can set v_1 = 0 (or any value) and it won't matter for score or feasibility.
    # We just need to ensure the returned values are valid.
    
    # So the problem reduces to:
    # Maximize - sum_{k=2}^K (ln k / k) v_k
    # Subject to:
    # For all x in 1..10K:
    # sum_{k=2}^K v_k * (floor(x/k) - x/k) <= 1.0001
    # And -10 <= v_k <= 10.
    
    # Let's construct the constraint matrix A_ub and b_ub.
    # Number of constraints: 10 * K
    # Number of variables: K (but v_1 is free, we can fix it to 0 and solve for v_2..v_K)
    # Let's just solve for all K variables, with v_1 coefficient 0 in objective and constraints.
    
    x_vals = np.arange(1, 10 * K + 1)
    num_constraints = len(x_vals)
    
    A_ub = np.zeros((num_constraints, K))
    b_ub = np.full(num_constraints, 1.0001)
    
    for i, x in enumerate(x_vals):
        for j, k in enumerate(domain):
            if k == 1:
                A_ub[i, j] = 0.0
            else:
                A_ub[i, j] = np.floor(x / k) - x / k
                
    # Bounds for variables
    bounds = [(-10, 10)] * K
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        values = res.x
        # Ensure values are within [-10, 10] (linprog should handle this, but clip just in case)
        values = np.clip(values, -10, 10)
        
        # Check if the solution is valid (M(x) <= 1.0001)
        # Calculate S
        S = np.sum(values / domain)
        values_adj = values.copy()
        values_adj[0] = values_adj[0] - S
        
        # Recalculate M to be sure
        max_M = -np.inf
        for x in x_vals:
            M_x = 0.0
            for k, v in zip(domain, values_adj):
                M_x += v * np.floor(x / k)
            if M_x > max_M:
                max_M = M_x
                
        if max_M <= 1.0001:
            return domain, values_adj
        else:
            # If LP failed to find a feasible solution due to numerical issues or tight constraints,
            # fall back to a simple valid solution.
            # The seed solution is valid.
            pass
            
    # Fallback to seed
    seed_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    seed_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return seed_domain, seed_values
