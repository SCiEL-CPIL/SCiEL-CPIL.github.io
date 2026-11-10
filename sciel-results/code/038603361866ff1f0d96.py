import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Define the support: 16 types (v1, v2) with v1, v2 in {0, 1, 2, 3}
    # Order: v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)
    n_goods = 2
    values = [0, 1, 2, 3]
    
    # Parse distribution_spec to get probabilities for each type
    probs = np.zeros(n_types)
    if isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
            elif str(t) in distribution_spec:
                probs[i] = distribution_spec[str(t)]
            else:
                probs[i] = 1.0 / n_types
    elif isinstance(distribution_spec, str):
        spec = distribution_spec.lower()
        if "uniform" in spec:
            probs = np.ones(n_types) / n_types
        elif "skewed" in spec or "biased" in spec:
            weights = np.array([(v1 + 1) * (v2 + 1) for v1, v2 in types])
            probs = weights / np.sum(weights)
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
        
    # Normalize probabilities
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types

    # Compute marginal distributions for each good
    # Good 1 depends on v1, Good 2 depends on v2
    # types[i] = (v1, v2) where v1 = i // 4, v2 = i % 4
    marginal_1 = np.zeros(4)
    marginal_2 = np.zeros(4)
    for i, (v1, v2) in enumerate(types):
        marginal_1[v1] += probs[i]
        marginal_2[v2] += probs[i]

    def solve_single_good(marginal_probs):
        """
        Solves the optimal single-good auction for values 0, 1, 2, 3.
        Returns optimal allocations q (size 4) and payments p (size 4).
        """
        n_vals = 4
        # Variables: q0, q1, q2, q3, p0, p1, p2, p3
        # Indices: 0-3 for q, 4-7 for p
        n_vars = 8
        c = np.zeros(n_vars)
        
        # Objective: Maximize Revenue = sum(mu[v] * p[v])
        # Minimize -Revenue
        for v in range(n_vals):
            c[4 + v] = -marginal_probs[v]
            
        A_ub_rows = []
        b_ub_vals = []
        
        # IR: p[v] <= v * q[v]  =>  p[v] - v * q[v] <= 0
        for v in range(n_vals):
            row = np.zeros(n_vars)
            row[4 + v] = 1.0
            row[v] = -values[v]
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
        # IC: p[v] - p[v'] <= v * q[v] - v * q[v']
        # => p[v] - p[v'] - v * q[v] + v * q[v'] <= 0
        for v in range(n_vals):
            for vp in range(n_vals):
                if v == vp:
                    continue
                row = np.zeros(n_vars)
                row[4 + v] = 1.0
                row[4 + vp] = -1.0
                row[v] = -values[v]
                row[vp] = values[v]
                A_ub_rows.append(row)
                b_ub_vals.append(0.0)
                
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
        bounds = []
        for i in range(n_vals):
            bounds.append((0, 1)) # q in [0, 1]
        for i in range(n_vals):
            bounds.append((0, None)) # p >= 0
            
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback: No sale
            q = np.zeros(n_vals)
            p = np.zeros(n_vals)
        else:
            q = res.x[:n_vals]
            p = res.x[n_vals:]
            q = np.clip(q, 0, 1)
            p = np.clip(p, 0, None)
            
        return q, p

    # Solve for Good 1
    q1, p1 = solve_single_good(marginal_1)
    # Solve for Good 2
    q2, p2 = solve_single_good(marginal_2)
    
    # Construct final allocations and payments
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    
    for i, (v1, v2) in enumerate(types):
        allocations[i, 0] = q1[v1]
        allocations[i, 1] = q2[v2]
        payments[i] = p1[v1] + p2[v2]
        
    return {
        'allocations': allocations,
        'payments': payments
    }
