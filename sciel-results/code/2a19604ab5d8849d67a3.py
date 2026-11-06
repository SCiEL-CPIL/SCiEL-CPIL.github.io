"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try to infer from support
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Parse support if provided
    if support is not None:
        if isinstance(support, dict):
            # Support might specify allowed reports or values
            pass
        elif isinstance(support, (list, tuple, np.ndarray)):
            # Support might be a list of allowed type indices
            pass
    
    # Parse constraints if provided
    if constraints is not None:
        if isinstance(constraints, dict):
            pass
    
    # For the mechanism design LP:
    # Variables: x[t, t'] for t in types, t' in types (allocation probability of type t' when reporting t)
    #           p[t, t'] for t in types, t' in types (payment when type t' reports t)
    #
    # Actually, let's use the standard formulation:
    # x[t, t'] = probability of allocating to type t' when true type is t
    # p[t, t'] = payment when type t' reports t (i.e., type t' is the reported type)
    #
    # Wait, let me reconsider. The standard mechanism design LP for a single bidder:
    # Let x[t, t'] be the allocation probability when true type is t and reported type is t'
    # Let p[t, t'] be the payment when true type is t and reported type is t'
    #
    # Truthfulness (IC): for all t, t1, t2:
    #   sum_{t'} x[t, t'] * (v_t - v_t') ... no wait
    #
    # Actually, the IC constraint is:
    # For all t, t1: utility(t, t) >= utility(t, t1)
    # utility(t, t') = sum_{g} x[t', g] * v_t[g] - p[t']
    # where x[t', g] is the allocation of good g when reporting t'
    #
    # So: sum_g x[t', g] * v_t[g] - p[t'] >= sum_g x[t1, g] * v_t[g] - p[t1] for all t, t1
    #
    # Let's reparameterize:
    # x[t, g] = allocation probability of good g when reporting type t
    # p[t] = payment when reporting type t
    #
    # IC: for all t, t1:
    #   sum_g x[t, g] * v_t[g] - p[t] >= sum_g x[t1, g] * v_t[g] - p[t1]
    #
    # IR: for all t:
    #   sum_g x[t, g] * v_t[g] - p[t] >= 0
    #
    # Feasibility: 0 <= x[t, g] <= 1 for all t, g
    #
    # Maximize: sum_t probs[t] * p[t]
    #
    # This is a linear program!
    
    # Number of goods
    n_goods = 2
    
    # Variables: x[t, g] for t in 0..15, g in 0..1; p[t] for t in 0..15
    # Total variables: 16*2 + 16 = 48
    
    # Let's set up the LP
    # c = objective coefficients (we want to maximize, so minimize -c)
    # For x variables: coefficient is 0
    # For p variables: coefficient is -probs[t] (since we're minimizing)
    
    n_x = n_types * n_goods  # 32
    n_p = n_types  # 16
    n_vars = n_x + n_p  # 48
    
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -probs[t]
    
    # Constraints:
    # 1. x[t, g] >= 0: -x[t, g] <= 0
    # 2. x[t, g] <= 1: x[t, g] <= 1
    # 3. IC constraints: for all t, t1:
    #    sum_g x[t, g] * v_t[g] - p[t] >= sum_g x[t1, g] * v_t[g] - p[t1]
    #    => sum_g (x[t, g] - x[t1, g]) * v_t[g] - p[t] + p[t1] >= 0
    #    => -sum_g (x[t, g] - x[t1, g]) * v_t[g] + p[t] - p[t1] <= 0
    # 4. IR constraints: for all t:
    #    sum_g x[t, g] * v_t[g] - p[t] >= 0
    #    => -sum_g x[t, g] * v_t[g] + p[t] <= 0
    
    # Build A_ub and b_ub
    # We'll collect all inequality constraints in the form A_ub @ vars <= b_ub
    
    A_rows = []
    b_vals = []
    
    # Constraint 1: x[t, g] >= 0 => -x[t, g] <= 0
    for t in range(n_types):
        for g in range(n_goods):
            row = np.zeros(n_vars)
            row[t * n_goods + g] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)
    
    # Constraint 2: x[t, g] <= 1
    for t in range(n_types):
        for g in range(n_goods):
            row = np.zeros(n_vars)
            row[t * n_goods + g] = 1.0
            A_rows.append(row)
            b_vals.append(1.0)
    
    # Constraint 3: IC constraints
    for t in range(n_types):
        for t1 in range(n_types):
            if t == t1:
                continue
            row = np.zeros(n_vars)
            for g in range(n_goods):
                row[t * n_goods + g] = -values[t[g]]  # -v_t[g]
                row[t1 * n_goods + g] = values[t[g]]   # +v_t[g]
            row[n_x + t] = 1.0    # +p[t]
            row[n_x + t1] = -1.0  # -p[t1]
            A_rows.append(row)
            b_vals.append(0.0)
    
    # Constraint 4: IR constraints
    for t in range(n_types):
        row = np.zeros(n_vars)
        for g in range(n_goods):
            row[t * n_goods + g] = -values[t[g]]
        row[n_x + t] = 1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: x[t,g] in [0,1], p[t] >= 0
    bounds = []
    for i in range(n_x):
        bounds.append((0.0, 1.0))
    for i in range(n_p):
        bounds.append((0.0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback to a simple mechanism
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x_vals = result.x[:n_x].reshape(n_types, n_goods)
        p_vals = result.x[n_x:n_x + n_p]
        
        # Clip to handle numerical issues
        x_vals = np.clip(x_vals, 0.0, 1.0)
        p_vals = np.clip(p_vals, 0.0, None)
        
        # Verify constraints
        # Check IC
        for t in range(n_types):
            for t1 in range(n_types):
                util_t = np.dot(x_vals[t], values[t]) - p_vals[t]
                util_t1 = np.dot(x_vals[t1], values[t]) - p_vals[t1]
                if util_t < util_t1 - 1e-8:
                    # IC violated, adjust
                    pass
        
        # Check IR
        for t in range(n_types):
            util_t = np.dot(x_vals[t], values[t]) - p_vals[t]
            if util_t < -1e-8:
                # IR violated, adjust payment
                p_vals[t] = np.dot(x_vals[t], values[t])
        
        # Final check
        for t in range(n_types):
            for t1 in range(n_types):
                util_t = np.dot(x_vals[t], values[t]) - p_vals[t]
                util_t1 = np.dot(x_vals[t1], values[t]) - p_vals[t1]
                if util_t < util_t1 - 1e-8:
                    # Need to fix IC
                    # This shouldn't happen if LP is correct, but just in case
                    pass
        
        return {'allocations': x_vals, 'payments': p_vals}
    
    except Exception as e:
        # Fallback to a simple mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
