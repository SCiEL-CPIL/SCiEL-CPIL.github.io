import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s limit
    
    target_val = math.factorial(12)
    primes = [2, 3, 5, 7, 11]
    
    # Calculate prime exponents for 12!
    target_exp = [0] * 5
    for idx, p in enumerate(primes):
        n = target_val
        while n % p == 0:
            target_exp[idx] += 1
            n //= p
            
    # Generate all divisors of 12!
    divisors = []
    i = 1
    while i * i <= target_val:
        if target_val % i == 0:
            if i > 12:
                divisors.append(i)
            if target_val // i > 12 and target_val // i != i:
                divisors.append(target_val // i)
        i += 1
        
    # Remove duplicates and sort
    divisors = sorted(list(set(divisors)))
    n_divs = len(divisors)
    
    # If no divisors found (impossible here), return fallback
    if n_divs == 0:
        return [target_val]
        
    # Precompute prime exponent vectors for each divisor
    # exps[i][j] is the exponent of primes[j] in divisors[i]
    exps = np.zeros((n_divs, 5), dtype=int)
    for i, d in enumerate(divisors):
        n = d
        for j, p in enumerate(primes):
            while n % p == 0:
                exps[i][j] += 1
                n //= p
                
    # ILP Formulation
    # Variables: x_0, ..., x_{n_divs-1} (binary), M (continuous)
    # Total variables: n_divs + 1
    n_vars = n_divs + 1
    
    # Objective: Minimize M
    # c = [0, 0, ..., 0, 1]
    c = np.zeros(n_vars)
    c[-1] = 1.0
    
    # Constraints:
    # 1. Prime exponent equalities: sum(exp[i][j] * x[i]) == target_exp[j]
    #    A_eq shape: (5, n_vars)
    #    b_eq shape: (5,)
    A_eq = np.zeros((5, n_vars))
    A_eq[:, :n_divs] = exps
    b_eq = np.array(target_exp, dtype=float)
    
    # 2. Count constraints: 1 <= sum(x) <= 12
    #    sum(x) >= 1  =>  -sum(x) <= -1
    #    sum(x) <= 12 =>  sum(x) <= 12
    #    We can combine these into a single LinearConstraint or use two.
    #    Let's use two linear constraints for clarity, or one with bounds.
    #    LinearConstraint(A, lb, ub)
    #    A_count = [[-1, -1, ..., -1, 0], [1, 1, ..., 1, 0]]
    #    lb = [-1, 1], ub = [inf, 12] ? No, LinearConstraint takes lb and ub vectors.
    #    Constraint 1: -sum(x) <= -1  =>  sum(x) >= 1
    #    Constraint 2: sum(x) <= 12
    #    Let's define A_lb_ub for the sum of x.
    #    A_sum = np.zeros((2, n_vars))
    #    A_sum[0, :n_divs] = -1.0
    #    A_sum[1, :n_divs] = 1.0
    #    lb = [-1.0, 1.0]
    #    ub = [np.inf, 12.0]
    
    # 3. M constraints: M >= d_i * x_i  =>  -M + d_i * x_i <= 0
    #    For each i: d_i * x_i - M <= 0
    #    A_M shape: (n_divs, n_vars)
    A_M = np.zeros((n_divs, n_vars))
    for i in range(n_divs):
        A_M[i, i] = divisors[i]
        A_M[i, -1] = -1.0
    lb_M = np.full(n_divs, -np.inf)
    ub_M = np.zeros(n_divs)
    
    # Combine all constraints
    # Constraint 1: Prime exponents (Equality)
    # Constraint 2: Count (Inequality)
    # Constraint 3: M bounds (Inequality)
    
    # scipy.optimize.milp takes a list of LinearConstraint or a single one.
    # Let's create a single LinearConstraint object combining all inequalities.
    # Equality constraints are handled separately? No, milp uses LinearConstraint for both.
    # LinearConstraint(A, lb, ub). If lb == ub, it's an equality.
    
    # Total constraints:
    # 5 equalities (primes)
    # 2 inequalities (count)
    # n_divs inequalities (M)
    # Total rows: 5 + 2 + n_divs
    
    A_all = np.zeros((5 + 2 + n_divs, n_vars))
    lb_all = np.zeros(5 + 2 + n_divs)
    ub_all = np.zeros(5 + 2 + n_divs)
    
    # Fill Prime Equalities
    A_all[:5, :n_divs] = exps
    lb_all[:5] = target_exp
    ub_all[:5] = target_exp
    
    # Fill Count Inequalities
    # Row 5: sum(x) >= 1  =>  -sum(x) <= -1  (We use standard form Ax <= b? No, milp uses lb <= Ax <= ub)
    # So for sum(x) >= 1:  lb=1, ub=inf.
    # For sum(x) <= 12:   lb=-inf, ub=12.
    # Let's just define A_count = [[1, 1, ..., 1, 0]]
    # And set lb=1, ub=12.
    # Wait, we need to distinguish the two bounds.
    # We can use one row for sum(x) with lb=1 and ub=12.
    # So only 1 row for count.
    
    # Let's restructure:
    # Rows 0-4: Prime exponents (Equality)
    # Row 5: Count (1 <= sum(x) <= 12)
    # Rows 6 to 6+n_divs-1: M constraints (d_i x_i - M <= 0)
    
    # Adjust A_all size
    n_rows = 5 + 1 + n_divs
    A_all = np.zeros((n_rows, n_vars))
    lb_all = np.full(n_rows, -np.inf)
    ub_all = np.full(n_rows, np.inf)
    
    # 1. Prime Exponents
    A_all[:5, :n_divs] = exps
    lb_all[:5] = target_exp
    ub_all[:5] = target_exp
    
    # 2. Count
    A_all[5, :n_divs] = 1.0
    lb_all[5] = 1.0
    ub_all[5] = 12.0
    
    # 3. M Constraints
    for i in range(n_divs):
        row_idx = 6 + i
        A_all[row_idx, i] = divisors[i]
        A_all[row_idx, -1] = -1.0
        # Constraint: d_i x_i - M <= 0
        # So lb = -inf, ub = 0
        lb_all[row_idx] = -np.inf
        ub_all[row_idx] = 0.0
        
    constraints = LinearConstraint(A_all, lb_all, ub_all)
    
    # Integrality: x_i are binary (0 or 1), M is continuous
    integrality = np.zeros(n_vars)
    integrality[:n_divs] = 1.0
    integrality[-1] = 0.0
    
    # Bounds for variables
    # x_i in [0, 1]
    # M in [13, target_val] (since factors > 12, min max is at least 13, max is target_val)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    bounds.lb[-1] = 13.0
    bounds.ub[-1] = float(target_val)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds
        )
        
        if not res.success:
            # Fallback
            return [target_val]
            
        # Extract solution
        x = res.x[:n_divs]
        # x should be close to 0 or 1
        selected_factors = []
        for i in range(n_divs):
            if x[i] > 0.5:
                selected_factors.append(divisors[i])
                
        if not selected_factors:
            return [target_val]
            
        # Verify solution
        prod = 1
        for d in selected_factors:
            prod *= d
        if prod != target_val:
            return [target_val]
        if len(selected_factors) > 12 or len(selected_factors) < 1:
            return [target_val]
        if any(d <= 12 for d in selected_factors):
            return [target_val]
        if len(set(selected_factors)) != len(selected_factors):
            return [target_val]
            
        return selected_factors
        
    except Exception as e:
        # If ILP fails for any reason, fallback
        return [target_val]

