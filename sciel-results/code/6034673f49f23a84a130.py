"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    from math import lcm

    # Denominators are 2, 3, ..., 48
    denominators = list(range(2, 49))
    n = len(denominators)  # 47 variables
    
    # Compute LCM of all denominators to convert to integer arithmetic
    # LCM of 2..48
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Each unit fraction 1/d becomes L/d in integer terms
    # We want to minimize 1 - sum(1/d_i) = (L - sum(L/d_i)) / L
    # So minimize (L - sum(L/d_i)) subject to sum(L/d_i) < L
    # i.e., sum(L/d_i) <= L - 1 (since integers)
    # And we want to maximize sum(L/d_i) subject to sum(L/d_i) <= L - 1
    # This is equivalent to minimizing (L - 1 - sum(L/d_i))
    
    # Objective: minimize L - 1 - sum(c_i * x_i) where c_i = L/d_i
    # Equivalent to: minimize (L - 1) - sum(c_i * x_i)
    # In MILP form: minimize c_obj^T x where c_obj = -c_i (to maximize sum)
    # But we need to be careful with the constant. Let's just maximize sum(c_i * x_i)
    # which is minimize -sum(c_i * x_i)
    
    c = np.array([L // d for d in denominators], dtype=np.int64)
    
    # Objective: minimize -c^T x (i.e., maximize c^T x)
    c_obj = -c
    
    # Constraints:
    # 1) sum(c_i * x_i) <= L - 1  (strictly below 1)
    # 2) sum(x_i) >= 1  (nonempty selection)
    # 3) x_i in {0, 1}
    
    A_ub = np.array([c], dtype=np.int64)
    b_ub = np.array([L - 1], dtype=np.int64)
    
    A_ge = np.array([np.ones(n, dtype=np.int64)])
    b_ge = np.array([1], dtype=np.int64)
    
    # Combine into LinearConstraint
    # A_ub @ x <= b_ub  ->  -A_ub @ x >= -b_ub
    # A_ge @ x >= b_ge
    A = np.vstack([-A_ub, A_ge])
    lb = np.array([-b_ub[0], b_ge[0]], dtype=np.int64)
    ub = np.array([np.inf, np.inf])
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: x_i in {0, 1}
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    
    # Integrality: all binary
    integrality = np.ones(n)
    
    result = milp(
        c=c_obj,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds
    )
    
    if not result.success:
        # Fallback to greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    x = np.round(result.x).astype(int)
    selected = [i for i in range(n) if x[i] == 1]
    
    if not selected:
        # Fallback to greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    return selected
# EVOLVE-BLOCK-END
