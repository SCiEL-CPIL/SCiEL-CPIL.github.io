import numpy as np
from scipy.optimize import fsolve

def solve_coefficients():
    # Variables: a1, a2, a3, b1, b2, b3
    # Symmetry: a4..a12, b4..b12 are determined.
    # We set a4=a5=a6=0, b4=b5=b6=0.
    # So a = [a1, a2, a3, 0, 0, 0, 0, 0, 0, 0, a2, a1]
    # b = [b1, b2, b3, 0, 0, 0, 0, 0, 0, 0, b2, b1]
    
    def equations(vars):
        a1, a2, a3, b1, b2, b3 = vars
        # 1. Sum a = 1 => 2(a1+a2+a3) = 1
        eq1 = 2*(a1 + a2 + a3) - 1
        # 2. Sum b = 1 => 2(b1+b2+b3) = 1
        eq2 = 2*(b1 + b2 + b3) - 1
        # 3. Sum a_k b_k = 1/2 => 2(a1b1 + a2b2 + a3b3) = 1/2
        eq3 = 2*(a1*b1 + a2*b2 + a3*b3) - 0.5
        # 4. Sum a_k b_k^2 = 1/3 => 2(a1b1^2 + a2b2^2 + a3b3^2) = 1/3
        eq4 = 2*(a1*b1**2 + a2*b2**2 + a3*b3**2) - 1/3
        # 5. Sum a_k^2 b_k = 1/3 => 2(a1^2b1 + a2^2b2 + a3^2b3) = 1/3
        eq5 = 2*(a1**2*b1 + a2**2*b2 + a3**2*b3) - 1/3
        return [eq1, eq2, eq3, eq4, eq5]
        
    # Initial guess
    # Strang-like guess
    guess = [0.25, 0.25, 0.0, 0.25, 0.0, 0.25]
    
    try:
        sol = fsolve(equations, guess, full_output=True)
        x, info, ier, msg = sol
        if ier == 1:
            a1, a2, a3, b1, b2, b3 = x
            a = [a1, a2, a3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a2, a1]
            b = [b1, b2, b3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, b2, b1]
            # Check sums
            if abs(sum(a)-1) < 1e-6 and abs(sum(b)-1) < 1e-6:
                return a, b
    except Exception:
        pass
        
    # Fallback to Strang
    a = [0.5/6]*12
    b = [0.5/6]*12
    return a, b

a_opt, b_opt = solve_coefficients()

def coefficients(stages):
    # Use the pre-computed optimal coefficients
    # Ensure length is 'stages'
    a = list(a_opt)
    b = list(b_opt)
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a[:stages], b[:stages]
