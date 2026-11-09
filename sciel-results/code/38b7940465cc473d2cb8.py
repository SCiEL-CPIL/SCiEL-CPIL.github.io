"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Return 12 double-root locations z_1..z_12 (list of floats).
    
    Strategy: Use a time-bounded local search to minimize B / (2*pi).
    B is determined by the largest real root r of q(x) where q(x) = g(x) / (x * prod_i (x - z_i)^2),
    with g constructed to satisfy g(0)=0, g'(0)=1, and g(z_i)=g'(z_i)=0.
    
    We use a heuristic approach:
    1. Start with an initial guess for z_i (e.g., evenly spaced or based on Laguerre node patterns)
    2. Evaluate the score (B / (2*pi)) using exact rational arithmetic
    3. If time permits, perform local perturbations to improve the score
    
    Due to the 600s time limit and the cost of exact rational evaluation,
    we limit the number of evaluations and use a simple hill-climbing approach.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin: 20s before the 600s kill
    
    # Initial guess: use a pattern that might work well
    # The roots should be positive and distinct. Let's start with values spread out.
    # Based on the problem structure, smaller z_i might lead to smaller B.
    # Try a geometric or arithmetic progression.
    
    def evaluate(z_list):
        """
        Evaluate the score for a given list of z_i values.
        Returns (score, valid) where score = B / (2*pi) and valid is a boolean.
        Uses exact rational arithmetic as specified.
        """
        import sympy as sp
        from sympy import Rational, symbols, L, diff, solve, Matrix, LUsolve, real_roots, nsimplify
        
        x = symbols('x')
        alpha = Rational(-1, 2)
        
        # Convert z_i to exact rationals
        z_rationals = [Rational(float(zi)) for zi in z_list]
        z_rationals = sorted(z_rationals)
        
        # Check pairwise distinctness
        for i in range(len(z_rationals)):
            for j in range(i+1, len(z_rationals)):
                if z_rationals[i] == z_rationals[j]:
                    return (float('inf'), False)
        
        # Check positivity
        for z in z_rationals:
            if z <= 0:
                return (float('inf'), False)
        
        # Build the system of equations
        # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{alpha}(x)
        # Unknowns: c_0, c_1, ..., c_25 (26 unknowns)
        # Equations:
        #   g(0) = 0
        #   g'(0) = 1
        #   g(z_i) = 0 for i=0..11 (12 equations)
        #   g'(z_i) = 0 for i=0..11 (12 equations)
        # Total: 2 + 12 + 12 = 26 equations
        
        c = sp.symbols('c0:26')
        
        # Build g(x)
        g = sum(c[k] * L(2*k, alpha, x) for k in range(26))
        
        # Build equations
        eqs = []
        eqs.append(sp.Eq(g.subs(x, 0), 0))
        eqs.append(sp.Eq(diff(g, x).subs(x, 0), 1))
        
        for z in z_rationals:
            eqs.append(sp.Eq(g.subs(x, z), 0))
            eqs.append(sp.Eq(diff(g, x).subs(x, z), 0))
        
        # Set up linear system
        # g is linear in c, so we can extract coefficients
        # g = sum_k c_k * L_{2k}^{alpha}(x)
        # We need to express each equation as a linear combination of c_k
        
        # Extract the coefficient of each c_k in each equation
        # For g(0) = 0: sum_k c_k * L_{2k}^{alpha}(0) = 0
        # For g'(0) = 1: sum_k c_k * L'_{2k}^{alpha}(0) = 1
        # For g(z_i) = 0: sum_k c_k * L_{2k}^{alpha}(z_i) = 0
        # For g'(z_i) = 0: sum_k c_k * L'_{2k}^{alpha}(z_i) = 0
        
        # Build the matrix A and vector b
        A = sp.Matrix([[0]*26 for _ in range(26)])
        b = sp.Matrix([0]*26)
        
        # Row 0: g(0) = 0
        for k in range(26):
            A[0, k] = L(2*k, alpha, 0)
        b[0] = 0
        
        # Row 1: g'(0) = 1
        for k in range(26):
            A[1, k] = diff(L(2*k, alpha, x), x).subs(x, 0)
        b[1] = 1
        
        # Rows 2-13: g(z_i) = 0
        for i, z in enumerate(z_rationals):
            row = 2 + i
            for k in range(26):
                A[row, k] = L(2*k, alpha, z)
            b[row] = 0
        
        # Rows 14-25: g'(z_i) = 0
        for i, z in enumerate(z_rationals):
            row = 14 + i
            for k in range(26):
                A[row, k] = diff(L(2*k, alpha, x), x).subs(x, z)
            b[row] = 0
        
        # Solve the linear system
        try:
            c_sol = LUsolve(A, b)
        except Exception as e:
            return (float('inf'), False)
        
        # Build g(x) with the solved coefficients
        g_sol = sum(c_sol[k] * L(2*k, alpha, x) for k in range(26))
        
        # Compute q(x) = g(x) / (x * prod_i (x - z_i)^2)
        # This should be an exact polynomial division
        denom = x * sp.prod((x - z)**2 for z in z_rationals)
        
        try:
            q, remainder = sp.div(g_sol, denom, x)
            if remainder != 0:
                return (float('inf'), False)
        except Exception as e:
            return (float('inf'), False)
        
        # Find real roots of q
        try:
            roots = real_roots(q, x)
        except Exception as e:
            return (float('inf'), False)
        
        # For each real root r, check if q has strictly opposite signs at r - 1e-198 and r + 1e-198
        # Note: 1e-198 is extremely small. We need to evaluate q at these points.
        # Since we're using exact arithmetic, we can use Rational(1, 10**198)
        
        eps = Rational(1, 10**198)
        counted_roots = []
        
        for r in roots:
            # Evaluate q at r - eps and r + eps
            try:
                q_minus = q.subs(x, r - eps)
                q_plus = q.subs(x, r + eps)
            except Exception as e:
                continue
            
            # Check if they have strictly opposite signs
            if q_minus * q_plus < 0:
                counted_roots.append(r)
        
        if not counted_roots:
            B = 0
        else:
            B = max(counted_roots)
        
        score = float(B) / (2 * math.pi)
        return (score, True)
    
    # Initial guess: start with small values since smaller z_i might lead to smaller B
    # Try a geometric progression
    best_z = [0.5 * (i + 1) for i in range(12)]
    best_score = float('inf')
    
    # Evaluate the initial guess
    score, valid = evaluate(best_z)
    if valid:
        best_score = score
    
    # If we have time, try some perturbations
    # Simple hill-climbing: perturb each z_i slightly and see if the score improves
    
    num_evaluations = 1
    max_evaluations = 20  # Limit to avoid timeout
    
    while num_evaluations < max_evaluations:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Try perturbing a random z_i
        import random
        idx = random.randint(0, 11)
        perturbation = random.uniform(-0.1, 0.1)
        new_z = best_z[:]
        new_z[idx] = max(0.01, new_z[idx] + perturbation)
        
        # Ensure distinctness
        valid_distinct = True
        for i in range(12):
            for j in range(i+1, 12):
                if abs(new_z[i] - new_z[j]) < 1e-6:
                    valid_distinct = False
                    break
            if not valid_distinct:
                break
        
        if not valid_distinct:
            num_evaluations += 1
            continue
        
        score, valid = evaluate(new_z)
        num_evaluations += 1
        
        if valid and score < best_score:
            best_score = score
            best_z = new_z
    
    return best_z
# EVOLVE-BLOCK-END
