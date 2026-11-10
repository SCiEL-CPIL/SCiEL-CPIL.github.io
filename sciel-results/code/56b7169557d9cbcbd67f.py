import time
import math
import numpy as np
import sympy as sp
from scipy.special import roots_genlaguerre, jn_zeros

def construct():
    """
    Returns 12 double-root locations z_1..z_12 for the Laguerre problem.
    Uses exact rational arithmetic for scoring and search.
    """
    
    # Set a global deadline with a safety margin (evaluator limit is 600s, use 570s)
    GLOBAL_DEADLINE = time.time() + 570.0
    
    def time_remaining():
        return GLOBAL_DEADLINE - time.time()

    def _compute_exact_score(z_floats):
        """
        Computes the exact score for a list of 12 float nodes.
        Returns float('inf') if invalid or if time is running out.
        """
        if time_remaining() < 5.0:
            return float('inf')

        # Validate inputs
        z_floats = sorted([float(z) for z in z_floats])
        if len(z_floats) != 12:
            return float('inf')
        if any(z <= 0 or not np.isfinite(z) for z in z_floats):
            return float('inf')
        
        # Check for distinctness with a small tolerance
        for i in range(11):
            if abs(z_floats[i] - z_floats[i+1]) < 1e-10:
                return float('inf')

        # Convert to exact rationals
        z_rats = [sp.Rational(z) for z in z_floats]
        
        # Mathematical Setup
        alpha = sp.Rational(-1, 2)
        x = sp.symbols('x')
        
        # Precompute Laguerre polynomials L_{2k}^{-1/2}(x) for k=0..25
        # We need L_0, L_2, ..., L_50
        L_poly = []
        L_poly_diff = []
        for k in range(26):
            n = 2 * k
            Lk = sp.laguerre(n, alpha, x)
            L_poly.append(Lk)
            L_poly_diff.append(sp.diff(Lk, x))
        
        # Construct the linear system A c = b
        # A is 26x26, b is 26x1
        A = sp.zeros(26, 26)
        b = sp.zeros(26, 1)
        
        # Row 0: g(0) = 0  => sum c_k L_{2k}(0) = 0
        for k in range(26):
            A[0, k] = L_poly[k].subs(x, 0)
        b[0, 0] = 0
        
        # Row 1: g'(0) = 1 => sum c_k L'_{2k}(0) = 1
        for k in range(26):
            A[1, k] = L_poly_diff[k].subs(x, 0)
        b[1, 0] = 1
        
        # Rows 2..25: For each i in 0..11, g(z_i)=0 and g'(z_i)=0
        for i in range(12):
            zi = z_rats[i]
            
            # g(z_i) = 0
            for k in range(26):
                A[2+2*i, k] = L_poly[k].subs(x, zi)
            b[2+2*i, 0] = 0
            
            # g'(z_i) = 0
            for k in range(26):
                A[3+2*i, k] = L_poly_diff[k].subs(x, zi)
            b[3+2*i, 0] = 0
            
            if time_remaining() < 10.0:
                return float('inf')

        # Solve the system
        try:
            c_sol = sp.LUsolve(A, b)
        except Exception:
            return float('inf')

        if time_remaining() < 5.0:
            return float('inf')

        # Construct g(x)
        g_x = sum(c_sol[k] * L_poly[k] for k in range(26))
        
        # Construct D(x) = x * prod (x - z_i)^2
        D_x = x
        for zi in z_rats:
            D_x *= (x - zi)**2
            
        # Perform polynomial division q(x) = g(x) / D(x)
        try:
            q_x, rem = sp.div(g_x, D_x, x)
            if rem != 0:
                if sp.simplify(rem) != 0:
                    return float('inf')
        except Exception:
            return float('inf')

        # Find real roots of q(x)
        try:
            roots_q = sp.real_roots(q_x, x)
        except Exception:
            return float('inf')

        # Check sign change at roots to count them
        # The condition is "strictly opposite signs at r - 1e-198 and r + 1e-198"
        eps = sp.Rational(1, 10)**198
        
        max_r = 0.0
        for r in roots_q:
            q_left = q_x.subs(x, r - eps)
            q_right = q_x.subs(x, r + eps)
            
            # Check if signs are opposite (product < 0)
            # Note: q_left and q_right are exact rationals
            if q_left * q_right < 0:
                r_float = float(r)
                if r_float > max_r:
                    max_r = r_float
        
        B = max(0.0, max_r)
        score = B / (2.0 * math.pi)
        
        return score

    # --- Candidate Generation ---
    
    def generate_bessel_nodes():
        # Zeros of J_0 squared? Or just Bessel zeros?
        # The task mentions "Bessel zeros squared" in the inspiration.
        # Let's try squares of the first 12 positive zeros of J_0.
        try:
            zeros = jn_zeros(0, 12)
            return [z*z for z in zeros]
        except:
            # Fallback to approximation
            return [((i-0.25)*math.pi)**2 for i in range(1, 13)]

    def generate_laguerre_nodes():
        # Roots of L_12^{-1/2}(x)
        try:
            r = roots_genlaguerre(12, -0.5)
            return sorted([float(x) for x in r.roots])
        except:
            return [float(i) for i in range(1, 13)]

    def generate_uniform_nodes():
        return [float(i) for i in range(1, 13)]

    def generate_geometric_nodes():
        return [0.5 * (1.3 ** i) for i in range(12)]

    initial_candidates = [
        generate_bessel_nodes(),
        generate_laguerre_nodes(),
        generate_uniform_nodes(),
        generate_geometric_nodes()
    ]
    
    best_z = None
    best_score = float('inf')
    
    # 1. Evaluate Initial Candidates
    for cand in initial_candidates:
        if time_remaining() < 10.0:
            break
        score = _compute_exact_score(cand)
        if score < best_score:
            best_score = score
            best_z = cand[:]
            
    if best_z is None:
        return generate_uniform_nodes()
        
    current_z = best_z[:]
    current_score = best_score
    
    # 2. Local Optimization
    # We use a coordinate descent method with adaptive steps.
    # Since the exact score is expensive, we limit the number of iterations.
    # We will try to improve each coordinate sequentially.
    
    max_passes = 5
    base_step = 0.1
    
    for _pass in range(max_passes):
        if time_remaining() < 30.0:
            break
            
        improved = False
        step = base_step / (_pass + 1)
        
        for i in range(12):
            if time_remaining() < 15.0:
                break
                
            # Try increasing z_i
            new_val_up = current_z[i] * (1 + step)
            if new_val_up > 0 and (i == 0 or new_val_up > current_z[i-1] * 1.01) and (i == 11 or new_val_up < current_z[i+1] * 0.99):
                test_z = current_z[:]
                test_z[i] = new_val_up
                test_z.sort()
                s = _compute_exact_score(test_z)
                if s < current_score:
                    current_score = s
                    current_z = test_z
                    best_score = s
                    best_z = test_z[:]
                    improved = True
                    continue # Restart index loop? No, just continue to next i
            
            # Try decreasing z_i
            new_val_down = current_z[i] * (1 - step)
            if new_val_down > 0 and (i == 0 or new_val_down > current_z[i-1] * 1.01) and (i == 11 or new_val_down < current_z[i+1] * 0.99):
                test_z = current_z[:]
                test_z[i] = new_val_down
                test_z.sort()
                s = _compute_exact_score(test_z)
                if s < current_score:
                    current_score = s
                    current_z = test_z
                    best_score = s
                    best_z = test_z[:]
                    improved = True
        
        if not improved:
            break
            
    return best_z

