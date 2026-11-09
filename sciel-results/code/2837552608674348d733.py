import sympy as sp
import numpy as np
import time
import random
import math

# Precompute Laguerre polynomials and their derivatives
x_sym = sp.Symbol('x')
alpha_val = sp.Rational(-1, 2)
Ls_precomputed = []
dLs_precomputed = []
for k in range(26):
    Lk = sp.Laguerre(2*k, alpha_val, x_sym)
    Ls_precomputed.append(Lk)
    dLs_precomputed.append(sp.diff(Lk, x_sym))

def evaluate_candidate(z_list):
    """
    Evaluates the candidate z_list exactly.
    Returns B if valid, None if invalid.
    """
    try:
        # Convert to exact rationals and sort
        z_list_sorted = sorted(z_list)
        z_rat = [sp.Rational(float(z)) for z in z_list_sorted]
        
        # Build the 26x26 linear system using precomputed Laguerre polynomials
        cs = sp.symbols('c0:26')
        
        M = sp.zeros(26, 26)
        rhs = sp.zeros(26, 1)
        
        # Eq 0: g(0) = 0
        for k in range(26):
            M[0, k] = Ls_precomputed[k].subs(x_sym, 0)
        rhs[0] = 0
        
        # Eq 1: g'(0) = 1
        for k in range(26):
            M[1, k] = dLs_precomputed[k].subs(x_sym, 0)
        rhs[1] = 1
        
        idx = 2
        for z in z_rat:
            # g(z) = 0
            for k in range(26):
                M[idx, k] = Ls_precomputed[k].subs(x_sym, z)
            rhs[idx] = 0
            idx += 1
            
            # g'(z) = 0
            for k in range(26):
                M[idx, k] = dLs_precomputed[k].subs(x_sym, z)
            rhs[idx] = 0
            idx += 1
            
        # Solve
        try:
            sol = M.LUsolve(rhs)
        except sp.matrices.exceptions.SingularMatrixException:
            return None
        except Exception:
            return None
            
        # Substitute cs back into g
        g_sol = sum(sol[k] * Ls_precomputed[k] for k in range(26))
        
        # q(x) = g(x) / (x * prod (x - z_i)^2)
        denom = x_sym
        for z in z_rat:
            denom *= (x_sym - z)**2
            
        # Polynomial division
        q = sp.div(g_sol, denom)
        q_poly = q[0]
        
        # Find real roots of q_poly
        try:
            roots = sp.real_roots(q_poly)
        except Exception:
            return None
            
        if not roots:
            return 0.0
            
        max_r = 0.0
        for r in roots:
            # r is a sympy expression
            try:
                r_float = float(r)
            except:
                continue
                
            if not np.isfinite(r_float) or r_float <= 0:
                continue
                
            # Check sign change at r +/- 1e-198
            delta = 1e-198
            try:
                val_minus = float(q_poly.subs(x_sym, r_float - delta))
                val_plus = float(q_poly.subs(x_sym, r_float + delta))
            except Exception:
                continue
                
            if val_minus * val_plus < 0:
                if r_float > max_r:
                    max_r = r_float
                    
        return float(max_r)
        
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 580.0  # Leave 50s margin for the evaluator's own overhead/kill signal
    
    # Generate diverse candidates with focus on promising regions
    candidates = []
    
    # 1. Seed (baseline) - keep this first as fallback
    candidates.append([4.0 * (i + 1) for i in range(12)])
    
    # 2. Very small z_i - hypothesis: might push residual roots to smaller values
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        candidates.append([scale * (i + 1) for i in range(12)])
    
    # 3. Geometric spacing with various bases and ratios
    for base in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for ratio in [1.3, 1.5, 2.0, 2.5]:
            z = [base * (ratio ** i) for i in range(12)]
            # Check if all are reasonable (not too large)
            if z[-1] < 1000:
                candidates.append(z)
    
    # 4. Power law distributions
    for p in [1.2, 1.5, 2.0, 2.5, 3.0]:
        candidates.append([(i + 1) ** p for i in range(12)])
    
    # 5. Roots of Laguerre L_12^(-1/2)(x) - natural nodes
    try:
        L12 = sp.Laguerre(12, alpha_val, x_sym)
        roots_L12 = sp.nroots(L12, n=50, maxsteps=100)
        real_roots = [float(r) for r in roots_L12 if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0]
        if len(real_roots) == 12:
            candidates.append(sorted(real_roots))
    except:
        pass
    
    # 6. Clustered distributions
    for center in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for spread in [0.1, 0.5, 1.0, 2.0]:
            z = [center + spread * (i - 6) for i in range(12)]
            if all(z_i > 0 for z_i in z):
                candidates.append(sorted(z))
    
    # 7. Random search with various ranges
    random.seed(42)
    for _ in range(20):
        # Mix of small and large
        z = [random.uniform(0.01, 20.0) for _ in range(12)]
        z = sorted(z)
        # Check distinctness
        valid = True
        for i in range(11):
            if abs(z[i+1] - z[i]) < 1e-6:
                valid = False
                break
        if valid:
            candidates.append(z)
    
    # 8. Logarithmically spaced
    for n_points in [12]:
        for log_min in [-2, -1, 0]:
            for log_max in [1, 2, 3]:
                z = [10 ** (log_min + (log_max - log_min) * i / (n_points - 1)) for i in range(n_points)]
                if all(z_i > 0 for z_i in z):
                    candidates.append(z)
    
    # Sort candidates by a heuristic: prefer those with smaller spread and centered values
    # This is a rough heuristic - smaller z_i might be better
    def heuristic_score(z_list):
        z_sorted = sorted(z_list)
        return z_sorted[-1]  # Use the largest z_i as a proxy
    
    candidates.sort(key=heuristic_score)
    
    best_B = float('inf')
    best_z = None
    evaluated_count = 0
    
    for z in candidates:
        if time.time() > deadline:
            break
            
        # Validate z
        if len(z) != 12:
            continue
        if any(z_i <= 0 for z_i in z):
            continue
        if len(set(z)) != 12:
            continue
        # Check distinctness with tolerance
        z_sorted = sorted(z)
        valid = True
        for i in range(11):
            if abs(z_sorted[i+1] - z_sorted[i]) < 1e-10:
                valid = False
                break
        if not valid:
            continue
            
        B = evaluate_candidate(z)
        evaluated_count += 1
        
        if B is not None and B < best_B:
            best_B = B
            best_z = z
            # If we find a very good solution, we can break early
            if best_B < 1.0:
                break
        
        # If we've evaluated many candidates and time is running out, stop
        if evaluated_count > 30 and time.time() > start_time + 300:
            break
    
    if best_z is None:
        # Fallback to seed if all failed
        return [4.0 * (i + 1) for i in range(12)]
        
    return best_z
