import sympy as sp
import numpy as np
import time

def evaluate_candidate(z_list):
    """
    Evaluates the candidate z_list exactly.
    Returns B if valid, None if invalid.
    """
    try:
        # Convert to exact rationals
        z_rat = [sp.Rational(float(z)) for z in z_list]
        
        # Define x and Laguerre polynomials
        x = sp.Symbol('x')
        alpha = sp.Rational(-1, 2)
        
        # Precompute L_2k(x) for k=0..25
        Ls = []
        for k in range(26):
            Ls.append(sp.Laguerre(2*k, alpha, x))
        
        # Build matrix M and rhs directly
        M = sp.zeros(26, 26)
        rhs = sp.zeros(26, 1)
        
        # Eq 0: g(0) = 0
        for k in range(26):
            M[0, k] = Ls[k].subs(x, 0)
        rhs[0] = 0
        
        # Eq 1: g'(0) = 1
        for k in range(26):
            M[1, k] = sp.diff(Ls[k], x).subs(x, 0)
        rhs[1] = 1
        
        idx = 2
        for z in z_rat:
            for k in range(26):
                M[idx, k] = Ls[k].subs(x, z)
            rhs[idx] = 0
            idx += 1
            
            for k in range(26):
                M[idx, k] = sp.diff(Ls[k], x).subs(x, z)
            rhs[idx] = 0
            idx += 1
        
        # Solve
        try:
            sol = M.LUsolve(rhs)
        except sp.matrices.exceptions.SingularMatrixException:
            return None
        except Exception:
            return None
        
        # Build g_sol
        cs = sp.symbols('c0:26')
        g = sum(cs[k] * Ls[k] for k in range(26))
        g_sol = g.subs(list(zip(cs, sol)))
        
        # q(x) = g(x) / (x * prod (x - z_i)^2)
        denom = x
        for z in z_rat:
            denom *= (x - z)**2
            
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
            try:
                r_float = float(r)
            except:
                continue
            if not np.isfinite(r_float) or r_float <= 0:
                continue
            
            delta = 1e-198
            try:
                val_minus = float(q_poly.subs(x, r_float - delta))
                val_plus = float(q_poly.subs(x, r_float + delta))
            except:
                continue
            
            if val_minus * val_plus < 0:
                if r_float > max_r:
                    max_r = r_float
                    
        return float(max_r)
        
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 570.0  # Leave 60s margin
    
    # Strategy: Use the 12 smallest positive zeros of the Bessel function J_{1/2}(x).
    # J_{1/2}(x) is proportional to sin(sqrt(x))/sqrt(x).
    # The zeros are at sqrt(x) = n*pi for n=1, 2, ..., 12.
    # Thus x = (n*pi)^2 = n^2 * pi^2.
    
    # The task requires z_i to be converted to Rational(float(z_i)).
    # So we generate the float values.
    
    candidates = []
    
    # Primary candidate: Bessel zeros
    bessel_zeros = []
    for n in range(1, 13):
        bessel_zeros.append((n * np.pi)**2)
    candidates.append(bessel_zeros)
    
    # Fallbacks in case Bessel zeros are invalid or time out (unlikely, but safe)
    # These are standard heuristics from the previous program, kept as backups.
    candidates.append([4.0 * (i + 1) for i in range(12)])
    candidates.append([1.0 * (i + 1) for i in range(12)])
    
    best_B = float('inf')
    best_z = None
    
    for z in candidates:
        if time.time() > deadline:
            break
            
        # Check if z is valid (positive, distinct)
        z_sorted = sorted(z)
        if len(set(z_sorted)) != 12:
            continue
        if any(z_i <= 0 for z_i in z_sorted):
            continue
        # Check pairwise distinct with tolerance
        valid = True
        for i in range(len(z_sorted) - 1):
            if abs(z_sorted[i+1] - z_sorted[i]) < 1e-10:
                valid = False
                break
        if not valid:
            continue
            
        B = evaluate_candidate(z)
        
        if B is not None and B < best_B:
            best_B = B
            best_z = z
            
    if best_z is None:
        # Fallback to seed if all failed
        return [4.0 * (i + 1) for i in range(12)]
        
    return best_z
