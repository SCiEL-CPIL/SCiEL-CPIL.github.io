import sympy as sp
import numpy as np
import time

def evaluate_candidate(z_list, Ls_cache=None, x_sym=None):
    """
    Evaluates the candidate z_list exactly.
    Returns B if valid, None if invalid.
    """
    try:
        # Convert to exact rationals
        z_rat = [sp.Rational(float(z)) for z in z_list]
        
        # Define x and Laguerre polynomials
        if x_sym is None:
            x_sym = sp.Symbol('x')
        x = x_sym
        alpha = sp.Rational(-1, 2)
        
        # Use cached Laguerre polynomials or compute
        if Ls_cache is None:
            Ls_cache = []
            for k in range(26):
                Ls_cache.append(sp.Laguerre(2*k, alpha, x))
        Ls = Ls_cache
        
        # g(x) = sum c_k L_2k(x)
        cs = sp.symbols('c0:26')
        g = sum(cs[k] * Ls[k] for k in range(26))
        
        # Build system of equations
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
            # g(z) = 0
            for k in range(26):
                M[idx, k] = Ls[k].subs(x, z)
            rhs[idx] = 0
            idx += 1
            
            # g'(z) = 0
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
            
        # Substitute cs back into g
        g_sol = g.subs(list(zip(cs, sol)))
        
        # q(x) = g(x) / (x * prod (x - z_i)^2)
        denom = x
        for z in z_rat:
            denom *= (x - z)**2
            
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
            
        # Filter roots with sign change
        max_r = 0.0
        for r in roots:
            r_float = float(r)
            if not np.isfinite(r_float) or r_float <= 0:
                continue
                
            delta = 1e-198
            val_minus = q_poly.subs(x, r_float - delta)
            val_plus = q_poly.subs(x, r_float + delta)
            
            try:
                v_minus = float(val_minus)
                v_plus = float(val_plus)
            except:
                continue
            
            if v_minus * v_plus < 0:
                if r_float > max_r:
                    max_r = r_float
                    
        return float(max_r)
        
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 570.0  # Leave 60s margin before 630s limit
    
    # Precompute Laguerre polynomials (independent of z)
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    Ls_cache = []
    for k in range(26):
        Ls_cache.append(sp.Laguerre(2*k, alpha, x))
    
    # Primary candidate: roots of Laguerre L_12^(-1/2)
    # This was measured to give B=0.403
    try:
        L12 = sp.Laguerre(12, alpha, x)
        roots_L12 = sp.nroots(L12, n=50, maxsteps=200)
        real_roots = sorted([float(r) for r in roots_L12 if abs(r.as_real_imag()[1]) < 1e-10 and r.as_real_imag()[0] > 0])
        if len(real_roots) != 12:
            real_roots = [4.0 * (i + 1) for i in range(12)]
    except:
        real_roots = [4.0 * (i + 1) for i in range(12)]
    
    best_B = float('inf')
    best_z = None
    
    # Evaluate the Laguerre root candidate first (best known)
    B = evaluate_candidate(real_roots, Ls_cache, x)
    if B is not None and B < best_B:
        best_B = B
        best_z = list(real_roots)
    
    # Local optimization: perturb the Laguerre roots
    # Try small perturbations to see if we can improve
    if best_z is not None and time.time() < deadline - 120:
        base = np.array(best_z, dtype=float)
        
        # Try coordinate-wise perturbations
        for dim in range(12):
            if time.time() > deadline - 60:
                break
            for delta_factor in [0.1, -0.1, 0.05, -0.05, 0.2, -0.2]:
                if time.time() > deadline - 60:
                    break
                perturbed = base.copy()
                perturbed[dim] *= (1 + delta_factor)
                if perturbed[dim] <= 0:
                    continue
                # Check distinctness
                if len(set(np.round(perturbed, 10))) != 12:
                    continue
                    
                B = evaluate_candidate(perturbed.tolist(), Ls_cache, x)
                if B is not None and B < best_B:
                    best_B = B
                    best_z = perturbed.tolist()
                    base = perturbed.copy()
                    break  # Restart from improved point
        
        # Try simultaneous small perturbations
        if time.time() < deadline - 60:
            for scale in [0.05, 0.1, 0.2]:
                if time.time() > deadline - 60:
                    break
                rng = np.random.RandomState(42)
                for trial in range(20):
                    if time.time() > deadline - 60:
                        break
                    perturb = base * (1 + scale * rng.randn(12))
                    if np.any(perturb <= 0):
                        continue
                    if len(set(np.round(perturb, 10))) != 12:
                        continue
                        
                    B = evaluate_candidate(perturb.tolist(), Ls_cache, x)
                    if B is not None and B < best_B:
                        best_B = B
                        best_z = perturb.tolist()
                        base = perturb.copy()
    
    # Try some other candidate families if time permits
    other_candidates = []
    # Geometric spacing
    other_candidates.append(list(np.exp(np.linspace(np.log(0.5), np.log(50), 12))))
    # Clustered
    other_candidates.append([0.5 * (i + 1) for i in range(12)])
    # Wider spread
    other_candidates.append([8.0 * (i + 1) for i in range(12)])
    
    for cand in other_candidates:
        if time.time() > deadline - 60:
            break
        if any(z <= 0 for z in cand):
            continue
        if len(set(np.round(cand, 10))) != 12:
            continue
            
        B = evaluate_candidate(cand, Ls_cache, x)
        if B is not None and B < best_B:
            best_B = B
            best_z = list(cand)
    
    if best_z is None:
        return real_roots
        
    return best_z
