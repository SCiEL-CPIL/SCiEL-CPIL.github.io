"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_B(z_list):
    """Compute B for given z locations using exact rational arithmetic."""
    import sympy as sp
    
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    
    # Build g(x) = sum_k c_k L_2k^(alpha)(x), k=0..25
    # We need to solve for c_k from the linear system
    # g(0) = 0, g'(0) = 1, g(z_i) = 0, g'(z_i) = 0 for i=1..12
    # Total: 2 + 24 = 26 equations for 26 unknowns
    
    # Precompute Laguerre polynomials L_2k^(alpha)(x) for k=0..25
    # L_d^(alpha)(x) for d = 0, 2, 4, ..., 50
    
    laguerre_polys = []
    for k in range(26):
        d = 2 * k
        L = sp.genlaguerre(d, alpha)  # L_d^(alpha)(x)
        laguerre_polys.append(L)
    
    # Build the linear system: A * c = b
    # Unknowns: c_0, c_1, ..., c_25 (26 unknowns)
    # Equations:
    #   g(0) = sum_k c_k L_2k(0) = 0
    #   g'(0) = sum_k c_k L_2k'(0) = 1
    #   For each i: g(z_i) = 0, g'(z_i) = 0
    
    A = sp.zeros(26, 26)
    b = sp.zeros(26, 1)
    
    # Equation 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_polys[k].subs(x, 0)
    b[0, 0] = 0
    
    # Equation 1: g'(0) = 1
    for k in range(26):
        A[1, k] = sp.diff(laguerre_polys[k], x).subs(x, 0)
    b[1, 0] = 1
    
    # Equations 2-25: for each z_i, g(z_i)=0 and g'(z_i)=0
    eq_idx = 2
    for i in range(12):
        z_i = sp.Rational(float(z_list[i]))
        # g(z_i) = 0
        for k in range(26):
            A[eq_idx, k] = laguerre_polys[k].subs(x, z_i)
        b[eq_idx, 0] = 0
        eq_idx += 1
        # g'(z_i) = 0
        for k in range(26):
            A[eq_idx, k] = sp.diff(laguerre_polys[k], x).subs(x, z_i)
        b[eq_idx, 0] = 0
        eq_idx += 1
    
    # Solve the linear system
    try:
        c = sp.LUdecomp(A).solve(b)
    except Exception:
        return float('inf')
    
    # Build g(x)
    g = sum(c[k] * laguerre_polys[k] for k in range(26))
    
    # Build the denominator: x * prod_i (x - z_i)^2
    denom = x
    for i in range(12):
        z_i = sp.Rational(float(z_list[i]))
        denom *= (x - z_i)**2
    
    # q(x) = g(x) / denom, exact polynomial division
    try:
        q, remainder = sp.div(g, denom, x)
        if remainder != 0:
            # Should be zero if system is consistent, but check
            pass
    except Exception:
        return float('inf')
    
    # Find real roots of q
    try:
        real_roots = sp.real_roots(q)
    except Exception:
        return float('inf')
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
    counted_roots = []
    for r in real_roots:
        r_val = float(r)
        if not np.isfinite(r_val):
            continue
        # Evaluate q at r - 1e-198 and r + 1e-198
        # Use a small epsilon for sign check
        eps = 1e-198
        try:
            q_minus = float(q.subs(x, sp.Rational(r_val - eps)))
            q_plus = float(q.subs(x, sp.Rational(r_val + eps)))
        except Exception:
            # Try with a larger epsilon if needed
            eps2 = 1e-15
            try:
                q_minus = float(q.subs(x, sp.Rational(r_val - eps2)))
                q_plus = float(q.subs(x, sp.Rational(r_val + eps2)))
            except Exception:
                continue
        
        if q_minus * q_plus < 0:
            counted_roots.append(r_val)
    
    if not counted_roots:
        B = 0.0
    else:
        B = max(0.0, max(counted_roots))
    
    return B

def construct():
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    # Initial guess: spread out positive values
    # Try different starting configurations
    best_B = float('inf')
    best_z = None
    
    # Start with evenly spaced positive values
    initial_z = [0.5 + i * 0.5 for i in range(12)]
    
    # First, evaluate the initial guess
    B_init = _compute_B(initial_z)
    if B_init < best_B:
        best_B = B_init
        best_z = initial_z[:]
    
    # Try a few different initial configurations
    configs = [
        [0.1 + i * 0.3 for i in range(12)],
        [0.5 + i * 1.0 for i in range(12)],
        [1.0 + i * 0.5 for i in range(12)],
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
    ]
    
    for config in configs:
        if time.time() > deadline - 120:
            break
        B_val = _compute_B(config)
        if B_val < best_B:
            best_B = B_val
            best_z = config[:]
    
    # Local optimization using Nelder-Mead
    if best_z is not None and time.time() < deadline - 180:
        def objective(z_flat):
            z_list = list(z_flat)
            # Ensure positivity
            if any(zi <= 0 for zi in z_list):
                return float('inf')
            # Ensure pairwise distinct
            for i in range(len(z_list)):
                for j in range(i+1, len(z_list)):
                    if abs(z_list[i] - z_list[j]) < 1e-10:
                        return float('inf')
            return _compute_B(z_list)
        
        # Try multiple restarts
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline - 120:
                break
            
            if restart == 0:
                x0 = np.array(best_z)
            else:
                # Perturb the best solution
                x0 = np.array(best_z) + np.random.normal(0, 0.1, size=12)
                x0 = np.maximum(x0, 0.01)  # Keep positive
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead', 
                                  options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if result.fun < best_B:
                    best_B = result.fun
                    best_z = list(result.x)
            except Exception:
                pass
    
    # Final polish: try to improve with smaller steps
    if best_z is not None and time.time() < deadline - 60:
        for _ in range(3):
            if time.time() > deadline - 30:
                break
            # Try small perturbations
            for step in [0.01, 0.005, 0.001]:
                if time.time() > deadline - 30:
                    break
                for i in range(12):
                    if time.time() > deadline - 30:
                        break
                    for direction in [-1, 1]:
                        z_trial = best_z[:]
                        z_trial[i] += direction * step
                        if z_trial[i] <= 0:
                            continue
                        B_trial = _compute_B(z_trial)
                        if B_trial < best_B:
                            best_B = B_trial
                            best_z = z_trial
    
    # Return the best z found
    if best_z is None:
        return initial_z
    
    # Ensure all values are positive and distinct
    result = []
    for i, zi in enumerate(best_z):
        if zi <= 0:
            zi = 0.1 + i * 0.1
        result.append(zi)
    
    # Ensure pairwise distinct
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if abs(result[i] - result[j]) < 1e-8:
                result[j] = result[i] + 1e-6
    
    return result
# EVOLVE-BLOCK-END
