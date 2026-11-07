"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precomputed coefficients for Physicists' Hermite Polynomials in terms of y = x^2
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12  => 16y^2 - 48y + 12
# H_8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168 => 256y^4 - 1344y^3 + 2688y^2 - 1344y + 168
# H_12(x) = 4096x^12 - 46080x^10 + 332160x^8 - 1003520x^6 + 1424640x^4 - 921600x^2 + 18480
#         => 4096y^6 - 46080y^5 + 332160y^4 - 1003520y^3 + 1424640y^2 - 921600y + 18480

# Coefficients in ascending power of y: [c0, c1, c2, c3, c4, c5, c6]
H0_y = [1.0]
H4_y = [12.0, -48.0, 16.0]
H8_y = [168.0, -1344.0, 2688.0, -1344.0, 256.0]
H12_y = [18480.0, -921600.0, 1424640.0, -1003520.0, 332160.0, -46080.0, 4096.0]

def compute_q_coeffs(c0, c1, c2):
    """
    Compute coefficients of q(y) = g(x)/x^2 as a polynomial in y=x^2.
    q(y) is degree 5.
    Returns array of coefficients [a0, a1, a2, a3, a4, a5] for a0 + a1*y + ... + a5*y^5
    """
    # c3 is fixed by g(0) = 0
    # g(0) = c0*1 + c1*12 + c2*168 + c3*18480 = 0
    c3 = -(c0 + 12.0*c1 + 168.0*c2) / 18480.0
    
    # g(y) = sum_{k=0}^6 b_k y^k
    # b_k = c0*(H0_y)_k + c1*(H4_y)_k + c2*(H8_y)_k + c3*(H12_y)_k
    # Note: H0_y has length 1, H4_y length 3, etc.
    
    # We construct b[0..6]
    b = np.zeros(7)
    
    # Add contributions
    b[0] += c0 * H0_y[0]
    
    for k in range(len(H4_y)):
        b[k] += c1 * H4_y[k]
        
    for k in range(len(H8_y)):
        b[k] += c2 * H8_y[k]
        
    for k in range(len(H12_y)):
        b[k] += c3 * H12_y[k]
        
    # g(0) should be b[0]. Due to floating point, it might not be exactly 0.
    # But mathematically, b[0] = 0.
    # q(y) = g(y)/y = b[1] + b[2]y + ... + b[6]y^5
    q_coeffs = b[1:7]
    
    return q_coeffs

def evaluate_objective(coeffs):
    """
    Objective function to minimize.
    Returns r^2, where r is the largest real root of q(x) with sign change.
    If no such root exists or value is undefined, returns a large penalty.
    """
    c0, c1, c2 = coeffs
    q_coeffs = compute_q_coeffs(c0, c1, c2)
    
    # Find roots of q(y) = 0
    # q_coeffs are in ascending order [a0, ..., a5]
    # np.roots expects descending order [a5, ..., a0]
    try:
        roots_y = np.roots(q_coeffs[::-1])
    except Exception:
        return 1e10
        
    best_r_sq = 0.0
    
    # We need to check each root
    for y_root in roots_y:
        # Filter: must be real and positive
        if abs(y_root.imag) > 1e-10:
            continue
        y_real = y_root.real
        if y_real <= 1e-12:
            continue
            
        # Check sign change at y_real
        # Evaluate q(y) at y_real - eps and y_real + eps
        # We can use polynomial evaluation
        # To avoid numerical issues with very small eps relative to large roots, 
        # we can just check the sign of the leading term or use a relative eps.
        
        # Let's evaluate q(y) using Horner's method on the ascending coeffs
        def q_eval(y_val):
            # Horner for ascending coeffs
            res = 0.0
            for c in q_coeffs[::-1]:
                res = res * y_val + c
            return res
            
        eps = 1e-8 * (1.0 + abs(y_real))
        y_minus = max(0.0, y_real - eps)
        y_plus = y_real + eps
        
        q_minus = q_eval(y_minus)
        q_plus = q_eval(y_plus)
        
        # Sign change if product < 0
        if q_minus * q_plus < 0:
            r_sq = y_real
            if r_sq > best_r_sq:
                best_r_sq = r_sq
                
    if best_r_sq == 0.0:
        # If no sign-changing positive root found, this configuration is likely invalid 
        # or the "largest root" logic implies something else. 
        # However, g(x) is a polynomial. If it doesn't cross zero (except at 0), 
        # the inequality might not hold in the standard way or the bound is infinity.
        # Returning a large value encourages the optimizer to find a valid crossing.
        return 1e10
        
    return best_r_sq

def construct():
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_coeffs = seed.copy()
    best_val = evaluate_objective(seed)
    
    # Local search using Nelder-Mead
    # We will perform multiple restarts to avoid local minima
    rng = np.random.RandomState(42)
    
    # Define a function to run a local minimization step
    def local_minimize(init_coeffs, max_time=2.0):
        start = time.time()
        try:
            res = minimize(
                evaluate_objective, 
                init_coeffs, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            # Check time
            if time.time() - start > max_time:
                # If it took too long, we might not use it, but Nelder-Mead usually stops fast
                pass
            if res.success or res.fun < best_val:
                return res.x, res.fun
        except Exception:
            pass
        return init_coeffs, best_val

    # Phase 1: Random restarts with local refinement
    # We'll try a few dozen random points in a reasonable range
    # The seed is around [-1.13, 0.04, 0.0003]
    # Let's explore a box around 0
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Generate random initial point
        # Mix of small perturbations and larger jumps
        if i < 10:
            # Small perturbations around seed
            init = seed + rng.normal(0, 0.1, size=3)
        elif i < 30:
            # Medium range
            init = rng.uniform(-5, 5, size=3)
        else:
            # Large range
            init = rng.uniform(-20, 20, size=3)
            
        x_opt, f_opt = local_minimize(init)
        
        if f_opt < best_val:
            best_val = f_opt
            best_coeffs = x_opt
            
    # Phase 2: Fine-tuning around the best found
    # Run a few more local optimizations with very small steps around best_coeffs
    if time.time() < deadline - 5.0:
        for i in range(10):
            init = best_coeffs + rng.normal(0, 1e-4, size=3)
            x_opt, f_opt = local_minimize(init)
            if f_opt < best_val:
                best_val = f_opt
                best_coeffs = x_opt

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
