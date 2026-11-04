"""Refined Cohn-Goncalves Laguerre double-root construction for Fourier sign uncertainty."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

# Precompute factorials up to a reasonable degree
# P(x) has degree 24 (12 double roots -> (x-r)^2 -> degree 2 per root -> 24 total)
# P(x)^2 has degree 48.
_MAX_DEG = 50
_FACT = np.array([math.factorial(i) for i in range(_MAX_DEG + 1)], dtype=np.float64)

def _compute_c4_bound(roots):
    """
    Compute the C4 bound for the given roots.
    The bound is proportional to ||P||_2 / |P(0)|.
    We return the value to be MINIMIZED.
    """
    m = len(roots)
    # P(x) = prod (x - r_i)^2
    # |P(0)| = prod r_i^2
    # log |P(0)| = 2 * sum log(r_i)
    
    # Construct P(x) coefficients
    # P(x) = (x-r_1)^2 ... (x-r_m)^2
    # It's more stable to compute Q(x) = prod (x-r_i) first, then P(x) = Q(x)^2
    q_coeffs = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)
        new_q = np.zeros(len(q_coeffs) + 1)
        new_q[1:] += q_coeffs
        new_q[:-1] -= r * q_coeffs
        q_coeffs = new_q
        
    # P(x) = Q(x)^2
    # Convolve q_coeffs with itself
    p_coeffs = np.convolve(q_coeffs, q_coeffs)
    
    # Ensure degree is at most _MAX_DEG
    if len(p_coeffs) > _MAX_DEG + 1:
        p_coeffs = p_coeffs[:_MAX_DEG + 1]
        
    # Integral of e^{-x} P(x) dx = sum c_k * k!
    norm_sq = np.dot(p_coeffs, _FACT[:len(p_coeffs)])
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e300, None
        
    # log |P(0)|
    log_p0 = 2.0 * np.sum(np.log(roots))
    
    # The bound C4 is roughly proportional to sqrt(norm_sq) / |P(0)|
    # Minimizing C4 is equivalent to minimizing log(norm_sq)/2 - log|P(0)|
    # Or maximizing log|P(0)| - 0.5*log(norm_sq)
    # Let's minimize - (log|P(0)| - 0.5*log(norm_sq))
    val = -log_p0 + 0.5 * np.log(norm_sq)
    
    # Gradient calculation
    # d/d(r_j) val = -d/d(r_j) log|P(0)| + 0.5 * d/d(r_j) log(norm_sq)
    # d/d(r_j) log|P(0)| = 2 * (1/r_j)
    
    # d/d(r_j) norm_sq
    # norm_sq = <P, P> where P = Q^2
    # This is tricky to do analytically fast without symbolic diff.
    # Let's use numerical gradient for the optimization if analytic is too complex,
    # or derive it.
    
    # Actually, let's just return the value and let minimize use numerical gradient
    # or provide a simple analytic gradient if possible.
    # Given the complexity, numerical gradient is safer for a first pass, 
    # but L-BFGS-B with numerical grad can be slow.
    # Let's try to provide a gradient.
    
    # Alternative: Use the log-space optimization from the seed but with the CORRECT objective.
    # The seed used quadrature. Let's stick to exact moments for speed and accuracy.
    
    # Gradient of norm_sq w.r.t r_j
    # P(x) = Q(x)^2. Q(x) = prod (x-r_i).
    # dQ/dr_j = -Q(x) / (x-r_j)
    # dP/dr_j = 2 Q(x) dQ/dr_j = -2 Q(x)^2 / (x-r_j) = -2 P(x) / (x-r_j)
    # d(norm_sq)/dr_j = <dP/dr_j, 1> * ... wait, norm_sq = int e^-x P(x) dx.
    # d(norm_sq)/dr_j = int e^-x (-2 P(x) / (x-r_j)) dx
    # This integral is not a simple moment.
    
    # Let's use a finite difference gradient for the optimization step, 
    # but evaluate the function exactly.
    # Or, better, use the fact that we are minimizing a smooth function.
    
    return val, None

def _objective_and_grad(roots):
    """
    Compute objective and gradient using central differences for gradient.
    """
    val, _ = _compute_c4_bound(roots)
    
    # Numerical gradient
    grad = np.zeros_like(roots)
    h = 1e-6
    for i in range(len(roots)):
        r_plus = roots.copy()
        r_plus[i] += h
        val_plus, _ = _compute_c4_bound(r_plus)
        
        r_minus = roots.copy()
        r_minus[i] -= h
        val_minus, _ = _compute_c4_bound(r_minus)
        
        grad[i] = (val_plus - val_minus) / (2 * h)
        
    return val, grad

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    m = 12
    
    # Generate multiple initializations
    inits = []
    
    # 1. Standard Laguerre zeros (L_12)
    try:
        x_lag, _ = np.polynomial.laguerre.laggauss(m)
        inits.append(x_lag.copy())
    except Exception:
        pass
        
    # 2. Dyson gas initialization (repulsion ~ 2 for simple roots, ~4 for double? 
    #    The potential is -2 sum log(x-r_i) for P, so repulsion is stronger)
    def _dyson_init(m, repulsion=4.0, iterations=500):
        roots = np.linspace(1.0, 20.0, m)
        for _ in range(iterations):
            forces = np.zeros(m)
            for i in range(m):
                # Attractive force from origin (log term)
                forces[i] += 1.0 / roots[i] 
                # Repulsive forces from other roots
                for j in range(m):
                    if i != j:
                        forces[i] -= repulsion / (roots[i] - roots[j])
            
            # Gradient descent step
            step = 0.01
            for i in range(m):
                roots[i] -= step * forces[i]
                roots[i] = max(roots[i], 0.1)
            roots = np.sort(roots)
        return roots

    inits.append(_dyson_init(m, repulsion=4.0))
    inits.append(_dyson_init(m, repulsion=6.0))
    
    # 3. Geometric sequences
    for ratio in [1.1, 1.2, 1.5]:
        inits.append(np.array([1.0 * (ratio ** i) for i in range(m)]))
        
    # 4. Uniform with different scales
    for scale in [1.0, 2.0, 5.0]:
        inits.append(np.linspace(0.5, 10.0 * scale, m))
        
    # Evaluate all inits
    scored_inits = []
    for init_roots in inits:
        init_roots = np.sort(init_roots)
        if any(r <= 0 for r in init_roots):
            continue
        val, _ = _compute_c4_bound(init_roots)
        if np.isfinite(val):
            scored_inits.append((val, init_roots))
            
    scored_inits.sort(key=lambda x: x[0])
    
    # Keep top 3 distinct initializations
    best_inits = []
    seen = set()
    for val, roots in scored_inits:
        key = tuple(np.round(roots, 1))
        if key not in seen:
            best_inits.append(roots)
            seen.add(key)
        if len(best_inits) >= 3:
            break
            
    if not best_inits:
        best_inits = [np.linspace(0.5, 10.0, m)]
        
    best_roots = best_inits[0].copy()
    best_val, _ = _compute_c4_bound(best_roots)
    
    # Optimize from each initial point
    for idx, init_roots in enumerate(best_inits):
        if time.time() - start_time > deadline - 30:
            break
            
        try:
            # Use L-BFGS-B
            # We minimize _objective_and_grad
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                init_roots,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=[(0.1, 100.0)] * m,
                options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if np.isfinite(result.fun):
                candidate_roots = np.sort(result.x)
                cand_val, _ = _compute_c4_bound(candidate_roots)
                if cand_val < best_val:
                    best_roots = candidate_roots
                    best_val = cand_val
                    
        except Exception:
            pass
        
        # Polish with tighter tolerances
        if time.time() - start_time > deadline - 20:
            break
            
        try:
            result2 = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_roots,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=[(0.1, 100.0)] * m,
                options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if np.isfinite(result2.fun):
                candidate_roots = np.sort(result2.x)
                cand_val, _ = _compute_c4_bound(candidate_roots)
                if cand_val < best_val:
                    best_roots = candidate_roots
                    best_val = cand_val
                    
        except Exception:
            pass
            
    # Final verification
    best_roots = np.sort(best_roots)
    if any(r <= 0 for r in best_roots) or len(best_roots) != m:
        best_roots = np.linspace(0.5, 10.0, m)
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
