import numpy as np
import time
import sys

N = 50

def _eval(h):
    """
    Compute the objective value for a given non-negative height vector h.
    The objective is scale-invariant, so we can work with any scaling.
    """
    h = np.asarray(h, dtype=float)
    # Ensure non-negative (clip tiny negatives from numerical noise)
    h = np.clip(h, 0, None)
    s = np.sum(h)
    if s == 0:
        return 0.0
    
    # Normalize to sum=1 for stable gradient calculations? 
    # Actually, the objective is scale invariant: f(ah) = f(h).
    # So we can just compute it directly.
    
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    # Points are np.linspace(-0.5, 0.5, m + 2)
    # Number of intervals = m + 1
    w = 1.0 / (m + 1)
    
    # l2 = sum_{i=0}^m (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    # Let A = y[:m+1], B = y[1:m+2]
    # terms = A^2 + A*B + B^2
    A = y[:m+1]
    B = y[1:m+2]
    terms = A**2 + A*B + B**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def _objective_and_grad(h):
    """
    Compute the objective value and its gradient with respect to h.
    We use finite differences for the gradient for simplicity and robustness,
    or analytic if possible. Given the complexity of l2 involving conv,
    analytic is better but finite diff with central differences on a 
    normalized simplex might be tricky.
    
    Let's try to derive or use a robust finite difference.
    Since we are doing gradient ascent, we need d(f)/d(h_i).
    
    Alternative: Since the problem is small (N=50), we can use a 
    parameterization that ensures non-negativity and sum=1, e.g. softmax,
    and optimize over the logits. This avoids projection issues.
    
    Let z be the logits. h_i = exp(z_i) / sum(exp(z_j)).
    We optimize z.
    """
    # This function is a placeholder if we switch to softmax parameterization.
    # For now, let's stick to the direct h-space but use a robust optimizer like Nelder-Mead
    # or a custom projected gradient method.
    
    # Actually, the prompt says "rethinking the approach".
    # The current approach is simple hill climbing with random perturbations on the simplex.
    # A "materially different" approach:
    # 1. Use a parametric family (e.g., Gaussian, Polynomial) to reduce dimensionality?
    #    - Inspiration 2 mentions 2D parametric family.
    #    - But the optimal shape might be irregular.
    # 2. Use scipy.optimize.minimize with a proper constraint or parameterization.
    #    - Minimize -value(h) subject to h >= 0, sum(h)=1.
    #    - Use SLSQP or trust-constr.
    # 3. Use a global optimizer like Differential Evolution or Basin Hopping?
    #    - Basin Hopping with L-BFGS-B polish is often very effective for these types of problems.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Strategy: Multi-start Basin Hopping / or just Multi-start L-BFGS-B with different initializations.
    # Since N=50 is small, L-BFGS-B is fast.
    # We will use scipy.optimize.minimize.
    
    from scipy.optimize import minimize
    
    def neg_value(h):
        # h is flattened
        return -_eval(h)
    
    # Constraints: sum(h) = 1, h >= 0
    # To use L-BFGS-B, we can parameterize h using softmax to automatically satisfy constraints.
    # Let z be the free variables. h = softmax(z).
    # This removes the need for explicit constraints in the optimizer, but the gradient
    # must be with respect to z.
    
    def softmax(z):
        # Stable softmax
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z)
    
    def neg_value_softmax(z):
        h = softmax(z)
        return -_eval(h)
    
    # We don't have an analytic gradient for z easily without deriving it.
    # Finite difference gradient for z?
    # Or just use a black-box optimizer like Nelder-Mead on z?
    # Nelder-Mead doesn't need gradients and is robust.
    
    # Let's try a mix:
    # 1. Generate diverse initial z vectors.
    # 2. Run Nelder-Mead on z for a fixed number of iterations or time.
    # 3. Keep the best h.
    
    best_h = np.ones(N) / N
    best_score = _eval(best_h)
    
    # Initial candidates in z-space (logits)
    # Note: z is defined up to a constant shift. We can fix z[0]=0 or just let it float.
    # Nelder-Mead will handle it.
    
    initial_zs = []
    
    # 1. Uniform h -> z = 0
    initial_zs.append(np.zeros(N))
    
    # 2. Gaussian peaks
    x = np.arange(N)
    for center in [N//4, N//2, 3*N//4]:
        for width in [2, 5, 10]:
            h_init = np.exp(-((x - center)**2) / (2*width**2))
            # Convert h to z? z = log(h) + const.
            # If h_i > 0, z_i = log(h_i).
            # If any h_i == 0, log is -inf.
            # For initialization, we can just use log(h + eps).
            z_init = np.log(h_init + 1e-10)
            initial_zs.append(z_init)
            
    # 3. Random h -> z
    for _ in range(5):
        h_init = np.random.rand(N)
        z_init = np.log(h_init + 1e-10)
        initial_zs.append(z_init)
        
    # 4. "Boxcar" shapes
    for length in [10, 20, 30]:
        for start in range(0, N-length+1, 10):
            h_init = np.zeros(N)
            h_init[start:start+length] = 1.0
            z_init = np.log(h_init + 1e-10)
            initial_zs.append(z_init)

    # Run optimization
    for z0 in initial_zs:
        if time.time() > deadline:
            break
            
        # Nelder-Mead parameters
        # maxiter: limit per run to allow multiple starts
        # maxfev: limit function evaluations
        options = {
            'maxiter': 500,
            'maxfev': 1000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'adaptive': True
        }
        
        try:
            res = minimize(neg_value_softmax, z0, method='Nelder-Mead', options=options)
            if res.success or res.fun < -best_score:
                h_candidate = softmax(res.x)
                score_candidate = -res.fun
                if score_candidate > best_score:
                    best_score = score_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
            
    # If we have time left, do a final polish with L-BFGS-B on the simplex?
    # Or just return the best found.
    
    # Ensure the output is a list of floats
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
