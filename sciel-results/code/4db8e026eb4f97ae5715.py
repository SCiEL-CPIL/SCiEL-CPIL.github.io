"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_value(h):
    """
    Compute the value metric for a given height vector h.
    h: array of length N (50)
    """
    h = np.asarray(h, dtype=float)
    # Convolve h with itself
    g = np.convolve(h, h)
    m = len(g)  # m = 2*N - 1 = 99
    
    # y = [0, g[0], ..., g[m-1], 0]
    # Length of y is m + 2
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    # Points: np.linspace(-0.5, 0.5, m + 2)
    # Spacing w
    x = np.linspace(-0.5, 0.5, m + 2)
    w = x[1] - x[0]
    
    # l2 calculation
    # sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # i goes from 0 to m.
    # y indices involved: y[0]..y[m+1]
    # Terms: y[i]^2 + y[i]*y[i+1] + y[i+1]^2
    
    # We can vectorize this.
    # Let A = y[0:m+1], B = y[1:m+2]
    A = y[0:m+1]
    B = y[1:m+2]
    terms = A**2 + A*B + B**2
    l2 = (w / 3.0) * np.sum(terms)
    
    # norm_1 = sum(abs(g)) / (m + 1)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # norm_inf = max(abs(g))
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Seed
    h_seed = np.ones(N)
    best_h = h_seed.copy()
    best_val = _compute_value(best_h)
    
    # Strategy: Low-rank superposition of DCT basis functions
    # h = sum_{k=0}^{K-1} c_k * phi_k
    # We want h >= 0.
    # Let's use a few low-frequency DCT basis vectors.
    # DCT-II basis: phi_k[n] = cos(pi * k * (2n + 1) / (2N))
    
    # Number of basis functions to use
    K = 5
    
    # Precompute basis matrix
    n = np.arange(N)
    basis = np.zeros((K, N))
    for k in range(K):
        basis[k] = np.cos(np.pi * k * (2*n + 1) / (2*N))
        
    # Normalize basis? Not strictly necessary, but helps with scaling.
    # Let's just optimize coefficients c.
    
    # Objective function for optimizer: maximize value -> minimize -value
    # Constraint: h >= 0.
    # h = basis @ c
    
    def objective(c):
        h = basis @ c
        # Enforce non-negativity softly or hard?
        # Hard constraint via penalty or bounds?
        # Differential Evolution can handle bounds on c, but h >= 0 is a linear constraint on c.
        # It's easier to use a penalty for negative h or clip h.
        # Clipping changes the gradient structure if we used grad-based, but DE is derivative-free.
        # However, clipping h = max(h, 0) makes the function non-smooth.
        # Alternative: Parameterize h with softplus or squares?
        # h = basis @ c doesn't guarantee h >= 0 easily with simple bounds on c.
        
        # Let's try a different parameterization:
        # h_i = exp(z_i) ? No, that restricts to positive, but loses the ability to be zero.
        # The seed is all 1s.
        # Let's stick to h = basis @ c and use a penalty for negative values.
        # Or, use bounds on c such that h is likely positive?
        
        # Better approach: Use a parameterization that ensures non-negativity.
        # h = (basis @ c)^2 ? No, that squares the signal.
        # h = basis @ c, and we clip h to 0.
        # Since DE is stochastic, clipping is a common and effective heuristic for constrained problems.
        
        h_clipped = np.maximum(h, 0.0)
        
        # Check time
        if time.time() > deadline:
            return best_val  # Return current best to stop optimization early effectively
        
        val = _compute_value(h_clipped)
        return -val
    
    # Bounds for coefficients c
    # The seed h=1 corresponds to some c.
    # Let's find c_seed that approximates h_seed.
    # Since basis[0] is constant (cos(0)=1), c[0] should be around 1.
    # Other coefficients 0.
    c_seed = np.zeros(K)
    c_seed[0] = 1.0
    
    # Set bounds for c
    # If c[0] is too large, h becomes large.
    # If c[k] for k>0 is large, h can go negative.
    # Let's set reasonable bounds.
    bounds = [(-2.0, 2.0)] * K
    
    # Run Differential Evolution
    # Use a smaller population to save time, but enough to explore.
    # Maxiter: limit based on time.
    
    # We will run DE with a callback to check time.
    
    def callback(xk, convergence=True):
        if time.time() > deadline:
            raise RuntimeError("Time limit reached")
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15, 
            tol=0.01,
            callback=callback,
            init='latinhypercube'
        )
    except RuntimeError:
        pass
    except Exception:
        # Fallback if DE fails
        pass
    
    if 'result' in locals() and result.success:
        c_opt = result.x
        h_opt = np.maximum(basis @ c_opt, 0.0)
        val_opt = _compute_value(h_opt)
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
    
    # Check if we have time for a local polish
    if time.time() < deadline - 10:
        # Use Nelder-Mead on the clipped h directly?
        # Or on c?
        # Let's try polishing on c using the clipped objective.
        # NM can handle non-smoothness reasonably well if steps are small.
        
        def obj_nm(c):
            h = np.maximum(basis @ c, 0.0)
            return -_compute_value(h)
        
        try:
            res_nm = minimize(
                obj_nm, 
                c_opt if 'c_opt' in locals() else c_seed, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            h_nm = np.maximum(basis @ res_nm.x, 0.0)
            val_nm = _compute_value(h_nm)
            if val_nm > best_val:
                best_val = val_nm
                best_h = h_nm
        except Exception:
            pass
    
    return list(best_h)

# EVOLVE-BLOCK-END
