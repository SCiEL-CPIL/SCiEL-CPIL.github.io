import numpy as np
import scipy.special as sc
import scipy.optimize as opt
import time
import math

def rudin_problem10() -> np.ndarray:
    # 1. Setup Quadrature
    # 70 Gauss-Legendre points in cos(theta)
    n_theta = 70
    t_nodes, w_theta = np.polynomial.legendre.leggauss(n_theta)
    
    # 130 equally spaced phi points
    n_phi = 130
    phi_nodes = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = 2 * np.pi / n_phi
    
    # Create grid
    t_grid, phi_grid = np.meshgrid(t_nodes, phi_nodes, indexing='ij')
    t_flat = t_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # Weights for each point
    w_grid = np.outer(w_theta, np.ones(n_phi) * w_phi)
    w_flat = w_grid.flatten()
    
    # 2. Compute Spherical Harmonics Y_m for m = -10..10
    k = 10
    ms = np.arange(-k, k + 1)
    n_points = n_theta * n_phi
    Y = np.zeros((n_points, 2 * k + 1), dtype=np.float64)
    
    # Precompute factorials for normalization
    fact = [math.factorial(i) for i in range(2 * k + 1)]
    
    for idx, m in enumerate(ms):
        abs_m = abs(m)
        # Condon-Shortley phase is included in lpmv
        P = sc.lpmv(abs_m, k, t_flat)
        
        # Normalization constant N_m
        norm_factor = math.sqrt((2 * k + 1) / (4 * np.pi) * fact[k - abs_m] / fact[k + abs_m])
        
        if m == 0:
            Y[:, idx] = norm_factor * P
        elif m > 0:
            Y[:, idx] = np.sqrt(2) * norm_factor * P * np.cos(m * phi_flat)
        else:
            Y[:, idx] = np.sqrt(2) * norm_factor * P * np.sin(abs_m * phi_flat)
            
    # 3. Define Objective Function
    # Minimize R^4 = ||f||_4^4 / ||f||_2^4
    def objective_from_c(c):
        f = Y @ c
        norm4_pow = np.sum(w_flat * f**4)
        norm2_pow = np.sum(w_flat * f**2)
        if norm2_pow == 0:
            return np.inf
        return norm4_pow / (norm2_pow**2)

    # 4. Search Strategy: Low-Rank Factorization + Nelder-Mead
    # We parameterize c = u * v^T where u, v are vectors? No, that's rank 1.
    # Let's use a small rank r, say r=3. c = U @ V.T where U, V are (21, r).
    # Total parameters: 2 * 21 * r = 42 * r. For r=2, 84 params. Still high.
    # Actually, the current dimension is 21. Low rank 2 gives 42 params. That's worse.
    # Wait, the "low rank" idea usually applies to matrices. Here c is a vector.
    # A different approach: Exploit symmetry or specific basis.
    
    # Let's rethink. The "flattest" harmonic likely has specific symmetries.
    # Degree 10 harmonics with high symmetry (e.g., only even m, or specific m values)
    # might be easier to optimize.
    
    # Alternative Strategy: Simulated Annealing or Basin Hopping on c directly,
    # but with a smarter initial guess based on "flat" distributions.
    # A uniform function on the sphere has minimum L4/L2 ratio.
    # The projection of a constant function onto the degree-10 subspace is 0.
    # So we need the degree-10 harmonic that looks most like a constant.
    # This is equivalent to maximizing the "flatness".
    
    # Let's try a different optimization: 
    # Use scipy.optimize.differential_evolution which is global and robust.
    # It might be slower per iteration but better at escaping local minima in 21D.
    # However, 21D is too high for DE to be fast enough for many restarts.
    
    # Let's stick to a refined Multi-Start Nelder-Mead but with BETTER initializations.
    # Current: Random unit vectors.
    # New: 
    # 1. Random unit vectors.
    # 2. Vectors concentrated on specific m values (symmetry breaking).
    # 3. Vectors from previous best runs (if we could cache, but we can't across calls easily without global state).
    
    # Actually, let's look at the inspiration: "Multi-restart Nelder-Mead with random initializations... Measured: 1 -> 1.093".
    # The current program uses Powell. Switching to Nelder-Mead is a different algorithm.
    # Nelder-Mead is often faster for low-to-medium dimensions and doesn't need gradients.
    # Let's implement Multi-Start Nelder-Mead.
    
    start_time = time.time()
    deadline = 590.0
    
    best_c = None
    best_score = np.inf
    
    # Helper to run Nelder-Mead
    def run_nm(c_init):
        try:
            res = opt.minimize(objective_from_c, c_init, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            return res.x, res.fun
        except Exception:
            return None, np.inf

    # Initial guesses
    # 1. All ones (normalized)
    c_init_1 = np.ones(2 * k + 1)
    c_init_1 = c_init_1 / np.linalg.norm(c_init_1)
    
    # 2. Random restarts
    num_restarts = 0
    max_restarts = 200 # Limit to ensure we finish in time
    
    while True:
        current_time = time.time()
        if current_time > deadline - 10.0:
            break
            
        # Generate initial guess
        if num_restarts == 0:
            c_guess = c_init_1
        else:
            # Mix of random and structured
            if num_restarts % 5 == 0:
                # Structured: emphasize low m
                c_guess = np.zeros(2 * k + 1)
                c_guess[10] = 1.0 # m=0
                c_guess[9] = 0.5 # m=-1
                c_guess[11] = 0.5 # m=1
                c_guess = c_guess / np.linalg.norm(c_guess)
            elif num_restarts % 5 == 1:
                # Structured: emphasize high m
                c_guess = np.zeros(2 * k + 1)
                c_guess[0] = 1.0 # m=-10
                c_guess[20] = 1.0 # m=10
                c_guess = c_guess / np.linalg.norm(c_guess)
            else:
                # Random
                c_guess = np.random.randn(2 * k + 1)
                c_guess = c_guess / np.linalg.norm(c_guess)
                
        c_opt, score_opt = run_nm(c_guess)
        
        if score_opt < best_score:
            best_score = score_opt
            best_c = c_opt
            
        num_restarts += 1
        
        # Check if we should stop early if convergence is good?
        # No, just keep going until time.
        
    # Ensure we have a valid result
    if best_c is None:
        best_c = c_init_1
    else:
        # Normalize to ensure valid range
        max_abs = np.max(np.abs(best_c))
        if max_abs > 1e-3 and max_abs < 1e5:
            best_c = best_c / max_abs
        else:
            norm = np.linalg.norm(best_c)
            if norm > 0:
                best_c = best_c / norm
            else:
                best_c = np.ones(2 * k + 1)
                
    return best_c
