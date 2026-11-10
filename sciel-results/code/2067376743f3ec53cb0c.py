import numpy as np
import scipy.optimize
import scipy.special

def rudin_problem10() -> np.ndarray:
    k = 10
    # Grid parameters
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in cos(theta)
    t, w = np.polynomial.legendre.leggauss(n_theta)
    # t is in [-1, 1], w sums to 2
    theta = np.arccos(t)
    
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Create meshgrid
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    T = np.cos(Theta)
    
    # Weights for each point
    # Integral is sum w_i * (2 pi / n_phi) * f(t_i, phi_j)
    # Total weight for point (i, j) is w_i * (2 pi / n_phi)
    W = np.tile(w[:, np.newaxis], (1, n_phi)) * (2 * np.pi / n_phi)
    W_flat = W.flatten()
    
    # Compute Spherical Harmonics Y_m for m = -10..10
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # Note: factorial can be large, use gammaln or logsumexp?
    # (10-m)!/(10+m)! for m=10 is 0! / 20! which is small.
    # Using scipy.special.gammaln is safe.
    m_vals = np.arange(-k, k + 1)
    
    # Calculate normalization constants N_m
    # N_m = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
    # For real harmonics, the normalization is slightly different in the definition given in the prompt.
    # Prompt: N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    # Y_0 = N_0 P^0
    # Y_m = sqrt(2) N_m P^m cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi) for m < 0
    
    # Let's compute P_l^m for all m
    # scipy.special.lpmv(m, l, x) gives P_l^m(x) with Condon-Shortley phase.
    
    A = np.zeros((n_theta * n_phi, 2 * k + 1), dtype=np.float64)
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        # Compute N_m
        # log N_m = 0.5 * (log(21) - log(4*pi) + gammaln(11-abs_m) - gammaln(11+abs_m))
        log_N = 0.5 * (np.log(21) - np.log(4 * np.pi) + scipy.special.gammaln(11 - abs_m) - scipy.special.gammaln(11 + abs_m))
        N_m = np.exp(log_N)
        
        # Compute P_10^abs_m(cos theta)
        # T is cos(theta)
        P_lm = scipy.special.lpmv(abs_m, k, T)
        
        if m == 0:
            Y_m = N_m * P_lm
        elif m > 0:
            Y_m = np.sqrt(2) * N_m * P_lm * np.cos(m * Phi)
        else:
            Y_m = np.sqrt(2) * N_m * P_lm * np.sin(abs_m * Phi)
            
        A[:, idx] = Y_m.flatten()
        
    # Now we have A. We want to minimize J(c) = sum W_flat * (A c)^4
    # Subject to c^T c = 1
    
    def objective(c):
        y = A.dot(c)
        return np.sum(W_flat * y**4)
        
    def constraint_eq(c):
        return np.sum(c**2) - 1.0
        
    # We will run several optimizations from random starting points
    best_c = None
    best_val = np.inf
    
    # Random seed for reproducibility in the function? 
    # The task doesn't specify, but deterministic is better for evaluation consistency if possible.
    # However, random restarts help escape local minima.
    # We'll use a fixed seed for the random number generator inside the function to be deterministic.
    rng = np.random.default_rng(42)
    
    n_restarts = 10
    for _ in range(n_restarts):
        # Random unit vector
        c0 = rng.normal(size=2 * k + 1)
        c0 /= np.linalg.norm(c0)
        
        # SLSQP
        try:
            res = scipy.optimize.minimize(
                objective, 
                c0, 
                method='SLSQP', 
                constraints=[{'type': 'eq', 'fun': constraint_eq}],
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success:
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_c = res.x
        except Exception:
            continue
            
    # If all failed, return the all-ones vector (normalized)
    if best_c is None:
        best_c = np.ones(2 * k + 1)
        best_c /= np.linalg.norm(best_c)
        
    # Ensure max|c| is within valid range
    # The problem states max|c| < 1e-3 or > 1e5 is invalid.
    # Our c is on the unit sphere, so max|c| <= 1. This is safe.
    
    return best_c
