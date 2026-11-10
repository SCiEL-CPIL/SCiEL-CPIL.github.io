# EVOLVE-BLOCK-START
import numpy as np
import time

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m (m=-10..10) for a degree-10 spherical harmonic
    that minimizes ||f||_4 / ||f||_2 (flattest |f| on the sphere).
    """
    t_start = time.time()
    DEADLINE = 600.0  # evaluator kills at 630s, leave 30s margin
    
    k = 10
    n_modes = 2 * k + 1  # 21
    
    # Build the quadrature grid: 70 Gauss-Legendre in cos(theta) x 130 equally spaced phi
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_theta)
    # x_gl = cos(theta), so dOmega = dphi * d(cos(theta))
    # phi is equally spaced
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    
    # 2D grid
    cos_theta = np.tile(x_gl, (n_phi, 1))  # shape (n_phi, n_theta)
    phi_grid = np.tile(phi[:, None], (1, n_theta))  # shape (n_phi, n_theta)
    
    # Weights: w_i = w_gl[j] * (2*pi/n_phi) for each (i,j)
    # Actually: dOmega = dphi * d(cos_theta), and total = 2*pi * 2 = 4*pi
    # phi spacing: dphi = 2*pi/n_phi
    dphi = 2.0 * np.pi / n_phi
    w_theta = w_gl  # these integrate to 2 over [-1,1]
    # Total weight per point: dphi * w_gl[j]
    # Sum over all points: sum_j w_gl[j] * sum_i dphi = 2 * 2*pi = 4*pi. Correct.
    
    # Build evaluation matrix A: A[i,j] = Y_j(x_i)
    # where i indexes the grid points, j indexes the modes m=-10..10
    
    # Compute associated Legendre polynomials P_10^m(cos(theta)) with Condon-Shortley phase
    # We need P_l^m for l=10, m=0,1,...,10
    
    def associated_legendre(l, m, x):
        """Compute P_l^m(x) with Condon-Shortley phase, for l=10, m=0..10."""
        # Use recurrence or scipy
        from scipy.special import lpmv
        return lpmv(m, l, x)
    
    # Precompute P_10^m for m=0..10 at all cos_theta values
    # P_m[mm, j] = P_10^mm(cos_theta[j])
    P = np.zeros((n_theta + 1, n_theta))  # P[mm, j] for mm=0..10
    for mm in range(k + 1):
        P[mm, :] = associated_legendre(k, mm, x_gl)
    
    # Normalization: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # 21 = (2*10+1)/2
    from math import factorial, pi
    
    def norm_factor(m):
        # N_m = sqrt((2*10+1)/(4*pi) * (10-|m|)!/(10+|m|)!)
        mm = abs(m)
        return np.sqrt((2 * k + 1) / (4.0 * pi) * factorial(k - mm) / factorial(k + mm))
    
    # Build A matrix: shape (n_points, n_modes)
    # Grid point index: i = iphi * n_theta + itheta
    n_points = n_phi * n_theta
    A = np.zeros((n_points, n_modes), dtype=np.float64)
    
    for iphi in range(n_phi):
        for itheta in range(n_theta):
            idx = iphi * n_theta + itheta
            ct = x_gl[itheta]
            ph = phi[iphi]
            for mm in range(k + 1):
                N_mm = norm_factor(mm)
                if mm == 0:
                    # Y_0 = N_0 P^0
                    A[idx, k] = N_mm * P[mm, itheta]  # m=0 is index k=10
                else:
                    # Y_+m = sqrt(2) N_m P^m cos(m*phi)  -> index k+m
                    # Y_-m = sqrt(2) N_m P^m sin(m*phi)  -> index k-m
                    A[idx, k + mm] = np.sqrt(2.0) * N_mm * P[mm, itheta] * np.cos(mm * ph)
                    A[idx, k - mm] = np.sqrt(2.0) * N_mm * P[mm, itheta] * np.sin(mm * ph)
    
    # Quadrature weights for each grid point
    # w[idx] = dphi * w_gl[itheta]
    w = np.zeros(n_points)
    for iphi in range(n_phi):
        for itheta in range(n_theta):
            idx = iphi * n_theta + itheta
            w[idx] = dphi * w_gl[itheta]
    
    # Now optimize: minimize ||f||_4 subject to ||v|| = 1
    # f = A @ v
    # ||f||_4^4 = sum_i w_i |f_i|^4
    # grad = 4 * sum_i w_i f_i^3 A[i,:] / 4 = sum_i w_i f_i^3 A[i,:]
    # (for the objective (1/4)||f||_4^4)
    
    def compute_f(v):
        return A @ v
    
    def objective(v):
        f = A @ v
        return np.sum(w * np.abs(f) ** 4)
    
    def gradient(v):
        f = A @ v
        # d/dv of sum_i w_i f_i^4 = sum_i 4 w_i f_i^3 A[i,:]
        # For (1/4) sum_i w_i f_i^4, grad = sum_i w_i f_i^3 A[i,:]
        return (w * f ** 3) @ A  # shape (n_modes,)
    
    # Sphere-constrained gradient descent with line search
    def sphere_descent(v_init, max_iter=5000, tol=1e-12, time_limit=None):
        v = v_init.copy()
        v = v / np.linalg.norm(v)
        
        best_v = v.copy()
        best_obj = objective(v)
        
        eta = 0.01  # initial step size
        t_local = time.time()
        
        for it in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
            
            f = A @ v
            g = (w * f ** 3) @ A  # gradient of (1/4)||f||_4^4
            
            # Project to tangent space
            g_tan = g - np.dot(g, v) * v
            g_norm = np.linalg.norm(g_tan)
            
            if g_norm < tol:
                break
            
            # Backtracking line search
            best_step_obj = objective(v)
            best_step_v = v.copy()
            
            for ls in range(30):
                v_try = v - eta * g_tan
                v_try = v_try / np.linalg.norm(v_try)
                obj_try = objective(v_try)
                
                if obj_try < best_obj - 1e-15:
                    best_step_obj = obj_try
                    best_step_v = v_try.copy()
                    break
                eta *= 0.5
            
            if np.allclose(best_step_v, v):
                break
            
            v = best_step_v
            obj_v = objective(v)
            
            if obj_v < best_obj:
                best_obj = obj_v
                best_v = v.copy()
            
            # Adaptive step size
            if obj_v < best_obj + 1e-10:
                eta = min(eta * 1.5, 1.0)
            else:
                eta = max(eta * 0.5, 1e-6)
            
            if it % 100 == 0:
                elapsed = time.time() - t_local
                if elapsed > 50:  # safety per iteration batch
                    break
        
        return best_v, best_obj
    
    # Generate diverse initial conditions
    # 1. All ones
    # 2. Random
    # 3. Symmetry-based: only even m, only odd m, etc.
    # 4. Single mode
    # 5. Pairs of modes
    
    np.random.seed(42)
    
    candidates = []
    
    # All ones
    candidates.append(np.ones(n_modes))
    
    # Random initializations
    for _ in range(20):
        candidates.append(np.random.randn(n_modes))
    
    # Only m=0
    v0 = np.zeros(n_modes)
    v0[k] = 1.0
    candidates.append(v0)
    
    # Only |m|=10
    v0 = np.zeros(n_modes)
    v0[0] = 1.0  # m=-10
    v0[-1] = 1.0  # m=+10
    candidates.append(v0)
    
    # Pairs: m and -m
    for mm in range(1, k + 1):
        v0 = np.zeros(n_modes)
        v0[k + mm] = 1.0
        v0[k - mm] = 1.0
        candidates.append(v0)
    
    # Alternating signs
    v0 = np.ones(n_modes)
    v0[::2] = -1.0
    candidates.append(v0)
    
    # More structured: cos(m*phi) type patterns
    for mm in range(1, k + 1):
        v0 = np.zeros(n_modes)
        v0[k + mm] = 1.0
        candidates.append(v0)
        v0 = np.zeros(n_modes)
        v0[k - mm] = 1.0
        candidates.append(v0)
    
    # Run optimization on each candidate
    best_v = None
    best_score = np.inf
    
    time_limit = t_start + DEADLINE - 30  # leave 30s buffer
    
    for ci, v_init in enumerate(candidates):
        if time.time() > time_limit:
            break
        
        v_opt, obj_val = sphere_descent(v_init, max_iter=3000, time_limit=time_limit)
        
        # Compute the actual R = ||f||_4 / ||f||_2
        f = A @ v_opt
        l2 = np.sqrt(np.sum(w * f ** 2))
        l4 = (np.sum(w * np.abs(f) ** 4)) ** 0.25
        R = l4 / l2
        
        if R < best_score:
            best_score = R
            best_v = v_opt.copy()
    
    # Final polish: more iterations on the best
    if best_v is not None and time.time() < time_limit - 5:
        best_v, _ = sphere_descent(best_v, max_iter=10000, tol=1e-14, time_limit=time_limit)
    
    if best_v is None:
        best_v = np.ones(n_modes)
    
    # Normalize to avoid issues (the ratio is scale-invariant, but let's keep it reasonable)
    v_norm = np.linalg.norm(best_v)
    if v_norm > 0:
        best_v = best_v / v_norm
    
    # Validate
    if not np.all(np.isfinite(best_v)):
        best_v = np.ones(n_modes)
    
    max_abs = np.max(np.abs(best_v))
    if max_abs < 1e-3:
        best_v = best_v / max_abs * 0.1
    elif max_abs > 1e5:
        best_v = best_v / max_abs * 1.0
    
    return best_v.astype(np.float64)

# EVOLVE-BLOCK-END
