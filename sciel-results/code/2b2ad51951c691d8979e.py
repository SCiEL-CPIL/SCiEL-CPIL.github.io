import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on the specified quadrature grid.
    2. Define objective R(c) = ||f||_4 / ||f||_2.
    3. Use Differential Evolution for global search to find promising basins.
    4. Polish the best candidate with L-BFGS-B using analytic gradient.
    5. Return the best valid coefficient vector.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin (630s limit, 50s buffer)

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in mu = cos(theta)
    mu, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    # Flatten grid
    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)
    
    # m = 0
    m = 0
    idx = 10 
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    # m > 0
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
        
    # 3. Objective and Gradient
    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Analytic gradient of R with respect to u
        sum_wf_A = A.T @ (w_flat * f_vals) * d_phi
        sum_wf3_A = A.T @ (w_flat * f_vals**3) * d_phi
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        # Chain rule for c -> u = c/||c||
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # 4. Global Search: Differential Evolution
    # Bounds: [-1, 1] is sufficient due to scale invariance
    bounds = [(-1.0, 1.0)] * n_m
    
    # Objective function for DE (no gradient)
    def de_objective(c):
        val, _ = objective_and_grad(c)
        return val

    best_c = None
    best_val = np.inf
    
    # Run DE
    # maxiter 100, popsize 20 -> ~2000 evaluations.
    # Each eval is ~21*9100 ops, very fast.
    # DE is stochastic, so we might run it multiple times if time permits.
    try:
        # DE is robust to scale, but normalizing inside objective helps.
        # We set seed for reproducibility if needed, but default is fine.
        de_res = differential_evolution(
            de_objective, 
            bounds, 
            maxiter=150, 
            popsize=30, 
            tol=1e-8,
            polish=False, # We will polish manually for better control
            update_interval=1,
            seed=42
        )
        
        if de_res.success or de_res.fun < best_val:
            best_val = de_res.fun
            best_c = de_res.x.copy()
            
    except Exception:
        pass

    # If time permits, run a second DE with different seed or parameters
    if time.time() < deadline - 60:
        try:
            de_res2 = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=100, 
                popsize=50, 
                tol=1e-8,
                polish=False,
                seed=123
            )
            if de_res2.fun < best_val:
                best_val = de_res2.fun
                best_c = de_res2.x.copy()
        except Exception:
            pass

    # 5. Polish with L-BFGS-B
    # If DE found a candidate, polish it
    if best_c is not None:
        try:
            res = minimize(
                objective_and_grad, 
                best_c, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Fallback if DE failed or found nothing
    if best_c is None or not np.all(np.isfinite(best_c)):
        # Try a few random restarts with L-BFGS-B as fallback
        rng = np.random.default_rng(42)
        for _ in range(5):
            c_rand = rng.standard_normal(n_m)
            c_rand /= np.linalg.norm(c_rand)
            try:
                res = minimize(
                    objective_and_grad, 
                    c_rand, 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x.copy()
            except Exception:
                pass
        
        if best_c is None:
            best_c = np.zeros(n_m)
            best_c[10] = 1.0

    # 6. Final Validity Check and Scaling
    if not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c

# EVOLVE-BLOCK-END
