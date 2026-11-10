import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on the specified quadrature grid.
    2. Define objective R(c) = ||f||_4 / ||f||_2 and its analytic gradient 
       with respect to normalized coefficients.
    3. Instead of global stochastic search (DE), use a structured set of 
       initial guesses based on symmetry classes:
       - Axisymmetric (m=0 dominant)
       - Single m modes (m=1..10)
       - Random but normalized
    4. Polish each candidate with L-BFGS-B using the analytic gradient.
    5. Return the best valid coefficient vector.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

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
        # R = ||f||_4 / ||f||_2
        # dR/du = (1/||f||_2) d||f||_4/du - (||f||_4 / ||f||_2^2) d||f||_2/du
        # d||f||_4/du = ||f||_4^3 * A^T (w f^3) d_phi
        # d||f||_2/du = (1/||f||_2) A^T (w f) d_phi
        
        sum_wf_A = A.T @ (w_flat * f_vals) * d_phi
        sum_wf3_A = A.T @ (w_flat * f_vals**3) * d_phi
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        # Chain rule for c -> u = c/||c||
        # grad_c R = (I - uu^T) @ grad_u / ||c||
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # 4. Generate Initial Candidates
    candidates = []
    
    # Candidate 1: Pure m=0 (Axisymmetric)
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # Candidates 2-11: Pure m>0 modes (cos and sin parts)
    for m in range(1, 11):
        c_cos = np.zeros(n_m)
        c_cos[m + 10] = 1.0
        candidates.append(c_cos)
        
        c_sin = np.zeros(n_m)
        c_sin[-m + 10] = 1.0
        candidates.append(c_sin)
        
    # Candidate 12: Sum of all m=0 and m=10 (often extremal)
    c_mix = np.zeros(n_m)
    c_mix[10] = 1.0
    c_mix[20] = 1.0
    c_mix /= np.linalg.norm(c_mix)
    candidates.append(c_mix)
    
    # Candidates 13-22: Random normalized vectors
    rng = np.random.default_rng(42)
    for _ in range(10):
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)

    # 5. Optimize Each Candidate
    best_c = None
    best_val = np.inf
    
    for i, c_init in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = res.fun
            if val < best_val:
                best_val = val
                best_c = res.x.copy()
        except Exception as e:
            # If optimization fails, check the initial point itself
            val, _ = objective_and_grad(c_init)
            if val < best_val:
                best_val = val
                best_c = c_init.copy()

    # 6. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        # Fallback to m=0
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
