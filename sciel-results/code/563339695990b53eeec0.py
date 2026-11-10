import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on the specified quadrature grid.
    2. Define objective R(c) = ||f||_4 / ||f||_2 and its analytic gradient.
    3. Generate a diverse set of initial candidates based on symmetry:
       - Pure m=0
       - Pure m=k for k=1..10
       - Combinations of m and -m
       - Random normalized vectors
    4. Polish each candidate with L-BFGS-B using the analytic gradient.
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
        # R = (sum w f^4)^0.25 / (sum w f^2)^0.5
        # dR/du_i = 0.25 (sum w f^4)^(-0.75) * sum w 4 f^3 A_i - (sum w f^4)^0.25 * 0.5 (sum w f^2)^(-1.5) * sum w 2 f A_i
        # Let S4 = sum w f^4, S2 = sum w f^2
        # dR/du_i = S4^(-0.75) * sum w f^3 A_i - S4^0.25 * S2^(-1.5) * sum w f A_i
        
        sum_wf_A = A.T @ (w_flat * f_vals) * d_phi
        sum_wf3_A = A.T @ (w_flat * f_vals**3) * d_phi
        
        grad_u = (norm4_4 ** -0.75) * sum_wf3_A - (N4 * norm2_sq ** -1.5) * sum_wf_A
        
        # Chain rule for c -> u = c/||c||
        # u = c / ||c||
        # du_i/dc_j = (delta_ij ||c|| - c_i c_j / ||c||) / ||c||^2 = (delta_ij - u_i u_j) / ||c||
        # grad_c_j = sum_i grad_u_i * du_i/dc_j
        # grad_c_j = (1/||c||) * (grad_u_j - u_j sum_i grad_u_i u_i)
        
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # 4. Generate Initial Candidates
    candidates = []
    
    # 1. Pure m=0
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # 2. Pure m=k (cos and sin)
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        
        c_pos = np.zeros(n_m)
        c_pos[idx_pos] = 1.0
        candidates.append(c_pos)
        
        c_neg = np.zeros(n_m)
        c_neg[idx_neg] = 1.0
        candidates.append(c_neg)
        
        # 3. Combinations of m and -m (e.g., cos + sin, cos - sin)
        c_combo1 = np.zeros(n_m)
        c_combo1[idx_pos] = 1.0 / np.sqrt(2)
        c_combo1[idx_neg] = 1.0 / np.sqrt(2)
        candidates.append(c_combo1)
        
        c_combo2 = np.zeros(n_m)
        c_combo2[idx_pos] = 1.0 / np.sqrt(2)
        c_combo2[idx_neg] = -1.0 / np.sqrt(2)
        candidates.append(c_combo2)

    # 4. Random candidates
    rng = np.random.default_rng(42)
    for _ in range(20):
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
        
    # 5. Add some structured random candidates mixing low and high m
    for _ in range(10):
        c_struct = np.zeros(n_m)
        # Pick 2-3 random m values
        m_vals = rng.choice(np.arange(1, k+1), size=3, replace=False)
        for m_val in m_vals:
            idx_pos = m_val + 10
            idx_neg = -m_val + 10
            # Random phase
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 1.0)
            c_struct[idx_pos] += amp * np.cos(phase)
            c_struct[idx_neg] += amp * np.sin(phase)
        
        # Maybe add m=0
        if rng.random() > 0.5:
            c_struct[10] += rng.uniform(-0.5, 0.5)
            
        norm_c = np.linalg.norm(c_struct)
        if norm_c > 1e-6:
            c_struct /= norm_c
            candidates.append(c_struct)

    # 5. Polish Candidates
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
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Fallback if all failed
    if best_c is None or not np.all(np.isfinite(best_c)):
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
