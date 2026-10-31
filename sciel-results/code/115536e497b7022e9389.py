import numpy as np
import time
from scipy.optimize import minimize, least_squares
from scipy.special import factorial

# Precompute Hermite polynomial coefficients
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 56700x^4 - 113400x^2 + 10395
# Coefficients for powers x^0, x^2, x^4, x^6, x^8, x^10, x^12
H_coeffs = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [3, -6, 1, 0, 0, 0, 0],
    [105, -420, 210, -28, 1, 0, 0],
    [10395, -113400, 56700, -12600, 1260, -60, 1]
], dtype=float)

# Precompute J_m = integral of x^(2m) * exp(-x^2) dx from -inf to inf
# J_m = (2m)! / (4^m * m!) * sqrt(pi)
J = np.zeros(15)
for m in range(15):
    J[m] = (factorial(2*m) / (4**m * factorial(m))) * np.sqrt(np.pi)

# Precompute the Gram matrix for P(x)^2 integrals
# G_N[i,j] = J[i+j] for N = integral P(x)^2 exp(-x^2) dx
# G_M2[i,j] = J[i+j+1] for M2 = integral x^2 P(x)^2 exp(-x^2) dx
G_N = np.zeros((7, 7))
G_M2 = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        G_N[i, j] = J[i + j]
        G_M2[i, j] = J[i + j + 1]

# Precompute the Q Gram matrix for D = integral Q(x)^2 exp(-x^2) dx
# Q(x) = sum q_k x^(2k+1), Q(x)^2 has terms x^(2(i+j+1))
# G_D[i,j] = J[i+j+1]
G_D = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        G_D[i, j] = J[i + j + 1]

# Precompute the transformation from p to q
# q_0 = 2*p_1 - p_0
# q_k = 2*(k+1)*p_{k+1} - p_k for k=1..5
# q_6 = -p_6
T_pq = np.zeros((7, 7))
T_pq[0, 0] = -1
T_pq[0, 1] = 2
for k in range(1, 6):
    T_pq[k, k] = -1
    T_pq[k, k+1] = 2*(k+1)
T_pq[6, 6] = -1

def compute_moments_vec(c0, c4, c8):
    """Vectorized computation of N, M2, D."""
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    c = np.array([c0, c4, c8, c12])
    p = c @ H_coeffs  # shape (7,)
    q = T_pq @ p      # shape (7,)
    
    N = p @ G_N @ p
    M2 = p @ G_M2 @ p
    D = q @ G_D @ q
    return N, M2, D, p, q

def objective_and_grad(c):
    """Compute objective value and gradient."""
    c0, c4, c8 = c
    c12 = -(c0 + 3*c4 + 105*c8) / 10395.0
    c = np.array([c0, c4, c8, c12])
    p = c @ H_coeffs
    q = T_pq @ p
    
    N = p @ G_N @ p
    M2 = p @ G_M2 @ p
    D = q @ G_D @ q
    
    if N <= 1e-12:
        return 1e10, np.zeros(3)
    
    obj = (M2 / N) * (D / N)
    
    # Gradients using precomputed matrices
    # dp_k/dc_m = H_coeffs[m, k] + H_coeffs[3, k] * dc12_dc_m
    dc12_dc = np.array([-1.0, -3.0, -105.0]) / 10395.0
    dp_dc = H_coeffs[:3, :] + np.outer(dc12_dc, H_coeffs[3, :])  # shape (3, 7)
    dq_dc = dp_dc @ T_pq.T  # shape (3, 7)
    
    # grad_N[m] = 2 * dp_dc[m] @ G_N @ p
    grad_N = 2.0 * (dp_dc @ G_N @ p)
    grad_M2 = 2.0 * (dp_dc @ G_M2 @ p)
    grad_D = 2.0 * (dq_dc @ G_D @ q)
    
    # obj = M2 * D / N^2
    # d(obj)/dc = (D/N^2)*dM2 + (M2/N^2)*dD - 2*(M2*D/N^3)*dN
    grad_obj = (D / (N**2)) * grad_M2 + (M2 / (N**2)) * grad_D - 2.0 * (M2 * D / (N**3)) * grad_N
    
    return obj, grad_obj

def gradient_only(c):
    """Return just the gradient for root-finding."""
    _, grad = objective_and_grad(c)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin (210s limit)
    
    initial_guesses = [
        np.array([-1.13, 0.04, 1.0/3240.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
        np.array([0.5, -0.5, 0.1]),
        np.array([-0.5, 0.5, -0.1]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 1.0, -0.5]),
    ]
    
    best_score = float('inf')
    best_c = initial_guesses[0].copy()
    
    # Phase 1: Gradient-based optimization from multiple starts
    for guess in initial_guesses:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                objective_and_grad, 
                guess, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Root-finding on gradient equations from best points
    # This directly targets stationary points and can escape local minima
    candidate_starts = [best_c]
    for guess in initial_guesses[:5]:
        candidate_starts.append(guess)
    
    for start in candidate_starts:
        if time.time() > deadline - 15:
            break
        try:
            res_root = least_squares(
                gradient_only,
                start,
                method='lm',  # Levenberg-Marquardt, no bounds needed
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=500
            )
            if res_root.success:
                obj_val, _ = objective_and_grad(res_root.x)
                if obj_val < best_score:
                    best_score = obj_val
                    best_c = res_root.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish from the best point
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                objective_and_grad, 
                best_c, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res_polish.success and res_polish.fun < best_score:
                best_score = res_polish.fun
                best_c = res_polish.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more root-find from the polished point
    if time.time() < deadline - 2:
        try:
            res_root2 = least_squares(
                gradient_only,
                best_c,
                method='lm',
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                max_nfev=200
            )
            if res_root2.success:
                obj_val, _ = objective_and_grad(res_root2.x)
                if obj_val < best_score:
                    best_score = obj_val
                    best_c = res_root2.x.copy()
        except Exception:
            pass
    
    return best_c
