# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv

def rudin_problem10() -> np.ndarray:
    k = 10
    n_mu = 70
    n_phi = 130
    
    # Gauss-Legendre points and weights for [-1, 1]
    mu, w_mu = np.polynomial.legendre.leggauss(n_mu)
    
    # Equally spaced phi points
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = 2 * np.pi / n_phi
    
    # Create meshgrid
    MU, PHI = np.meshgrid(mu, phi, indexing='ij')
    MU_flat = MU.flatten()
    PHI_flat = PHI.flatten()
    
    # Weights
    W = np.outer(w_mu, w_phi).flatten()
    
    # Precompute basis functions Y_m at quadrature points
    n_points = n_mu * n_phi
    Y = np.zeros((n_points, 2 * k + 1))
    
    # Normalization constants N_m
    N = np.zeros(2 * k + 1)
    for m in range(-k, k + 1):
        abs_m = abs(m)
        num = np.math.factorial(10 - abs_m)
        den = np.math.factorial(10 + abs_m)
        N[m + k] = np.sqrt((21.0 / (4.0 * np.pi)) * (num / den))
        
    # Evaluate Y
    for idx, m in enumerate(range(-k, k + 1)):
        if m == 0:
            P = lpmv(0, 10, MU_flat)
            Y[:, idx] = N[idx] * P
        elif m > 0:
            P = lpmv(m, 10, MU_flat)
            Y[:, idx] = np.sqrt(2) * N[idx] * P * np.cos(m * PHI_flat)
        else:
            abs_m = -m
            P = lpmv(abs_m, 10, MU_flat)
            Y[:, idx] = np.sqrt(2) * N[abs_m + k] * P * np.sin(abs_m * PHI_flat)

    def compute_f(c):
        return Y.T @ c

    def compute_norm4(c):
        f = compute_f(c)
        return np.sum(W * (f ** 4)) ** 0.25

    def compute_v(c):
        f = compute_f(c)
        f3 = f ** 3
        return Y.T @ (W * f3)

    def objective_and_grad(c):
        f = compute_f(c)
        norm4_sq = np.sum(W * (f ** 4))
        norm4 = norm4_sq ** 0.25
        v = compute_v(c)
        grad = v / (norm4 ** 3)
        return norm4, grad

    best_c = None
    best_val = np.inf
    
    # Initial guesses
    guesses = []
    guesses.append(np.ones(2 * k + 1))
    
    c_zonal = np.zeros(2 * k + 1)
    c_zonal[k] = 1.0
    guesses.append(c_zonal)
    
    rng = np.random.default_rng(42)
    for _ in range(20):
        c_rand = rng.normal(size=2 * k + 1)
        guesses.append(c_rand)
        
    for c0 in guesses:
        c = c0 / np.linalg.norm(c0)
        current_val = compute_norm4(c)
        
        for i in range(500):
            val, grad = objective_and_grad(c)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Backtracking line search
            alpha = 0.1
            c_new = c - alpha * grad
            c_new = c_new / np.linalg.norm(c_new)
            val_new = compute_norm4(c_new)
            
            # If it didn't improve, reduce step size
            if val_new > val:
                # Simple backtracking
                alpha = 0.01
                c_new = c - alpha * grad
                c_new = c_new / np.linalg.norm(c_new)
                val_new = compute_norm4(c_new)
                if val_new > val:
                    alpha = 0.001
                    c_new = c - alpha * grad
                    c_new = c_new / np.linalg.norm(c_new)
                    val_new = compute_norm4(c_new)
                    if val_new > val:
                        # Give up on this step
                        break

            c = c_new
            val = val_new
            
        if val < best_val:
            best_val = val
            best_c = c
            
    return best_c
# EVOLVE-BLOCK-END
