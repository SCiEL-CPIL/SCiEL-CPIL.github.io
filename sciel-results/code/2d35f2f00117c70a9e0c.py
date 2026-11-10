import numpy as np
import time
from scipy.special import eval_legendre, gammaln
from scipy.optimize import minimize

def _precompute():
    """Precompute quadrature nodes, weights, and spherical harmonic basis."""
    k = 10
    
    # Gauss-Legendre quadrature for cos(theta) in [-1, 1]
    n_theta = 70
    x, w = np.polynomial.legendre.leggauss(n_theta)
    
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Precompute P_10^m(cos(theta)) for m = 0..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = eval_legendre(10, m, x)
    
    # Normalization constants
    N = np.zeros(11)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4 * np.pi)) + gammaln(11 - m) - gammaln(11 + m))
        N[m] = np.exp(log_N)
    
    # Build basis matrix B of shape (n_theta * n_phi, 21)
    total_points = n_theta * n_phi
    B = np.zeros((total_points, 21))
    
    idx = 0
    for i in range(n_theta):
        for j in range(n_phi):
            for m_idx in range(21):
                m = m_idx - 10
                if m == 0:
                    B[idx, m_idx] = N[0] * P[0, i]
                elif m > 0:
                    B[idx, m_idx] = np.sqrt(2) * N[m] * P[m, i] * np.cos(m * phi[j])
                else:
                    am = abs(m)
                    B[idx, m_idx] = np.sqrt(2) * N[am] * P[am, i] * np.sin(am * phi[j])
            idx += 1
    
    # Weights vector
    W = np.zeros(total_points)
    idx = 0
    for i in range(n_theta):
        for j in range(n_phi):
            W[idx] = w[i] * dphi
            idx += 1
    
    return B, W

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    B, W = _precompute()
    dim = 21
    n_points = B.shape[0]
    
    def objective(c):
        f = B @ c
        f2 = f * f
        return np.sum(W * f2 * f2)
    
    def gradient(c):
        f = B @ c
        f2 = f * f
        return 4.0 * (B.T @ (W * f * f2))
    
    # Projected Gradient Descent on the sphere
    def pgd_optimize(c0, max_iter=1000, tol=1e-10):
        c = c0.copy()
        c = c / np.linalg.norm(c)
        prev_obj = objective(c)
        step_size = 0.01
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
            grad = gradient(c)
            # Normalize gradient to stabilize step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            # Update
            c_new = c - step_size * grad
            # Project to sphere
            c_new_norm = np.linalg.norm(c_new)
            if c_new_norm == 0:
                break
            c_new = c_new / c_new_norm
            
            obj_new = objective(c_new)
            if obj_new < prev_obj:
                c = c_new
                prev_obj = obj_new
                # Adaptive step size: increase if improving
                if i % 10 == 0 and i > 0:
                    step_size *= 1.05
            else:
                # Decrease step size if not improving
                step_size *= 0.8
                if step_size < 1e-6:
                    break
            
            if abs(prev_obj - obj_new) < tol:
                break
                
        return c, prev_obj

    # Seeds
    seeds = []
    # Pure m=0
    c0 = np.zeros(dim); c0[10] = 1.0; seeds.append(c0)
    # Pure m=10
    c1 = np.zeros(dim); c1[20] = 1.0; seeds.append(c1)
    # Pure m=-10
    c2 = np.zeros(dim); c2[0] = 1.0; seeds.append(c2)
    # Mix m=10, -10 (cosine)
    c3 = np.zeros(dim); c3[20] = 1/np.sqrt(2); c3[0] = 1/np.sqrt(2); seeds.append(c3)
    # Mix m=10, -10 (sine-like phase shift? No, just linear combo)
    # Random seeds
    np.random.seed(42)
    for _ in range(20):
        cr = np.random.randn(dim)
        cr /= np.linalg.norm(cr)
        seeds.append(cr)
        
    best_c = None
    best_val = np.inf
    
    for s in seeds:
        if time.time() > deadline - 10:
            break
        c_opt, val = pgd_optimize(s)
        if val < best_val:
            best_val = val
            best_c = c_opt.copy()
            
    # Refine best candidate with SLSQP
    if best_c is not None and time.time() < deadline - 20:
        constraint = {'type': 'eq', 'fun': lambda c: np.sum(c*c) - 1.0, 'jac': lambda c: 2.0 * c}
        try:
            res = minimize(objective, best_c, method='SLSQP', jac=gradient, 
                           constraints=[constraint], options={'maxiter': 200, 'ftol': 1e-14})
            if res.success:
                c_refined = res.x
                c_refined /= np.linalg.norm(c_refined)
                val_refined = objective(c_refined)
                if val_refined < best_val:
                    best_c = c_refined
                    best_val = val_refined
        except Exception:
            pass
            
    # Final validity check
    if best_c is None:
        best_c = np.ones(dim) / np.sqrt(dim)
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c *= (1e-3 / max_abs)
    elif max_abs > 1e5:
        best_c *= (1e5 / max_abs)
        
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(dim) / np.sqrt(dim)
        
    return best_c.astype(np.float64)
