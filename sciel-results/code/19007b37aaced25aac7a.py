import numpy as np
import time
import scipy.optimize as opt
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """
    Compute the ratio: L2^2 / (L1 * Sup(AutoConv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse random (Sidon-like)
    np.random.seed(42)
    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Gaussian
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 4. Triangular
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # --- Phase 2: Optimization using Epigraph Formulation ---
    # Variables: x = [h_0, ..., h_{N-1}, M]
    # Objective: Minimize M (x[N])
    # Constraints:
    # 1. sum(h) - 1 == 0
    # 2. h_i >= 0
    # 3. M - conv_k >= 0 for all k
    
    def objective(x):
        return x[N]
        
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
        
    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0
        
    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac
        
    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
        
    def constraint_ineq_jac(x):
        h = x[:N]
        num_conv = 2 * N - 1
        jac = np.zeros((num_conv, N + 1))
        # d(M - conv_k)/dM = 1
        jac[:, N] = 1.0
        # d(M - conv_k)/dh_j = -2 * h_{k-j}
        # Efficiently compute the Jacobian block for h
        G = np.zeros((2 * N - 1, N))
        for j in range(N):
            k_start = j
            k_end = min(j + N, 2 * N - 1)
            if k_start < 2 * N - 1 and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                G[indices_k, j] = 2.0 * h[indices_h]
        jac[:, :N] = -G
        return jac
        
    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]
    
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        if np.sum(h_init) == 0:
            continue
            
        h_init = h_init / np.sum(h_init)
        
        # Initial M guess: max convolution
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        
        x0 = np.concatenate((h_init, [M_init]))
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                # Ensure non-negativity and normalization
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # --- Phase 3: Final Polish ---
    # Use the best solution found so far to run a high-precision optimization
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        conv_start = np.convolve(h_start, h_start, mode='full')
        M_start = np.max(conv_start) + 1e-8
        x0 = np.concatenate((h_start, [M_start]))
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': 2000, 'ftol': 1e-14}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
