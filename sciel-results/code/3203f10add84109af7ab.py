import numpy as np
import time
import scipy.optimize as opt
from scipy.signal import fftconvolve

N = 50
DELTA = 0.5 / N

def _score_raw(h):
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
    deadline = start_time + 110.0  # Safety margin for 150s kill
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
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
        
    def ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
        
    def ineq_jac(x):
        h = x[:N]
        num_conv = 2 * N - 1
        jac = np.zeros((num_conv, N + 1))
        # d(M - conv_k)/dM = 1
        jac[:, N] = 1.0
        # d(M - conv_k)/dh_j = -2 * h_{k-j}
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
        
    constraint_eq = {
        'type': 'eq',
        'fun': constraint_eq_fun,
        'jac': constraint_eq_jac
    }
    
    constraint_ineq = {
        'type': 'ineq',
        'fun': ineq_fun,
        'jac': ineq_jac
    }
    
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    # Initializations
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
            h_g = np.exp(-0.5 * ((np.arange(N) - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 4. Triangular
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(np.arange(N) - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)

    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        x0 = np.concatenate((h_init, [1.0])) # Initial M guess
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[constraint_eq, constraint_ineq],
                jac=objective_grad,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                # Ensure non-negativity and normalization (SLSQP might drift slightly)
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception as e:
            pass
            
    # Block-coordinate descent to refine the best solution
    if time.time() < deadline - 5:
        h = best_h.copy()
        # Identify the dominant autoconvolution peak
        conv = np.convolve(h, h, mode='full')
        peak_idx = np.argmax(conv)
        
        # Perform coordinate descent on pairs of indices that contribute to the peak
        # to reduce the peak while maintaining L2 norm
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            conv = np.convolve(h, h, mode='full')
            peak_idx = np.argmax(conv)
            peak_val = conv[peak_idx]
            
            # Find indices that contribute most to the peak
            # conv_k = sum_i h_i h_{k-i}
            # For fixed k, the terms are h_i h_{k-i}
            # We want to reduce h_i h_{k-i} for large i, k-i
            
            # Try reducing the largest h_i that is paired with another large h_j where i+j=k
            # For simplicity, try reducing h[peak_idx // 2] and h[peak_idx - peak_idx // 2] if they exist
            
            i1 = peak_idx // 2
            i2 = peak_idx - i1
            
            if i1 < N and i2 < N and i1 != i2:
                # Try reducing h[i1] slightly and increasing h[i2] slightly to maintain sum
                # This is a heuristic; in practice, we should solve a small subproblem
                
                # For now, just try a small perturbation
                for eps in [0.01, 0.005, 0.001]:
                    h_test = h.copy()
                    h_test[i1] -= eps * h[i1]
                    h_test[i2] += eps * h[i1]
                    h_test = np.clip(h_test, 0, None)
                    s = np.sum(h_test)
                    if s > 0:
                        h_test /= s
                        s_test = _score_raw(h_test)
                        if s_test > best_score:
                            best_score = s_test
                            best_h = h_test
                            h = h_test
                            break
                            
    return list(best_h)
