"""Combined program for the second autocorrelation inequality task."""
import time
import numpy as np
import scipy.optimize as opt

N = 50

# Precomputed 7-element Sidon set within {0, ..., 49}
SIDON_SET = [0, 1, 4, 10, 18, 30, 46]

def _score_from_heights(h):
    """Compute the raw score: (sum h^2) / (L1 * sup |autoconv|)"""
    h = np.asarray(h, dtype=np.float64)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    sum_sq = np.sum(h ** 2)
    # Autoconvolution (full)
    ac = np.convolve(h, h, mode='full')
    sup_ac = np.max(np.abs(ac))
    if sup_ac <= 0:
        return 0.0
    return sum_sq / (l1 * sup_ac)

def _autoconv(h):
    """Compute full autoconvolution."""
    return np.convolve(h, h, mode='full')

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    # Objective: minimize M subject to M >= (h*h)_k for all k, sum(h)=1, h>=0
    # Ratio = sum(h^2) / (sum(h) * M) = sum(h^2) / M when sum(h)=1
    
    def neg_ratio_obj(x):
        # x = [h_0, ..., h_{N-1}, M]
        # We want to maximize ratio = sum(h^2)/M, so minimize -ratio
        # But the formulation in inspiration 1 minimizes M directly
        # Let's stick with minimizing M for simplicity and robustness
        return x[N]
    
    def neg_ratio_obj_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
    
    def constraint_sum(x):
        return np.sum(x[:N]) - 1.0
    
    def constraint_sum_grad(x):
        grad = np.zeros(N + 1)
        grad[:N] = 1.0
        return grad
    
    def constraints_autoconv(x):
        h = x[:N]
        M = x[N]
        c = _autoconv(h)
        return M - c
    
    def constraints_autoconv_grad(x):
        h = x[:N]
        grad = np.zeros((2*N - 1, N + 1))
        for k in range(2*N - 1):
            derivs = np.zeros(N)
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    derivs[i] = 2.0 * h[j]
            grad[k, :N] = -derivs
            grad[k, N] = 1.0
        return grad
    
    bounds = [(0.0, None)] * N + [(0.0, None)]
    
    cons = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
        {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
    ]
    
    def calculate_ratio(h):
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < -1e-12):
            return 0.0
        h = np.maximum(h, 0.0)
        l1 = np.sum(h)
        if l1 <= 0:
            return 0.0
        l2_sq = np.sum(h**2)
        c = _autoconv(h)
        peak = np.max(c)
        if peak <= 0:
            return 0.0
        return l2_sq / (l1 * peak)
    
    best_h = np.array([1.0 / 7.0 if i in SIDON_SET else 0.0 for i in range(N)])
    best_score = calculate_ratio(best_h)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Sidon set with equal weights
    inits.append(best_h.copy())
    
    # 2. Sidon set with skewed weights
    indices = SIDON_SET
    n_idx = len(indices)
    for base in [0.5, 0.7, 0.9, 1.2, 1.5]:
        raw_weights = [base ** i for i in range(n_idx)]
        total = sum(raw_weights)
        h_s = np.zeros(N)
        for idx, w in zip(indices, raw_weights):
            h_s[idx] = w / total
        inits.append(h_s)
        # Reverse
        raw_weights_rev = [base ** (n_idx - 1 - i) for i in range(n_idx)]
        total_rev = sum(raw_weights_rev)
        h_r = np.zeros(N)
        for idx, w in zip(indices, raw_weights_rev):
            h_r[idx] = w / total_rev
        inits.append(h_r)
    
    # 3. Uniform
    inits.append(np.ones(N) / N)
    
    # 4. Random sparse sets
    np.random.seed(42)
    for _ in range(30):
        k = np.random.randint(3, 12)
        idx = np.random.choice(N, size=k, replace=False)
        h_rand = np.zeros(N)
        h_rand[idx] = 1.0
        h_rand /= np.sum(h_rand)
        inits.append(h_rand)
    
    # 5. Gaussian-like bumps
    x_idx = np.arange(N)
    for c in np.linspace(2, N-3, 5):
        for w in np.linspace(1, N/3, 3):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
    
    # Run SLSQP on each initialization
    for i, h_init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        c_init = _autoconv(h_init)
        M_init = np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])
        
        try:
            res = opt.minimize(
                neg_ratio_obj, 
                x0, 
                method='SLSQP', 
                jac=neg_ratio_obj_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    continue
                h_res = np.clip(h_res, 0, None)
                h_res = h_res / np.sum(h_res)
                
                ratio_res = calculate_ratio(h_res)
                if ratio_res > best_score:
                    best_score = ratio_res
                    best_h = h_res.copy()
                    
        except Exception:
            continue
    
    # If time remains, do local refinement on the best solution found
    if time.time() < deadline - 5.0:
        c_best = _autoconv(best_h)
        M_best = np.max(c_best)
        x_best = np.concatenate([best_h, [M_best]])
        
        try:
            res = opt.minimize(
                neg_ratio_obj, 
                x_best, 
                method='SLSQP', 
                jac=neg_ratio_obj_grad,
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if res.success:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    h_res = best_h
                h_res = np.clip(h_res, 0, None)
                h_res = h_res / np.sum(h_res)
                
                ratio_res = calculate_ratio(h_res)
                if ratio_res > best_score:
                    best_score = ratio_res
                    best_h = h_res.copy()
        except Exception:
            pass
    
    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
# EVOLVE-BLOCK-END
