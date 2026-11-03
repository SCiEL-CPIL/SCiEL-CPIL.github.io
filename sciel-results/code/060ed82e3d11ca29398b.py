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
    
    # 6. Alternating patterns
    for period in [2, 3, 4, 5, 7]:
        h_alt = np.zeros(N)
        for i in range(N):
            if i % period == 0:
                h_alt[i] = 1.0
        if np.sum(h_alt) > 0:
            h_alt /= np.sum(h_alt)
            inits.append(h_alt)
    
    # 7. Power law decay from center
    center = N // 2
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        h_pw = np.maximum(0, 1 - np.abs(x_idx - center) / N)**alpha
        if np.sum(h_pw) > 0:
            h_pw /= np.sum(h_pw)
            inits.append(h_pw)

    # --- Phase 2: Optimization using Epigraph Formulation ---
    num_conv = 2 * N - 1
    
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
        jac = np.zeros((num_conv, N + 1))
        jac[:, N] = 1.0
        
        # Vectorized Jacobian computation
        # conv_k = sum_i h_i * h_{k-i}
        # d(conv_k)/dh_j = h_{k-j} + h_j * (1 if 0<=k-j<N and k-j!=j else 0)
        # If k-j == j, d(conv_k)/dh_j = 2*h_j
        
        # Create index arrays
        k_vals = np.arange(num_conv)  # shape (num_conv,)
        j_vals = np.arange(N)  # shape (N,)
        
        # For each (k, j), compute k-j
        kj = k_vals[:, None] - j_vals[None, :]  # shape (num_conv, N)
        
        # Mask for valid k-j (0 <= k-j < N)
        valid = (kj >= 0) & (kj < N)
        
        # h_{k-j} for valid entries
        h_kj = np.zeros_like(kj, dtype=np.float64)
        h_kj[valid] = h[kj[valid]]
        
        # Base contribution: h_{k-j}
        G = h_kj.copy()
        
        # Additional contribution: h_j if valid and k-j != j
        # k-j != j means k != 2*j
        not_equal = (kj != j_vals[None, :])
        extra = valid & not_equal
        G[extra] += h[j_vals[None, :]][extra]
        
        # For k-j == j (i.e., k=2*j), the derivative is 2*h_j, not h_j + h_j
        # But our formula already gives h_j + h_j = 2*h_j for that case
        # Wait: if k-j == j, then h_{k-j} = h_j, and the extra term would add h_j again,
        # giving 2*h_j. But that's correct! Because d(h_j^2)/dh_j = 2*h_j.
        # So the formula G = h_{k-j} + h_j * (valid & (k-j != j)) is WRONG for k-j==j.
        # For k-j==j: G should be 2*h_j
        # Our formula gives: h_j + 0 = h_j (since extra is False when k-j==j)
        # That's wrong!
        
        # Let me redo: 
        # d(conv_k)/dh_j = h_{k-j} (from differentiating h_j in term i=j)
        #                  + h_j (from differentiating h_{k-j} in term i=k-j, if k-j != j and valid)
        # If k-j == j: d(h_j * h_j)/dh_j = 2*h_j
        # If k-j != j and valid: d(h_j * h_{k-j} + h_{k-j} * h_j)/dh_j = h_{k-j} + h_j
        #   Wait, no. conv_k = sum_i h_i * h_{k-i}. The terms involving h_j are:
        #   - i=j: h_j * h_{k-j}
        #   - i=k-j: h_{k-j} * h_j (if k-j != j)
        #   If k-j == j, these are the same term: h_j * h_j
        #   So d(conv_k)/dh_j = h_{k-j} + h_j * (1 if k-j != j and valid else 0)
        #   If k-j == j and valid: d(h_j^2)/dh_j = 2*h_j
        #   Our formula: h_{k-j} + h_j * (k-j != j) = h_j + 0 = h_j. WRONG.
        
        # Correct formula:
        # d(conv_k)/dh_j = h_{k-j} if valid
        #                 + h_j if valid and k-j != j
        #                 + h_j if k-j == j (to make it 2*h_j)
        # Simplified: 
        # If valid:
        #   If k-j == j: 2*h_j
        #   Else: h_{k-j} + h_j
        # If not valid: 0
        
        G = np.zeros((num_conv, N))
        for k in range(num_conv):
            for j in range(N):
                kj = k - j
                if 0 <= kj < N:
                    if kj == j:
                        G[k, j] = 2.0 * h[j]
                    else:
                        G[k, j] = h[kj] + h[j]
        
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
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # --- Phase 3: Final Polish with tighter tolerances ---
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
    
    # --- Phase 4: Second polish from perturbed start ---
    if time.time() < deadline - 3:
        h_start = best_h / np.sum(best_h)
        np.random.seed(123)
        h_pert = h_start + 0.01 * np.random.randn(N)
        h_pert = np.clip(h_pert, 0, None)
        h_pert /= np.sum(h_pert)
        conv_pert = np.convolve(h_pert, h_pert, mode='full')
        M_pert = np.max(conv_pert) + 1e-8
        x0 = np.concatenate((h_pert, [M_pert]))
        
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
