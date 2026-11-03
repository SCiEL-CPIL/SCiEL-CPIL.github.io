import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
NUM_CONV = 2 * N - 1

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse structured (Sidon-like intuition)
    np.random.seed(42)
    for trial in range(50):
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Alternating patterns (helps reduce auto-correlation peaks)
    for start in range(N):
        for gap in [2, 3, 4, 5]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                inits.append(h)
                
    # 7. Log-sum-exp based smooth initialization (new structural idea)
    # We want to maximize L2^2 / (L1 * M) where M is peak autoconv.
    # With L1=1, maximize L2^2 / M.
    # The log-sum-exp approximation makes the peak constraint smooth.
    
    # --- Phase 2: Smooth Log-Sum-Exp Optimization ---
    # This is the new structural change: use L-BFGS-B with a smooth penalty
    # instead of SLSQP with hard constraints.
    
    def smooth_max_conv(h, beta):
        """Smooth approximation of max autoconvolution using log-sum-exp."""
        h_conv = np.convolve(h, h, mode='full')
        m = np.max(h_conv)
        if m <= 0:
            return 0.0, 0.0
        # Log-sum-exp: log(sum(exp(x_i))) = m + log(sum(exp(x_i - m)))
        # We want (1/beta) * log(sum(exp(beta * x_i))) which approximates max as beta->inf
        # For numerical stability:
        lse = m + np.log(np.sum(np.exp((h_conv - m) * beta))) / beta
        # Gradient of lse w.r.t. h:
        # d/dh_j [ (1/beta) * log(sum_k exp(beta * conv_k)) ]
        # = (1/beta) * sum_k [ beta * exp(beta * conv_k) / sum_k' exp(beta * conv_k') * d(conv_k)/dh_j ]
        # = sum_k [ exp(beta * conv_k) / sum_k' exp(beta * conv_k') * d(conv_k)/dh_j ]
        # Let w_k = exp(beta * (conv_k - m)) / sum_k' exp(beta * (conv_k' - m))
        # Then gradient = sum_k w_k * d(conv_k)/dh_j
        weights = np.exp(beta * (h_conv - m))
        weights /= np.sum(weights)
        
        # Compute gradient of weighted sum of convolutions
        grad = np.zeros(N)
        for j in range(N):
            # d(conv_k)/dh_j for k in [0, 2N-2]
            # conv_k = sum_{m=0}^{N-1} h_m h_{k-m}
            # d(conv_k)/dh_j = h_{k-j} + h_{j-k} (where indices valid)
            val = 0.0
            # Part 1: k-j in [0, N-1] => k in [j, j+N-1]
            k_start = j
            k_end = min(j + N, NUM_CONV)
            if k_start < NUM_CONV and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                val += np.sum(weights[indices_k] * h[indices_h])
            # Part 2: j-k in [0, N-1] => k in [j-N+1, j]
            k_start2 = max(0, j - N + 1)
            k_end2 = min(j, NUM_CONV)
            if k_start2 < NUM_CONV and k_end2 >= 0:
                indices_k = np.arange(k_start2, k_end2 + 1)
                indices_h = j - indices_k
                val += np.sum(weights[indices_k] * h[indices_h])
            grad[j] = val
        return lse, grad
    
    def smooth_objective_and_grad(h, beta):
        """
        Minimize: M_smooth - lambda * L2^2
        where M_smooth is log-sum-exp approximation of max autoconv.
        With L1=1 constraint handled by normalizing h in the objective.
        Actually, we work with unnormalized h and include L1 in the ratio.
        
        Better: minimize M_smooth / L1 - lambda * L2^2 / L1
        But this is complex. Let's fix L1=1 by normalizing inside.
        
        Actually, let's use: minimize M_smooth - lambda * (L2^2 / L1)
        This encourages large L2^2/L1 (which is good) and small M_smooth.
        """
        h = np.maximum(0, h)
        l1 = np.sum(h)
        if l1 <= 1e-15:
            return 1e10, np.zeros(N)
        
        h_norm = h / l1
        l2_sq = np.sum(h_norm**2)
        
        m_smooth, grad_m = smooth_max_conv(h_norm, beta)
        
        # Objective: m_smooth - lambda * l2_sq
        lam = 1.0  # Balance between minimizing peak and maximizing L2
        obj = m_smooth - lam * l2_sq
        
        # Gradient: grad_m - lam * grad(l2_sq)
        # d(l2_sq)/dh_j = 2 * h_j / l1 - 2 * h_j * sum(h) / l1^2 = 2*h_j/l1 - 2*h_j/l1 = 0?
        # Wait, h_norm = h/l1. d(h_norm_j)/dh_k = delta_jk/l1 - h_j * delta_k_sum / l1^2
        # d(l2_sq)/dh_k = 2 * sum_j h_norm_j * d(h_norm_j)/dh_k
        # = 2 * sum_j (h_j/l1) * (delta_jk/l1 - h_j/l1^2)
        # = 2 * (h_k/l1^2 - (sum_j h_j^2)/l1^2)
        # = 2 * (h_k - l2_sq) / l1^2
        grad_l2 = 2.0 * (h - l2_sq) / (l1 * l1)
        
        # But grad_m is gradient w.r.t. h_norm, not h.
        # d(m_smooth)/dh_k = sum_j (d(m_smooth)/d(h_norm_j)) * (d(h_norm_j)/dh_k)
        # = sum_j grad_m_j * (delta_jk/l1 - h_j/l1^2)
        # = grad_m_k/l1 - (sum_j grad_m_j * h_j)/l1^2
        sum_grad_h = np.sum(grad_m * h)
        grad_obj = grad_m / l1 - sum_grad_h / (l1 * l1) - lam * grad_l2
        
        return obj, grad_obj
    
    # Run smooth optimization from top initializations
    scored_inits = []
    for h_init in inits:
        s = _score_raw(h_init)
        scored_inits.append((s, h_init))
    scored_inits.sort(key=lambda x: -x[0])
    
    max_smooth_runs = min(len(scored_inits), 40)
    
    for i in range(max_smooth_runs):
        if time.time() > deadline - 30:
            break
        s_init, h_init = scored_inits[i]
        
        # Try multiple beta values for different smoothness levels
        for beta in [10.0, 50.0, 100.0]:
            if time.time() > deadline - 25:
                break
            
            def obj_wrapper(h, beta_val=beta):
                obj, grad = smooth_objective_and_grad(h, beta_val)
                return obj, grad
            
            try:
                res = minimize(obj_wrapper, h_init, method='L-BFGS-B',
                             jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < 1e9:
                    h_candidate = res.x
                    h_candidate = np.maximum(0, h_candidate)
                    if np.sum(h_candidate) > 1e-15:
                        h_candidate /= np.sum(h_candidate)
                        s_candidate = _score_raw(h_candidate)
                        if s_candidate > best_score:
                            best_score = s_candidate
                            best_h = h_candidate.copy()
            except Exception:
                pass
    
    # --- Phase 3: Refinement using SLSQP on best result ---
    if best_h is not None and time.time() < deadline - 15:
        h_start = best_h / np.sum(best_h)
        
        def neg_obj_h(h):
            l1 = np.sum(h)
            if l1 <= 1e-15:
                return 0.0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv <= 1e-15:
                return 0.0
            return -(l2_sq / (l1 * DELTA * max_conv))
        
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 300, 'ftol': 1e-12})
            if res.success and res.fun < 0:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
        except Exception:
            pass
    
    # --- Phase 4: Final local refinement via coordinate descent ---
    if best_h is not None and time.time() < deadline - 5:
        h = best_h.copy()
        h = np.maximum(0, h)
        if np.sum(h) > 0:
            h /= np.sum(h)
        
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            improved = False
            conv = np.convolve(h, h, mode='full')
            M_val = np.max(conv)
            
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                # Optimize h[i] with others fixed
                h_old = h[i]
                
                def obj_1d(hi):
                    h_test = h.copy()
                    h_test[i] = hi
                    l1 = np.sum(h_test)
                    if l1 <= 1e-15:
                        return 0
                    l2_sq = np.sum(h_test**2)
                    h_conv = np.convolve(h_test, h_test, mode='full')
                    max_c = np.max(h_conv)
                    if max_c <= 1e-15:
                        return 0
                    return l2_sq / (l1 * DELTA * max_c)
                
                # Golden section search
                a, b = 0, 1.0
                for _ in range(25):
                    m1 = a + (b - a) * 0.382
                    m2 = a + (b - a) * 0.618
                    if obj_1d(m1) < obj_1d(m2):
                        b = m2
                    else:
                        a = m1
                    if b - a < 1e-10:
                        break
                h_new = (a + b) / 2
                h[i] = h_new
                if abs(h_new - h_old) > 1e-10:
                    improved = True
            
            if not improved:
                break
        
        h = np.maximum(0, h)
        if np.sum(h) > 0:
            h /= np.sum(h)
            s_final = _score_raw(h)
            if s_final > best_score:
                best_score = s_final
                best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
