import numpy as np
import time
import scipy.optimize as opt

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

def _log_score_smooth(h, temperature=1.0):
    """
    Compute a smooth approximation of the log-ratio objective.
    Maximizing this is equivalent to maximizing the original ratio.
    
    Ratio R = L2_sq / (L1 * L_inf_conv)
    log(R) = log(L2_sq) - log(L1) - log(L_inf_conv)
    
    We approximate log(L_inf_conv) using log-sum-exp (softmax) on the convolution.
    log(L_inf_conv) ≈ log(sum(exp(conv_k / T))) * T  (scaled log-sum-exp)
    Actually, log-sum-exp(x) = T * log(sum(exp(x_k / T)))
    As T -> 0, this approaches max(x).
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-12:
        return -np.inf
    l2_sq = np.sum(h**2)
    if l2_sq <= 1e-12:
        return -np.inf
        
    conv = np.convolve(h, h, mode='full')
    
    # Log-sum-exp trick for numerical stability
    max_conv = np.max(conv)
    if max_conv <= 1e-12:
        return -np.inf
    
    # Approximate log(max(conv))
    # log(sum(exp(conv/T))) * T is not exactly log(max).
    # Let's use: log(max_conv) + log(sum(exp((conv - max_conv)/T))) * (1/T) * T ? No.
    # Standard: log(sum exp(x)) = max_x + log(sum exp(x - max_x)).
    # We want to approximate max_x.
    # max_x ≈ T * log(sum(exp(x/T)))
    # Let y = conv / T
    # max(conv) ≈ T * log(sum(exp(conv/T)))
    
    # However, we need the gradient to be useful.
    # Let f(T) = T * log(sum(exp(conv_k / T)))
    # df/dT is complex, but we can treat T as a fixed parameter during optimization of h.
    # We just need d/dh [ T * log(sum(exp(conv_k / T))) ]
    
    # Let S = sum(exp(conv_k / T))
    # Obj = - T * log(S)
    # d Obj / dh_i = - T * (1/S) * d S / dh_i
    # d S / dh_i = sum_k exp(conv_k / T) * (1/T) * d conv_k / dh_i
    # d conv_k / dh_i = h_{k-i} + h_{i-k} (with bounds checks)
    
    # So d Obj / dh_i = - sum_k [ exp(conv_k / T) / S ] * (h_{k-i} + h_{i-k})
    # The term in brackets is a probability distribution p_k = exp(conv_k/T) / S.
    
    # This gradient points in the direction that minimizes the smoothed max.
    # We want to MAXIMIZE the ratio, so we MINIMIZE the negative log-ratio.
    # Negative Log Ratio = -log(L2_sq) + log(L1) + log(L_inf_conv_smooth)
    
    # Let's compute the value first.
    log_l2_sq = np.log(l2_sq)
    log_l1 = np.log(l1)
    
    # Smoothed log max
    # Using stable log-sum-exp
    conv_scaled = conv / temperature
    max_scaled = np.max(conv_scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(conv_scaled - max_scaled)))
    log_max_approx = temperature * log_sum_exp
    
    return log_l2_sq - log_l1 - log_max_approx

def _negative_log_score_smooth(h, temperature=1.0):
    val = _log_score_smooth(h, temperature)
    if np.isinf(val):
        return 1e10
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse random
    np.random.seed(42)
    for _ in range(30):
        h_s = np.zeros(N)
        k = np.random.randint(3, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Random continuous
    for _ in range(20):
        h_r = np.random.exponential(1.0, N)
        h_r /= np.sum(h_r)
        inits.append(h_r)
        
    # 4. Gaussian
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/3, 5):
            h_g = np.exp(-0.5 * ((np.arange(N) - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)

    # Optimization using L-BFGS-B on the negative smooth log-score
    # We need to handle the constraint sum(h) = 1.
    # L-BFGS-B doesn't support equality constraints directly.
    # We can parameterize h such that sum(h)=1 is automatically satisfied?
    # Or use a penalty method.
    # Or use SLSQP with the smooth objective? SLSQP supports constraints.
    # But the directive suggests L-BFGS-B.
    # Let's stick to L-BFGS-B but handle the normalization constraint via penalty or by working in a simplex?
    # Actually, the ratio is scale-invariant: R(c*h) = R(h).
    # So we can fix sum(h) = 1 without loss of generality.
    # If we use L-BFGS-B, we can minimize the penalty:
    # J(h) = -log_score_smooth(h) + lambda * (sum(h) - 1)^2
    
    lambda_penalty = 1000.0
    
    def objective_lbfgsb(x):
        h = np.clip(x, 1e-12, None)
        # Add penalty for normalization
        sum_h = np.sum(h)
        penalty = lambda_penalty * (sum_h - 1.0)**2
        val = _negative_log_score_smooth(h, temperature=0.5)
        return val + penalty
        
    def gradient_lbfgsb(x):
        h = np.clip(x, 1e-12, None)
        # Gradient of penalty
        sum_h = np.sum(h)
        grad_penalty = 2 * lambda_penalty * (sum_h - 1.0) * np.ones(N)
        
        # Gradient of smooth log score
        # We need d/dh [ -log(L2_sq) + log(L1) + T*log(sum(exp(conv/T))) ]
        
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        
        if l2_sq < 1e-12 or l1 < 1e-12:
            return grad_penalty
            
        # d/dh (-log(L2_sq)) = - (1/L2_sq) * 2h = -2h/L2_sq
        grad_l2 = -2.0 * h / l2_sq
        
        # d/dh (log(L1)) = (1/L1) * 1 = 1/L1
        grad_l1 = np.ones(N) / l1
        
        # d/dh (T * log(sum(exp(conv/T))))
        # Let S = sum(exp(conv_k/T))
        # Term = T * log(S)
        # d Term / dh_i = T * (1/S) * sum_k exp(conv_k/T) * (1/T) * d conv_k / dh_i
        # = (1/S) * sum_k exp(conv_k/T) * (h_{k-i} + h_{i-k})
        
        conv = np.convolve(h, h, mode='full')
        T = 0.5
        conv_scaled = conv / T
        max_scaled = np.max(conv_scaled)
        exp_scaled = np.exp(conv_scaled - max_scaled)
        S = np.sum(exp_scaled)
        
        # p_k = exp(conv_k/T) / S
        p = exp_scaled / S
        
        # grad_smooth_max[i] = sum_k p_k * (h_{k-i} + h_{i-k})
        # Note: h is 0-indexed. conv index k corresponds to sum_{j} h_j h_{k-j}.
        # d conv_k / dh_i = h_{k-i} + h_{i-k} (if indices valid)
        
        grad_smooth_max = np.zeros(N)
        for i in range(N):
            val = 0.0
            # k ranges from 0 to 2N-2
            # h_{k-i} is valid if 0 <= k-i < N => i <= k < i+N
            # h_{i-k} is valid if 0 <= i-k < N => i-N < k <= i => k in [max(0, i-N+1), i]
            
            # Efficient calculation:
            # sum_k p_k h_{k-i} = (p * h)[i] ? No, it's a cross-correlation.
            # Let's just loop for simplicity if N=50 is small enough, or use fftconvolve.
            # N=50, loop is 50*50 = 2500 ops. Very fast.
            
            for k in range(2*N - 1):
                term1 = 0.0
                term2 = 0.0
                idx1 = k - i
                idx2 = i - k
                if 0 <= idx1 < N:
                    term1 = h[idx1]
                if 0 <= idx2 < N:
                    term2 = h[idx2]
                val += p[k] * (term1 + term2)
            grad_smooth_max[i] = val
            
        total_grad = grad_l2 + grad_l1 + grad_smooth_max
        return total_grad + grad_penalty

    # Try L-BFGS-B from multiple initializations
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        x0 = h_init.copy()
        
        try:
            res = opt.minimize(
                objective_lbfgsb, 
                x0, 
                method='L-BFGS-B', 
                jac=gradient_lbfgsb,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_candidate = np.clip(res.x, 0, None)
            if np.sum(h_candidate) > 0:
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Refine with SLSQP on the original non-smooth objective (or smooth with lower T)
    # SLSQP can handle the equality constraint sum(h)=1 properly.
    # We use the smooth objective but with a smaller temperature for sharper peak approximation.
    
    def objective_slsqp(x):
        return _negative_log_score_smooth(x, temperature=0.1)
        
    def gradient_slsqp(x):
        # Reuse the gradient calculation logic but with T=0.1
        # To avoid code duplication, we can just call the lbfgsb gradient with a modified T?
        # No, the lambda penalty is inside. Let's write a clean one.
        
        h = np.clip(x, 1e-12, None)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        
        if l2_sq < 1e-12 or l1 < 1e-12:
            return np.zeros(N)
            
        grad_l2 = -2.0 * h / l2_sq
        grad_l1 = np.ones(N) / l1
        
        conv = np.convolve(h, h, mode='full')
        T = 0.1
        conv_scaled = conv / T
        max_scaled = np.max(conv_scaled)
        exp_scaled = np.exp(conv_scaled - max_scaled)
        S = np.sum(exp_scaled)
        p = exp_scaled / S
        
        grad_smooth_max = np.zeros(N)
        for i in range(N):
            val = 0.0
            for k in range(2*N - 1):
                term1 = 0.0
                term2 = 0.0
                idx1 = k - i
                idx2 = i - k
                if 0 <= idx1 < N:
                    term1 = h[idx1]
                if 0 <= idx2 < N:
                    term2 = h[idx2]
                val += p[k] * (term1 + term2)
            grad_smooth_max[i] = val
            
        return grad_l2 + grad_l1 + grad_smooth_max

    constraint_eq = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(N)}
    bounds = [(0.0, 1.0)] * N
    
    # Start refinement from best_h and a few other good candidates
    refine_starts = [best_h]
    # Add a few random sparse starts
    for _ in range(5):
        h_s = np.zeros(N)
        k = np.random.randint(3, 10)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        refine_starts.append(h_s)
        
    for h_init in refine_starts:
        if time.time() > deadline - 5:
            break
        try:
            res = opt.minimize(
                objective_slsqp, 
                h_init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[constraint_eq],
                jac=gradient_slsqp,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            h_candidate = np.clip(res.x, 0, None)
            if np.sum(h_candidate) > 0:
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
