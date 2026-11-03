import numpy as np
from scipy.optimize import minimize
import time

N = 50
_TIME_LIMIT = 135.0
_start_time = time.time()

def _score(h):
    """
    Score function: (sum h_i^2) / ( (sum h_i) * max_k conv_k )
    where conv is the linear autoconvolution.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _smooth_score(h, beta=10.0):
    """
    Smooth surrogate for score using log-sum-exp for the max.
    score = s2 / (s1 * soft_max(conv))
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0, 0.0
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    if np.max(conv) < 1e-12:
        return 0.0, 0.0
    # Log-sum-exp for soft max
    max_c = np.max(conv)
    if max_c < 1e-12:
        return 0.0, 0.0
    shifted = conv - max_c
    log_sum_exp = max_c + np.log(np.sum(np.exp(shifted * beta)) / len(conv)) / beta
    score = s2 / (s1 * log_sum_exp)
    return score, log_sum_exp

def _stochastic_subgradient(x, beta=10.0, lr=0.01, n_samples=5):
    """
    Stochastic subgradient step for x (where h = x^2).
    Samples the active lag of the supremum norm to compute gradients.
    """
    h = x ** 2
    s1 = np.sum(h)
    if s1 < 1e-12:
        return x, 0.0, 0.0
    s2 = np.dot(h, h)
    
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return x, 0.0, 0.0
    
    # Find the active lag (index of max)
    active_idx = np.argmax(conv)
    
    # Compute gradient of log_sum_exp w.r.t. h
    # d(log_sum_exp)/dh_i = sum_j conv_j * d(conv_j)/dh_i / sum(exp(beta*conv_j))
    # conv_j = sum_{i+k=j} h_i * h_k
    # d(conv_j)/dh_i = sum_k (delta_{i+k,j} * h_k + h_i * delta_{i+k,j})
    # = h_{j-i} + h_{j-i} if j-i != i, else 2*h_i
    
    # For the active lag, we can compute the gradient contribution
    # Simplified: use the subgradient at the active point
    # d(max)/dh_i ≈ sum_k h_{active_idx - i} (where active_idx - i is in [0, N-1])
    
    # Compute gradient of score w.r.t. h
    # score = s2 / (s1 * M) where M = max_conv
    # d(score)/dh = (2*h * s1*M - s2*(sum(h)*M + s1*dM/dh)) / (s1*M)^2
    
    # For stochastic, sample a few lags
    rng = np.random.default_rng()
    grad_h = np.zeros(N)
    
    for _ in range(n_samples):
        # Sample a lag
        lag = rng.integers(0, len(conv))
        # Compute d(conv_lag)/dh
        dconv_dh = np.zeros(N)
        for i in range(N):
            j = lag - i
            if 0 <= j < N:
                dconv_dh[i] += h[j]
                if j != i:
                    dconv_dh[i] += h[j]  # double count for i != j
        
        # Simplified gradient
        M = conv[lag]
        if M < 1e-12:
            continue
        
        # d(score)/dh_i = (2*h_i*s1*M - s2*(s1*M + s1*dM_dh_i)) / (s1*M)^2
        # = (2*h_i*M - s2/s1 * (M + dM_dh_i)) / (s1*M^2)
        grad_h += (2*h*M - (s2/s1)*(M + dconv_dh)) / (s1*M**2)
    
    grad_h /= n_samples
    
    # Gradient w.r.t. x: d(score)/dx = d(score)/dh * 2*x
    grad_x = 2*x*grad_h
    
    # Update
    x_new = x - lr*grad_x
    
    # Compute new score
    h_new = x_new**2
    s1_new = np.sum(h_new)
    s2_new = np.dot(h_new, h_new)
    if s1_new < 1e-12:
        return x_new, 0.0, 0.0
    conv_new = np.convolve(h_new, h_new, mode='full')
    max_conv_new = np.max(conv_new)
    if max_conv_new < 1e-12:
        return x_new, 0.0, 0.0
    score_new = s2_new / (s1_new * max_conv_new)
    
    return x_new, score_new, max_conv_new

def _optimize_with_subgradient(h0, n_iters=200, lr=0.01, beta=10.0):
    """
    Optimize using stochastic subgradient with h = x^2.
    """
    h0 = np.array(h0, dtype=float)
    h0 = np.clip(h0, 0, None)
    
    # Initialize x from h0
    x = np.sqrt(h0)
    
    best_h = h0.copy()
    best_score = _score(h0)
    
    for i in range(n_iters):
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        
        x, score, _ = _stochastic_subgradient(x, beta=beta, lr=lr, n_samples=5)
        
        h = x**2
        h = np.clip(h, 0, None)
        
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        # Occasionally do a random restart or perturbation
        if i % 50 == 0 and i > 0:
            rng = np.random.default_rng(i)
            noise = rng.normal(0, 0.05, N)
            x_pert = x + noise
            h_pert = x_pert**2
            h_pert = np.clip(h_pert, 0, None)
            s_pert = _score(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                x = np.sqrt(best_h)
    
    return best_h, best_score

def construct():
    # Known best vector
    h0 = [0.0] * N
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = np.array(h0)
    best_score = _score(best_h)
    
    # Try different support sizes
    for support_size in [2, 3, 4, 5, 6]:
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        
        # Try different initializations for this support size
        for trial in range(3):
            if time.time() - _start_time > _TIME_LIMIT - 10:
                break
            
            if trial == 0:
                # Uniform
                init = [1.0/support_size]*support_size + [0.0]*(N-support_size)
            elif trial == 1:
                # Decaying
                weights = np.linspace(1.0, 0.3, support_size)
                weights /= np.sum(weights)
                init = list(weights) + [0.0]*(N-support_size)
            else:
                # Random
                rng = np.random.default_rng(trial)
                weights = rng.random(support_size)
                weights /= np.sum(weights)
                init = list(weights) + [0.0]*(N-support_size)
            
            h_opt, s_opt = _optimize_with_subgradient(init, n_iters=100, lr=0.01)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
    
    # Final polish with smaller learning rate
    if time.time() - _start_time < _TIME_LIMIT - 5:
        h_polish, s_polish = _optimize_with_subgradient(best_h, n_iters=50, lr=0.001)
        if s_polish > best_score:
            best_score = s_polish
            best_h = h_polish
    
    return [float(x) for x in best_h]
