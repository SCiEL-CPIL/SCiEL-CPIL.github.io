import numpy as np
from scipy.optimize import minimize
import time

N = 50
_TIME_LIMIT = 145.0
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
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-12:
        return 0.0
    return s2 / (s1 * max_conv)

def _neg_score_smooth(h, beta=50.0):
    """
    Negative of a smooth surrogate for the score.
    Uses log-sum-exp to approximate max(conv) smoothly.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    # Log-sum-exp for smooth max
    max_c = np.max(conv)
    if max_c < 1e-15:
        return 0.0
    # soft_max = max_c * exp(sum(exp((conv - max_c)/beta)) / len(conv))
    shifted = (conv - max_c) / beta
    log_sum_exp = max_c / beta + np.log(np.sum(np.exp(shifted))) - np.log(len(conv))
    soft_max = np.exp(log_sum_exp)
    if soft_max < 1e-12:
        return 0.0
    return -(s2 / (s1 * soft_max))

def _neg_score_grad(h, beta=50.0):
    """
    Gradient of the negative smooth score.
    """
    h = np.asarray(h, dtype=float)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 < 1e-12:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    max_c = np.max(conv)
    if max_c < 1e-15:
        return np.zeros_like(h)
    
    # Soft max weights
    shifted = (conv - max_c) / beta
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp < 1e-30:
        return np.zeros_like(h)
    w = exp_shifted / sum_exp  # weights for soft max
    
    soft_max = max_c + beta * np.log(sum_exp / len(conv))
    if soft_max < 1e-12:
        return np.zeros_like(h)
    
    # Score = s2 / (s1 * soft_max)
    # d(score)/d(h_i) = [2*h_i * s1 * soft_max - s2 * (h_i * soft_max + s1 * d(soft_max)/d(h_i))] / (s1^2 * soft_max^2)
    # = [2*h_i / s1 - s2/s1^2] / soft_max - s2/(s1^2 * soft_max^2) * d(soft_max)/d(h_i)
    
    # d(soft_max)/d(h_i) = sum_k w_k * d(conv_k)/d(h_i)
    # conv_k = sum_j h_j * h_{k-j} (for linear conv, with appropriate indices)
    # d(conv_k)/d(h_i) = h_{k-i} + h_{k-i} if k-i is valid... 
    # Actually for linear conv: conv[k] = sum_j h[j] * h[k-j] where 0<=j<N, 0<=k-j<N
    # d(conv[k])/d(h[i]) = h[k-i] (if 0<=k-i<N) + h[k-i] (if 0<=k-i<N) -- no, it's just h[k-i] + h[k-i]
    # Wait: conv[k] = sum_j h[j]*h[k-j]. d/d(h[i]) = h[k-i] (from j=i term) + h[k-i] (from k-j=i term)
    # = 2*h[k-i] if 0 <= k-i < N, else h[k-i] if only one term is valid
    
    # Let's compute d(soft_max)/d(h) more carefully
    # conv has length 2N-1, indices 0..2N-2
    # conv[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j] * h[k-j]
    # d(conv[k])/d(h[i]) = h[k-i] (if 0<=k-i<N) + h[k-i] (if 0<=k-i<N)
    # Actually: d(conv[k])/d(h[i]) = h[k-i] + h[k-i] where each term exists if index is valid
    # = h[k-i] * (1 if 0<=k-i<N else 0) + h[k-i] * (1 if 0<=k-i<N else 0)
    # No wait. conv[k] = sum_j h[j]*h[k-j]. 
    # d/d(h[i]) = sum_j [delta(j,i)*h[k-j] + h[j]*delta(k-j,i)]
    # = h[k-i] (if 0<=k-i<N) + h[k-i] (if 0<=k-i<N)
    # = 2*h[k-i] if 0<=k-i<N, else 0
    
    # So d(soft_max)/d(h[i]) = sum_k w[k] * 2*h[k-i] * I(0<=k-i<N)
    # = 2 * sum_k w[k] * h[k-i] * I(0<=k-i<N)
    
    # Let's compute this via correlation
    # d_sm[i] = 2 * sum_k w[k] * h[k-i] for 0<=k-i<N
    # This is a cross-correlation of w and h
    
    # Build the gradient
    grad_score = np.zeros(N)
    for i in range(N):
        # d(soft_max)/d(h[i]) = 2 * sum_{k} w[k] * h[k-i] where 0 <= k-i < N
        # k-i in [0, N-1] means k in [i, i+N-1]
        lo = i
        hi = min(2*N-2, i + N - 1)
        if lo <= hi:
            d_sm_i = 2.0 * np.dot(w[lo:hi+1], h[0:hi-lo+1])
        else:
            d_sm_i = 0.0
        
        ds2_dhi = 2.0 * h[i]
        ds1_dhi = 1.0
        
        # score = s2 / (s1 * sm)
        # d(score)/d(h_i) = (ds2_dhi * s1 * sm - s2 * (ds1_dhi * sm + s1 * d_sm_i)) / (s1^2 * sm^2)
        num = ds2_dhi * s1 * soft_max - s2 * (ds1_dhi * soft_max + s1 * d_sm_i)
        den = s1 * s1 * soft_max * soft_max
        grad_score[i] = num / den
    
    # Return negative gradient (since we're minimizing neg_score)
    return -grad_score

def _optimize_from_init(h0, beta=50.0, max_iter=200):
    """
    Optimize from a given initialization using L-BFGS-B with smooth surrogate.
    """
    h0 = np.asarray(h0, dtype=float)
    h0 = np.clip(h0, 0, None)
    
    if np.sum(h0) < 1e-12:
        return h0, _score(h0)
    
    best_h = h0.copy()
    best_score = _score(h0)
    
    try:
        res = minimize(
            _neg_score_smooth,
            h0,
            jac=_neg_score_grad,
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        h_opt = np.clip(res.x, 0, None)
        s_opt = _score(h_opt)
        
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    except Exception:
        pass
    
    # Polish with higher beta (sharper max approximation)
    if time.time() - _start_time < _TIME_LIMIT - 5:
        try:
            res2 = minimize(
                _neg_score_smooth,
                best_h,
                jac=_neg_score_grad,
                args=(200.0,),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt2 = np.clip(res2.x, 0, None)
            s_opt2 = _score(h_opt2)
            if s_opt2 > best_score:
                best_score = s_opt2
                best_h = h_opt2
        except Exception:
            pass
    
    return best_h, best_score

def construct():
    # Known best vector
    h0 = [0.0] * N
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    best_h = np.array(h0)
    best_score = _score(best_h)
    
    # Diverse initializations
    inits = [
        h0,
        [1.0/2, 1.0/2] + [0.0]*(N-2),
        [1.0/3]*3 + [0.0]*(N-3),
        [0.4, 0.3, 0.3] + [0.0]*(N-3),
        [0.5, 0.25, 0.25] + [0.0]*(N-3),
        [0.6, 0.2, 0.2] + [0.0]*(N-3),
        [0.7, 0.15, 0.15] + [0.0]*(N-3),
        [0.8, 0.1, 0.1] + [0.0]*(N-3),
        [0.5, 0.5] + [0.0]*(N-2),
        [0.3, 0.3, 0.2, 0.2] + [0.0]*(N-4),
        [0.25]*4 + [0.0]*(N-4),
        [0.2]*5 + [0.0]*(N-5),
        [1.0/N]*N,  # uniform
        [0.5, 0.3, 0.1, 0.1] + [0.0]*(N-4),
        [0.4, 0.4, 0.2] + [0.0]*(N-3),
    ]
    
    # Add some random initializations
    rng = np.random.default_rng(123)
    for _ in range(10):
        h_rand = rng.random(N)
        h_rand = h_rand / np.sum(h_rand) if np.sum(h_rand) > 0 else h_rand
        inits.append(h_rand.tolist())
    
    for init in inits:
        if time.time() - _start_time > _TIME_LIMIT - 5:
            break
        h_opt, s_opt = _optimize_from_init(init, beta=50.0, max_iter=150)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Final random perturbation search around best
    rng2 = np.random.default_rng(42)
    for _ in range(100):
        if time.time() - _start_time > _TIME_LIMIT - 2:
            break
        h_pert = np.array(best_h, dtype=float)
        noise = rng2.normal(0, 0.03, N)
        h_pert = h_pert + noise
        h_pert = np.clip(h_pert, 0, None)
        s_pert = _score(h_pert)
        if s_pert > best_score:
            best_score = s_pert
            best_h = h_pert
    
    # Final polish
    if time.time() - _start_time < _TIME_LIMIT - 3:
        h_opt, s_opt = _optimize_from_init(best_h, beta=200.0, max_iter=100)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    return [float(x) for x in best_h]
