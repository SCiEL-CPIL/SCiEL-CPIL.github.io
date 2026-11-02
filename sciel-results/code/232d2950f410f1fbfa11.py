import numpy as np
from scipy.optimize import minimize
import time

N = 200

# EVOLVE-BLOCK-START

def _compute_overlaps(h):
    """
    Compute all circular cross-correlation overlaps h[i]*(1-h[i+k]).
    Returns an array of length N where overlaps[k] = sum_i h[i] * (1 - h[i+k]).
    """
    N = len(h)
    # Use FFT for speed? 
    # O_k = sum_i h_i - sum_i h_i h_{i+k} = N/2 - C_k where C_k is autocorrelation of h.
    # Actually, sum h_i is not necessarily N/2 during optimization if we don't enforce it strictly in the objective.
    # Let's stick to direct computation for correctness, but vectorize.
    # h_i (1 - h_{i+k}) = h_i - h_i h_{i+k}
    # Sum_i h_i = S. Sum_i h_i h_{i+k} = autocorr_k.
    # O_k = S - autocorr_k.
    
    # Autocorrelation via FFT
    f_h = np.fft.fft(h)
    power = f_h * np.conj(f_h)
    autocorr = np.fft.ifft(power).real
    # autocorr[k] is sum_i h_i h_{i+k} (circular)
    S = np.sum(h)
    overlaps = S - autocorr
    return overlaps

def _objective_and_grad(h, beta, mu, target_sum):
    """
    Compute objective value and gradient.
    Objective: log-sum-exp(beta * overlaps) + mu * (sum(h) - target_sum)^2
    """
    overlaps = _compute_overlaps(h)
    
    # Log-sum-exp stability
    max_ov = np.max(overlaps)
    # If beta is large, exp(beta * (ov - max_ov)) might underflow for small ov, which is fine.
    # We want to minimize the max, so we use log-sum-exp of beta * overlaps.
    # As beta -> inf, this approaches max(overlaps).
    
    # To avoid overflow in exp, subtract max
    scaled = beta * (overlaps - max_ov)
    # If beta is very large, scaled might have large negative values, exp -> 0.
    # exp(0) = 1 for the max element.
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    
    obj_lse = max_ov * beta + np.log(sum_exp)
    
    # Gradient of log-sum-exp
    # d/dh log(sum exp(beta * O_k)) = sum_k (beta * exp(beta * O_k) / sum exp) * dO_k/dh
    # weights w_k = exp(beta * O_k) / sum exp
    # We computed exp_scaled = exp(beta * (O_k - max_ov))
    # So w_k = exp_scaled_k / sum_exp
    
    weights = exp_scaled / sum_exp
    
    # Gradient of overlaps w.r.t h
    # O_k = sum_i h_i (1 - h_{i+k})
    # dO_k/dh_j = (1 - h_{j+k}) - h_{j-k}
    # We need sum_k weights[k] * ((1 - h_{j+k}) - h_{j-k})
    
    # Let's compute this efficiently.
    # Term 1: sum_k weights[k] * (1 - h_{j+k}) = 1 - sum_k weights[k] h_{j+k}
    # This is 1 - (weights * h) circular convolution at j?
    # Let w_rev be reversed weights?
    # Circular correlation: (w * h)_j = sum_k w_k h_{j+k}
    # np.convolve(w, h, mode='full')? No, circular.
    # Use FFT: ifft(fft(w) * fft(h)) gives circular convolution (w*h)_j = sum_k w_k h_{j-k}.
    # We want sum_k w_k h_{j+k}. This is correlation.
    # corr(j) = sum_k w_k h_{j+k} = sum_k w_{k} h_{j+k}.
    # Let m = -k. sum_m w_{-m} h_{j-m}.
    # So we need ifft(fft(w_rev) * fft(h)) where w_rev[k] = w[-k].
    
    w_rev = np.roll(weights, -1) # w_rev[k] = w[k-1]? 
    # Let's check indices.
    # We want G1_j = sum_k w_k h_{j+k}.
    # Let y = ifft(fft(w) * fft(h)). y_j = sum_k w_k h_{j-k}.
    # We want h_{j+k}. Let l = -k. sum_l w_{-l} h_{j+l}.
    # This is ifft(fft(w_neg) * fft(h)) where w_neg[l] = w_{-l}.
    # w_neg is weights reversed and rolled?
    # w_neg[0] = w[0], w_neg[1] = w[N-1], etc.
    # np.roll(weights, 1) shifts right. weights[1] -> weights[0].
    # Let's just use the direct formula with FFT for speed.
    
    # G1_j = 1 - corr_w_h_j
    # corr_w_h = ifft(fft(weights) * np.conj(fft(h)))? No.
    # Correlation of w and h: sum_k w_k h_{j+k}.
    # This is ifft(fft(w) * conj(fft(h)))?
    # Let's test.
    # Actually, simpler: 
    # sum_k w_k h_{j+k} is the circular correlation of w and h.
    # scipy.signal.correlate(w, h, mode='full')?
    # With FFT: ifft(fft(w) * conj(fft(h))) gives sum_k w_k h_{j-k}.
    # We want h_{j+k}.
    # Let h' = h reversed?
    # Let's stick to the gradient formula derived:
    # grad[j] = sum_k w_k (1 - h_{j+k}) - h_{j-k}
    # grad[j] = (sum_k w_k) - sum_k w_k h_{j+k} - sum_k w_k h_{j-k}
    # sum_k w_k = 1.
    # Term A: sum_k w_k h_{j+k}
    # Term B: sum_k w_k h_{j-k}
    
    # Term B is standard circular convolution (w*h)_j.
    # ifft(fft(w) * fft(h))
    
    # Term A is circular cross-correlation.
    # (w * h)_j = sum_k w_k h_{j+k}.
    # This is ifft(fft(w) * conj(fft(h)))?
    # Let's verify.
    # X = fft(h), W = fft(w).
    # ifft(W * conj(X))_j = sum_k w_k h_{j+k}. Yes.
    
    W = np.fft.fft(weights)
    H = np.fft.fft(h)
    
    term_b = np.fft.ifft(W * H).real
    term_a = np.fft.ifft(W * np.conj(H)).real
    
    grad_ov = 1.0 - term_a - term_b
    
    # Gradient of penalty term
    sum_h = np.sum(h)
    grad_pen = 2.0 * mu * (sum_h - target_sum) * np.ones(N)
    
    total_grad = beta * grad_ov + grad_pen
    total_obj = obj_lse + mu * (sum_h - target_sum)**2
    
    return total_obj, total_grad

def _project_to_simplex(h, target_sum):
    """Project h onto [0,1]^N with sum = target_sum."""
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    for _ in range(50):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            free = h < 1.0 - 1e-10
        else:
            free = h > 1e-10
        if not np.any(free):
            break
        h[free] += diff / np.sum(free)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    deadline = time.time() + 140.0
    target_sum = N / 2.0
    
    # Initial guess: uniform
    best_h = np.full(N, 0.5)
    best_score = np.max(_compute_overlaps(best_h))
    
    rng = np.random.RandomState(42)
    
    # Parameters
    betas = [1.0, 5.0, 20.0, 50.0, 100.0]
    mus = [10.0, 100.0, 1000.0]
    
    # Try different initializations
    inits = []
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    # 2. Random
    for _ in range(3):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project_to_simplex(h_rand, target_sum)
        inits.append(h_rand)
    # 3. Spectral guess: single harmonic
    # h_i = 0.5 + A cos(2 pi i / N)
    for A in [0.1, 0.2, 0.3]:
        h_spec = 0.5 + A * np.cos(2 * np.pi * np.arange(N) / N)
        h_spec = _project_to_simplex(h_spec, target_sum)
        inits.append(h_spec)
    
    for h0 in inits:
        if time.time() > deadline - 20.0:
            break
            
        current_h = h0.copy()
        
        for beta in betas:
            for mu in mus:
                if time.time() > deadline - 10.0:
                    break
                
                def obj_func(h):
                    return _objective_and_grad(h, beta, mu, target_sum)[0]
                
                def grad_func(h):
                    return _objective_and_grad(h, beta, mu, target_sum)[1]
                
                try:
                    res = minimize(
                        obj_func,
                        current_h,
                        jac=grad_func,
                        method='L-BFGS-B',
                        bounds=[(0.0, 1.0)] * N,
                        options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    
                    if res.success:
                        h_opt = res.x
                        # Project to ensure constraints are met exactly
                        h_opt = _project_to_simplex(h_opt, target_sum)
                        score = np.max(_compute_overlaps(h_opt))
                        
                        if score < best_score:
                            best_score = score
                            best_h = h_opt.copy()
                            current_h = h_opt.copy()
                            
                except Exception:
                    pass

    # Final polish: Direct optimization of max overlap using sub-gradient or local search
    # We can try a simple local search on the best_h found
    # Or just return best_h
    
    return best_h

# EVOLVE-BLOCK-END
