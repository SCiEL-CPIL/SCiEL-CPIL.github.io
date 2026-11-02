import numpy as np
from scipy.signal import fftconvolve
import time

N = 600
# Interval width
w = (1/4 - (-1/4)) / N  # 0.5 / 600

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    h = np.asarray(heights, dtype=float)
    h = np.maximum(h, 0)
    integral = np.sum(h) * w
    if integral < 1e-12:
        return 1e10
    autoconv = fftconvolve(h, h, mode='full')
    max_autoconv = np.max(autoconv)
    return max_autoconv / (integral * integral)

def smooth_max_surrogate(values, beta=200.0):
    """Smooth approximation of the maximum using log-sum-exp."""
    max_val = np.max(values)
    if max_val < 1e-30:
        return max_val
    shifted = values - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

def objective(h, beta=200.0):
    """Objective: smooth max of autoconvolution, normalized."""
    h = np.maximum(h, 0)
    integral = np.sum(h) * w
    if integral < 1e-12:
        return 1e10
    autoconv = fftconvolve(h, h, mode='full')
    max_autoconv = smooth_max_surrogate(autoconv, beta)
    return max_autoconv / (integral * integral)

def objective_grad(h, beta=200.0):
    """Gradient of the smooth objective with respect to h."""
    h = np.maximum(h, 0)
    integral = np.sum(h) * w
    if integral < 1e-12:
        return np.zeros_like(h), 1e10
    
    autoconv = fftconvolve(h, h, mode='full')
    
    # Compute log-sum-exp gradient
    max_val = np.max(autoconv)
    if max_val < 1e-30:
        grad_ac = np.zeros_like(autoconv)
        smooth_max = max_val
    else:
        shifted = autoconv - max_val
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        grad_ac = weights
        smooth_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # d(autoconv)/d(h[k]) = autoconv of e_k with h, which is just h shifted
    # autoconv[m] = sum_i h[i] * h[m-i]
    # d(autoconv[m])/d(h[k]) = h[m-k] + h[m-k] (since it's a product) -- wait
    # autoconv = h * h (convolution), so d(autoconv)/d(h[k]) = 2 * (e_k * h)[m] = 2*h[m-k]
    # where (e_k * h)[m] = h[m-k]
    
    # The gradient of the smooth max w.r.t. autoconv is grad_ac
    # So d(objective)/d(h[k]) = (1/(integral^2)) * sum_m grad_ac[m] * d(autoconv[m])/d(h[k]) - 2*(1/(integral^3)) * (sum_m grad_ac[m] * autoconv[m]) * (w * d(integral)/d(h[k]))
    # d(integral)/d(h[k]) = w
    # sum_m grad_ac[m] * autoconv[m] is not quite right; let me redo.
    
    # objective = smooth_max(autoconv) / integral^2
    # d(objective)/d(h[k]) = [d(smooth_max)/d(h[k]) * integral^2 - smooth_max * 2*integral*w] / integral^4
    #                       = [d(smooth_max)/d(h[k]) * integral - 2*smooth_max*w] / integral^3
    
    # d(smooth_max)/d(h[k]) = sum_m grad_ac[m] * d(autoconv[m])/d(h[k])
    # d(autoconv[m])/d(h[k]) = 2 * h[m-k] (with h out of range = 0)
    # So d(smooth_max)/d(h[k]) = 2 * sum_m grad_ac[m] * h[m-k]
    # = 2 * (grad_ac * h)[k] (discrete convolution, but careful with indexing)
    
    # Actually, (grad_ac * h)[k] = sum_m grad_ac[m] * h[k-m] = sum_m grad_ac[m] * h[m-k] (by symmetry of convolution)
    # Wait, convolution (a*b)[n] = sum_k a[k] b[n-k]
    # sum_m grad_ac[m] * h[m-k] = sum_m grad_ac[m] * h[m-k]
    # Let j = m-k, m = j+k: sum_j grad_ac[j+k] * h[j] = (grad_ac * h)[k] if we define it properly.
    # Actually this is the cross-correlation or convolution depending on convention.
    # Let me just compute it directly.
    
    # d(smooth_max)/d(h[k]) = 2 * sum_m grad_ac[m] * h[m-k] for valid indices
    # This is 2 * (grad_ac flipped * h)[k] ... let me use fftconvolve
    
    # sum_m grad_ac[m] * h[m-k]: let's think of it as correlation
    # = sum_m grad_ac[m] * h[m-k]
    # If I define g = grad_ac, then this is (g * h_flipped)[k] where h_flipped[i] = h[N-1-i]
    # Or more simply: it's the convolution of grad_ac with h, but with a shift.
    
    # Let me just compute it via FFT for efficiency.
    # d_smooth_max_dh[k] = 2 * sum_m grad_ac[m] * h[m-k]
    # = 2 * (grad_ac ⊛ h)[k] where ⊛ denotes cross-correlation
    # Cross-correlation can be computed as convolution with flipped signal.
    
    # Actually, let me just do it with a direct approach using FFT.
    # sum_m grad_ac[m] * h[m-k] for m=0..len(grad_ac)-1
    # This is the (k-th) element of the cross-correlation of grad_ac and h.
    # np.correlate(grad_ac, h, mode='full')[k] gives sum_m grad_ac[m] * h[m-k]
    
    # But grad_ac has length 2N-1 and h has length N.
    # The cross-correlation will have length 2N-1 + N - 1 = 3N-2.
    # We want the values for k=0..N-1.
    
    # Let me use a simpler approach: 
    # d_smooth_max_dh[k] = 2 * sum_{m=k}^{k+N-1} grad_ac[m] * h[m-k]  (approximately, for the relevant range)
    
    # Actually, the cleanest way: 
    # autoconv[m] = sum_i h[i] * h[m-i]
    # d(autoconv[m])/d(h[k]) = h[m-k] + h[k-m] = h[m-k] + h[|m-k|]... no.
    # d(autoconv[m])/d(h[k]) = h[m-k] + h[m-k] = 2*h[m-k] if m-k is in [0,N-1], else 0.
    # Wait, autoconv[m] = sum_i h[i]*h[m-i]. d/d(h[k]) = h[m-k] (from i=k term) + h[k] (from m-i=k term, i.e., i=m-k).
    # So d(autoconv[m])/d(h[k]) = h[m-k] + h[m-k] = 2*h[m-k] when m-k is in [0,N-1].
    # But h[m-k] is only defined for m-k in [0,N-1]. So:
    # d(autoconv[m])/d(h[k]) = 2*h[m-k] if 0 <= m-k < N, else 0.
    
    # So d_smooth_max_dh[k] = 2 * sum_m grad_ac[m] * h[m-k] * 1_{0<=m-k<N}
    # = 2 * sum_{j=0}^{N-1} grad_ac[k+j] * h[j]  (letting j = m-k)
    # = 2 * sum_{j=0}^{N-1} h[j] * grad_ac[k+j]
    
    # This is a valid computation. Let me compute it for all k.
    
    d_smooth_max_dh = np.zeros(N)
    for k in range(N):
        # sum_{j=0}^{N-1} h[j] * grad_ac[k+j]
        # grad_ac has length 2N-1, so k+j ranges from k to k+N-1
        # We need k+N-1 < 2N-1, i.e., k < N, which is true.
        d_smooth_max_dh[k] = 2.0 * np.dot(h, grad_ac[k:k+N])
    
    # Now compute the full gradient
    # d(objective)/d(h[k]) = [d_smooth_max_dh[k] * integral - 2*smooth_max*w] / integral^3
    
    grad = (d_smooth_max_dh * integral - 2.0 * smooth_max * w) / (integral ** 3)
    
    # Project to nonnegative orthant (simple projection for subgradient)
    grad = np.where(h > 0, grad, 0)  # Only consider positive components for simplicity
    
    return grad, objective(h, beta)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Start with a cosine-shaped initial guess
    x = np.linspace(0, 1, N)
    h = 0.5 * (1 + np.cos(2 * np.pi * x))
    h = np.maximum(h, 0)
    
    best_h = h.copy()
    best_val = compute_score(best_h)
    
    # Phase 1: Gradient descent with adaptive step size
    beta = 100.0
    lr = 0.1
    patience = 0
    max_patience = 30
    
    while time.time() < deadline - 30:
        grad, val = objective_grad(best_h, beta)
        
        if not np.isfinite(val) or val > best_val * 1.1:
            break
        
        # Adaptive step size
        step = lr / (np.max(np.abs(grad)) + 1e-10)
        new_h = best_h - step * grad
        new_h = np.maximum(new_h, 0)
        
        new_val = compute_score(new_h)
        
        if new_val < best_val:
            best_h = new_h
            best_val = new_val
            patience = 0
            lr *= 1.05  # Slightly increase step size
        else:
            patience += 1
            lr *= 0.9  # Decrease step size
            if patience > max_patience:
                break
        
        # Increase beta to sharpen the max approximation
        beta *= 1.1
        if beta > 1000:
            beta = 1000
    
    # Phase 2: Coordinate descent refinement
    # Use the smooth objective for faster evaluation
    beta_cd = 500.0
    
    def compute_value_smooth(h):
        return objective(h, beta_cd)
    
    best_h_cd = best_h.copy()
    best_val_cd = compute_value_smooth(best_h_cd)
    
    # Multiple passes of coordinate descent
    for pass_num in range(5):
        if time.time() > deadline - 10:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 10:
                break
            
            current_val = best_h_cd[i]
            
            # Try a range of values
            if current_val > 0:
                candidates = [0.0]
                for frac in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                    candidates.append(current_val * frac)
            else:
                candidates = [0.0, 0.01, 0.05, 0.1]
            
            best_candidate_val = current_val
            best_candidate_obj = best_val_cd
            
            for cand in candidates:
                if cand < 0:
                    continue
                trial_h = best_h_cd.copy()
                trial_h[i] = cand
                trial_obj = compute_value_smooth(trial_h)
                if trial_obj < best_candidate_obj:
                    best_candidate_obj = trial_obj
                    best_candidate_val = cand
            
            if best_candidate_val != current_val:
                best_h_cd[i] = best_candidate_val
                best_val_cd = best_candidate_obj
                improved = True
        
        if not improved:
            break
    
    # Phase 3: Final polish with fine-tuning
    if time.time() < deadline - 5:
        for i in range(N):
            if time.time() > deadline - 5:
                break
            current_val = best_h_cd[i]
            if current_val == 0:
                continue
            for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
                trial_val = current_val + delta
                if trial_val < 0:
                    continue
                trial_h = best_h_cd.copy()
                trial_h[i] = trial_val
                trial_obj = compute_score(trial_h)
                if trial_obj < best_val_cd:
                    best_h_cd[i] = trial_val
                    best_val_cd = trial_obj
    
    # Use the best result from either phase
    final_val = compute_score(best_h_cd)
    if final_val < best_val:
        best_h = best_h_cd
        best_val = final_val
    
    return list(best_h)
