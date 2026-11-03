"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _overlap_vector(h, N):
    """
    Compute the overlap vector c where c[k] = sum_i h[i] * h[(i+k) % N]
    for k = 0, ..., N-1. This is the circular autocorrelation.
    """
    # Use FFT for efficient computation
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h[i] * h[(i-k) % N]
    # We want c[k] = sum_i h[i] * h[(i+k) % N] = autocorr[N-k] for k>0, autocorr[0] for k=0
    # Actually, let's be careful with the definition.
    # The problem asks for overlap of h with shifted 1-h.
    # Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i] * h[(i+k) % N]
    # = 1 - autocorr_forward[k]
    # where autocorr_forward[k] = sum_i h[i] * h[(i+k) % N]
    
    # np.fft.ifft(H * np.conj(H)) gives circular autocorrelation r[m] = sum_i h[i] h[i-m]
    # So r[0] = sum h[i]^2
    # r[1] = sum h[i] h[i-1] = sum h[i+1] h[i] (by reindexing)
    # We want sum_i h[i] h[i+k]. This is r[-k] = r[N-k].
    
    # Let's compute the forward autocorrelation properly.
    # c[k] = sum_i h[i] h[(i+k) % N]
    # This is the cross-correlation of h with itself, shifted by k.
    # Using FFT: c = ifft(fft(h) * conj(fft(h))) gives r[m] = sum_i h[i] h[i-m].
    # So c[k] = r[-k] = r[N-k] (for k != 0), and c[0] = r[0].
    
    r = np.fft.ifft(H * np.conj(H)).real
    # c[k] = r[(N-k) % N]
    c = r[(N - np.arange(N)) % N]
    
    # Due to floating point, c might have tiny imaginary parts or numerical noise.
    # Also, c[0] should be sum h[i]^2.
    return c

def _max_overlap(h, N):
    """
    Compute the maximum overlap of h with shifted 1-h.
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N]) = 1 - c[k]
    where c[k] = sum_i h[i] h[(i+k) % N]
    So max_overlap = 1 - min_k c[k]
    We want to minimize this, so maximize min_k c[k].
    """
    c = _overlap_vector(h, N)
    min_c = np.min(c)
    return 1.0 - min_c

def _project(h, N):
    """
    Project h onto the constraint set:
    1. 0 <= h[i] <= 1
    2. sum h[i] = N/2
    """
    h = np.clip(h, 0.0, 1.0)
    # Adjust sum to be exactly N/2
    target_sum = N / 2.0
    current_sum = np.sum(h)
    if current_sum > 0 and current_sum < N:
        # We need to adjust. If sum is too high, decrease; if too low, increase.
        # A simple way: add a constant offset, then re-clip.
        # But this might not converge in one step. Let's use a more robust method.
        # We want to find h' = h + delta such that sum(h') = N/2 and 0 <= h' <= 1.
        # This is a projection onto the hyperplane intersected with the box.
        # We can use a bisection on the offset.
        
        # If all h are at bounds, we can't adjust.
        if np.all(h == 0.0) or np.all(h == 1.0):
            return h
        
        # Try to find an offset d such that sum(clip(h+d, 0, 1)) = N/2
        # The function f(d) = sum(clip(h+d, 0, 1)) is continuous and non-decreasing in d.
        # f(min_d) = 0, f(max_d) = N.
        # We can bisect on d.
        
        lo = -1.0
        hi = 1.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            h_new = np.clip(h + mid, 0.0, 1.0)
            s = np.sum(h_new)
            if s < target_sum:
                lo = mid
            else:
                hi = mid
        
        h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final check: ensure sum is exactly N/2 by adjusting the least constrained element
    # This is a fallback if the bisection didn't work perfectly
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        # Find an element that can be adjusted
        if diff > 0:
            # Need to increase sum
            # Find indices where h < 1
            candidates = np.where(h < 1.0)[0]
            if len(candidates) > 0:
                # Add diff to the first candidate (or distribute)
                h[candidates[0]] += diff
                h = np.clip(h, 0.0, 1.0)
        else:
            # Need to decrease sum
            candidates = np.where(h > 0.0)[0]
            if len(candidates) > 0:
                h[candidates[0]] += diff
                h = np.clip(h, 0.0, 1.0)
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe
    
    # Start with the constant solution
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _max_overlap(h, N)  # This should be 1 - 0.25 = 0.75
    
    # We want to minimize max_overlap = 1 - min_c, so maximize min_c.
    # Let's define f(h) = -min_c(h) and minimize f(h).
    # Or directly minimize max_overlap(h).
    
    def objective(x):
        return _max_overlap(x, N)
    
    # Use scipy.optimize.minimize
    # Since the objective is non-smooth (min of linear functions), we can use Nelder-Mead
    # or a gradient-based method with subgradients.
    # Let's try Nelder-Mead first, then polish with a more careful method.
    
    try:
        # Initial perturbation
        np.random.seed(42)
        h_init = np.full(N, 0.5)
        h_init += np.random.normal(0, 0.05, N)
        h_init = _project(h_init, N)
        
        res = opt.minimize(
            objective,
            h_init,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res.fun < best_score:
            best_h = _project(res.x, N)
            best_score = _max_overlap(best_h, N)
        
        # Check time
        if time.time() > deadline - 20.0:
            return best_h
        
        # Now do a local polish using subgradient method
        # The objective is g(h) = 1 - min_k c_k(h)
        # The subgradient of -min_k c_k is -c_k / min_k c_k for the argmin k
        # So the subgradient of g(h) is c_k / min_k c_k for the argmin k
        # Wait, g(h) = 1 - min c_k.
        # Let m(h) = min c_k. g(h) = 1 - m(h).
        # The subdifferential of m(h) at h is the convex hull of gradients of c_k for k in argmin.
        # grad c_k = 2 * (shifted h)
        # So subgrad of g(h) = - subgrad of m(h) = - convex hull of 2 * shifted h for k in argmin.
        # To minimize g(h), we want to move in the negative subgradient direction.
        # So direction = - subgrad g(h) = subgrad m(h).
        # We want to increase m(h), so we move in the direction of the subgradient of m(h).
        
        # Let's implement a simple subgradient method.
        h_curr = best_h.copy()
        step_size = 0.01
        for i in range(10000):
            if time.time() > deadline - 5.0:
                break
            
            c = _overlap_vector(h_curr, N)
            min_c = np.min(c)
            argmin_k = np.argmin(c)
            
            # Gradient of c_k with respect to h
            # c_k = sum_i h[i] h[(i+k) % N]
            # d c_k / d h[j] = h[(j+k) % N] + h[(j-k) % N]
            # (because h[j] appears in term i=j and in term i such that (i+k)%N = j, i.e., i=(j-k)%N)
            
            grad_c_k = np.zeros(N)
            for j in range(N):
                grad_c_k[j] = h_curr[(j + argmin_k) % N] + h_curr[(j - argmin_k) % N]
            
            # We want to increase min_c, so move in direction grad_c_k
            h_new = h_curr + step_size * grad_c_k
            h_new = _project(h_new, N)
            
            new_score = _max_overlap(h_new, N)
            if new_score < best_score:
                best_h = h_new.copy()
                best_score = new_score
                h_curr = h_new.copy()
                # Decrease step size slowly
                step_size *= 0.999
            else:
                # Rejected, decrease step size
                step_size *= 0.95
            
            if step_size < 1e-8:
                break
        
    except Exception as e:
        # If anything goes wrong, return the best we have
        pass
    
    return best_h
# EVOLVE-BLOCK-END
