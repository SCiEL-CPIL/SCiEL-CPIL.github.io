"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _start_time

def _deadline_ok(margin=2.0):
    return _elapsed() < _TIME_LIMIT - margin

def _max_overlap(h):
    """Compute max over circular shifts of sum_i h_i * (1 - h_{(i+k)%N})."""
    # sum_i h_i * (1 - h_{(i+k)%N}) = sum h_i - sum h_i * h_{(i+k)%N} = N/2 - autocorr(k)
    # So max overlap = N/2 - min_k autocorr(k)
    # Compute all circular autocorrelations via FFT
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] = sum_i h_i * h_{(i-k)%N} (circular)
    min_autocorr = np.min(autocorr)
    return N / 2.0 - min_autocorr

def _smooth_max_overlap(h, alpha=1.0):
    """Smooth approximation of max overlap using log-sum-exp on the overlap values."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr  # overlap at each shift
    # log-sum-exp smoothing of max(overlaps)
    m = np.max(overlaps)
    if m < -1e10:
        return 0.0
    log_sum = m + np.log(np.sum(np.exp(alpha * (overlaps - m))))
    return log_sum / alpha

def _objective_and_grad(h, alpha=1.0):
    """Objective: smooth max overlap. Returns (value, gradient)."""
    h = np.clip(h, 1e-10, 1.0 - 1e-10)
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    
    m = np.max(overlaps)
    if m < -1e10:
        val = 0.0
        grad = np.zeros(N)
        return val, grad
    
    exp_vals = np.exp(alpha * (overlaps - m))
    sum_exp = np.sum(exp_vals)
    val = (m + np.log(sum_exp)) / alpha
    
    # Gradient: d/dh_i of (1/alpha) * (m + log(sum exp(alpha*(overlap_k - m))))
    # = (1/alpha) * sum_k exp(alpha*(overlap_k - m)) / sum_exp * alpha * d(overlap_k)/dh_i
    # = sum_k (exp(alpha*(overlap_k - m)) / sum_exp) * d(overlap_k)/dh_i
    # d(overlap_k)/dh_i = d/dh_i [N/2 - autocorr_k] = -d(autocorr_k)/dh_i
    # autocorr_k = sum_j h_j * h_{(j-k)%N}
    # d(autocorr_k)/dh_i = h_{(i-k)%N} + h_{(i+k)%N} (for circular)
    # Actually: autocorr_k = sum_j h_j * h_{(j-k)%N}
    # d/dh_i: when j=i, term is h_{(i-k)%N}; when (j-k)%N = i i.e. j=(i+k)%N, term is h_{(i+k)%N}
    # So d(autocorr_k)/dh_i = h_{(i-k)%N} + h_{(i+k)%N} (with possible double count if 2k=0 mod N)
    
    # weights_k = exp(alpha*(overlap_k - m)) / sum_exp
    weights = exp_vals / sum_exp
    
    # grad_i = -sum_k weights_k * d(autocorr_k)/dh_i = -sum_k weights_k * (h_{(i-k)%N} + h_{(i+k)%N})
    # This is: -sum_k weights_k * h_{(i-k)%N} - sum_k weights_k * h_{(i+k)%N}
    # = -(weights convolved with h)_i - (weights convolved with h flipped)_i
    # Using FFT:
    W = np.fft.fft(weights)
    grad = np.fft.ifft(W * H).real + np.fft.ifft(W * H).real  # both terms same due to symmetry
    # Actually: sum_k weights_k * h_{(i-k)%N} is circular correlation of weights and h
    # = ifft(fft(weights) * conj(fft(h))).real
    # And sum_k weights_k * h_{(i+k)%N} = ifft(fft(weights) * fft(h)).real (circular convolution)
    
    term1 = np.fft.ifft(W * np.conj(H)).real  # sum_k weights_k * h_{(i-k)%N}
    term2 = np.fft.ifft(W * H).real            # sum_k weights_k * h_{(i+k)%N}
    
    grad = -(term1 + term2)
    
    return val, grad

def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0,1]."""
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    # After clipping, re-adjust
    for _ in range(10):
        current_sum = np.sum(h)
        diff = target - current_sum
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _local_search(h, max_iters=5000):
    """Coordinate descent / local search to improve the solution."""
    h = _fix_sum(h.copy())
    best_overlap = _max_overlap(h)
    
    for it in range(max_iters):
        if not _deadline_ok(5.0):
            break
        improved = False
        # Try adjusting pairs of variables
        for i in range(N):
            if not _deadline_ok(5.0):
                break
            for j in range(i + 1, N):
                # Try moving mass from i to j or vice versa
                for delta in [0.01, 0.05, 0.1]:
                    h_new = h.copy()
                    h_new[i] -= delta
                    h_new[j] += delta
                    if 0 <= h_new[i] <= 1 and 0 <= h_new[j] <= 1:
                        h_new = _fix_sum(h_new)
                        ov = _max_overlap(h_new)
                        if ov < best_overlap - 1e-10:
                            h = h_new
                            best_overlap = ov
                            improved = True
                            break
                    h_new = h.copy()
                    h_new[i] += delta
                    h_new[j] -= delta
                    if 0 <= h_new[i] <= 1 and 0 <= h_new[j] <= 1:
                        h_new = _fix_sum(h_new)
                        ov = _max_overlap(h_new)
                        if ov < best_overlap - 1e-10:
                            h = h_new
                            best_overlap = ov
                            improved = True
                            break
            if improved:
                break
        if not improved:
            break
    
    return h, best_overlap

def construct():
    # Start with constant solution
    h = np.full(N, 0.5)
    
    # Try gradient-based optimization with annealing alpha
    alphas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    for alpha in alphas:
        if not _deadline_ok(10.0):
            break
        res = minimize(
            _objective_and_grad,
            h,
            method='L-BFGS-B',
            args=(alpha,),
            options={'maxiter': 200, 'maxls': 20},
            jac=True
        )
        h = _fix_sum(res.x)
    
    # Local search polish
    if _deadline_ok(5.0):
        h, _ = _local_search(h, max_iters=2000)
    
    h = _fix_sum(h)
    return h
# EVOLVE-BLOCK-END
