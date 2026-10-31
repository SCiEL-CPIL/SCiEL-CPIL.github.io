import time
import math
import numpy as np

N = 100
TARGET_SUM = N / 2.0

def _compute_overlaps(h):
    """Compute overlap[k] = sum_i h[i]*(1-h[(i+k)%N]) for all k."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H) ** 2).real
    overlaps = TARGET_SUM - autocorr
    return overlaps

def _max_overlap(h):
    return np.max(_compute_overlaps(h))

def _smooth_min_autocorr(h, tau=1.0):
    """Smooth approximation of min_k autocorr[k] using log-sum-exp of -autocorr."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H) ** 2).real
    # min autocorr ≈ -logsumexp(-autocorr/tau)*tau
    # For stability, subtract max
    neg_ac = -autocorr
    m = np.max(neg_ac)
    if m > 700:
        m = 700
    lse = m + tau * math.log(np.sum(np.exp((neg_ac - m) / tau)))
    return -lse / tau  # This is -tau*log(sum(exp(-ac/tau))) which approximates min

def _grad_smooth_min_autocorr(h, tau=1.0):
    """Gradient of the smooth min of autocorr w.r.t. h."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H) ** 2).real
    neg_ac = -autocorr
    m = np.max(neg_ac)
    if m > 700:
        m = 700
    weights = np.exp((neg_ac - m) / tau)
    weights /= np.sum(weights)
    # d/dh of -autocorr[k] = -2*IFFT(H * conj(FFT(h_shifted_k)))... 
    # Actually, autocorr[k] = sum_i h[i] h[(i+k)%N]
    # d autocorr[k] / dh[j] = h[(j-k)%N] + h[(j+k)%N] (for circular, but careful)
    # More precisely: autocorr[k] = sum_i h[i] h[(i+k)%N]
    # d/dh[j]: term i=j gives h[(j+k)%N], term i+k=j i.e. i=j-k gives h[j-k]
    # So d autocorr[k]/dh[j] = h[(j+k)%N] + h[(j-k)%N]
    # Gradient of -sum_k w_k * autocorr[k] w.r.t. h[j] = -sum_k w_k (h[(j+k)%N] + h[(j-k)%N])
    # = -2 * sum_k w_k h[(j+k)%N] (since w is symmetric? Not necessarily)
    # Let's compute directly:
    # grad[j] = -sum_k w_k * (h[(j+k)%N] + h[(j-k)%N])
    # = -sum_k w_k h[(j+k)%N] - sum_k w_k h[(j-k)%N]
    # Second term: let m=-k, = -sum_m w_{-m} h[(j+m)%N]
    # So grad[j] = -sum_m (w_m + w_{-m}) h[(j+m)%N]
    # This is a circular convolution of h with (w + w_reversed)
    w_sym = weights + weights[::-1]  # w_m + w_{-m}, note weights[::-1][m] = weights[N-m] = w_{-m}
    # grad = -IFFT(FFT(h) * FFT(w_sym))
    grad = -np.fft.ifft(np.fft.fft(h) * np.fft.fft(w_sym)).real
    return grad

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with theoretical two-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    half = N // 2
    h = np.zeros(N)
    h[:half] = A
    h[half:] = B
    # Fix sum exactly
    diff = TARGET_SUM - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    # Re-fix sum after clip
    diff = TARGET_SUM - np.sum(h)
    # Distribute to elements with most room
    if diff > 0:
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room > 0:
            h += diff * room / total_room
    else:
        total_room = np.sum(h)
        if total_room > 0:
            h += diff * h / total_room
    h = np.clip(h, 0.0, 1.0)
    
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    
    # Subgradient / projected gradient descent on smooth maximin
    # We want to maximize min_k autocorr[k], equivalently minimize max_k overlap[k]
    # Use smooth approximation with decreasing temperature
    
    tau_start = 2.0
    tau_end = 0.05
    n_iters = 3000
    lr_start = 0.005
    lr_end = 0.0005
    
    for it in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        # Anneal temperature and learning rate
        frac = it / max(1, n_iters - 1)
        tau = tau_start * (tau_end / tau_start) ** frac
        lr = lr_start * (lr_end / lr_start) ** frac
        
        # Compute gradient of smooth min autocorr
        g = _grad_smooth_min_autocorr(h, tau)
        
        # We want to maximize min autocorr, so ascent direction is -g
        # But we need to maintain sum constraint and bounds
        # Projected gradient: h_new = h + lr * (-g), then project
        
        # To maintain sum, remove mean of gradient
        g_centered = g - np.mean(g)
        
        h_new = h + lr * (-g_centered)
        
        # Project to [0,1]
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Fix sum
        diff = TARGET_SUM - np.sum(h_new)
        if diff > 0:
            room = 1.0 - h_new
            total_room = np.sum(room)
            if total_room > 1e-10:
                h_new += diff * room / total_room
            else:
                h_new += diff / N
        else:
            total_room = np.sum(h_new)
            if total_room > 1e-10:
                h_new += diff * h_new / total_room
            else:
                h_new += diff / N
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Final sum fix
        diff = TARGET_SUM - np.sum(h_new)
        if abs(diff) > 1e-12:
            # Add to element with most room (or least for negative)
            if diff > 0:
                idx = np.argmax(1.0 - h_new)
                add = min(diff, 1.0 - h_new[idx])
                h_new[idx] += add
                remaining = diff - add
                if remaining > 1e-12:
                    # Distribute among others
                    mask = np.ones(N, dtype=bool)
                    mask[idx] = False
                    room = 1.0 - h_new[mask]
                    total_room = np.sum(room)
                    if total_room > 0:
                        h_new[mask] += remaining * room / total_room
            else:
                idx = np.argmin(h_new)
                sub = min(-diff, h_new[idx])
                h_new[idx] -= sub
                remaining = -diff - sub
                if remaining > 1e-12:
                    mask = np.ones(N, dtype=bool)
                    mask[idx] = False
                    total_room = np.sum(h_new[mask])
                    if total_room > 0:
                        h_new[mask] -= remaining * h_new[mask] / total_room
        
        h_new = np.clip(h_new, 0.0, 1.0)
        
        score_new = _max_overlap(h_new)
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
        
        h = h_new
    
    # Final polish with a few more iterations at low tau
    if time.time() < deadline - 10:
        for it in range(500):
            if time.time() > deadline - 3:
                break
            tau = 0.03
            lr = 0.0002
            g = _grad_smooth_min_autocorr(h, tau)
            g_centered = g - np.mean(g)
            h_new = h + lr * (-g_centered)
            h_new = np.clip(h_new, 0.0, 1.0)
            diff = TARGET_SUM - np.sum(h_new)
            if diff > 0:
                room = 1.0 - h_new
                total_room = np.sum(room)
                if total_room > 1e-10:
                    h_new += diff * room / total_room
            else:
                total_room = np.sum(h_new)
                if total_room > 1e-10:
                    h_new += diff * h_new / total_room
            h_new = np.clip(h_new, 0.0, 1.0)
            diff = TARGET_SUM - np.sum(h_new)
            if abs(diff) > 1e-12:
                if diff > 0:
                    idx = np.argmax(1.0 - h_new)
                    h_new[idx] += min(diff, 1.0 - h_new[idx])
                else:
                    idx = np.argmin(h_new)
                    h_new[idx] += max(diff, -h_new[idx])
            h_new = np.clip(h_new, 0.0, 1.0)
            
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
            h = h_new
    
    # Final enforcement
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-10:
        if diff > 0:
            room = 1.0 - best_h
            total_room = np.sum(room)
            if total_room > 0:
                best_h += diff * room / total_room
        else:
            total_room = np.sum(best_h)
            if total_room > 0:
                best_h += diff * best_h / total_room
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # Verify
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Heights out of bounds"
    assert abs(np.sum(best_h) - TARGET_SUM) < 1e-8, f"Sum constraint violated: {np.sum(best_h)} vs {TARGET_SUM}"
    
    return best_h
