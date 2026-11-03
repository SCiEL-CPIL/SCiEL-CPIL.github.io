import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _overlap_vector(h):
    """Compute overlap at all circular shifts."""
    # h is shape (N,)
    # overlap at shift s = sum_i h[i] * (1 - h[(i+s) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i+s)%N]
    # sum_i h[i] = N/2 (constant)
    # So O(s) = N/2 - autocorr(s) where autocorr(s) = sum_i h[i]*h[(i+s)%N]
    # Using FFT: autocorr via circular correlation
    H = np.fft.fft(h)
    corr = np.fft.ifft(H * np.conj(H)).real
    # corr[s] = sum_i h[i] * h[(i-s)%N] ... need to check convention
    # Actually np.fft.ifft(fft(x)*conj(fft(x))) gives circular autocorrelation
    # corr[k] = sum_i x[i] * x[(i-k) % N]
    # We want sum_i h[i]*h[(i+s)%N] = corr[-s % N]
    # But since h is real, corr is real and symmetric: corr[k] = corr[N-k]
    # So corr[s] = sum_i h[i]*h[(i-s)%N] = sum_i h[i]*h[(i+s)%N] by symmetry
    return N / 2.0 - corr

def _score(h):
    """Max overlap value."""
    ov = _overlap_vector(h)
    return np.max(ov)

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s
    
    best_h = np.full(N, 0.5)
    best_val = _score(best_h)  # N/4 = 25.0
    
    def objective(x):
        # x is flat, enforce sum constraint via projection
        h = x
        # Project to feasible: clip to [0,1] and adjust to sum N/2
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            # Add diff/N to all, then re-clip
            h = h + diff / N
            h = np.clip(h, 0, 1)
            # If still not exact, do iterative projection
            for _ in range(50):
                diff = N / 2.0 - np.sum(h)
                if abs(diff) < 1e-12:
                    break
                # Adjust only elements that can move
                mask = (h > 0) & (h < 1)
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0, 1)
                else:
                    break
        ov = _overlap_vector(h)
        # Smooth max approximation
        beta = 20.0
        m = np.max(ov)
        if beta > 0:
            smooth_max = m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta
        else:
            smooth_max = m
        return smooth_max
    
    def objective_and_grad(x):
        h = x.copy()
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
        if abs(diff) > 1e-12:
            h = h + diff / N
            h = np.clip(h, 0, 1)
            for _ in range(50):
                diff = N / 2.0 - np.sum(h)
                if abs(diff) < 1e-12:
                    break
                mask = (h > 0) & (h < 1)
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0, 1)
                else:
                    break
        
        ov = _overlap_vector(h)
        beta = 20.0
        m = np.max(ov)
        exp_terms = np.exp(beta * (ov - m))
        smooth_max = m + np.log(np.sum(exp_terms)) / beta
        
        # Gradient of smooth max w.r.t. ov
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of ov w.r.t. h
        # ov[s] = N/2 - corr[s] where corr[s] = sum_i h[i]*h[(i+s)%N]
        # d(ov[s])/d(h[j]) = -d(corr[s])/d(h[j])
        # corr[s] = sum_i h[i]*h[(i+s)%N]
        # d(corr[s])/d(h[j]) = h[(j+s)%N] + h[(j-s)%N]  (by symmetry of the product)
        # Actually: corr[s] = sum_i h[i]*h[(i+s)%N]
        # d/d(h[j]) = h[(j+s)%N] + sum_i h[i]*d(h[(i+s)%N])/d(h[j])
        # d(h[(i+s)%N])/d(h[j]) = 1 if (i+s)%N == j, i.e., i = (j-s)%N
        # So d(corr[s])/d(h[j]) = h[(j+s)%N] + h[(j-s)%N]
        
        # grad_ov[j] = sum_s weights[s] * (-h[(j+s)%N] - h[(j-s)%N])
        # = -sum_s weights[s] * h[(j+s)%N] - sum_s weights[s] * h[(j-s)%N]
        # By symmetry of weights (if weights is symmetric), these are the same
        # But weights may not be symmetric. Let's compute directly.
        
        # grad_ov[j] = -sum_s weights[s] * (h[(j+s)%N] + h[(j-s)%N])
        # = -sum_s weights[s]*h[(j+s)%N] - sum_s weights[s]*h[(j-s)%N]
        # Let u = -s, then second term = sum_u weights[-u]*h[(j+u)%N]
        # If weights is symmetric (weights[s]=weights[-s]), then grad_ov[j] = -2*sum_s weights[s]*h[(j+s)%N]
        
        # Compute via FFT for efficiency
        # sum_s weights[s]*h[(j+s)%N] is circular cross-correlation of weights and h
        W = np.fft.fft(weights)
        H = np.fft.fft(h)
        conv = np.fft.ifft(W * H).real  # conv[j] = sum_s weights[s]*h[(j+s)%N]
        
        # For the second term: sum_s weights[s]*h[(j-s)%N]
        # This is circular cross-correlation with reversed weights
        W_rev = np.fft.fft(weights[::-1]) if N > 1 else W
        # Actually, sum_s weights[s]*h[(j-s)%N] = conv of weights and h with reversed index
        # Let's just compute: this equals sum_s weights[s]*h[(j-s)%N]
        # If we define w_rev[s] = weights[-s % N], then sum_s w_rev[s]*h[(j+s)%N] = sum_s weights[-s]*h[(j+s)%N] = sum_u weights[u]*h[(j-u)%N] where u=-s
        # So it's the same as conv with reversed weights
        
        # Simpler: just compute both terms
        # term1[j] = sum_s weights[s]*h[(j+s)%N]
        # term2[j] = sum_s weights[s]*h[(j-s)%N]
        
        # term1 = ifft(fft(weights) * fft(h))
        # term2 = ifft(fft(weights) * fft(h)) with conjugate? No.
        # term2[j] = sum_s weights[s]*h[(j-s)%N]
        # Let k = j-s, s = j-k: term2[j] = sum_k weights[j-k]*h[k]
        # This is circular convolution of weights and h: (weights * h)[j]
        # while term1[j] = sum_s weights[s]*h[(j+s)%N] = sum_s weights[s]*h[(j+s)%N]
        # Let t = j+s: term1[j] = sum_t weights[j-t]*h[t] ... same as term2!
        # Wait: term1[j] = sum_s weights[s]*h[(j+s)%N]. Let u = (j+s)%N, s = (u-j)%N
        # term1[j] = sum_u weights[(u-j)%N]*h[u]
        # term2[j] = sum_s weights[s]*h[(j-s)%N]. Let v = (j-s)%N, s = (j-v)%N
        # term2[j] = sum_v weights[(j-v)%N]*h[v]
        # These are the same: term1[j] = sum_u weights[(u-j)%N]*h[u] = sum_u weights[(j-u)%N]*h[u] (since weights is real, weights[-x]=weights[N-x]... not necessarily)
        
        # Actually for real signals, the circular autocorrelation is symmetric, but cross-correlation is not.
        # Let me just compute term1 and term2 separately using FFT.
        
        # term1[j] = sum_s weights[s] * h[(j+s)%N]
        # This is the circular cross-correlation. Using FFT:
        # term1 = ifft(fft(weights) * conj(fft(h)))? Let's check.
        # ifft(fft(a) * conj(fft(b)))[k] = sum_i a[i] * b[(i-k)%N]
        # So ifft(fft(weights) * conj(fft(h)))[j] = sum_i weights[i] * h[(i-j)%N] = sum_i weights[i]*h[(i-j)%N]
        # Let s = i: = sum_s weights[s]*h[(s-j)%N] = sum_s weights[s]*h[(-j+s)%N]
        # We want sum_s weights[s]*h[(j+s)%N]. Let t = -s: sum_t weights[-t]*h[(j-t)%N]
        # So term1[j] = ifft(fft(weights_flipped) * conj(fft(h)))[j] where weights_flipped[s] = weights[-s%N]
        
        # This is getting complex. Let me just use a direct approach for N=100 which is fast enough.
        
        # For N=100, direct computation is O(N^2) = 10000 per gradient eval, which is fine.
        grad_ov = np.zeros(N)
        for j in range(N):
            s = np.arange(N)
            term1 = np.dot(weights, h[(j + s) % N])
            term2 = np.dot(weights, h[(j - s) % N])
            grad_ov[j] = -(term1 + term2)
        
        return smooth_max, grad_ov
    
    # Multi-restart optimization
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if r == 0:
            x0 = np.full(N, 0.5)
        else:
            # Random perturbation
            x0 = np.random.uniform(0, 1, N)
            # Project to sum = N/2
            x0 = np.clip(x0, 0, 1)
            for _ in range(100):
                diff = N / 2.0 - np.sum(x0)
                if abs(diff) < 1e-12:
                    break
                mask = (x0 > 0) & (x0 < 1)
                if np.any(mask):
                    x0[mask] += diff / np.sum(mask)
                    x0 = np.clip(x0, 0, 1)
                else:
                    break
        
        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            continue
        
        h_opt = res.x
        h_opt = np.clip(h_opt, 0, 1)
        # Enforce exact sum
        for _ in range(100):
            diff = N / 2.0 - np.sum(h_opt)
            if abs(diff) < 1e-14:
                break
            mask = (h_opt > 1e-15) & (h_opt < 1 - 1e-15)
            if np.any(mask):
                h_opt[mask] += diff / np.sum(mask)
                h_opt = np.clip(h_opt, 0, 1)
            else:
                break
        
        val = _score(h_opt)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Polish pass with higher beta
    if time.time() < deadline - 5:
        def polish_objective(x):
            h = x
            h = np.clip(h, 0, 1)
            diff = N / 2.0 - np.sum(h)
            if abs(diff) > 1e-12:
                h = h + diff / N
                h = np.clip(h, 0, 1)
                for _ in range(50):
                    diff = N / 2.0 - np.sum(h)
                    if abs(diff) < 1e-12:
                        break
                    mask = (h > 0) & (h < 1)
                    if np.any(mask):
                        h[mask] += diff / np.sum(mask)
                        h = np.clip(h, 0, 1)
                    else:
                        break
            ov = _overlap_vector(h)
            beta = 100.0
            m = np.max(ov)
            exp_terms = np.exp(beta * (ov - m))
            smooth_max = m + np.log(np.sum(exp_terms)) / beta
            return smooth_max
        
        try:
            res2 = minimize(polish_objective, best_h, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            h_polish = np.clip(res2.x, 0, 1)
            for _ in range(100):
                diff = N / 2.0 - np.sum(h_polish)
                if abs(diff) < 1e-14:
                    break
                mask = (h_polish > 1e-15) & (h_polish < 1 - 1e-15)
                if np.any(mask):
                    h_polish[mask] += diff / np.sum(mask)
                    h_polish = np.clip(h_polish, 0, 1)
                else:
                    break
            val_polish = _score(h_polish)
            if val_polish < best_val:
                best_val = val_polish
                best_h = h_polish
        except Exception:
            pass
    
    # Final projection to ensure exact sum and bounds
    h_final = np.clip(best_h, 0, 1)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h_final)
        if abs(diff) < 1e-15:
            break
        mask = (h_final > 0) & (h_final < 1)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0, 1)
        else:
            break
    
    return h_final
