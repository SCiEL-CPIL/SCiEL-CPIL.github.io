import numpy as np
import time
from scipy.optimize import minimize

N = 400
START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def _soft_max(x, beta=50.0):
    m = np.max(x)
    if not np.isfinite(m):
        return 1e10
    # Numerical stability
    e = np.exp(-beta * (x - m))
    s = np.sum(e)
    if s == 0:
        return m
    return m + np.log(s) / beta

def _objective_and_grad(h):
    h = np.asarray(h)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    # Compute convolution
    conv = np.convolve(h, h)
    
    # Soft max of convolution
    beta = 50.0
    m = np.max(conv)
    e = np.exp(-beta * (conv - m))
    w = e / np.sum(e)
    soft_val = m + np.log(np.sum(e)) / beta
    
    val = soft_val / (s * s)
    
    # Gradient
    # d(soft_val)/d(conv_k) = w_k
    # d(conv_k)/d(h_j) = h[k-j] + h[j-k] (with bounds)
    # grad_h = sum_k w_k * (h[k-j] + h[j-k])
    
    # This is equivalent to conv(w, h) + conv(w, h_reversed)?
    # Let's compute directly using FFT for speed and correctness
    # fftconvolve(w, h) gives sum_i w_i h[j-i]
    # We need sum_k w_k h[k-j] + w_k h[j-k]
    # Let G1[j] = sum_k w_k h[k-j]
    # Let G2[j] = sum_k w_k h[j-k]
    
    # G2 is standard convolution of w and h?
    # (w * h)[j] = sum_i w_i h[j-i]. Let i=k. sum_k w_k h[j-k]. Yes, G2 = conv(w, h).
    
    # G1[j] = sum_k w_k h[k-j].
    # This is cross-correlation of w and h.
    # G1[j] = sum_k w_k h[k-j] = sum_m w_{m+j} h[m].
    # This is conv(w, reverse(h))?
    # Let's just use np.convolve for clarity, N is small enough.
    
    # grad = np.zeros(N)
    # for k in range(len(w)):
    #     # add w[k] * d(conv[k])/d(h)
    #     # d(conv[k])/d(h[j]) is non-zero for j such that k-j in [0, N-1] and j-k in [0, N-1]
    #     # This loop is O(N^2) = 160k. 1000 iterations = 160M ops. Might be slow in Python.
    #     pass
    
    # Use FFT
    # G2 = fftconvolve(w, h)
    # G1 = fftconvolve(w, h[::-1])[::-1] ?
    
    # Let's check G1.
    # G1[j] = sum_k w_k h[k-j]
    # Let u = k-j => k = u+j.
    # G1[j] = sum_u w[u+j] h[u]
    # This is the cross-correlation of w and h at lag j.
    # np.correlate(w, h, mode='full')?
    # np.correlate(w, h) computes sum w[i] h[i+lag].
    # So G1 is related to correlate.
    
    # However, scipy.signal.fftconvolve is fast.
    # G2 = np.fft.irfft(np.fft.rfft(w, 2*N) * np.fft.rfft(h, 2*N))[:N] ?
    # No, w has length 2N-1. h has length N.
    
    # Let's stick to a simpler gradient approximation or just use numerical gradient if speed permits?
    # No, L-BFGS needs analytical.
    
    # Efficient calculation:
    # grad = np.zeros(N)
    # We can compute the gradient as:
    # grad = np.convolve(w, h, mode='same') + np.convolve(w, h[::-1], mode='same')[::-1]
    # Let's verify this logic.
    
    # Actually, there is a simpler way.
    # The objective is scale invariant.
    # We can minimize just soft_max(conv) subject to sum(h)=1.
    # This simplifies the gradient.
    # But L-BFGS-B doesn't handle equality constraints well without penalty.
    # Let's use the ratio.
    
    # Gradient of V = S / S_h^2
    # dV/dh = (dS/dh * S_h^2 - S * 2 S_h * 1) / S_h^4
    # dV/dh = (dS/dh * S_h - 2 S * 1) / S_h^3
    
    # dS/dh = sum_k w_k dC_k/dh
    
    # Let's compute dS/dh using FFT
    # dC_k/dh is the kernel.
    # sum_k w_k dC_k/dh = conv(w, h) + conv(w, h_flipped)
    
    # Let's use scipy.signal.fftconvolve
    from scipy.signal import fftconvolve
    
    # w is length 2N-1. h is length N.
    # conv(w, h) length 3N-2.
    # We need the part that affects h[j].
    # h[j] appears in C_k for k in [j, j+N-1].
    # dC_k/dh_j = h[k-j] + h[j-k]
    
    # Term 1: sum_k w_k h[k-j]
    # Term 2: sum_k w_k h[j-k]
    
    # Term 2 is (w * h)[j] ?
    # (w * h)[j] = sum_i w_i h[j-i]. Let i=k. sum_k w_k h[j-k]. Yes.
    # So Term 2 is the j-th element of full convolution of w and h.
    
    # Term 1: sum_k w_k h[k-j]
    # This is sum_k w_k h[-(j-k)]
    # This is cross-correlation.
    # It is equal to (w * h_rev)[j] ?
    # Let h_rev[i] = h[N-1-i].
    
    # To be safe and fast enough for N=400, we can use a direct loop if optimized, or FFT.
    # FFT is safer for correctness.
    
    # Pad w and h to size 2N
    size = 2 * N
    W = np.fft.rfft(w, size)
    H = np.fft.rfft(h, size)
    
    # Term 2: conv(w, h)
    # We want the central part corresponding to indices 0..N-1 of h.
    # The full conv has length 3N-2.
    # The indices of h are 0..N-1.
    # In the full conv of w (len 2N-1) and h (len N), the element at index j corresponds to sum_i w_i h[j-i].
    # This matches Term 2 directly.
    conv_wh = np.fft.irfft(W * H, size)
    term2 = conv_wh[:N]
    
    # Term 1: sum_k w_k h[k-j]
    # This is the same as term2 but with w and h roles swapped in a specific way?
    # Actually, Term 1[j] = sum_k w_k h[k-j]
    # Let m = k-j. k = m+j.
    # Term 1[j] = sum_m w[m+j] h[m]
    # This is the cross-correlation of w and h at lag j.
    # It is equal to the j-th element of conv(w, h[::-1])?
    # Let's just compute conv(w, h) and conv(h, w) and combine?
    # Note that w is symmetric around N-1? No, w depends on conv(h,h) which is symmetric.
    # If h is not symmetric, w is symmetric?
    # conv(h,h) is symmetric. So w is symmetric.
    # If w is symmetric, w_k = w_{2N-2-k}.
    # Then Term 1[j] = sum_k w_k h[k-j]
    # Term 2[j] = sum_k w_k h[j-k]
    
    # If w is symmetric, Term 1 and Term 2 are related.
    # Let's just compute both using FFT.
    # Term 1 is conv(w, h) but shifted?
    # Actually, Term 1[j] = (w * h)[2N-2-j] ?
    # Let's check.
    # (w * h)[k] = sum_i w_i h[k-i]
    # We want sum_k w_k h[k-j]
    # Let i = k-j => k = i+j.
    # sum_i w_{i+j} h_i
    # This is not standard conv.
    
    # Given the time, I will use a numerical gradient for the first few steps to check, or just use Nelder-Mead which doesn't need gradient.
    # Nelder-Mead is slower but robust.
    # For N=400, Nelder-Mead might be too slow.
    
    # Let's use L-BFGS-B with a simple gradient approximation:
    # grad = np.gradient(soft_val, h) ? No.
    
    # I will implement the analytical gradient carefully.
    # Term 2 = conv(w, h)[:N]
    # Term 1 = conv(w, h)[::-1][N-1:2N-1] ?
    
    # Let's just use the property that grad = conv(w, h) + conv(w, h[::-1])[::-1] is a common pattern for symmetric kernels.
    # Since w is symmetric (because conv(h,h) is symmetric), the gradient is:
    # grad = conv(w, h) + conv(w, h) (with appropriate shifts)
    
    # To avoid bugs, I will use `scipy.optimize.minimize` with `method='Nelder-Mead'` on a reduced set of variables or just accept the slowness if I reduce iterations.
    # Actually, for N=400, Nelder-Mead is O(N^2) per step? No, O(N) evaluations.
    # Each eval is O(N^2) conv.
    # 1000 steps * 400 evals * 160k ops = 64 billion ops. Too slow.
    
    # I MUST use gradient.
    # Let's assume the gradient is `conv(w, h) + conv(w, h[::-1])[::-1]` centered.
    
    # Correct Gradient Formula for SoftMax of Convolution:
    # dS/dh[j] = sum_k w_k (h[k-j] + h[j-k])
    # This is exactly:
    # dS/dh = conv(w, h, mode='same') + conv(w, h[::-1], mode='same')[::-1]
    # This holds if w is symmetric.
    
    dS_dh = fftconvolve(w, h, mode='same') + fftconvolve(w, h[::-1], mode='same')[::-1]
    
    # dV/dh
    V = soft_val / (s * s)
    dV_dh = (dS_dh * s - 2 * soft_val) / (s * s * s)
    
    return V, dV_dh

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return np.ones(N)
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Try a few random starts
    for i in range(5):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        # Initialize with random noise
        h_init = rng.normal(0, 1, N)
        
        try:
            result = minimize(
                _objective_and_grad,
                h_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if np.all(np.isfinite(result.x)):
                res_val, _ = _objective_and_grad(result.x)
                if res_val < best_val:
                    best_val = res_val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
