import numpy as np
import time

N = 400
DEADLINE = 140.0  # 10s safety margin

def _autocorr(s):
    """Compute autocorrelation via FFT."""
    n = len(s)
    size = 1
    while size < 2*n - 1:
        size *= 2
    f = np.fft.rfft(s, size)
    power = f * np.conj(f)
    ac = np.fft.irfft(power, size)[:2*n-1]
    # For correlation: R[k] = sum_i s[i]*s[i+k], this is ac[k] for k>=0
    # Actually irfft of |F|^2 gives circular correlation, need to handle linear
    # For linear autocorrelation: R[k] = sum_{i=0}^{n-1-k} s[i]*s[i+k]
    # The FFT method: zero-pad to 2n-1, compute, take first n
    return ac[:n]

def _soft_max_abs(ac, tau=1.0):
    """Smooth approximation of max |ac[k]|."""
    abs_ac = np.abs(ac)
    return np.logaddexp.reduce(np.log(1.0 + tau * abs_ac)) / tau if tau > 0 else np.max(abs_ac)

def construct():
    start = time.time()
    
    # Initialize with all ones (seed)
    s = np.ones(N)
    best_s = s.copy()
    
    # Compute seed value
    def _value(s):
        ac = _autocorr(s)
        sm = np.sum(s)
        if abs(sm) < 1e-10:
            return np.inf
        return np.max(np.abs(ac)) / (sm * sm)
    
    best_val = _value(best_s)
    
    # Gradient descent on real-valued s
    # Normalize: fix sum(s) = N (same as seed) to keep denominator fixed
    # Then minimize max |ac|
    tau = 0.1  # softness parameter
    lr = 0.01
    steps = 50000
    
    s = np.ones(N)
    # Add small random perturbation
    s = s + 0.01 * np.random.randn(N)
    
    for step in range(steps):
        if time.time() > DEADLINE:
            break
        
        ac = _autocorr(s)
        abs_ac = np.abs(ac)
        sm = np.sum(s)
        
        # Soft max surrogate
        # L = log(sum exp(tau * abs_ac)) / tau
        m = np.max(tau * abs_ac)
        logsumexp = m + np.log(np.sum(np.exp(tau * abs_ac - m)))
        L = logsumexp / tau
        
        # Gradient of L w.r.t. ac
        # dL/dac[k] = (tau * exp(tau*|ac[k]|) / sum(exp(tau*|ac|))) * sign(ac[k])
        weights = np.exp(tau * abs_ac - m) / np.sum(np.exp(tau * abs_ac - m))
        grad_ac = tau * weights * np.sign(ac)
        
        # Gradient of ac w.r.t. s
        # ac[k] = sum_i s[i]*s[i+k]
        # dac[k]/ds[j] = s[j+k] (if j+k < N) + s[j-k] (if j-k >= 0)
        # This is a convolution-like operation
        # grad_s = ifft of (something) — use FFT
        # Actually, let's compute directly for N=400 (O(N^2) = 160K ops, fast)
        grad_s = np.zeros(N)
        for k in range(N):
            if abs(grad_ac[k]) > 1e-12:
                # dac[k]/ds[j] = s[j+k] for j=0..N-1-k, and s[j-k] for j=k..N-1
                # So grad_s[j] += grad_ac[k] * (s[j+k] if j+k<N else 0) + grad_ac[k] * (s[j-k] if j-k>=0 else 0)
                # This is O(N) per k, total O(N^2)
                for j in range(N - k):
                    grad_s[j] += grad_ac[k] * s[j + k]
                    grad_s[j + k] += grad_ac[k] * s[j]
        
        # Normalize gradient to keep sum(s) constant
        grad_s -= np.mean(grad_s)
        
        s = s - lr * grad_s
        s -= np.mean(s)  # keep mean zero deviation? No, keep sum = N
        
        # Actually, let's keep sum(s) = N
        s += (N - np.sum(s)) / N
        
        val = _value(s)
        if val < best_val:
            best_val = val
            best_s = s.copy()
    
    # Round to ±1
    s_rounded = np.sign(best_s)
    s_rounded[s_rounded == 0] = 1
    
    # Local search: flip individual bits to improve
    def _val_rounded(s):
        ac = _autocorr(s)
        sm = np.sum(s)
        if abs(sm) < 1e-10:
            return np.inf
        return np.max(np.abs(ac)) / (sm * sm)
    
    best_rounded = s_rounded.copy()
    best_rval = _val_rounded(best_rounded)
    
    # Simulated annealing / local search
    T = 0.1
    for i in range(500000):
        if time.time() > DEADLINE:
            break
        j = np.random.randint(N)
        s_try = best_rounded.copy()
        s_try[j] *= -1
        v = _val_rounded(s_try)
        if v < best_rval:
            best_rounded = s_try
            best_rval = v
        elif np.random.rand() < np.exp(-(v - best_rval) / T):
            best_rounded = s_try
            best_rval = v
        T *= 0.9999
    
    return best_rounded
