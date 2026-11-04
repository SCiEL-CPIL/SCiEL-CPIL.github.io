"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft


def compute_value(h):
    """Compute the objective value for a given h."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))


def construct():
    """Construct step heights to minimize the autocorrelation inequality constant."""
    deadline = time.time() + 120.0  # 120s safety margin
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    # Try various structured patterns
    
    # Pattern 1: Alternating signs
    h_alt = np.array([1.0, -1.0] * (N // 2))
    val = compute_value(h_alt)
    if val < best_val:
        best_val = val
        best_h = h_alt.copy()
    
    # Pattern 2: Block structure - positive block then negative block
    for split in range(1, N):
        h_block = np.zeros(N)
        h_block[:split] = 1.0
        h_block[split:] = -1.0
        val = compute_value(h_block)
        if val < best_val:
            best_val = val
            best_h = h_block.copy()
    
    # Pattern 3: Ramp
    h_ramp = np.linspace(1, -1, N)
    val = compute_value(h_ramp)
    if val < best_val:
        best_val = val
        best_h = h_ramp.copy()
    
    # Pattern 4: Sinusoidal
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h_sin = np.cos(t)
    val = compute_value(h_sin)
    if val < best_val:
        best_val = val
        best_h = h_sin.copy()
    
    h_sin2 = np.sin(t)
    val = compute_value(h_sin2)
    if val < best_val:
        best_val = val
        best_h = h_sin2.copy()
    
    # Pattern 5: Two-frequency
    h_2f = np.cos(t) + 0.5 * np.cos(3 * t)
    val = compute_value(h_2f)
    if val < best_val:
        best_val = val
        best_h = h_2f.copy()
    
    # Pattern 6: Random with optimization
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        h_rand = np.random.randn(N)
        # Normalize to have non-zero sum
        s = np.sum(h_rand)
        if abs(s) < 1e-10:
            continue
        h_norm = h_rand / s * N  # Scale so sum = N
        
        # Local optimization using scipy
        def objective(x):
            h = x
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10
            conv = np.convolve(h, h, mode='full')
            return abs(2 * N * np.max(conv) / (s ** 2))
        
        try:
            res = minimize(objective, h_norm, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_h = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 20:
            break
    
    # Pattern 7: Try to make convolve entries as negative as possible
    # If we can make all convolve entries negative, max would be close to 0
    # But sum(h) != 0 required
    
    # Try: h with large negative autocorrelation
    # Use FFT-based approach to design h
    # We want max of conv(h,h) to be as small as possible (ideally negative)
    
    # Key insight: conv(h,h)[k] = sum_i h[i] * h[i+k-N+1] (with appropriate indexing)
    # The center element (index N-1) is sum(h^2) which is always >= 0
    # So max(conv) >= sum(h^2) > 0 always
    # Therefore value >= 2*N*sum(h^2)/sum(h)^2 >= 2 by Cauchy-Schwarz
    
    # Wait, this means 2 is indeed the lower bound!
    # But p6 got 1.1774... Let me re-examine
    
    # Actually, for the full convolve, the center is sum(h^2), but there might be
    # larger values at other lags if h has structure
    
    # Let me try: h that is mostly zero with a few large values
    for n_nonzero in [2, 3, 4, 5, 10, 20, 50]:
        if time.time() > deadline - 10:
            break
        for trial in range(3):
            idx = np.random.choice(N, n_nonzero, replace=False)
            h_spar = np.zeros(N)
            h_spar[idx] = np.random.choice([-1, 1], n_nonzero)
            val = compute_value(h_spar)
            if val < best_val:
                best_val = val
                best_h = h_spar.copy()
    
    # Pattern 8: Use the fact that we can make some convolve entries very negative
    # while keeping sum(h) positive
    # Try: h = [1]*k + [-a]*(N-k) where we optimize a and k
    
    for k in range(1, N, 5):
        if time.time() > deadline - 5:
            break
        # Optimize a for this k
        best_a = 1.0
        best_val_k = np.inf
        for a in np.linspace(0.1, 5.0, 50):
            h_test = np.zeros(N)
            h_test[:k] = 1.0
            h_test[k:] = -a
            val = compute_value(h_test)
            if val < best_val_k:
                best_val_k = val
                best_a = a
        if best_val_k < best_val:
            best_val = best_val_k
            h_test = np.zeros(N)
            h_test[:k] = 1.0
            h_test[k:] = -best_a
            best_h = h_test.copy()
    
    return best_h
# EVOLVE-BLOCK-END
