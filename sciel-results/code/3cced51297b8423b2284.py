"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_lin(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    conv = np.fft.irfft(X * X.conj(), size)
    return conv[:2 * n - 1]

def _metric_offdiag(x):
    """Max |c_k| for k != 0, divided by (sum x)^2."""
    c = _autoconv_lin(x)
    # c[0] is zero-lag; off-diagonal are c[1:] and c[-1:] symmetric
    # For linear conv of length 2N-1, lags are -(N-1)..(N-1)
    # c[k] for k=1..N-1 are positive lags, c[2N-2-k] for k=1..N-1 are negative lags
    n = len(x)
    off = np.concatenate([c[1:n], c[n+1:2*n-1]])
    max_abs = np.max(np.abs(off)) if len(off) > 0 else 0.0
    s = np.sum(x)
    if abs(s) < 1e-12:
        return max_abs
    return max_abs / (s * s)

def _objective_offdiag(x):
    """Same as metric but for optimization (avoid division issues)."""
    return _metric_offdiag(x)

def construct():
    deadline = time.time() + 110.0  # safety margin
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_score = _metric_offdiag(best_x)
    
    # Phase 1: Spectral flatness gradient descent in continuous domain
    # Goal: make |X(f)|^2 as flat as possible to spread autoconvolution energy
    x = rng.standard_normal(N)
    # Ensure nonzero sum
    x += np.mean(x) * 0  # keep as is
    
    lr = 0.05
    for step in range(20000):
        if time.time() > deadline - 30:
            break
        
        X = np.fft.rfft(x)
        mag2 = np.abs(X) ** 2
        target = np.mean(mag2)
        
        # Gradient of sum (mag2 - target)^2 w.r.t. x
        # d/dx |X_k|^2 = 2 Re(X_k * conj(dX_k/dx))
        # dX_k/dx_j = exp(-2 pi i k j / N)
        # Gradient = 4 * IFFT((mag2 - target) * X) real part
        grad = 4.0 * np.fft.irfft((mag2 - target) * X, N)
        
        x -= lr * grad
        
        # Normalize to keep sum ~ N (large integral)
        s = np.sum(x)
        if abs(s) > 1e-6:
            x = x * (N / abs(s)) * np.sign(s)
        
        # Periodic check
        if step % 2000 == 0:
            score = _metric_offdiag(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
    
    # Phase 2: Round to signs and do local flip search
    x_sign = np.sign(x)
    # Ensure at least some +1s to keep sum large
    if np.sum(x_sign) <= 0:
        x_sign = -x_sign
    
    # Local search: try flipping individual bits
    def _try_flips(x_sign, budget_seconds):
        nonlocal best_x, best_score
        cur = x_sign.copy().astype(float)
        cur_score = _metric_offdiag(cur)
        if cur_score < best_score:
            best_score = cur_score
            best_x = cur.copy()
        
        t_end = time.time() + budget_seconds
        improved = True
        while improved and time.time() < t_end:
            improved = False
            # Try random subset of positions to flip
            n_tries = min(500, N)
            for _ in range(n_tries):
                if time.time() > t_end:
                    break
                i = rng.integers(0, N)
                cur[i] = -cur[i]
                new_score = _metric_offdiag(cur)
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_x = cur.copy()
                else:
                    cur[i] = -cur[i]
        
        return cur
    
    remaining = deadline - time.time()
    if remaining > 5:
        _try_flips(x_sign, remaining - 5)
    
    # Phase 3: Multi-start with different random seeds if time permits
    remaining = deadline - time.time()
    if remaining > 20:
        for seed in [1, 7, 13, 99]:
            if time.time() > deadline - 15:
                break
            rng2 = np.random.default_rng(seed)
            x2 = rng2.standard_normal(N)
            lr2 = 0.05
            for step in range(5000):
                if time.time() > deadline - 15:
                    break
                X2 = np.fft.rfft(x2)
                mag2 = np.abs(X2) ** 2
                target = np.mean(mag2)
                grad2 = 4.0 * np.fft.irfft((mag2 - target) * X2, N)
                x2 -= lr2 * grad2
                s2 = np.sum(x2)
                if abs(s2) > 1e-6:
                    x2 = x2 * (N / abs(s2)) * np.sign(s2)
            
            x2_sign = np.sign(x2)
            if np.sum(x2_sign) <= 0:
                x2_sign = -x2_sign
            _try_flips(x2_sign, min(10, deadline - time.time() - 15))
    
    # Return as signed step heights
    return best_x.astype(float)
# EVOLVE-BLOCK-END
