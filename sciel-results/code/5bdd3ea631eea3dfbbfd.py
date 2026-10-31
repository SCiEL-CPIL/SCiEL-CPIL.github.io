"""Combined program: Legendre symbol initialization + local sign-flip + smooth min-max polish.

Strategy:
1. Initialize with Legendre symbol sequences (proven low off-peak autocorrelation)
2. Apply local sign-flip optimization to refine the ±1 sequence
3. Optionally apply smooth min-max polish for continuous refinement
"""

import time
import numpy as np

N = 400
_DEADLINE = time.time() + 140.0
_EPS = 1e-12


def _remaining():
    return _DEADLINE - time.time()


def _autoconv(x):
    """Circular autoconvolution via FFT (real, symmetric)."""
    n = x.size
    X = np.fft.rfft(x)
    r = np.fft.irfft(X * np.conj(X), n=n)
    return r


def _integral(x):
    return float(np.sum(x))


def _score(x):
    """C3' = max |autoconv| / (sum x)^2  (lower is better)."""
    s = _integral(x)
    if abs(s) < _EPS:
        return np.inf
    r = _autoconv(x)
    return float(np.max(np.abs(r))) / (s * s)


def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p."""
    if p == 2:
        return 1 if n % 4 == 1 else -1
    # Use Euler's criterion: (n/p) = n^((p-1)/2) mod p
    result = pow(n % p, (p - 1) // 2, p)
    return 1 if result == 1 else -1


def _build_legendre_sequence(length, prime):
    """Build a Legendre symbol sequence of given length from a prime."""
    # Use the first 'length' non-zero residues modulo prime
    seq = []
    for i in range(1, length + 1):
        if i < prime:
            seq.append(_legendre_symbol(i, prime))
        else:
            # Wrap around or use a different approach
            seq.append(_legendre_symbol(i % prime, prime) if i % prime != 0 else 1)
    return np.array(seq[:length], dtype=np.float64)


def _find_good_prime(length):
    """Find a prime slightly larger than length for Legendre symbol construction."""
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(np.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    p = length + 1
    while not is_prime(p):
        p += 1
    return p


def _local_sign_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    """
    h = h.copy()
    best_val = _score(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping sign
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val


def _smooth_minmax_polish(x, budget_s):
    """Smooth min-max (log-sum-exp) local polish for continuous values."""
    best_x = x.copy()
    best_val = _score(best_x)
    s0 = _integral(x)

    def grad_L(xk, tau, beta):
        r = _autoconv(xk)
        a = np.abs(r)
        m = a.max()
        z = np.exp((a - m) / tau)
        lse = m / tau + np.log(z.sum())
        w = z / z.sum() * (1.0 / tau)
        sign_r = np.sign(r)
        G = np.fft.fft(w * sign_r)
        Xf = np.fft.fft(xk)
        grad = np.real(np.fft.ifft(G * Xf))
        ssq = float(np.sum(xk * xk)) + _EPS
        grad = grad + (2.0 * beta * xk) / ssq
        return grad, lse

    tau = max(0.05, 0.2 * (np.max(np.abs(_autoconv(x))) + _EPS))
    beta = 1e-4
    step = 0.02
    while _remaining() > budget_s:
        g, _ = grad_L(x, tau, beta)
        g = g - g.mean()
        x_new = x - step * g
        x_new = x_new + (s0 - _integral(x_new)) / N
        if not np.all(np.isfinite(x_new)):
            break
        v = _score(x_new)
        if v < best_val - 1e-12:
            best_x = x_new.copy()
            best_val = v
            x = x_new
            step = min(step * 1.05, 0.5)
        else:
            step *= 0.8
            if step < 1e-5:
                tau *= 0.85
                step = 0.05
        if tau < 1e-3:
            break
    return best_x, best_val


def construct():
    t_start = time.time()
    
    # 1) Build Legendre symbol sequence
    prime = _find_good_prime(N)
    x_legendre = _build_legendre_sequence(N, prime)
    
    # Ensure positive integral
    if _integral(x_legendre) < 0:
        x_legendre = -x_legendre
    
    best_x = x_legendre.copy()
    best_val = _score(best_x)
    
    # 2) Local sign-flip optimization on the ±1 sequence
    if _remaining() > 30.0:
        deadline_flip = t_start + 110.0  # Leave time for polish
        x_flip, val_flip = _local_sign_flip(x_legendre, max_passes=20, deadline=deadline_flip)
        if val_flip < best_val:
            best_x = x_flip
            best_val = val_flip
    
    # 3) Smooth min-max polish for continuous refinement (if time remains)
    if _remaining() > 15.0:
        # Convert to continuous values for polish
        x_continuous = best_x.astype(np.float64)
        # Add a small constant to make it near-constant positive
        x_continuous = x_continuous - x_continuous.mean() + 1.0
        
        polish_budget = min(10.0, _remaining() - 5.0)
        if polish_budget > 2.0:
            px, pv = _smooth_minmax_polish(x_continuous, polish_budget)
            if pv < best_val:
                best_x = px
                best_val = pv
    
    # 4) Final re-centering
    if _remaining() > 2.0:
        s0 = _integral(best_x)
        best_x = best_x + (s0 - _integral(best_x)) / N
    
    # Guard
    if (not np.all(np.isfinite(best_x))) or abs(_integral(best_x)) < _EPS:
        return np.ones(N)
    
    return best_x
