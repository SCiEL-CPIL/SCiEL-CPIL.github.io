"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv(x):
    """Full (non-periodic) autoconvolution of x via FFT. Returns array of length 2N-1
    indexed so that AC[0] = lag 0, AC[k] = lag k (k>=1), AC[2N-1-k] = lag -k."""
    n = len(x)
    size = 2 * n
    f = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    # ac[0..n-1] are lags 0..n-1 ; ac[n..2n-1] are lags -(n-1)..-1 (reversed)
    # Build full lag array from -(n-1) to (n-1)
    full = np.empty(2 * n - 1)
    full[0:n] = ac[0:n]          # lags 0..n-1
    full[n:2*n-1] = ac[n-1:0:-1] # lags -(n-1)..-1
    return full


def _autoconv_grad(x):
    """Gradient of each AC lag w.r.t. x[i]. Returns shape (2N-1, N):
    dAC[lag]/dx[i]. For lag L (can be negative), AC(L) = sum_i x[i]*x[i+L] (with x out of range = 0).
    d/dx[i] = x[i+L] + x[i-L]."""
    n = len(x)
    # We'll compute for all lags - (n-1) .. (n-1)
    # Represent lag L as index L + (n-1) in [0, 2n-2]
    grad = np.zeros((2 * n - 1, n))
    # For a given lag L: for i in [0, n): x[i+L] contributes if 0<=i+L<n, x[i-L] contributes if 0<=i-L<n
    # Vectorize: create shifted versions of x
    for L in range(-(n - 1), n):
        idx = L + (n - 1)
        # x[i+L]: i from max(0,-L) to n-1-L ... 
        a = np.zeros(n)
        b = np.zeros(n)
        if L >= 0:
            # x[i+L] valid for i in [0, n-1-L]; x[i-L] valid for i in [L, n-1]
            a[0:n - L] = x[L:n]
            b[L:n] = x[0:n - L]
        else:
            LL = -L
            # x[i+L]=x[i-LL] valid for i in [LL, n-1]; x[i-L]=x[i+LL] valid for i in [0, n-1-LL]
            a[LL:n] = x[0:n - LL]
            b[0:n - LL] = x[LL:n]
        grad[idx, :] = a + b
    return grad


def _soft_max_abs(ac, tau):
    """Smooth surrogate for max|ac| using log-sum-exp over |ac|/tau."""
    a = np.abs(ac) / tau
    # numerical stability
    amax = np.max(a)
    return tau * (amax + np.log(np.sum(np.exp(a - amax))))


def _soft_max_abs_grad(ac, tau):
    """Gradient of soft_max_abs w.r.t. ac."""
    a = np.abs(ac) / tau
    amax = np.max(a)
    e = np.exp(a - amax)
    w = e / np.sum(e)  # softmax weights
    # d/dac = w * sign(ac) / tau * tau = w * sign(ac)
    return w * np.sign(ac)


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s

    n = N
    # Initialize with a smooth sinusoid (one full period) to spread autoconv energy
    i = np.arange(n)
    x = np.sin(2 * np.pi * i / n)
    # Normalize so sum x = 1 (scale invariance of objective)
    s = np.sum(x)
    if abs(s) < 1e-9:
        x = x + 1e-3  # tiny DC to make sum nonzero
        s = np.sum(x)
    x = x / s

    # Precompute gradient structure (dAC/dx) once — it depends only on x values,
    # but x changes, so we recompute the shifted arrays each step. Instead, we
    # compute grad of AC on the fly using the same loop but only need the
    # weighting by the soft-max weights.
    
    def ac_and_grad(x):
        ac = _autoconv(x)
        # gradient of soft_max_abs w.r.t. x:
        # dF/dx[i] = sum_L (dF/dac[L]) * (dAC[L]/dx[i])
        # dF/dac = w * sign(ac)
        w = _soft_max_abs_grad(ac, tau)
        # Now dF/dx[i] = sum_L w[L] * sign(ac[L]) * (x[i+L] + x[i-L])
        # = sum_L v[L] * (x[i+L]+x[i-L]) where v[L]=w[L]*sign(ac[L])
        v = w * np.sign(ac)
        # This is a weighted sum of shifted x. Compute via convolution-like:
        # dF/dx[i] = sum_L v[L] x[i+L] + sum_L v[L] x[i-L]
        # = (v * x)[i] + (v_rev * x)[i]  appropriately
        # Use FFT for speed
        size = 2 * n
        vfull = np.zeros(size)
        # v is indexed 0..2n-2 for lags -(n-1)..(n-1)
        # Place into FFT-friendly: vfull[0] = v[lag 0], vfull[k] = v[lag k], vfull[size-k] = v[lag -k]
        vfull[0] = v[n - 1]  # lag 0
        for k in range(1, n):
            vfull[k] = v[n - 1 + k]      # lag +k
            vfull[size - k] = v[n - 1 - k] # lag -k
        vf = np.fft.rfft(vfull, n=size)
        xf = np.fft.rfft(x, n=size)
        # (v * x) circular conv, but we want linear; size 2n is enough for lags up to n-1
        conv1 = np.fft.irfft(vf * xf, n=size)  # conv1[i] = sum_k vfull[k] x[i-k] mod size
        # We need sum_L v[L] x[i+L]. Note conv1[i] = sum_k vfull[k] x[(i-k) mod size].
        # This is messy. Let me just do direct but vectorized over i using the precomputed shift idea.
        # Given n=400, direct O(n^2) per step = 160k ops, fine.
        dFdx = np.zeros(n)
        for L in range(-(n - 1), n):
            c = v[L + (n - 1)]
            if c == 0.0:
                continue
            if L >= 0:
                dFdx[0:n - L] += c * x[L:n]
                dFdx[L:n] += c * x[0:n - L]
            else:
                LL = -L
                dFdx[LL:n] += c * x[0:n - LL]
                dFdx[0:n - LL] += c * x[LL:n]
        return ac, dFdx

    # Adaptive temperature for soft-max
    tau = 5.0
    # Gradient descent with backtracking
    best_x = x.copy()
    best_val = np.inf
    lr = 1e-3
    it = 0
    while time.time() < deadline:
        ac, dFdx = ac_and_grad(x)
        # enforce sum x = 1 by projecting gradient
        dFdx = dFdx - np.mean(dFdx)  # remove component along 1-vector (constraint)
        # backtracking line search
        f0 = _soft_max_abs(ac, tau)
        step = lr
        improved = False
        for _ in range(20):
            x_try = x - step * dFdx
            x_try = x_try - np.mean(x_try - 1.0)  # re-normalize to sum=1
            ac_try = _autoconv(x_try)
            f_try = _soft_max_abs(ac_try, tau)
            if f_try < f0 - 1e-12:
                x = x_try
                improved = True
                break
            step *= 0.5
        if not improved:
            lr *= 0.5
            if lr < 1e-12:
                break
        # track best by TRUE max|AC|
        ac_now = _autoconv(x)
        val = np.max(np.abs(ac_now))
        if val < best_val:
            best_val = val
            best_x = x.copy()
        # anneal temperature
        tau = max(1.0, tau * 0.98)
        it += 1

    # Final high-precision polish: local sign-flip / coordinate descent on true max|AC|
    x = best_x
    # Coordinate descent: for each i, try flipping sign or adjusting, keep if reduces max|AC|
    for _ in range(3):
        ac = _autoconv(x)
        cur = np.max(np.abs(ac))
        changed = False
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            orig = x[i]
            # try a few candidate values: flip, and small perturbations
            cands = [-orig, orig * 0.5, orig * 1.5, 0.0]
            best_c = orig
            best_v = cur
            for c in cands:
                x[i] = c
                # renormalize
                s = np.sum(x)
                if abs(s) < 1e-12:
                    x[i] = best_c
                    continue
                x = x / s
                ac2 = _autoconv(x)
                v2 = np.max(np.abs(ac2))
                if v2 < best_v - 1e-12:
                    best_v = v2
                    best_c = c
            x[i] = best_c
            if best_c != orig:
                changed = True
            s = np.sum(x)
            if abs(s) > 1e-12:
                x = x / s
        if not changed:
            break

    # Final normalization to sum = 1
    s = np.sum(x)
    if abs(s) > 1e-12:
        x = x / s
    return x
# EVOLVE-BLOCK-END
