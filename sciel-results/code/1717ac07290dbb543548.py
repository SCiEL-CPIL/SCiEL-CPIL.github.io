import numpy as np

N = 400


def construct():
    # Design a step-height vector whose autoconvolution has a small maximum
    # relative to (sum h)^2 / N, aiming to beat the theoretical constant-2 bound
    # via a structured sign-changing pattern with a small net sum.
    #
    # Strategy: use a low-frequency cosine pattern that is positive in the
    # first half and negative in the second half, with a tiny DC offset so the
    # sum is small but non-zero. The autoconvolution of such a pattern will
    # have its largest magnitude at non-zero lags, and the peak at lag 0 will
    # be reduced relative to the sum squared.
    #
    # h[i] = cos(2*pi*i/N) + eps, with eps small.
    # sum(h) = sum(cos) + N*eps = 0 + N*eps = N*eps (since sum of cos over a full period is ~0)
    # The autoconvolution at lag 0: sum(h^2) = sum(cos^2) + 2*eps*sum(cos) + N*eps^2 = N/2 + 0 + N*eps^2
    # The autoconvolution at lag k: sum_{i} h[i] h[i+k] = sum cos(i)cos(i+k) + eps*sum cos(i) + eps*sum cos(i+k) + N*eps^2
    # For k != 0, sum cos(i)cos(i+k) = (N/2)cos(k*2pi/N) for k < N, and similar for k > N.
    # The maximum of (N/2)cos(2*pi*k/N) over k is N/2 (at k=0), so lag 0 is still the max.
    #
    # Actually, let's try a different approach: use a pattern where the autoconvolution
    # peaks at a non-zero lag. Consider h[i] = 1 for i in [0, 199], -1 for i in [200, 399],
    # plus a small offset. But sum is 0, so add a small offset to all entries.
    #
    # Let h[i] = 1 for i < 200, h[i] = -1 for i >= 200, then add delta to all.
    # sum = 200*1 + 200*(-1) + 400*delta = 400*delta
    # autoconv at lag 0: 200*1^2 + 200*(-1)^2 + 2*delta*sum(original) + 400*delta^2 = 400 + 0 + 400*delta^2
    # autoconv at lag k: for k < 200, sum_{i=0}^{199-k} 1*1 + sum_{i=200}^{399-k} (-1)*(-1) + ... 
    # This is getting complex. Let's just try a few patterns numerically.
    #
    # Try: h = cos(2*pi*i/N) + 0.01
    idx = np.arange(N)
    h = np.cos(2 * np.pi * idx / N) + 0.01
    # Ensure sum is not zero
    s = np.sum(h)
    if abs(s) < 1e-12:
        h += 1e-6
    return h
