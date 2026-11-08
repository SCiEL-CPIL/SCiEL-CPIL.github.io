"""Erdos minimum overlap: choose step heights on [0,2] to minimize max overlap of h with shifted 1-h."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def _score(h):
    """Compute the max overlap of h with shifted 1-h over all shifts."""
    n = len(h)
    # h is defined on [0, n) with step 1/n. Shift by k/n means h(x) * (1 - h(x + k/n))
    # Using circular convolution: overlap(k) = sum_i h[i] * (1 - h[(i+k) % n])
    # We want max over k of overlap(k)
    # overlap(k) = sum_i h[i] - sum_i h[i]*h[(i+k)%n]
    # = (n/2) - autocorr(k) where autocorr(k) = sum_i h[i]*h[(i+k)%n]
    # So minimizing max overlap = maximizing min autocorr(k)
    # autocorr via FFT
    H = np.fft.fft(h)
    ac = np.real(np.fft.ifft(H * np.conj(H)))
    ac = np.round(ac * n) / n  # normalize: ac[k] = (1/n) sum_i h[i]*h[(i+k)%n]
    # overlap(k) = 0.5 - ac[k]  (since sum h = n/2, so (1/n)*sum h = 0.5)
    ov = 0.5 - ac
    return float(np.max(ov))


def _construct_kronecker(alpha, n=N, levels=None):
    """Construct h[i] = 1 if {i*alpha} < 0.5 else 0, with rounding to fix sum."""
    h = np.zeros(n)
    x = np.arange(n) * alpha
    frac = np.mod(x, 1.0)
    h[frac < 0.5] = 1.0
    # Fix sum to be exactly n/2
    target = n / 2.0
    current = h.sum()
    diff = target - current
    if abs(diff) > 1e-12:
        # Adjust by flipping some entries
        # Find entries to flip
        if diff > 0:
            # Need to add 'diff' ones: flip 0->1
            zeros = np.where(h == 0)[0]
            for idx in zeros[:int(np.ceil(diff))]:
                h[idx] = 1.0
        else:
            ones = np.where(h == 1)[0]
            for idx in ones[:int(np.ceil(-diff))]:
                h[idx] = 0.0
    return h


def construct():
    start_time = time.time()
    deadline = 150.0
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)  # 0.5
    
    # Try various Kronecker sequences with different alpha
    # The golden ratio conjugate is a good candidate for low-discrepancy
    golden = (1 + np.sqrt(5)) / 2
    candidates = [
        1.0 / golden,          # ~0.618
        1.0 / (golden ** 2),   # ~0.382
        1.0 / (golden ** 3),   # ~0.236
        0.5 + 1.0 / (golden ** 3),  # ~0.736
        1.0 / 3.0,
        2.0 / 5.0,
        3.0 / 7.0,
        1.0 / (np.sqrt(2)),    # ~0.707
        1.0 / (np.sqrt(3)),    # ~0.577
        0.5 + 1.0 / (golden ** 4),
        1.0 / (golden ** 5),
    ]
    
    # Also try some rational approximations that might give better structure
    for denom in range(2, 30):
        for numer in range(1, denom):
            if np.gcd(numer, denom) == 1:
                candidates.append(numer / denom)
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = round(c, 10)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    for alpha in unique_candidates:
        if time.time() > deadline - 10.0:
            break
        h = _construct_kronecker(alpha)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Try to improve with local search on the best candidate
    if time.time() < deadline - 20.0:
        # Local search: flip individual bits to reduce max overlap
        h = best_h.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            n = len(h)
            for i in range(n):
                if time.time() > deadline - 10.0:
                    break
                old = h[i]
                h[i] = 1.0 - old
                new_score = _score(h)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    h[i] = old
    
    return best_h
# EVOLVE-BLOCK-END
