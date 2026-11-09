import numpy as np
import time

N = 100

def compute_all_overlaps(h):
    """Return array of overlaps for all shifts s=0..N-1."""
    # overlap_s = sum_i h[i]*(1-h[i+s]) = N/2 - autocorr_s
    # autocorr_s = sum_i h[i]*h[(i+s)%N]
    H = np.fft.fft(h)
    power = np.abs(H)**2
    autocorr = np.fft.ifft(power).real
    # autocorr[s] = sum_i h[i]*h[(i+s)%N]  (circular)
    overlap = N/2.0 - autocorr
    return overlap

def compute_score(h):
    ov = compute_all_overlaps(h)
    return float(np.max(ov))

def project(h):
    """Project to [0,1] with sum exactly N/2. Simple: clip then adjust."""
    h = np.clip(h, 0.0, 1.0)
    # fix sum via bisection on a threshold (water-filling)
    target = N/2.0
    cur = h.sum()
    if abs(cur - target) < 1e-12:
        return h
    # adjust: add uniform delta, then clip, iterate
    for _ in range(200):
        delta = (target - h.sum())/N
        h2 = np.clip(h + delta, 0.0, 1.0)
        if abs(h2.sum() - target) < 1e-12:
            return h2
        h = h2
    return h

def smooth_max(ov, beta):
    """Log-sum-exp style smooth max."""
    m = np.max(ov)
    return m + np.log(np.mean(np.exp(beta*(ov - m))))

def smooth_grad(ov, beta):
    m = np.max(ov)
    w = np.exp(beta*(ov - m))
    w = w / w.sum()
    return w  # weights over shifts

def spectral_init():
    """Initialize with a flat profile: try to make overlap_s constant.
    Use a few low-frequency modes to reduce peak."""
    h = np.full(N, 0.5)
    # We want overlap_s = N/4 - sum_m (N/4) A_m^2 cos(2 pi k_m s/N)
    # To flatten, we cannot cancel (all A_m^2 >= 0 with positive cos weight).
    # So best is small amplitude. Return uniform-ish with tiny perturbation.
    return h

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    np.random.seed(42)

    # Start: uniform
    h = np.full(N, 0.5)
    best_score = compute_score(h)
    best_h = h.copy()

    # Spectral init: small sinusoid to break symmetry, then binary local search
    # Try a binary alternating pattern
    h_alt = np.tile([1.0, 0.0], N//2)
    s_alt = compute_score(h_alt)
    if s_alt < best_score:
        best_score = s_alt
        best_h = h_alt.copy()

    # Binary local search with fast FFT score
    def local_search_binary(h_bin, max_iters):
        current = compute_score(h_bin)
        ones = np.where(h_bin == 1)[0]
        zeros = np.where(h_bin == 0)[0]
        stall = 0
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if len(ones) == 0 or len(zeros) == 0:
                break
            i = ones[np.random.randint(len(ones))]
            j = zeros[np.random.randint(len(zeros))]
            h_bin[i] = 0.0
            h_bin[j] = 1.0
            new = compute_score(h_bin)
            if new < current - 1e-12:
                current = new
                # update sets
                ones = np.where(h_bin == 1)[0]
                zeros = np.where(h_bin == 0)[0]
                stall = 0
            else:
                h_bin[i] = 1.0
                h_bin[j] = 0.0
                stall += 1
            if stall > 300:
                break
        return h_bin, current

    # Multiple restarts of binary local search
    n_restarts = 60
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        h_bin = np.zeros(N)
        ones_positions = np.random.choice(N, N//2, replace=False)
        h_bin[ones_positions] = 1.0
        h_bin, cur = local_search_binary(h_bin, max_iters=400)
        if cur < best_score:
            best_score = cur
            best_h = h_bin.copy()

    # Continuous polish: gradient descent on smooth max of overlaps,
    # with projection to [0,1] and sum constraint.
    if time.time() < deadline - 5:
        h_c = best_h.astype(float).copy()
        lr = 0.05
        beta = 20.0
        cur = compute_score(h_c)
        stall = 0
        for it in range(2000):
            if time.time() > deadline - 3:
                break
            ov = compute_all_overlaps(h_c)
            w = smooth_grad(ov, beta)
            # d overlap_s / d h: overlap_s = N/2 - sum_i h[i]h[i+s]
            # d overlap_s/d h[i] = -sum_s? Actually gradient w.r.t h[i]:
            # overlap_s = sum_i h[i] - sum_i h[i]h[i+s]
            # d overlap_s / d h[i] = 1 - (h[i+s] + h[(i-s)%N])
            # So grad_h[i] = sum_s w_s * (1 - h[i+s] - h[i-s])
            grad = np.zeros(N)
            for s in range(1, N):
                if w[s] < 1e-9:
                    continue
                grad += w[s] * (1.0 - np.roll(h_c, s) - np.roll(h_c, -s))
            h_c = h_c - lr * grad
            h_c = project(h_c)
            new = compute_score(h_c)
            if new < cur - 1e-10:
                cur = new
                stall = 0
                if cur < best_score:
                    best_score = cur
                    best_h = h_c.copy()
            else:
                # revert
                h_c = best_h.astype(float).copy()
                stall += 1
                if stall > 50:
                    break
        # final projection
        best_h = project(best_h)

    # Ensure exact sum and bounds
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N/2.0 - best_h.sum()
    best_h += diff / N
    best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
