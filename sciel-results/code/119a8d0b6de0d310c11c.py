"""Erdos minimum overlap: minimize max linear overlap of h with shifted complement.
Uses cosine-basis representation (guarantees sum=N/2), log-sum-exp smooth max,
multi-start, adaptive beta annealing, and coordinate-descent polish."""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    Using binary search on the shift parameter tau.
    """
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0

    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum

    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid

    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)


def _smooth_max_grad_cs(cs, beta):
    """
    Compute value and gradient weights for log-sum-exp smooth max of cs.
    Returns (value, weights) where weights sum to 1.
    """
    m = np.max(cs)
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    return val, w


def _dcs_dh(w, n):
    """
    Given weights w over C_k values, compute dC/dh (gradient of weighted C w.r.t. h).
    C_k = 1 - (2/n) * r_k where r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k else 0 + h_{j+k} if j+k<n else 0)
    """
    grad = np.zeros(n)
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n - k] if False else 0  # placeholder
    # Actually let me redo this properly
    return grad


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    # Cosine basis setup
    M = N // 2
    i_idx = np.arange(N)
    j_idx = np.arange(1, M + 1)
    C = np.cos(2 * np.pi * np.outer(i_idx, j_idx) / N)  # shape (N, M)

    def h_from_a(a):
        return 0.5 + C @ a

    def compute_cs(h):
        return _linear_overlap_values(h)

    def smooth_max_and_grad_a(a, beta):
        """
        Compute smooth max value and gradient w.r.t. a in cosine basis.
        """
        h = h_from_a(a)
        cs = compute_cs(h)
        val, w = _smooth_max_grad_cs(cs, beta)

        # dC_k/dh_j for each k:
        # C_k = 1 - (2/n) * sum_{i=k}^{n-1} h[i]*h[i-k]
        # dC_k/dh_j = -(2/n) * [h_{j-k} (if j>=k) + h_{j+k} (if j+k<n)]
        #
        # d(objective)/dh_j = sum_k w[k] * dC_k/dh_j
        # = -(2/n) * sum_k w[k] * [h_{j-k} (if j>=k) + h_{j+k} (if j+k<n)]
        #
        # For fixed j: sum_k w[k]*h_{j-k} (k<=j) + sum_k w[k]*h_{j+k} (k<n-j)
        # = sum_{k=0}^{j} w[k]*h[j-k] + sum_{k=0}^{n-1-j} w[k]*h[j+k]
        #
        # This is a cross-correlation-like operation.
        # Let's compute it efficiently.

        dh = np.zeros(N)
        # For each k, contribution to dh:
        # dh[j] += -(2/n)*w[k] * h[j-k] for j>=k
        # dh[j] += -(2/n)*w[k] * h[j+k] for j+k<n i.e. j<n-k
        #
        # Vectorized: for each k, we add -(2/n)*w[k] * (h shifted right by k) to dh[k:]
        # and -(2/n)*w[k] * (h shifted left by k) to dh[:n-k]
        #
        # Let's use FFT-based convolution for efficiency
        # Actually for N=100, direct loop is fine

        for k in range(N):
            if w[k] == 0.0:
                continue
            factor = -(2.0 / N) * w[k]
            if k > 0:
                dh[k:] += factor * h[:N - k]
            if k < N:
                dh[:N - k] += factor * h[k:]

        da = C.T @ dh
        return val, da

    def optimize_a(a0, beta_schedule, steps_per_phase, deadline):
        """
        Optimize a using projected gradient descent in cosine space.
        beta_schedule: list of (beta, steps) pairs
        """
        a = a0.copy()
        best_a = a.copy()
        best_val = float('inf')

        for beta, steps in beta_schedule:
            if time.time() > deadline:
                break
            a_prev = a.copy()
            lr = 0.05
            for it in range(steps):
                if time.time() > deadline:
                    break

                val, da = smooth_max_and_grad_a(a, beta)

                # Nesterov momentum
                if it > 0:
                    momentum = (1 + np.sqrt(1 + 2 * it)) / (1 + np.sqrt(1 + 2 * (it + 1)))
                    a_nest = a + momentum * (a - a_prev)
                else:
                    a_nest = a

                # Gradient step
                val_n, da_n = smooth_max_and_grad_a(a_nest, beta)
                a_new = a_nest - lr * da_n

                # Project: ensure 0 <= h[i] <= 1
                h_new = h_from_a(a_new)
                if np.any(h_new < -1e-12) or np.any(h_new > 1 + 1e-12):
                    # Binary search for valid step
                    lo, hi = 0.0, 1.0
                    for _ in range(12):
                        mid = (lo + hi) / 2
                        a_test = a_nest + mid * (a_new - a_nest)
                        h_test = h_from_a(a_test)
                        if np.all(h_test >= -1e-12) and np.all(h_test <= 1 + 1e-12):
                            lo = mid
                        else:
                            hi = mid
                    a_new = a_nest + lo * (a_new - a_nest)

                a_prev = a
                a = a_new

                # Track best
                h_curr = h_from_a(a)
                cs_curr = compute_cs(h_curr)
                true_max = np.max(cs_curr)
                if true_max < best_val:
                    best_val = true_max
                    best_a = a.copy()

                # Adaptive learning rate
                if it % 20 == 0:
                    lr = max(lr * 0.95, 0.001)

        return best_a, best_val

    def coordinate_descent_polish(h, deadline, max_rounds=50):
        """
        Coordinate descent polish: for each position, try to reduce the max overlap
        by adjusting that single coordinate while keeping sum constant.
        """
        h = h.copy()
        target_sum = N / 2.0
        best_score = _score(h)

        for round_num in range(max_rounds):
            if time.time() > deadline:
                break

            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break

                # Try adjusting h[idx]
                # To keep sum constant, we need to adjust another coordinate
                # Simple approach: try small perturbations and project
                for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                    h_test = h.copy()
                    h_test[idx] += delta
                    h_test = _project_sum_box(h_test, target_sum)
                    s = _score(h_test)
                    if s < best_score - 1e-10:
                        h = h_test
                        best_score = s
                        improved = True
                        break

            if not improved:
                break

        return h, best_score

    # Generate initial candidates
    candidates = []

    # Cosine waves
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        a = np.zeros(M)
        if freq <= M:
            a[freq - 1] = 0.4
        h = h_from_a(a)
        h = _project_sum_box(h, N / 2.0)
        candidates.append(h)

    # Random starts
    for _ in range(12):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        candidates.append(h_rand)

    # Evaluate and pick best initial
    best_h = None
    best_score = float('inf')
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Convert best_h to a-space
    # a = argmin ||C@a - (h - 0.5)||^2
    # a = (C^T C)^{-1} C^T (h - 0.5)
    # Use least squares
    rhs = best_h - 0.5
    a0, _, _, _ = np.linalg.lstsq(C, rhs, rcond=None)

    # Beta schedule: start coarse, end fine
    beta_schedule = [
        (10.0, 200),
        (30.0, 200),
        (80.0, 200),
        (200.0, 200),
        (500.0, 150),
    ]

    # Main optimization
    a_opt, val_opt = optimize_a(a0, beta_schedule, None, _DEADLINE - 30)

    h_opt = h_from_a(a_opt)
    h_opt = _project_sum_box(h_opt, N / 2.0)
    score_opt = _score(h_opt)

    if score_opt < best_score:
        best_score = score_opt
        best_h = h_opt.copy()

    # Coordinate descent polish
    if time.time() < _DEADLINE - 5:
        h_polished, score_polished = coordinate_descent_polish(
            best_h, _DEADLINE - 2, max_rounds=30
        )
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished.copy()

    # Final projection to ensure exact sum and bounds
    best_h = _project_sum_box(best_h, N / 2.0)

    return best_h
