"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Grid: N points on [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N)
    dx = x[1] - x[0] if N > 1 else 1.0

    def autoconv(h):
        return np.convolve(h, h, mode='full')

    def objective(h):
        """Return the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf). Higher is better."""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h * h)
        if norm2_sq < 1e-30:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-30:
            return 0.0
        ac = autoconv(h)
        ac_max = np.max(np.abs(ac))
        if ac_max < 1e-30:
            return 0.0
        return norm2_sq / (norm1 * ac_max)

    def make_two_lobe_h(delta, sigma, center=0.0):
        """Create heights from two Gaussian lobes at center +/- delta with width sigma."""
        lobe1 = np.exp(-((x - (center + delta))**2) / (2.0 * sigma**2))
        lobe2 = np.exp(-((x - (center - delta))**2) / (2.0 * sigma**2))
        h = lobe1 + lobe2
        return np.maximum(h, 0.0)

    def score_from_params(delta, sigma, center=0.0):
        """Compute the objective for given two-lobe parameters."""
        h = make_two_lobe_h(delta, sigma, center)
        return objective(h)

    best_score = 0.0
    best_h = None

    # Strategy 1: Grid search over delta and sigma for two symmetric lobes
    # delta: offset from center (0 to ~0.2 to stay in [-0.25, 0.25])
    # sigma: width of each Gaussian

    delta_range = np.linspace(0.0, 0.22, 40)
    sigma_range = np.linspace(0.005, 0.12, 30)

    for delta in delta_range:
        if time.time() > deadline:
            break
        for sigma in sigma_range:
            if time.time() > deadline:
                break
            h = make_two_lobe_h(delta, sigma)
            sc = objective(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()

    # Strategy 2: Local refinement around the best grid point using coordinate descent
    # on the full 50-dimensional space, starting from the best two-lobe configuration

    if best_h is not None and time.time() < deadline - 20:
        h_curr = best_h.copy()
        curr_val = best_score

        # Coordinate descent with decreasing step sizes
        step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]

        for step in step_sizes:
            if time.time() > deadline - 10:
                break
            improved = True
            max_iters = 500
            iters = 0
            while improved and iters < max_iters and time.time() < deadline - 10:
                improved = False
                iters += 1
                for i in range(N):
                    # Try increasing
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    trial_score = objective(h_trial)
                    if trial_score > curr_val + 1e-14:
                        h_curr = h_trial
                        curr_val = trial_score
                        improved = True
                        continue
                    # Try decreasing
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    trial_score = objective(h_trial)
                    if trial_score > curr_val + 1e-14:
                        h_curr = h_trial
                        curr_val = trial_score
                        improved = True

        if curr_val > best_score:
            best_score = curr_val
            best_h = h_curr.copy()

    # Strategy 3: Try asymmetric lobes (different widths or offsets)
    if time.time() < deadline - 30:
        # Asymmetric: left lobe at -d1 with sigma1, right lobe at +d2 with sigma2
        d1_range = np.linspace(0.0, 0.22, 20)
        d2_range = np.linspace(0.0, 0.22, 20)
        s1_range = np.linspace(0.005, 0.10, 15)
        s2_range = np.linspace(0.005, 0.10, 15)

        for d1 in d1_range[::2]:  # Sparse grid for asymmetric
            if time.time() > deadline - 15:
                break
            for d2 in d2_range[::2]:
                if time.time() > deadline - 15:
                    break
                for s1 in s1_range[::2]:
                    if time.time() > deadline - 15:
                        break
                    for s2 in s2_range[::2]:
                        if time.time() > deadline - 15:
                            break
                        lobe1 = np.exp(-((x - d2)**2) / (2.0 * s2**2))
                        lobe2 = np.exp(-((x + d1)**2) / (2.0 * s1**2))
                        h = lobe1 + lobe2
                        sc = objective(h)
                        if sc > best_score:
                            best_score = sc
                            best_h = h.copy()

    # Strategy 4: Try single narrow lobe (delta=0 limit) and very narrow spikes
    if time.time() < deadline - 10:
        # Very narrow Gaussian centered
        for sigma in np.linspace(0.003, 0.03, 10):
            h = np.exp(-x**2 / (2.0 * sigma**2))
            sc = objective(h)
            if sc > best_score:
                best_score = sc
                best_h = h.copy()

        # Sparse: only a few nonzero entries
        for k in range(1, 8):
            h_sparse = np.zeros(N)
            h_sparse[N//2 - k//2 : N//2 - k//2 + k] = 1.0
            sc = objective(h_sparse)
            if sc > best_score:
                best_score = sc
                best_h = h_sparse.copy()

    # Final fine polish
    if best_h is not None and time.time() < deadline - 5:
        h_curr = best_h.copy()
        curr_val = best_score
        for step in [0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 3:
                break
            improved = True
            iters = 0
            while improved and iters < 200 and time.time() < deadline - 3:
                improved = False
                iters += 1
                for i in range(N):
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    trial_score = objective(h_trial)
                    if trial_score > curr_val + 1e-14:
                        h_curr = h_trial
                        curr_val = trial_score
                        improved = True
                        continue
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    trial_score = objective(h_trial)
                    if trial_score > curr_val + 1e-14:
                        h_curr = h_trial
                        curr_val = trial_score
                        improved = True

        if curr_val > best_score:
            best_score = curr_val
            best_h = h_curr.copy()

    if best_h is None:
        return [1.0] * N

    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
