"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def autoconv(h):
        return np.convolve(h, h, mode='full')

    def score_fn(h):
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

    def make_lobes(centers, widths, heights):
        """Create a signal from Gaussian lobes at given centers with given widths and heights."""
        x = np.linspace(-0.25, 0.25, N)
        h = np.zeros(N)
        for c, w, ht in zip(centers, widths, heights):
            h += ht * np.exp(-((x - c) ** 2) / (2 * w ** 2))
        return h

    # Soft-max smoothed objective using log-sum-exp for the autoconvolution peak
    # We maximize log(||h||_2^2) - log(||h||_1) - softmax_log(||h*h||_inf)
    # where softmax_log(a) = log(sum(exp(a/beta))) * beta approximates max(a) as beta->0

    def _soft_max_log(ac, beta):
        """Compute beta * log-sum-exp(ac / beta) which approximates max(ac) for small beta."""
        ac_clipped = np.clip(ac, -1e30, 1e30)
        shifted = ac_clipped / beta
        m = np.max(shifted)
        if m > 700:
            shifted = shifted - m
        s = np.sum(np.exp(shifted))
        if s < 1e-30:
            return 1e-30
        return beta * (m + np.log(s))

    def smooth_log_ratio(h, beta):
        """
        Return log(||h||_2^2) - log(||h||_1) - softmax_log(||h*h||_inf).
        This is a smooth surrogate for log(ratio). Higher is better.
        """
        h = np.maximum(h, 1e-15)
        norm2_sq = np.sum(h * h)
        norm1 = np.sum(h)
        if norm2_sq < 1e-30 or norm1 < 1e-30:
            return -1e10
        ac = autoconv(h)
        ac_abs = np.abs(ac)
        smax = _soft_max_log(ac_abs, beta)
        if smax < 1e-30:
            return -1e10
        return np.log(norm2_sq) - np.log(norm1) - np.log(smax)

    def neg_smooth_log_ratio(h, beta):
        """Negative of smooth log ratio for minimization."""
        return -smooth_log_ratio(h, beta)

    best_h = None
    best_score = 0.0

    rng = np.random.RandomState(123)

    # Phase 1: Generate diverse starting points
    starts = []

    # Structured starts
    x = np.linspace(-0.25, 0.25, N)

    # Sparse center
    h_sp = np.zeros(N)
    h_sp[N//2-2:N//2+3] = 1.0
    h_sp = h_sp / np.sqrt(np.sum(h_sp*h_sp))
    starts.append(h_sp.copy())

    # Two narrow lobes
    h_2l = make_lobes([-0.1, 0.1], [0.02, 0.02], [1.0, 1.0])
    h_2l = h_2l / np.sqrt(np.sum(h_2l*h_2l))
    starts.append(h_2l.copy())

    # Three lobes
    h_3l = make_lobes([-0.15, 0.0, 0.15], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0])
    h_3l = h_3l / np.sqrt(np.sum(h_3l*h_3l))
    starts.append(h_3l.copy())

    # Four lobes
    h_4l = make_lobes([-0.18, -0.06, 0.06, 0.18], [0.015, 0.015, 0.015, 0.015], [1.0, 1.0, 1.0, 1.0])
    h_4l = h_4l / np.sqrt(np.sum(h_4l*h_4l))
    starts.append(h_4l.copy())

    # Random starts
    for _ in range(10):
        h_r = np.maximum(rng.random(N), 0.01)
        h_r = h_r / np.sqrt(np.sum(h_r*h_r))
        starts.append(h_r.copy())

    # Lobe-based random starts
    for n_lobe in [2, 3, 4, 5, 6]:
        for trial in range(5):
            if time.time() > deadline - 60:
                break
            centers = np.sort(rng.uniform(-0.22, 0.22, n_lobe))
            widths = rng.uniform(0.008, 0.035, n_lobe)
            heights = rng.uniform(0.5, 1.5, n_lobe)
            h0 = make_lobes(centers, widths, heights)
            h0 = np.maximum(h0, 0.0)
            if np.sum(h0 * h0) < 1e-30:
                continue
            h0 = h0 / np.sqrt(np.sum(h0 * h0))
            starts.append(h0.copy())

    # Phase 2: Optimize each start using SLSQP with soft-max smoothed objective
    try:
        from scipy.optimize import minimize

        beta_values = [0.05, 0.02, 0.01, 0.005, 0.002]

        for h0 in starts:
            if time.time() > deadline - 20:
                break

            h0 = np.maximum(h0, 0.0)
            n2 = np.sqrt(np.sum(h0 * h0))
            if n2 < 1e-30:
                continue
            h0 = h0 / n2

            # Use a moderate beta for SLSQP (too small makes it nearly non-smooth)
            beta = 0.01

            def obj(x, beta=beta):
                return neg_smooth_log_ratio(x, beta)

            def jac(x, beta=beta):
                # Numerical gradient
                eps = 1e-7
                g = np.zeros_like(x)
                for i in range(N):
                    xp = x.copy()
                    xp[i] += eps
                    xm = x.copy()
                    xm[i] -= eps
                    g[i] = (obj(xp) - obj(xm)) / (2 * eps)
                return g

            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x**2) - 1.0,
                     'jac': lambda x: 2 * x}]
            bounds = [(0.0, None)] * N

            try:
                res = minimize(
                    obj,
                    h0,
                    jac=jac,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 300, 'ftol': 1e-12}
                )

                h_opt = np.maximum(res.x, 0.0)
                n2 = np.sqrt(np.sum(h_opt * h_opt))
                if n2 > 1e-30:
                    h_opt = h_opt / n2
                s = score_fn(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()

            except Exception:
                pass

    except (ImportError, Exception):
        pass

    # Phase 3: Anneling beta downward from the best solution found
    if best_h is not None and time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize

            h_curr = best_h.copy()
            curr_score = score_fn(h_curr)

            for beta in [0.005, 0.002, 0.001, 0.0005, 0.0002]:
                if time.time() > deadline - 5:
                    break

                def obj(x, beta=beta):
                    return neg_smooth_log_ratio(x, beta)

                def jac(x, beta=beta):
                    eps = 1e-8
                    g = np.zeros_like(x)
                    for i in range(N):
                        xp = x.copy()
                        xp[i] += eps
                        xm = x.copy()
                        xm[i] -= eps
                        g[i] = (obj(xp) - obj(xm)) / (2 * eps)
                    return g

                cons = [{'type': 'eq', 'fun': lambda x: np.sum(x**2) - 1.0,
                         'jac': lambda x: 2 * x}]
                bounds = [(0.0, None)] * N

                try:
                    res = minimize(
                        obj,
                        h_curr,
                        jac=jac,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 200, 'ftol': 1e-14}
                    )

                    h_opt = np.maximum(res.x, 0.0)
                    n2 = np.sqrt(np.sum(h_opt * h_opt))
                    if n2 > 1e-30:
                        h_opt = h_opt / n2
                    s = score_fn(h_opt)
                    if s > curr_score:
                        curr_score = s
                        h_curr = h_opt.copy()

                except Exception:
                    pass

            if curr_score > best_score:
                best_score = curr_score
                best_h = h_curr.copy()

        except (ImportError, Exception):
            pass

    # Phase 4: Fine coordinate descent polish on the exact (non-smooth) objective
    if best_h is not None and time.time() < deadline - 3:
        h_curr = best_h.copy()
        curr_score = score_fn(h_curr)

        for step in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 2:
                break
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(N):
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-14:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
                        continue
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    n2 = np.sqrt(np.sum(h_trial * h_trial))
                    if n2 > 1e-30:
                        h_trial = h_trial / n2
                    s = score_fn(h_trial)
                    if s > curr_score + 1e-14:
                        h_curr = h_trial
                        curr_score = s
                        improved = True

        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()

    if best_h is None:
        return [1.0] * N

    return [float(x) for x in best_h]

# EVOLVE-BLOCK-END
