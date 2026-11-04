import numpy as np
import scipy.optimize
import time
from scipy.special import logsumexp

N = 100
TARGET_SUM = N / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude shift 0 (index N-1)
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def max_overlap(h):
        """Return the maximum linear overlap value."""
        overlaps = get_correlations(h)
        return float(np.max(overlaps))

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        Uses bisection on a uniform shift for exact sum.
        """
        h = np.clip(h, 0.0, 1.0).copy()
        target = TARGET_SUM

        def sum_at_delta(delta):
            return np.sum(np.clip(h + delta, 0.0, 1.0))

        s0 = np.sum(h)
        if abs(s0 - target) < 1e-13:
            return h

        lo, hi = -1.5, 1.5
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            s = sum_at_delta(mid)
            if s < target:
                lo = mid
            else:
                hi = mid

        delta = 0.5 * (lo + hi)
        h = np.clip(h + delta, 0.0, 1.0)

        # Distribute any tiny residual to interior elements.
        diff = target - np.sum(h)
        if abs(diff) > 1e-14:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        return h

    def epigraph_objective(x, beta, lam_sum):
        """
        Epigraph objective. x = [h_0..h_{N-1}, t].
        Minimize t + (lse(overlaps) - t)_+^2 + sum_penalty.
        """
        h = np.clip(x[:N], 0.0, 1.0)
        t = x[N]

        overlaps = get_correlations(h)
        if len(overlaps) == 0:
            lse = 0.0
        else:
            max_ov = np.max(overlaps)
            shifted = overlaps - max_ov
            shifted = np.clip(shifted, -50.0 / beta, 0.0)
            lse = max_ov + logsumexp(beta * shifted) / beta

        if lse > t:
            penalty = (lse - t) ** 2
        else:
            penalty = 0.0

        sum_err = np.sum(h) - TARGET_SUM
        sum_penalty = lam_sum * (sum_err ** 2)
        return t + penalty + sum_penalty

    def coordinate_search(h, dl, max_rounds=300):
        """
        Local search: adjust one coordinate, re-project, keep if max overlap drops.
        """
        current_h = h.copy()
        current_score = max_overlap(current_h)

        base_steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]

        for round_num in range(max_rounds):
            if time.time() > dl - 3.0:
                break
            improved = False
            scale = max(0.1, 1.0 - round_num / max_rounds)
            steps = [s * scale for s in base_steps]

            for i in range(N):
                for delta in steps:
                    for sign in (1, -1):
                        new_h = current_h.copy()
                        new_h[i] += sign * delta
                        new_h = project_to_feasible(new_h)
                        new_score = max_overlap(new_h)
                        if new_score < current_score - 1e-13:
                            current_h = new_h
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return current_h, current_score

    def logistic_map_sequence(params):
        """
        Generate a sequence using a chaotic dynamical system (logistic map).
        params = [r, x0, a, b, c, d] where:
        - r: logistic map parameter (3.5-4.0)
        - x0: initial condition
        - a, b, c, d: linear transformation params to map to [0,1] with mean ~0.5
        """
        r, x0, a, b, c, d = params
        x = np.zeros(N)
        x[0] = x0
        for i in range(1, N):
            x[i] = r * x[i-1] * (1.0 - x[i-1])
        # Map to [0,1] with controlled mean
        h = a * x + b * x**2 + c * x**3 + d
        return h

    def generate_candidates(rng):
        """Generate diverse initial candidates."""
        candidates = []
        t = np.arange(N)

        # Uniform
        candidates.append(np.full(N, 0.5))

        # Alternating patterns
        for hi, lo in [(0.6, 0.4), (0.8, 0.2), (0.9, 0.1), (0.7, 0.3), (0.95, 0.05)]:
            h_alt = np.full(N, 0.5)
            h_alt[::2] = hi
            h_alt[1::2] = lo
            candidates.append(h_alt)

        # Block patterns
        for block_size in [2, 3, 4, 5, 7, 10]:
            for hi, lo in [(0.7, 0.3), (0.8, 0.2), (0.6, 0.4)]:
                h_block = np.zeros(N)
                for i in range(0, N, block_size * 2):
                    h_block[i:i+block_size] = hi
                    h_block[i+block_size:i+2*block_size] = lo
                candidates.append(h_block)

        # Sinusoidal
        for freq in range(1, 15):
            for amp in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
                for phase in [0.0, np.pi / 4, np.pi / 2, 3*np.pi/4]:
                    candidates.append(0.5 + amp * np.sin(2 * np.pi * freq * t / N + phase))

        # Sturmian word candidates
        for alpha in [0.6180339887, 0.7071067812, 0.75]:
            n_vals = np.arange(N)
            h_sturm = np.floor((n_vals + 1) * alpha) - np.floor(n_vals * alpha)
            h_sturm = h_sturm.astype(float)
            candidates.append(h_sturm)

        # Chaotic logistic map candidates
        for r in [3.5, 3.6, 3.7, 3.8, 3.9, 3.95, 3.99, 3.999]:
            for x0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                # Simple linear mapping: h = a*x + b, choose a,b to get mean ~0.5
                x_seq = np.zeros(N)
                x_seq[0] = x0
                for i in range(1, N):
                    x_seq[i] = r * x_seq[i-1] * (1.0 - x_seq[i-1])
                mean_x = np.mean(x_seq)
                std_x = np.std(x_seq)
                # Map so that mean is 0.5 and std is ~0.25
                a = 0.25 / std_x if std_x > 1e-10 else 1.0
                b = 0.5 - a * mean_x
                h_chaos = a * x_seq + b
                candidates.append(h_chaos)

        # Random
        for _ in range(20):
            candidates.append(rng.uniform(0.0, 1.0, size=N))

        return candidates

    rng = np.random.default_rng(42)

    # --- Generate and evaluate diverse candidates ---
    candidates = generate_candidates(rng)

    best_h = None
    best_score = np.inf
    for h_cand in candidates:
        h_proj = project_to_feasible(h_cand)
        score = max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj

    # --- Try to improve the best chaotic candidate via parameter optimization ---
    if time.time() < deadline - 40.0:
        # Find which candidate was best and if it's chaotic, try local param search
        # For now, just add a few more chaotic variants around the best
        if best_h is not None:
            # Try small perturbations of the best chaotic-like structure
            for perturb_scale in [0.01, 0.02, 0.05]:
                h_perturbed = best_h + rng.normal(0, perturb_scale, N)
                h_proj = project_to_feasible(h_perturbed)
                score = max_overlap(h_proj)
                if score < best_score:
                    best_score = score
                    best_h = h_proj

    # --- Epigraph optimization with increasing beta ---
    t_init = best_score + 0.1
    bounds = [(0.0, 1.0)] * N + [(0.0, 10.0)]

    betas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    for beta in betas:
        if time.time() > deadline - 20.0:
            break
        lam_sum = 100.0 + 10.0 * beta
        x_init = np.concatenate([best_h, [t_init]])

        def obj_func(x):
            return epigraph_objective(x, beta, lam_sum)

        try:
            result = scipy.optimize.minimize(
                obj_func, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 40}
            )
            h_new = project_to_feasible(result.x[:N])
            score_new = max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                t_init = max(0.0, result.x[N])
        except Exception:
            pass

    # --- High-precision polish with very high beta ---
    if time.time() < deadline - 10.0:
        x_init = np.concatenate([best_h, [t_init]])

        def obj_polish(x):
            return epigraph_objective(x, 20000.0, 10000.0)

        try:
            result = scipy.optimize.minimize(
                obj_polish, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            h_polished = project_to_feasible(result.x[:N])
            score_polished = max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass

    # --- Coordinate local search on the true (nonsmooth) max objective ---
    if time.time() < deadline - 5.0:
        best_h, best_score = coordinate_search(best_h, deadline)

    # --- Final high-precision polish with very tight tolerances ---
    if time.time() < deadline - 3.0:
        x_init = np.concatenate([best_h, [best_score]])

        def obj_final_polish(x):
            return epigraph_objective(x, 50000.0, 50000.0)

        try:
            result = scipy.optimize.minimize(
                obj_final_polish, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            h_final = project_to_feasible(result.x[:N])
            score_final = max_overlap(h_final)
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
        except Exception:
            pass

    # --- Final exact-sum correction ---
    best_h = project_to_feasible(best_h)
    sum_h = np.sum(best_h)
    diff = TARGET_SUM - sum_h
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-9) & (best_h < 1.0 - 1e-9)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
