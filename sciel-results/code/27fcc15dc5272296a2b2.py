"""Program for Erdos minimum overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 135  # safety margin before 150s kill


def _overlap_vector(h):
    """Compute overlap for all N shifts: ov[k] = sum_i h[i]*(1-h[(i+k)%N])."""
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    overlaps = np.empty(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    return overlaps


def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    return float(np.max(_overlap_vector(h)))


def _objective_smooth(h, beta=50.0):
    """Smooth approximation of max overlap via log-sum-exp."""
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    overlaps = np.empty(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    m = np.max(overlaps)
    if m < 1e-15:
        return 0.0
    # log-sum-exp / N gives a smooth upper bound on max
    return m + np.log(np.sum(np.exp(beta * (overlaps - m))) / N) / beta


def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(20):
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            mask = h > 1e-15
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    # Final exact fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        # Add to elements with most room
        idx = np.argsort(1.0 - h)
        for i in idx:
            if diff > 0:
                add = min(diff, 1.0 - h[i])
                h[i] += add
                diff -= add
            elif diff < 0:
                sub = min(-diff, h[i])
                h[i] -= sub
                diff += sub
            if abs(diff) < 1e-14:
                break
    return h


def _epigraph_objective(x, beta=50.0):
    """Epigraph formulation: x = [h (N vars), t (1 var)].
    Minimize t subject to overlap_k(h) <= t for all k, 0<=h<=1, sum(h)=N/2.
    We use smooth constraints: overlap_k - t + (1/beta)*log(1+exp(beta*(overlap_k - t))) <= 0
    Actually simpler: minimize t + penalty for violations.
    """
    h = x[:N]
    t = x[N]
    one_minus_h = 1.0 - h
    # Compute all overlaps
    violations = np.empty(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        ov = np.dot(h, shifted)
        violations[k] = ov - t
    # Smooth max of violations (should be <= 0)
    m = np.max(violations)
    if m < -1e-10:
        return t
    # Soft penalty: log-sum-exp of violations
    penalty = m + np.log(np.sum(np.exp(beta * (violations - m))) / N) / beta
    return t + 100.0 * max(0.0, penalty)


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # Generate diverse initial candidates
    candidates = []

    # 1. Uniform baseline
    candidates.append(np.full(N, 0.5))

    # 2. Linear ramp
    candidates.append(np.array([(2 * i + 1) / (2 * N) for i in range(N)]))

    # 3. Sinusoidal patterns
    for freq in [1, 2, 3]:
        for A in [0.2, 0.35, 0.5]:
            h = 0.5 + A * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h = _project(h)
            candidates.append(h)
            h2 = 0.5 + A * np.cos(2 * np.pi * freq * np.arange(N) / N)
            h2 = _project(h2)
            candidates.append(h2)

    # 4. Random candidates
    for _ in range(10):
        h = rng.uniform(0.0, 1.0, size=N)
        h = _project(h)
        candidates.append(h)

    # Evaluate all candidates
    best_h = None
    best_val = float('inf')
    for h in candidates:
        val = _max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    elapsed = time.time() - t0

    # Phase 1: L-BFGS-B with smooth log-sum-exp objective
    if elapsed < DEADLINE - 20:
        try:
            h0 = best_h.copy()
            res = minimize(
                _objective_smooth,
                h0,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq',
                              'fun': lambda x: np.sum(x) - N / 2.0,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_cand = _project(res.x)
            val = _max_overlap(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        except Exception:
            pass

    elapsed = time.time() - t0

    # Phase 2: SLSQP polish with direct max-overlap objective
    if elapsed < DEADLINE - 15:
        try:
            h0 = best_h.copy()
            res = minimize(
                _max_overlap,
                h0,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq',
                              'fun': lambda x: np.sum(x) - N / 2.0,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            h_cand = _project(res.x)
            val = _max_overlap(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        except Exception:
            pass

    elapsed = time.time() - t0

    # Phase 3: Epigraph formulation (minimize t with smooth constraints)
    if elapsed < DEADLINE - 10:
        try:
            # x = [h, t], t initialized to current best max overlap + margin
            h0 = best_h.copy()
            t0_val = _max_overlap(h0) + 0.1
            x0 = np.concatenate([h0, [t0_val]])

            def _epi_obj(x):
                h = x[:N]
                t = x[N]
                one_minus_h = 1.0 - h
                violations = np.empty(N)
                for k in range(N):
                    shifted = np.roll(one_minus_h, k)
                    violations[k] = np.dot(h, shifted) - t
                m = np.max(violations)
                if m < -1e-10:
                    return t
                penalty = m + np.log(np.sum(np.exp(50.0 * (violations - m))) / N) / 50.0
                return t + 200.0 * max(0.0, penalty)

            res = minimize(
                _epi_obj,
                x0,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N + [(0.0, float('inf'))],
                constraints=[{'type': 'eq',
                              'fun': lambda x: np.sum(x[:N]) - N / 2.0,
                              'jac': lambda x: np.concatenate([np.ones(N), [0.0]])}],
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            h_cand = _project(res.x[:N])
            val = _max_overlap(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        except Exception:
            pass

    elapsed = time.time() - t0

    # Phase 4: Local refinement - perturb top shifts and re-optimize
    if elapsed < DEADLINE - 5:
        try:
            ov_vec = _overlap_vector(best_h)
            # Find the shifts that achieve max overlap
            max_ov = np.max(ov_vec)
            # Try small perturbations focused on reducing the worst shifts
            for _ in range(20):
                h_pert = best_h.copy()
                n_pert = rng.integers(3, 15)
                idx = rng.choice(N, size=n_pert, replace=False)
                h_pert[idx] += rng.normal(0, 0.05, size=n_pert)
                h_pert = _project(h_pert)
                val = _max_overlap(h_pert)
                if val < best_val:
                    best_val = val
                    best_h = h_pert.copy()
        except Exception:
            pass

    # Final projection to ensure exact constraints
    final_h = _project(best_h)
    # Verify sum is exactly N/2
    assert abs(np.sum(final_h) - N / 2) < 1e-10, f"Sum constraint violated: {np.sum(final_h)}"
    assert np.all(final_h >= 0) and np.all(final_h <= 1), "Bounds violated"

    return final_h


if __name__ == "__main__":
    t_start = time.time()
    h = construct()
    t_end = time.time()
    print(f"Time: {t_end - t_start:.2f}s")
    print("Sum:", np.sum(h))
    print("Min:", np.min(h), "Max:", np.max(h))
    print("Max overlap:", _max_overlap(h))
    baseline = _max_overlap(np.full(100, 0.5))
    print("Baseline overlap:", baseline)
    print("Score (seed/value):", baseline / _max_overlap(h))
