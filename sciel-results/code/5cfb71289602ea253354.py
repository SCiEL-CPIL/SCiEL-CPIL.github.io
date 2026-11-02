"""Erdos minimum overlap: bang-bang (quantized) waveform + local search.

Math: overlap[k] = sum_i h[i]*(1-h[(i+k)%N]) = N/2 - C[k], where C is the
circular autocorrelation of h.  Minimize max overlap  <=>  maximize min C[k].
In the frequency domain C = IFFT(|FFT(h)|^2); flattening C requires concentrating
|H|^2 in few modes, which a quantized (bang-bang) shape does better than a pure
sinusoid.  We search over {0,0.5,1}-valued patterns (mean 0.5) with a
flip-pair local search that directly minimizes the max overlap.
"""
import numpy as np
import time

N = 100
DEADLINE = 135  # safety margin before 150s kill


def _time_left(t0):
    return DEADLINE - (time.time() - t0)


def _overlap_vector(h):
    """ov[k] = sum_i h[i]*(1-h[(i+k)%N]) via FFT cross-correlation."""
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    H = np.fft.fft(h)
    OmH = np.fft.fft(one_minus_h)
    ov = np.fft.ifft(np.conj(H) * OmH).real
    return ov


def _max_overlap(h):
    return float(np.max(_overlap_vector(h)))


def _project(h):
    """Project to [0,1]^N with sum exactly N/2."""
    h = np.clip(np.asarray(h, dtype=np.float64), 0.0, 1.0)
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
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        idx = np.argsort(1.0 - h) if diff > 0 else np.argsort(h)
        for i in idx:
            if diff > 0:
                add = min(diff, 1.0 - h[i])
                h[i] += add
                diff -= add
            else:
                sub = min(-diff, h[i])
                h[i] -= sub
                diff += sub
            if abs(diff) < 1e-14:
                break
    return h


def _quantize_to_bangbang(h):
    """Snap to {0,0.5,1} with exactly N/2 total mass (mean 0.5)."""
    h = np.asarray(h, dtype=np.float64)
    # distance to each level
    d0 = h
    d1 = np.abs(h - 0.5)
    d2 = 1.0 - h
    best = np.zeros(N, dtype=np.int8)
    d = d0.copy()
    better = d1 < d
    best[better] = 1
    d[better] = d1[better]
    better2 = d2 < d
    best[better2] = 2
    # assign levels
    q = np.empty(N, dtype=np.float64)
    q[best == 0] = 0.0
    q[best == 1] = 0.5
    q[best == 2] = 1.0
    # fix total mass: need sum == N/2 => sum(q) == N/2
    # sum in units of 0.5: need integer count M = N (since N/2 / 0.5 = N)
    # current mass sum S; adjust by flipping individual levels in 0.5 steps
    target_mass = N / 2.0
    for _ in range(2 * N + 5):
        S = np.sum(q)
        if abs(S - target_mass) < 1e-12:
            break
        if S < target_mass:
            # need +0.5: pick a 0.5->1 or 0->0.5 that is "cheapest" (closest to target)
            cand = np.where(q == 0.5)[0]
            if len(cand):
                # flip the one closest to 1 (i.e. largest q already 0.5 -> trivial)
                # prefer flipping 0.5->1 for a 0.5 (no cost) -- just do it
                i = cand[0]
                q[i] = 1.0
            else:
                cand = np.where(q == 0.0)[0]
                if len(cand):
                    # flip 0->0.5 the one closest to 0.5 (largest h)
                    i = cand[np.argmax(h[cand])]
                    q[i] = 0.5
                else:
                    break
        else:
            cand = np.where(q == 0.5)[0]
            if len(cand):
                i = cand[0]
                q[i] = 0.0
            else:
                cand = np.where(q == 1.0)[0]
                if len(cand):
                    i = cand[np.argmin(h[cand])]
                    q[i] = 0.5
                else:
                    break
    return q


def _flip_pair_delta(h, a, b):
    """Return new max overlap if we flip h[a]->1-h[a] and h[b]->1-h[b].
    Efficient: only shifts affected are k in {b-a, a-b, b+a, a+b} mod N.
    Returns (new_max, changed_shifts) or None if infeasible (sum not preserved
    is fine because flip-pair preserves sum)."""
    # flip-pair preserves sum automatically.
    h2 = h.copy()
    h2[a] = 1.0 - h2[a]
    h2[b] = 1.0 - h2[b]
    ov = _overlap_vector(h2)
    return float(np.max(ov))


def _local_search(h, t0, max_iter=200000):
    """Flip-pair local search minimizing max overlap. Sum preserved by pairs."""
    h = h.copy()
    cur = _max_overlap(h)
    ov = _overlap_vector(h)
    rng = np.random.default_rng(7)
    it = 0
    while it < max_iter and _time_left(t0) > 10:
        it += 1
        # pick a random pair; bias toward high/low elements for interesting flips
        a = rng.integers(N)
        b = rng.integers(N)
        if a == b:
            continue
        new = _flip_pair_delta(h, a, b)
        if new < cur - 1e-15:
            h[a] = 1.0 - h[a]
            h[b] = 1.0 - h[b]
            cur = new
    return h, cur


def _local_search_sa(h, t0, max_iter=300000):
    """Simulated-annealing flip-pair search (accepts small worsenings)."""
    h = h.copy()
    cur = _max_overlap(h)
    rng = np.random.default_rng(99)
    T = 0.02
    it = 0
    best_h = h.copy()
    best_val = cur
    while it < max_iter and _time_left(t0) > 12:
        it += 1
        a = rng.integers(N)
        b = rng.integers(N)
        if a == b:
            continue
        new = _flip_pair_delta(h, a, b)
        if new < cur:
            h[a] = 1.0 - h[a]
            h[b] = 1.0 - h[b]
            cur = new
            if cur < best_val:
                best_val = cur
                best_h = h.copy()
        else:
            # accept with probability exp(-(new-cur)/T)
            if rng.random() < np.exp(-(new - cur) / max(T, 1e-9)):
                h[a] = 1.0 - h[a]
                h[b] = 1.0 - h[b]
                cur = new
        if it % 5000 == 0:
            T *= 0.97
    return best_h, best_val


def _sinusoid_init(freq, A):
    h = 0.5 + A * np.sin(2 * np.pi * freq * np.arange(N) / N)
    return h


def construct():
    t0 = time.time()
    rng = np.random.default_rng(2024)

    best_h = None
    best_val = float('inf')

    def consider(h):
        nonlocal best_h, best_val
        h = _project(h)
        v = _max_overlap(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # --- Initial candidates ---
    consider(np.full(N, 0.5))

    # Sinusoids (tie ~0.5 but give structure)
    for freq in [1, 2, 3, 4, 5]:
        for A in [0.3, 0.4, 0.5]:
            consider(_sinusoid_init(freq, A))

    # Random continuous -> quantize
    for _ in range(8):
        h = rng.uniform(0, 1, N)
        consider(_quantize_to_bangbang(h))

    # Structured random bang-bang: random phase + low-pass
    for _ in range(8):
        h = 0.5 + 0.5 * np.sign(rng.normal(0, 1, N))
        # low-pass to concentrate energy
        H = np.fft.fft(h)
        H[8:] = 0.0
        H[:8] *= 0.3
        h2 = np.fft.ifft(H).real
        consider(_quantize_to_bangbang(h2))

    # Ensure we have a bang-bang best for local search
    base = best_h.copy()
    if not np.allclose(np.round(base * 2) / 2, base, atol=1e-9):
        base = _quantize_to_bangbang(base)
        consider(base)

    # --- Phase 1: deterministic flip-pair local search ---
    if _time_left(t0) > 25:
        h, v = _local_search(base, t0, max_iter=150000)
        consider(h)

    # --- Phase 2: simulated annealing from a few restarts ---
    restarts = [best_h.copy()]
    for _ in range(3):
        h = best_h.copy()
        # random perturbation (flip several pairs)
        for _ in range(20):
            a = rng.integers(N)
            b = rng.integers(N)
            if a != b:
                h[a] = 1.0 - h[a]
                h[b] = 1.0 - h[b]
        restarts.append(_project(h))

    for r in restarts:
        if _time_left(t0) > 20:
            h, v = _local_search_sa(r, t0, max_iter=200000)
            consider(h)

    # --- Phase 3: final polish with continuous local search on top ---
    if _time_left(t0) > 12:
        try:
            from scipy.optimize import minimize
            h0 = best_h.copy()
            res = minimize(
                _max_overlap,
                h0,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq',
                              'fun': lambda x: np.sum(x) - N / 2.0,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
            )
            hc = _project(res.x)
            consider(hc)
            # also try quantizing the polished result and re-searching
            if _time_left(t0) > 12:
                hq = _quantize_to_bangbang(hc)
                consider(hq)
                h, v = _local_search(hq, t0, max_iter=80000)
                consider(h)
        except Exception:
            pass

    # Final projection and validation
    final_h = _project(best_h)
    assert abs(np.sum(final_h) - N / 2) < 1e-9, f"Sum violated: {np.sum(final_h)}"
    assert np.all(final_h >= -1e-12) and np.all(final_h <= 1.0 + 1e-12), "Bounds violated"
    final_h = np.clip(final_h, 0.0, 1.0)
    # re-fix sum after clip
    final_h = _project(final_h)
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
