import numpy as np
import time

N = 50

def _autoconv(h):
    """Full autoconvolution (length 2N-1)."""
    return np.correlate(h, h, mode='full')

def _ratio(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    l1 = h.sum()
    if l1 < 1e-15:
        return 0.0
    l2sq = float(np.dot(h, h))
    c = _autoconv(h)
    cmax = float(c.max())
    if cmax < 1e-15:
        return 0.0
    return l2sq / (l1 * cmax)

def _find_sidon_set(size, N, seed=0):
    """Find `size` distinct indices in [0,N) that form a Sidon set
    (all pairwise differences distinct). Returns list or None."""
    import random
    rng = random.Random(seed)
    for attempt in range(20000):
        S = set(rng.sample(range(N), size))
        diffs = set()
        ok = True
        for i, a in enumerate(S):
            for b in S:
                if a == b:
                    continue
                d = a - b
                if d in diffs:
                    ok = False
                    break
                diffs.add(d)
            if not ok:
                break
        if ok:
            return sorted(S)
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def timed_ok():
        return time.time() < deadline - 5.0

    best_h = np.zeros(N)
    best_score = 0.0

    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s

    # Baselines
    h2 = np.zeros(N); h2[0] = 2.0/3.0; h2[1] = 1.0/3.0
    try_candidate(h2)
    try_candidate(np.ones(N)/N)

    # ---- Build the core 2-element structure (a,b) that maximizes R ----
    # For support {0,1}: R(a,b) = (a^2+b^2)/( (a+b)(a^2+2ab+b^2) )
    # Let t = b/a. R = (1+t^2)/((1+t)(1+t)^2) * (a^2)/(a^2) ... actually scale-invariant.
    # R(t) = (1+t^2)/(1+t)^3. Maximize: derivative zero at t satisfying
    # 2t(1+t)^3 - (1+t^2)3(1+t)^2 = 0 -> 2t(1+t) - 3(1+t^2) = 0
    # -> 2t+2t^2 -3 -3t^2 = 0 -> -t^2+2t-3 = 0 -> no real root? 
    # Recompute: d/dt ln R = 2t/(1+t^2) - 3/(1+t) = 0
    # 2t(1+t) = 3(1+t^2) -> 2t+2t^2 = 3+3t^2 -> -t^2+2t-3=0 -> disc = 4-12 <0.
    # So R(t) is monotonic on (0,inf)? Check t->0: R->1; t->inf: R->0; t=1: R=2/8=0.25.
    # Hmm that's <1. So 2-element adjacent is NOT the 1.1944 source. 
    # The 1.1944 = 11/9 must come from a different config. Let's just trust the seed
    # and instead rely on the Sidon-spike mechanism + local polish.

    # ---- Sidon spike construction ----
    # Start with a 2-element core that we will polish, plus disjoint unit spikes.
    # We'll search over core positions and spike counts.

    # First, try pure disjoint spikes (R ~ 1) as a floor, then add a strong core.
    # The winning idea: take a high-R core (2-3 elements) and add disjoint spikes
    # far from it so they don't interact (autoconv peak stays at core's value).

    # Let's find the best small-core structure by brute force over a few supports.
    def brute_core(support):
        """Brute-force 2-3 element core on given support indices to maximize R
        with no spikes. Returns (values dict, score)."""
        # We'll use a small grid + local refinement.
        idx = list(support)
        k = len(idx)
        best = None
        bestv = -1
        # grid over simplex-ish
        if k == 2:
            for a in np.linspace(0.05, 3.0, 60):
                for b in np.linspace(0.05, 3.0, 60):
                    h = np.zeros(N)
                    h[idx[0]] = a; h[idx[1]] = b
                    s = _ratio(h)
                    if s > bestv:
                        bestv = s; best = (a, b)
            return best, bestv
        elif k == 3:
            # 3D grid coarser
            for a in np.linspace(0.05, 2.0, 30):
                for b in np.linspace(0.05, 2.0, 30):
                    for c in np.linspace(0.05, 2.0, 30):
                        h = np.zeros(N)
                        h[idx[0]] = a; h[idx[1]] = b; h[idx[2]] = c
                        s = _ratio(h)
                        if s > bestv:
                            bestv = s; best = (a, b, c)
            return best, bestv
        return None, -1

    # Evaluate a few core supports
    core_candidates = []
    for sup in [[0,1],[0,2],[1,2],[0,1,2],[0,1,3],[0,2,4],[1,3,5],[0,1,4]]:
        if not timed_ok():
            break
        vals, sc = brute_core(sup)
        if vals is not None:
            core_candidates.append((sc, sup, vals))
    core_candidates.sort(reverse=True)

    # ---- Combine best core with disjoint Sidon spikes ----
    for sc_core, sup, vals in core_candidates:
        if not timed_ok():
            break
        core_h = np.zeros(N)
        for pos, v in zip(sup, vals):
            core_h[pos] = v
        core_cmax = float(_autoconv(core_h).max())

        # Build a Sidon set of spike positions disjoint from sup.
        used = set(sup)
        # Try increasing spike counts
        for m in range(1, 12):
            if not timed_ok():
                break
            sidon = _find_sidon_set(m + len(sup), N, seed=m)
            if sidon is None:
                continue
            spike_pos = [p for p in sidon if p not in used]
            if len(spike_pos) < m:
                continue
            h = core_h.copy()
            for p in spike_pos[:m]:
                h[p] = 1.0
            # Verify peak didn't grow (spikes disjoint & Sidon => cross terms <=1,
            # but core*spike cross terms could exceed core_cmax? core_cmax is from core only;
            # adding spikes adds cross terms h_core * 1 which are <= max_core_value * ... 
            # To be safe, just measure.
            s = _ratio(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
            # Also try varying spike amplitude slightly
            for amp in [0.5, 1.0, 1.5]:
                h2v = core_h.copy()
                for p in spike_pos[:m]:
                    h2v[p] = amp
                try_candidate(h2v)

    # ---- Local polish of best_h via coordinate ascent on the ratio ----
    def polish(h, max_iter=400):
        h = h.copy()
        cur = _ratio(h)
        for it in range(max_iter):
            if not timed_ok():
                break
            improved = False
            for i in range(N):
                old = h[i]
                if old < 1e-12:
                    # try turning on
                    for v in [0.05, 0.1, 0.2, 0.5, 1.0]:
                        h[i] = v
                        s = _ratio(h)
                        if s > cur:
                            cur = s; improved = True
                    h[i] = 0.0
                    continue
                # multiplicative + additive perturbations
                for factor in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]:
                    h[i] = old * factor
                    s = _ratio(h)
                    if s > cur:
                        cur = s; improved = True
                for delta in [-0.1, -0.02, 0.02, 0.1, 0.3]:
                    nv = old + delta
                    if nv < 0:
                        continue
                    h[i] = nv
                    s = _ratio(h)
                    if s > cur:
                        cur = s; improved = True
                if not improved:
                    h[i] = old
            if not improved:
                break
        return h, cur

    if timed_ok():
        h_pol, s_pol = polish(best_h, max_iter=300)
        if s_pol > best_score:
            best_score = s_pol
            best_h = h_pol.copy()

    # Second polish round with finer steps
    if timed_ok():
        def polish_fine(h, max_iter=200):
            h = h.copy()
            cur = _ratio(h)
            for it in range(max_iter):
                if not timed_ok():
                    break
                improved = False
                for i in range(N):
                    old = h[i]
                    if old < 1e-12:
                        for v in [0.01, 0.05, 0.1]:
                            h[i] = v
                            s = _ratio(h)
                            if s > cur:
                                cur = s; improved = True
                        h[i] = 0.0
                        continue
                    for factor in [0.9, 0.95, 1.0, 1.05, 1.1]:
                        h[i] = old * factor
                        s = _ratio(h)
                        if s > cur:
                            cur = s; improved = True
                    if not improved:
                        h[i] = old
                if not improved:
                    break
            return h, cur
        h_f, s_f = polish_fine(best_h, max_iter=200)
        if s_f > best_score:
            best_score = s_f
            best_h = h_f.copy()

    return list(best_h)
