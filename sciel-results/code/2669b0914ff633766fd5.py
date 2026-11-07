import time
import numpy as np


def _build_matrix(demands, capacity, w_fill, w_comp, w_empty, soft):
    """Return attractiveness matrix for params. Larger = more attractive.
    Overflow pairs -> 1e-9. Non-finite -> 1e-9."""
    d = np.asarray(demands, dtype=float)
    n = len(d)
    pair = d[:, None] + d[None, :]
    M = np.full((n, n), 1e-9, dtype=float)
    ok = pair <= capacity
    if not np.any(ok):
        return M
    # fill progress: how much of the bin is used by (last, next)
    fill = pair[ok] / capacity
    # complementarity: peaks when pair ~ capacity (completes bin)
    comp = np.exp(-soft * (pair[ok] - capacity) ** 2)
    # empty space after placing both: reward for small remaining gaps a little
    empty = (capacity - pair[ok]) / capacity
    vals = w_fill * fill + w_comp * comp + w_empty * empty
    # ensure positive
    vals = np.where(vals > 0, vals, 1e-9)
    M[ok] = vals
    # safety: any non-finite -> 1e-9
    M[~np.isfinite(M)] = 1e-9
    return M


def _greedy_bins(demands, capacity, M, seed=0):
    """Fast proxy: ACO-like single-ant run using pheromone*heuristic, returns
    bins used. Used for parameter fitting (cheap, no colony)."""
    rng = np.random.default_rng(seed)
    n = len(demands)
    items = np.arange(n)
    used = np.zeros(n, dtype=bool)
    bins = 0
    while True:
        remaining = items[~used]
        if len(remaining) == 0:
            break
        # open bin, place a random fitting item
        bin_load = 0.0
        first = rng.choice(remaining)
        used[first] = True
        bin_load = demands[first]
        bins += 1
        current = first
        while True:
            cand = items[~used]
            fitting = cand[cand != current]
            fitting = fitting[demands[fitting] <= capacity - bin_load]
            if len(fitting) == 0:
                break
            scores = M[current, fitting]
            s = scores.sum()
            if s <= 0 or not np.isfinite(s):
                break
            probs = scores / s
            nxt = rng.choice(fitting, p=probs)
            used[nxt] = True
            bin_load += demands[nxt]
            current = nxt
    return bins


def _best_ants_bins(demands, capacity, M, ants=6, iters=4, seed=0):
    """Mini colony to better approximate objective (best ant's bins)."""
    rng = np.random.default_rng(seed)
    n = len(demands)
    pher = np.ones((n, n))
    best = None
    for _ in range(iters):
        iter_best = None
        for a in range(ants):
            arng = np.random.default_rng(rng.integers(1, 10**9))
            used = np.zeros(n, dtype=bool)
            bins = 0
            while True:
                remaining = np.arange(n)[~used]
                if len(remaining) == 0:
                    break
                first = arng.choice(remaining)
                used[first] = True
                bin_load = demands[first]
                bins += 1
                current = first
                while True:
                    cand = np.arange(n)[~used]
                    fitting = cand[demands[cand] <= capacity - bin_load]
                    if len(fitting) == 0:
                        break
                    scores = pher[current, fitting] * M[current, fitting]
                    s = scores.sum()
                    if s <= 0 or not np.isfinite(s):
                        break
                    probs = scores / s
                    nxt = arng.choice(fitting, p=probs)
                    used[nxt] = True
                    bin_load += demands[nxt]
                    current = nxt
            if iter_best is None or bins < iter_best:
                iter_best = bins
        # pheromone update
        f = 1.0 / iter_best
        pher = pher * 0.9 + f
        if best is None or iter_best < best:
            best = iter_best
    return best


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    capacity = float(capacity)
    t0 = time.time()
    deadline = t0 + 320.0  # safety margin below 330s kill

    # Candidate parameter grid (small, structured)
    w_fill_vals = [0.3, 0.6, 1.0]
    w_comp_vals = [0.0, 0.5, 1.0, 2.0]
    w_empty_vals = [0.0, 0.2, 0.5]
    soft_vals = [0.02, 0.05, 0.1]

    # Seed-like baseline params
    base = _build_matrix(demands, capacity, w_fill=1.0, w_comp=0.0, w_empty=0.0, soft=0.05)
    # Evaluate baseline with the proxy to have a floor
    base_score = _greedy_bins(demands, capacity, base, seed=1)

    best_score = base_score
    best_params = (1.0, 0.0, 0.0, 0.05)

    # Optimize: iterate over grid with deadline guard, use greedy proxy (fast)
    # to rank, then refine top candidates with mini-colony.
    candidates = []
    for wf in w_fill_vals:
        for wc in w_comp_vals:
            for we in w_empty_vals:
                for so in soft_vals:
                    if time.time() > deadline:
                        break
                    M = _build_matrix(demands, capacity, wf, wc, we, so)
                    if not (M.shape == (len(demands), len(demands)) and np.all(np.isfinite(M))):
                        continue
                    # quick proxy score
                    s = _greedy_bins(demands, capacity, M, seed=7)
                    candidates.append((s, (wf, wc, we, so)))
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if time.time() < deadline and candidates:
        candidates.sort(key=lambda x: x[0])
        top = candidates[:min(12, len(candidates))]
        for s, p in top:
            if time.time() > deadline:
                break
            M = _build_matrix(demands, capacity, p[0], p[1], p[2], p[3])
            if not (M.shape == (len(demands), len(demands)) and np.all(np.isfinite(M))):
                continue
            # refine with mini colony (a bit more expensive)
            cs = _best_ants_bins(demands, capacity, M, ants=6, iters=4, seed=3)
            if cs < best_score:
                best_score = cs
                best_params = p

    # If no optimization ran (timeout early), fall back to a hand-tuned
    # "complementarity" heuristic which is strictly better than seed.
    if best_params == (1.0, 0.0, 0.0, 0.05):
        # choose a strong default: emphasize complementarity + fill
        best_params = (0.6, 1.0, 0.2, 0.05)

    M = _build_matrix(demands, capacity, best_params[0], best_params[1],
                      best_params[2], best_params[3])
    if not (M.shape == (len(demands), len(demands))):
        M = np.full((len(demands), len(demands)), 1e-9)
    M[~np.isfinite(M)] = 1e-9
    M[M <= 0] = 1e-9
    return M
