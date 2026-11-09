# EVOLVE-BLOCK-START
import numpy as np

_N = 64
_A = (np.arange(1, _N + 1, dtype=np.float64)) / _N
_DA = np.diff(_A)                      # d_j = a_{j+1} - a_j > 0
_T = np.cumsum(_DA)                    # T_j = a_j - a_0
_T = _T - _T[0]
_T = _T / _DA                          # c_j = T_j / d_j (roots of the min switch)
_K = _N - 1


def _area_fast(x):
    """Analytic area of the union of the 64 Kakeya triangles (lower is better)."""
    order = np.argsort(x)
    xs = x[order]
    idx = order[1:]
    g = xs[1:] - xs[:-1]               # consecutive gaps
    d = _DA[idx]                        # slopes of the left edges
    c = _T[idx]                         # switch points
    w = 1.0 / _N
    t = np.clip(c, 0.0, 1.0)
    min_t = np.minimum(g, d * t)
    term = min_t + (w - d * t) * np.maximum(g - d * t, 0.0)
    return w + term.sum()


def _init_population(popsize, seed):
    rng = np.random.default_rng(seed)
    pop = np.empty((popsize, _N), dtype=np.float64)
    for p in range(popsize):
        base = rng.uniform(-1.0, 1.0, size=_N)
        base -= base.mean()
        pop[p] = base
    return pop


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the union area of 64 Kakeya needle triangles via differential
    evolution (JADE-style), using a fast analytic area evaluation.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    budget = 300000.0
    popsize = 24
    F = 0.5
    CR = 0.7

    pop = _init_population(popsize, 12345)
    evals = popsize
    scores = np.array([_area_fast(p) for p in pop])
    best_i = int(np.argmin(scores))
    best_x = pop[best_i].copy()
    best_score = scores[best_i]
    mean_score = scores.mean()
    rng = np.random.default_rng(987654321)

    while evals < budget:
        # JADE adaptation of F and CR
        F = 0.5 + 0.3 * (0.5 + 0.5 * np.sin(0.001 * evals))
        CR = 0.5 + 0.2 * (0.5 + 0.5 * np.cos(0.0013 * evals))

        for p in range(popsize):
            if evals >= budget:
                break
            # pick three distinct indices different from p
            while True:
                r1, r2, r3 = (int(v) for v in rng.integers(0, popsize, size=3))
                if r1 != p and r2 != p and r3 != p and r1 != r2 and r1 != r3 and r2 != r3:
                    break
            v = pop[p] + F * (pop[r1] - pop[r2]) + 0.1 * (best_x - pop[p])
            # binomial crossover
            mask = rng.random(_N) < CR
            if not mask.any():
                mask[rng.integers(_N)] = True
            trial = np.where(mask, v, pop[p])
            trial_score = _area_fast(trial)
            evals += 1
            if trial_score < scores[p]:
                scores[p] = trial_score
                pop[p] = trial
                if trial_score < best_score:
                    best_score = trial_score
                    best_x = trial.copy()
        mean_score = scores.mean()

    return np.asarray(best_x, dtype=np.float64).ravel()


# EVOLVE-BLOCK-END
