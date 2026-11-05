"""Autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 via Chebyshev basis + DE + polish."""
import numpy as np
import time

N = 400
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under 150s kill

# Chebyshev basis of degree 0..K on x in [-1,1] mapped from indices
def _chebyshev_matrix(K):
    idx = np.arange(N, dtype=np.float64)
    x = 2.0 * idx / (N - 1.0) - 1.0  # in [-1,1]
    T = np.zeros((K + 1, N), dtype=np.float64)
    T[0] = 1.0
    if K >= 1:
        T[1] = x
    for k in range(2, K + 1):
        T[k] = 2.0 * x * T[k - 1] - T[k - 2]
    return T

_T = None
_K = 12  # degrees 0..12 -> 13 coefficients

def _get_T():
    global _T
    if _T is None:
        _T = _chebyshev_matrix(_K)
    return _T

def _value(h):
    s = np.sum(h)
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _h_from_coefs(c):
    T = _get_T()
    return T @ c

def _objective(c):
    h = _h_from_coefs(c)
    return _value(h)

def _de_search(c0, max_iter, pop, tol, F=0.8, CR=0.9, max_wall=110.0):
    """Differential evolution on coefficient space, with wall-clock guard."""
    dim = len(c0)
    pop = max(pop, dim + 1)
    # Initial population: c0 plus random perturbations
    rng = np.random.default_rng(12345)
    P = np.empty((pop, dim), dtype=np.float64)
    P[0] = c0
    for i in range(1, pop):
        P[i] = c0 + rng.normal(0.0, 0.5, size=dim)
    # Evaluate
    Fvals = np.array([_objective(P[i]) for i in range(pop)], dtype=np.float64)
    # Replace any inf with large penalty
    Fvals = np.where(np.isfinite(Fvals), Fvals, 1e12)
    best_idx = int(np.argmin(Fvals))
    best_c = P[best_idx].copy()
    best_f = Fvals[best_idx]
    no_improve = 0
    it = 0
    while it < max_iter:
        if time.time() > max_wall:
            break
        for i in range(pop):
            if time.time() > max_wall:
                break
            # Choose 3 distinct indices != i
            while True:
                a, b, c = rng.integers(0, pop, size=3)
                if a != i and b != i and c != i and a != b and b != c and a != c:
                    break
            trial = P[a] + F * (P[b] - P[c])
            # Mix with CR
            r = rng.random()
            j0 = int(rng.integers(0, dim))
            for j in range(dim):
                if rng.random() < CR or j == j0:
                    pass
            mask = rng.random(dim) < CR
            mask[j0] = True
            trial = np.where(mask, trial, P[i])
            ft = _objective(trial)
            if not np.isfinite(ft):
                ft = 1e12
            if ft <= Fvals[i]:
                P[i] = trial
                Fvals[i] = ft
                if ft < best_f:
                    best_f = ft
                    best_c = trial.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        it += 1
        if no_improve > pop * 3 and best_f < 2.0:
            # Try to restart with tighter perturbation around best
            for i in range(1, pop):
                P[i] = best_c + rng.normal(0.0, 0.1, size=dim)
            Fvals[0] = best_f
            P[0] = best_c
            for i in range(1, pop):
                Fvals[i] = _objective(P[i])
                if not np.isfinite(Fvals[i]):
                    Fvals[i] = 1e12
            best_idx = int(np.argmin(Fvals))
            best_f = Fvals[best_idx]
            best_c = P[best_idx].copy()
            no_improve = 0
    return best_c, best_f

def _local_polish(c, steps=200, tol=1e-10):
    """Coordinate descent + small random search to polish."""
    best_c = c.copy()
    best_f = _objective(best_c)
    if not np.isfinite(best_f):
        return best_c, best_f
    rng = np.random.default_rng(999)
    dim = len(best_c)
    for step in range(steps):
        if time.time() > _DEADLINE - 10.0:
            break
        improved = False
        # Coordinate descent
        for j in range(dim):
            best_local = best_f
            best_val = best_c[j]
            # Try a small set of perturbations
            for delta in [1e-4, -1e-4, 1e-3, -1e-3, 5e-3, -5e-3, 1e-2, -1e-2]:
                trial = best_c.copy()
                trial[j] += delta
                f = _objective(trial)
                if np.isfinite(f) and f < best_local:
                    best_local = f
                    best_val = trial[j]
            if best_local < best_f - 1e-12:
                best_c[j] = best_val
                best_f = best_local
                improved = True
        # Random small moves
        for _ in range(3):
            trial = best_c + rng.normal(0.0, 1e-3, size=dim)
            f = _objective(trial)
            if np.isfinite(f) and f < best_f:
                best_c = trial
                best_f = f
                improved = True
        if not improved and step > 20:
            break
    return best_c, best_f

def construct():
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.ones(N)
    T = _get_T()
    dim = _K + 1
    # Initial: uniform -> c0 with only T[0] nonzero
    c0 = np.zeros(dim, dtype=np.float64)
    c0[0] = 1.0  # gives h = 1s
    # First, try a quick DE to get a good starting point
    c_best, f_best = _de_search(c0, max_iter=120, pop=40, tol=1e-8, max_wall=90.0)
    # Polish
    c_polished, f_polished = _local_polish(c_best, steps=150)
    h = _h_from_coefs(c_polished)
    # Ensure sum != 0 and finite
    if not np.all(np.isfinite(h)) or np.sum(h) == 0.0:
        # Fallback: try a few simple structured candidates
        candidates = []
        # Alternating sign blocks
        for block in [10, 20, 50, 100]:
            h_cand = np.ones(N)
            for i in range(0, N, block):
                h_cand[i:i+block] = -1.0
            candidates.append(h_cand)
        # Linear ramp
        h_cand = np.linspace(1, -1, N)
        candidates.append(h_cand)
        # Cosine
        h_cand = np.cos(2 * np.pi * np.arange(N) / N)
        candidates.append(h_cand)
        best_h = None
        best_v = np.inf
        for hc in candidates:
            v = _value(hc)
            if v < best_v:
                best_v = v
                best_h = hc
        if best_h is not None:
            return best_h
        return np.ones(N)
    # Verify improvement over seed
    v = _value(h)
    if v >= 2.0:
        # Try seed
        return np.ones(N)
    return h
