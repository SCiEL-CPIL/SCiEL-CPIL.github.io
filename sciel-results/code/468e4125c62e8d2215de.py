"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import time

_N = 20
_MAX_VAL = 10000
_DEADLINE = time.time() + 85.0  # safety margin under 90s kill

def _score(a):
    """Return max(|A+A|, |A*A|) for sorted distinct positive integers a."""
    a = np.asarray(a, dtype=np.int64)
    if len(a) != _N or np.any(a < 1) or np.any(a > _MAX_VAL):
        return 10**9
    if len(set(a.tolist())) != _N:
        return 10**9
    s = a[:, None] + a[None, :]
    p = a[:, None] * a[None, :]
    return max(len(np.unique(s)), len(np.unique(p)))

def _is_valid(a):
    a = list(a)
    return (len(a) == _N and len(set(a)) == _N and
            all(isinstance(x, int) and 1 <= x <= _MAX_VAL for x in a))

def _from_gp(a, b):
    """Generate 20 distinct integers from GP a*b^k, clipped to [1, 10000]."""
    if a < 1 or b <= 0:
        return None
    vals = a * (b ** np.arange(_N))
    # Round to nearest int, ensure positive
    vals = np.maximum(1, np.round(vals).astype(np.int64))
    # Clip to max
    vals = np.minimum(vals, _MAX_VAL)
    # Sort and deduplicate; if not 20 distinct, pad or fail
    vals = np.sort(vals)
    # If duplicates, try small perturbations
    if len(np.unique(vals)) < _N:
        return None
    # If still not exactly 20 after dedup (shouldn't happen if unique count == 20)
    unique = np.unique(vals)
    if len(unique) != _N:
        return None
    return unique.tolist()

def _gp_objective(params):
    """Objective for optimizer: minimize max sumset/product set size of rounded GP."""
    log_a, log_b = params
    a = np.exp(log_a)
    b = np.exp(log_b)
    if a < 0.5 or b < 0.5:
        return 10**9
    vals = a * (b ** np.arange(_N))
    vals = np.maximum(1, np.round(vals).astype(np.int64))
    vals = np.minimum(vals, _MAX_VAL)
    vals = np.sort(vals)
    if len(np.unique(vals)) < _N:
        return 10**9
    s = vals[:, None] + vals[None, :]
    p = vals[:, None] * vals[None, :]
    return max(len(np.unique(s)), len(np.unique(p)))

def _local_search(a, max_iter=500):
    """Local swap search: try swapping elements to reduce max(|A+A|,|A*A|)."""
    a = np.array(a, dtype=np.int64)
    best_score = _score(a)
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(_N):
            for j in range(i + 1, _N):
                # Try swapping a[i] and a[j] with small perturbations
                for delta in [-2, -1, 1, 2]:
                    new_a = a.copy()
                    new_a[i] = a[i] + delta
                    if new_a[i] < 1 or new_a[i] > _MAX_VAL:
                        continue
                    if new_a[i] in set(new_a.tolist()) - {a[i]}:
                        continue
                    sc = _score(new_a)
                    if sc < best_score:
                        a = new_a
                        best_score = sc
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return a.tolist(), best_score

def construct():
    start_time = time.time()
    best_set = None
    best_score = 10**9

    # Multi-start Nelder-Mead on log-space GP parameters
    random.seed(42)
    np.random.seed(42)

    # Generate diverse starting points in log-space
    # a in [1, 100], b in [1.01, 2.0]
    starts = []
    for _ in range(30):
        log_a = np.log(random.uniform(1.0, 50.0))
        log_b = np.log(random.uniform(1.02, 1.8))
        starts.append([log_a, log_b])
    # Add some structured starts
    for base in [1.1, 1.2, 1.3, 1.5, 1.7]:
        for a0 in [2, 5, 10, 20, 50]:
            starts.append([np.log(float(a0)), np.log(base)])

    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_gp_objective, start, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-4})
            log_a, log_b = res.x
            a = np.exp(log_a)
            b = np.exp(log_b)
            vals = a * (b ** np.arange(_N))
            vals = np.maximum(1, np.round(vals).astype(np.int64))
            vals = np.minimum(vals, _MAX_VAL)
            vals = np.sort(vals)
            unique = np.unique(vals)
            if len(unique) != _N:
                continue
            candidate = unique.tolist()
            if not _is_valid(candidate):
                continue
            sc = _score(candidate)
            if sc < best_score:
                best_score = sc
                best_set = candidate
        except Exception:
            continue

    # Also try a few random sets as fallback
    for _ in range(50):
        if time.time() > _DEADLINE - 5:
            break
        candidate = sorted(random.sample(range(1, _MAX_VAL + 1), _N))
        sc = _score(candidate)
        if sc < best_score:
            best_score = sc
            best_set = candidate

    # Local search polish on best
    if best_set is not None:
        polished, sc = _local_search(best_set, max_iter=300)
        if sc < best_score:
            best_score = sc
            best_set = polished

    # Fallback
    if best_set is None:
        best_set = list(range(1, 21))

    return best_set
# EVOLVE-BLOCK-END
