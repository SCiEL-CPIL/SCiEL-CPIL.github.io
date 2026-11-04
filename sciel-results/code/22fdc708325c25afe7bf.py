import time
import numpy as np
from itertools import combinations

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _autoconv_lags(h):
    """Compute all lag values of autoconvolution (without the w factor).
    c[m] = sum_j h[j]*h[j+m] for m=0..N-1
    """
    c = np.zeros(N)
    c[0] = np.dot(h, h)
    for m in range(1, N):
        c[m] = np.dot(h[:N - m], h[m:])
    return c


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) using hard max."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    c = _autoconv_lags(h)
    acmax = w * c.max()
    if acmax <= 0:
        return 0.0
    # ||h||_2^2 = w * s2, ||h||_1 = w * s1, ||h*h||_inf = acmax
    # ratio = (w*s2) / (w*s1 * acmax) = s2 / (s1 * acmax)
    return s2 / (s1 * acmax)


def _is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums/differences are distinct).
    For autoconvolution, we need all pairwise differences to be distinct in magnitude.
    """
    indices = sorted(indices)
    if len(indices) <= 2:
        return True
    diffs = set()
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = indices[j] - indices[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True


def _find_sidon_sets(max_size=10, rng=None):
    """Find Sidon sets of various sizes using greedy and random search."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    sidon_sets = []
    
    # Greedy: build Sidon set by adding elements one at a time
    for start in range(N):
        current = [start]
        for candidate in range(N):
            if candidate in current:
                continue
            if _is_sidon_set(current + [candidate]):
                current.append(candidate)
                if len(current) > max_size:
                    break
        if len(current) > 1:
            sidon_sets.append(tuple(current))
    
    # Random search for larger Sidon sets
    for _ in range(200):
        size = rng.integers(4, min(12, N))
        indices = tuple(sorted(rng.choice(N, size=size, replace=False)))
        if _is_sidon_set(indices):
            sidon_sets.append(indices)
    
    # Deduplicate
    unique_sets = list(set(sidon_sets))
    
    return unique_sets


def _spike_pattern(indices):
    """Create a spike pattern at given indices with unit height."""
    h = np.zeros(N)
    for i in indices:
        h[i] = 1.0
    return h


def _refine_spikes(h, max_rounds=50, tol=1e-10):
    """Refine spike heights using coordinate descent."""
    h = np.array(h, dtype=float)
    best_f = _ratio(h)
    
    for rnd in range(max_rounds):
        if _elapsed() > DEADLINE - 10:
            break
        improved = False
        for i in range(N):
            if _elapsed() > DEADLINE - 10:
                break
            if h[i] == 0:
                # Try turning on this spike
                for val in [0.1, 0.5, 1.0, 2.0]:
                    h[i] = val
                    f = _ratio(h)
                    if f > best_f:
                        best_f = f
                        improved = True
                    else:
                        h[i] = 0
            else:
                # Try adjusting height
                orig = h[i]
                for val in [0.0, 0.5*orig, 0.1*orig, 2*orig, 5*orig]:
                    h[i] = val
                    f = _ratio(h)
                    if f > best_f:
                        best_f = f
                        improved = True
                    else:
                        h[i] = orig
                # Try turning off
                h[i] = 0
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    improved = True
                else:
                    h[i] = orig
        if not improved:
            break
    
    return h, best_f


def _refine_positions(h, max_rounds=20):
    """Try shifting spikes to better positions."""
    h = np.array(h, dtype=float)
    best_f = _ratio(h)
    active = np.where(h > 0)[0]
    
    for rnd in range(max_rounds):
        if _elapsed() > DEADLINE - 10:
            break
        improved = False
        for idx in active:
            if _elapsed() > DEADLINE - 10:
                break
            orig_val = h[idx]
            for new_idx in range(max(0, idx-3), min(N, idx+4)):
                if new_idx == idx:
                    continue
                # Check if new position is free
                if h[new_idx] > 0:
                    continue
                h[idx] = 0
                h[new_idx] = orig_val
                f = _ratio(h)
                if f > best_f:
                    best_f = f
                    improved = True
                    active[active == idx] = new_idx
                else:
                    h[new_idx] = 0
                    h[idx] = orig_val
        if not improved:
            break
    
    return h, best_f


def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_f = _ratio(best_h)
    
    # Phase 1: Try all single spikes
    for i in range(N):
        h = _spike_pattern([i])
        f = _ratio(h)
        if f > best_f:
            best_f = f
            best_h = h.copy()
    
    # Phase 2: Try pairs of spikes
    for i in range(N):
        for j in range(i+1, N):
            if _elapsed() > DEADLINE - 60:
                break
            h = _spike_pattern([i, j])
            f = _ratio(h)
            if f > best_f:
                best_f = f
                best_h = h.copy()
    
    # Phase 3: Try Sidon sets
    if _elapsed() < DEADLINE - 40:
        rng = np.random.default_rng(42)
        sidon_sets = _find_sidon_sets(max_size=10, rng=rng)
        
        for sset in sidon_sets:
            if _elapsed() > DEADLINE - 30:
                break
            h = _spike_pattern(list(sset))
            f = _ratio(h)
            if f > best_f:
                best_f = f
                best_h = h.copy()
    
    # Phase 4: Random spike patterns
    if _elapsed() < DEADLINE - 25:
        rng = np.random.default_rng(99)
        for _ in range(500):
            if _elapsed() > DEADLINE - 20:
                break
            size = rng.integers(2, min(15, N))
            indices = rng.choice(N, size=size, replace=False)
            h = _spike_pattern(list(indices))
            f = _ratio(h)
            if f > best_f:
                best_f = f
                best_h = h.copy()
    
    # Phase 5: Refine best using coordinate descent
    if _elapsed() < DEADLINE - 15:
        best_h, best_f = _refine_spikes(best_h, max_rounds=50)
    
    # Phase 6: Try position refinement
    if _elapsed() < DEADLINE - 10:
        best_h, best_f = _refine_positions(best_h, max_rounds=20)
    
    # Phase 7: Final refinement
    if _elapsed() < DEADLINE - 5:
        best_h, best_f = _refine_spikes(best_h, max_rounds=100)
    
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]
