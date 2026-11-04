"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Strategy: Sidon-set structured initialization + differential evolution global search
+ local L-BFGS-B polish. Sidon sets minimize additive collisions in the autoconvolution,
directly reducing the peak value.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # (1/2)/DH

_DEADLINE = 150.0
_SAFETY = 15.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _sidon_set(n, max_val, rng):
    """Generate a Sidon set of size n from [0, max_val)."""
    # A Sidon set has the property that all pairwise sums are distinct.
    # Greedy construction: add elements that don't create duplicate sums.
    s = []
    used_sums = set()
    candidates = list(range(max_val))
    rng.shuffle(candidates)
    
    for c in candidates:
        if len(s) >= n:
            break
        # Check if adding c creates any duplicate sum
        ok = True
        for i in range(len(s)):
            for j in range(i, len(s)):
                ssum = s[i] + s[j]
                if ssum in used_sums:
                    ok = False
                    break
                used_sums.add(ssum)
            if not ok:
                break
        if ok:
            s.append(c)
    
    # If we didn't get enough, fill with random distinct values
    if len(s) < n:
        s = sorted(set(s + list(rng.choice(max_val, size=n, replace=False))))[:n]
    
    return sorted(s)


def _sidon_heights(s, rng):
    """Create nonnegative heights from a Sidon set with random amplitudes."""
    h = np.zeros(N)
    for pos in s:
        if 0 <= pos < N:
            h[pos] = rng.uniform(0.5, 2.0)
    return h


def _tapered_plateau(rng):
    """Raised-cosine tapered plateau shape."""
    center = N // 2
    width = rng.randint(10, 25)
    h = np.zeros(N)
    for i in range(N):
        d = abs(i - center)
        if d <= width:
            h[i] = 0.5 * (1 + np.cos(np.pi * d / width))
        else:
            h[i] = 0.0
    # Add small random perturbation
    h += rng.random(N) * 0.1
    return np.maximum(h, 0.0)


def _random_sparse(rng, density=0.3):
    """Random sparse nonnegative vector."""
    mask = rng.random(N) < density
    h = np.zeros(N)
    h[mask] = rng.uniform(0.1, 1.0, size=int(np.sum(mask)))
    return h


def _dft_flat_spectrum(rng):
    """Generate h from a flat power spectrum: h = |IFFT(z)|^2 with random phases."""
    z = np.zeros(N, dtype=complex)
    for k in range(N):
        z[k] = 1.0 + 1j * rng.random()
    h = np.abs(np.fft.ifft(z)) ** 2
    return h


def _local_polish(h0, time_budget):
    """Local L-BFGS-B polish on h-space."""
    t0 = time.time()
    
    def _over():
        return (time.time() - t0) > time_budget
    
    def obj(x):
        if _over():
            return np.nan
        x = np.maximum(x, 0.0)
        r = _ratio(x)
        return -r if r > 0 else 1e12
    
    def jac(x):
        if _over():
            return np.nan, np.zeros_like(x)
        x = np.maximum(x, 0.0)
        # Numerical gradient
        g = np.zeros_like(x)
        eps = 1e-6
        for i in range(N):
            xp = x.copy()
            xp[i] = max(xp[i] + eps, 0.0)
            xm = x.copy()
            xm[i] = max(xm[i] - eps, 0.0)
            g[i] = (obj(xp) - obj(xm)) / (2 * eps)
        return obj(x), g
    
    bounds = [(0.0, 10.0)] * N
    x0 = np.clip(h0, 0.0, 10.0)
    
    try:
        res = minimize(
            obj, x0,
            method='L-BFGS-B',
            jac=jac,
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
        )
        if res.x is not None:
            h_new = np.maximum(res.x, 0.0)
            v_new = _ratio(h_new)
            if v_new > 0:
                return h_new, v_new
    except Exception:
        pass
    
    return h0, _ratio(h0)


def construct():
    global _start
    _start = time.time()
    
    rng = np.random.default_rng(42)
    best_h = None
    best_v = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # Sidon-set based initializations
    for trial in range(5):
        s = _sidon_set(8, N, rng)
        h = _sidon_heights(s, rng)
        inits.append(h)
    
    # Tapered plateaus
    for trial in range(3):
        h = _tapered_plateau(rng)
        inits.append(h)
    
    # Random sparse
    for trial in range(3):
        h = _random_sparse(rng, density=0.25)
        inits.append(h)
    
    # DFT flat spectrum
    for trial in range(3):
        h = _dft_flat_spectrum(rng)
        inits.append(h)
    
    # Evaluate all initializations
    scored = []
    for h in inits:
        v = _ratio(h)
        scored.append((v, h))
        if v > best_v:
            best_v = v
            best_h = h.copy()
    scored.sort(key=lambda t: -t[0])
    
    # Local polish top initializations
    for v0, h0 in scored[:8]:
        if _time_left() <= 10.0:
            break
        h_pol, v_pol = _local_polish(h0, time_budget=min(8.0, _time_left() - 5.0))
        if v_pol > best_v:
            best_v = v_pol
            best_h = h_pol.copy()
    
    # Differential evolution global search
    if _time_left() > 40.0 and best_h is not None:
        try:
            # Use bounds informed by best solution
            h_best = best_h
            max_val = max(1.0, float(np.max(h_best))) * 2
            bounds_de = [(0.0, max_val)] * N
            
            def de_obj(x):
                x = np.maximum(x, 0.0)
                return -_ratio(x)
            
            # Run DE with moderate settings
            maxiter = 25
            popsize = 12
            de_res = differential_evolution(
                de_obj, bounds_de,
                seed=42, maxiter=maxiter, popsize=popsize,
                tol=1e-6, polish=False,
                updating='immediate',
                init='latinhypercube',
            )
            if de_res.x is not None:
                h_de = np.maximum(de_res.x, 0.0)
                v_de = _ratio(h_de)
                if v_de > best_v:
                    best_v = v_de
                    best_h = h_de.copy()
                    # Polish the DE result
                    if _time_left() > 15.0:
                        h_pol, v_pol = _local_polish(h_de, time_budget=min(10.0, _time_left() - 5.0))
                        if v_pol > best_v:
                            best_v = v_pol
                            best_h = h_pol.copy()
        except Exception:
            pass
    
    # Second round of local polish on current best with different budget
    if best_h is not None and _time_left() > 20.0:
        h_pol, v_pol = _local_polish(best_h, time_budget=min(15.0, _time_left() - 5.0))
        if v_pol > best_v:
            best_v = v_pol
            best_h = h_pol.copy()
    
    # Final safety: ensure valid output
    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
