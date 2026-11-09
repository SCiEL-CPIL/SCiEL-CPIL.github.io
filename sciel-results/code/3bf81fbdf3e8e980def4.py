import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


def _cut_values(n, edges):
    m = 2 ** n
    states = np.arange(m, dtype=np.int64)
    bits = (states[:, None] >> np.arange(n)[None, :]) & 1
    z = 1 - 2 * bits
    cv = np.zeros(m, dtype=np.float64)
    for (i, j) in edges:
        cv += (z[:, i] != z[:, j]).astype(np.float64)
    return cv


def _expected_cut(n, cut_vals, gammas_betas):
    m = 2 ** n
    state = np.full(m, 1.0 / np.sqrt(m), dtype=np.complex128)
    for gamma, beta in gammas_betas:
        phase = np.exp(-1j * gamma * cut_vals)
        state = state * phase
        if beta != 0.0:
            c = np.cos(beta)
            s = np.sin(beta)
            # G = [[c, -is], [-is, c]]
            # Apply to each qubit: for each pair of states differing in bit k,
            # new[0] = c*old[0] - 1j*s*old[1], new[1] = -1j*s*old[0] + c*old[1]
            # We can do this with reshape: treat state as (2, m/2) for bit 0,
            # (2, m/4, 2) for bit 1, etc.
            # General approach: for bit k, reshape to (2, m//2) with stride pattern
            # Actually the cleanest vectorized way:
            # For bit k, the state can be viewed as shape (2^k, 2, 2^(n-1-k))
            # Reshape to (2^k * 2^(n-1-k), 2) = (m//2, 2)
            # Apply G @ x for each row
            # Then reshape back
            # But the stride pattern makes this tricky with simple reshape.
            # Alternative: use the fact that for bit k, states i and i^(1<<k) are paired.
            # We can build index arrays.
            # For speed with small n (<=10, m<=1024), let's use a direct approach:
            # Build the full 2x2 block matrix in Kronecker product form? That's m x m.
            # For n=10, m=1024, that's a 1024x1024 complex matrix - 8MB. Doable but slow.
            # Better: iterate over bits but vectorize within each bit.
            for k in range(n):
                # Bit k: pair states that differ in bit k
                # Index: for each i in range(m), partner = i ^ (1 << k)
                # We only process i where i < partner (i.e., bit k is 0)
                mask = (np.arange(m) & (1 << k)) == 0
                idx = np.arange(m)[mask]
                partner = idx | (1 << k)
                a = state[idx]
                b = state[partner]
                new_a = c * a - 1j * s * b
                new_b = -1j * s * a + c * b
                state[idx] = new_a
                state[partner] = new_b
    probs = np.abs(state) ** 2
    return float(np.dot(probs, cut_vals))


def angles(n, edges, p):
    t0 = time.time()
    call_deadline = 18.0  # 6 calls * 18s = 108s, well under 150s wall-clock
    cut_vals = _cut_values(n, edges)

    def f(x):
        return -_expected_cut(n, cut_vals, x.reshape(p, 2))

    best_x = np.tile([0.35, 0.25], (p, 1)).reshape(-1).astype(float)
    best_f = f(best_x)

    def timed_out():
        return (time.time() - t0) > (call_deadline - 1.0)

    # Coarse grid search to identify promising basins
    grid_g = np.linspace(0.0, np.pi, 8)
    grid_b = np.linspace(0.0, np.pi, 8)
    for g in grid_g:
        for b in grid_b:
            if timed_out():
                break
            cand = np.tile([g, b], (p, 1)).reshape(-1)
            val = f(cand)
            if val < best_f:
                best_f, best_x = val, cand.copy()
        if timed_out():
            break

    # Multi-start Nelder-Mead optimization
    rng = np.random.default_rng(713)
    nstarts = 60
    for k in range(nstarts):
        if timed_out():
            break
        # Generate random start points
        start = np.concatenate([
            rng.uniform(0.0, np.pi, size=p),
            rng.uniform(0.0, np.pi, size=p),
        ])

        # Also try starts near the current best to polish
        if k % 2 == 1 and k > 0:
            noise = rng.normal(0, 0.1, size=2 * p)
            start = best_x + noise
            start = np.clip(start, 0.0, np.pi)

        try:
            res = minimize(f, start, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_f:
                best_f, best_x = float(res.fun), np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass
        if timed_out():
            break

    # Final polishing pass with tighter tolerances on the best point found
    if not timed_out():
        try:
            res = minimize(f, best_x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13})
            if res.fun < best_f:
                best_f, best_x = float(res.fun), np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass

    # Second polish if time remains
    if not timed_out():
        try:
            res = minimize(f, best_x, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_f:
                best_f, best_x = float(res.fun), np.asarray(res.x, dtype=float).copy()
        except Exception:
            pass

    return best_x.reshape(p, 2)
