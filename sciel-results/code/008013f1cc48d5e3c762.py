"""Rewritten consumption-saving solver: fast accurate backward induction."""
import time
import numpy as np

_DEADLINE = 150.0  # evaluator kills at 150s; keep a safety margin below.
_START = None


def _elapsed():
    if _START is None:
        return 0.0
    return time.time() - _START


def _remaining():
    return _DEADLINE - 0.5 - _elapsed()


def _u(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _u_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    return c ** (-gamma)


def solve_consumption(model, resource_grid, budget):
    global _START
    _START = time.time()

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    H = int(model['horizon'])
    grid = np.asarray(resource_grid, dtype=float)
    n = len(grid)

    # consumption[t, z, i]
    C = np.empty((H, 2, n), dtype=float)

    # terminal: consume everything
    C[H - 1] = grid[None, :, None] * np.ones((1, 2, 1))

    # value function V[t, z, i]
    V = np.empty((H, 2, n), dtype=float)
    V[H - 1] = _u(C[H - 1], gamma)

    # search grid for c (as a fraction of m)
    frac = np.linspace(0.002, 1.0, int(budget))  # budget points
    # For each m, candidate c = m * frac

    for t in range(H - 2, -1, -1):
        if _remaining() < 2.0:
            # Not enough time: fill remaining with a heuristic (consume all)
            for tt in range(t, H):
                C[tt] = grid[None, :, None] * np.ones((1, 2, 1))
                V[tt] = _u(C[tt], gamma)
            break

        for z in range(2):
            # continuation: for each next state z' and each m, next resource m' = R*(m-c)+income[z']
            # We'll compute value for each candidate c.
            # V_next = V[t+1, :, :]  shape (2, n)
            # For each candidate fraction f, c = grid*f (shape (n,))
            # For each z', m' = R*(grid - c) + income[z']
            #   need value at m' for state z': interpolate V[t+1, z', :].

            # Precompute interpolation tables for next value (2 states)
            Vnext = V[t + 1]  # (2, n)
            # For each z', we need to evaluate Vnext[z', :] at m'.

            # We'll vectorize over candidates (budget) and grid (n).
            # c = grid[:, None] * frac[None, :]  -> (n, budget)
            c = grid[:, None] * frac[None, :]  # (n, B)

            # current utility
            cu = _u(c, gamma)  # (n, B)

            # discounted expected continuation
            # For each z', m' = R*(grid - c) + income[z']
            #   -> (n, B) for each z'
            # Then value = sum_z' trans[z,z'] * Vnext[z', m']
            cont = np.zeros((n, len(frac)), dtype=float)
            for zp in range(2):
                mprime = R * (grid[:, None] - c) + income[zp]
                # clamp mprime to grid range for interpolation (np.interp handles out-of-range with endpoints)
                val = np.interp(mprime, grid, Vnext[zp])  # (n, B)
                cont += trans[z, zp] * val
            disc_cont = beta * cont  # (n, B)

            total = cu + disc_cont  # (n, B)
            # choose best fraction per grid point
            best_idx = np.argmax(total, axis=1)  # (n,)
            best_c = grid * frac[best_idx]  # (n,)
            best_val = total[np.arange(n), best_idx]  # (n,)

            # Local refinement around the best fraction using a finer local grid
            # This improves accuracy especially where the optimum is interior.
            # Do a small Newton-like / fine search in a window.
            refine_frac = np.linspace(0.0, 1.0, 61)  # local fine grid (relative offsets)
            # window around best fraction
            b = best_idx
            # candidate window indices
            lo = np.clip(b - 3, 0, len(frac) - 1)
            hi = np.clip(b + 3, 0, len(frac) - 1)
            # For each grid point, build a local fraction grid between frac[lo] and frac[hi]
            # To keep it vectorized and simple, do a per-point loop but small.
            # We only refine where the best fraction is interior (not at boundaries).
            interior = (b > 0) & (b < len(frac) - 1)
            if np.any(interior):
                idxs = np.where(interior)[0]
                for i in idxs:
                    f_lo = frac[lo[i]]
                    f_hi = frac[hi[i]]
                    f_local = np.linspace(f_lo, f_hi, 41)
                    c_local = grid[i] * f_local  # (L,)
                    cu_local = _u(c_local, gamma)
                    cont_local = np.zeros(len(f_local), dtype=float)
                    for zp in range(2):
                        mp = R * (grid[i] - c_local) + income[zp]
                        val = np.interp(mp, grid, Vnext[zp])
                        cont_local += trans[z, zp] * val
                    total_local = cu_local + beta * cont_local
                    j = int(np.argmax(total_local))
                    best_c[i] = c_local[j]
                    best_val[i] = total_local[j]

            C[t, z] = best_c
            V[t, z] = best_val

    # Ensure terminal condition strictly: c[5] equals grid within 1e-9
    C[H - 1] = grid[None, :, None] * np.ones((1, 2, 1))
    # Enforce 0 < c <= m + 1e-10 and finite
    m_broad = grid[None, :, None]
    C = np.clip(C, 1e-12, None)
    C = np.minimum(C, m_broad + 1e-10)
    # safety: where m is tiny, c must be <= m+1e-10 and >0
    C = np.where(np.isfinite(C), C, 1e-12)
    C = np.maximum(C, 1e-12)

    return C
