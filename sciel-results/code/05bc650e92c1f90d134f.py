# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_length_1d(starts):
    """Compute the length of the union of intervals [starts[i], starts[i]+w] for w=1/8."""
    n = len(starts)
    if n == 0:
        return 0.0
    order = np.argsort(starts)
    s_sorted = starts[order]
    w = 1.0 / 8.0
    # Greedy merge: union length = sum of (end of merged - start of merged)
    total = 0.0
    cur_start = s_sorted[0]
    cur_end = cur_start + w
    for k in range(1, n):
        s_k = s_sorted[k]
        e_k = s_k + w
        if s_k <= cur_end:
            if e_k > cur_end:
                cur_end = e_k
        else:
            total += cur_end - cur_start
            cur_start = s_k
            cur_end = e_k
    total += cur_end - cur_start
    return total


def _volume_from_bases(x_bases, y_bases):
    """Compute the union volume given 8 x-bases and 8 y-bases."""
    lx = _union_length_1d(x_bases)
    ly = _union_length_1d(y_bases)
    return lx * ly


def _build_positions(x_bases, y_bases):
    """Build the (64, 2) array from 8 x-bases and 8 y-bases."""
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = x_bases[i_flat]
    base_y = y_bases[j_flat]
    return np.stack([base_x, base_y], axis=1)


def _validate_and_volume(positions):
    """Check validity and compute the exact Monte Carlo volume as specified."""
    if not np.all(np.isfinite(positions)):
        return None
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    N = 8
    i = positions[:, 0]  # placeholder, recompute properly
    # Recompute i, j from row index
    idx = np.arange(64)
    i_idx = idx // N
    j_idx = idx % N
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i_idx + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j_idx + 1) / N)
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return None
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0, 1, 500000)
    # Check if each point is inside at least one tube
    # Tube (i,j) at height c: x in [x_k + c*i/8, x_k + c*i/8 + 1/8], y in [y_k + c*j/8, y_k + c*j/8 + 1/8]
    # For efficiency, vectorize over tubes for each point is too slow; instead check per-tube
    # Actually with 500000 points and 64 tubes, we can do it in chunks
    inside = np.zeros(500000, dtype=bool)
    for k in range(64):
        ik = i_idx[k]
        jk = j_idx[k]
        xk = x_k[k]
        yk = y_k[k]
        x_lo = xk + c * ik / N
        x_hi = x_lo + 1.0 / N
        y_lo = yk + c * jk / N
        y_hi = y_lo + 1.0 / N
        in_tube = (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
        inside |= in_tube
    frac = np.mean(inside)
    V = frac * area
    if V <= 0:
        return None
    return V


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds before deadline to stop

    # Keich binary construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    best_positions = _build_positions(keich, keich)
    best_volume = _volume_from_bases(keich, keich)
    if best_volume is None or best_volume <= 0:
        # Fallback
        best_positions = _build_positions(np.zeros(N), np.zeros(N))
        best_volume = _volume_from_bases(np.zeros(N), np.zeros(N))

    # Objective: minimize product of 1D union lengths
    def objective(params):
        x_b = params[:N]
        y_b = params[N:]
        return _volume_from_bases(x_b, y_b)

    # Multi-start Nelder-Mead optimization
    n_params = 2 * N
    rng_opt = np.random.default_rng(42)

    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break

        if r == 0:
            x0 = np.concatenate([keich, keich])
        else:
            # Perturb from best or from keich
            if r % 3 == 0:
                x0 = np.concatenate([keich, keich]) + rng_opt.normal(0, 0.05, n_params)
            elif r % 3 == 1:
                # Random in a reasonable range
                x0 = rng_opt.uniform(-0.5, 1.5, n_params)
            else:
                # Perturb from best
                best_params = np.concatenate([best_positions[::N, 0][:N], best_positions[::N, 1][:N]])
                x0 = best_params + rng_opt.normal(0, 0.03, n_params)

        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.success or res.fun < best_volume:
                new_vol = res.fun
                if new_vol < best_volume:
                    x_b = res.x[:N]
                    y_b = res.x[N:]
                    best_volume = new_vol
                    best_positions = _build_positions(x_b, y_b)
        except Exception:
            pass

    # Final validation with the exact Monte Carlo method
    final_V = _validate_and_volume(best_positions)
    if final_V is None:
        # Return the keich construction as fallback
        return _build_positions(keich, keich)

    return best_positions


# EVOLVE-BLOCK-END
