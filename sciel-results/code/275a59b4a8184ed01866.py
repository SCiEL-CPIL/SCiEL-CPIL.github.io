# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _soft_inside(x, a, b, s):
    """Smooth indicator that x is in [a,b]."""
    # sigmoid
    t1 = s * (x - a)
    t2 = s * (b - x)
    # stable sigmoid
    sig1 = np.where(t1 >= 0, 1.0 / (1.0 + np.exp(-t1)), np.exp(t1) / (1.0 + np.exp(t1)))
    sig2 = np.where(t2 >= 0, 1.0 / (1.0 + np.exp(-t2)), np.exp(t2) / (1.0 + np.exp(t2)))
    return sig1 * sig2


def _union_len_soft(pos1d, c, N, s, grid):
    """Union length (soft) of N intervals [pos1d[i] + c*i/N, +1/N] at height c."""
    N = int(N)
    a = pos1d + c * np.arange(N) / N
    b = a + 1.0 / N
    # product of (1 - inside_i) over i, then 1 - that
    # inside_i(x) for all x on grid
    # grid shape (G,), a,b shape (N,)
    # inside shape (G, N)
    inside = _soft_inside(grid[:, None], a[None, :], b[None, :], s)
    prod = np.prod(1.0 - inside, axis=1)
    union_indicator = 1.0 - prod
    # integrate over grid
    dx = grid[1] - grid[0] if grid.size > 1 else 1.0
    return np.sum(union_indicator) * dx


def _objective(vars, N, s, c_grid, x_grid, y_grid):
    """Objective = integral of Ux(c)*Uy(c) dc."""
    px = vars[:N]
    py = vars[N:]
    total = 0.0
    dc = c_grid[1] - c_grid[0] if c_grid.size > 1 else 1.0
    for c in c_grid:
        ux = _union_len_soft(px, c, N, s, x_grid)
        uy = _union_len_soft(py, c, N, s, y_grid)
        total += ux * uy * dc
    return total


def _keich(N):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    DEADLINE = 600.0
    N = 8

    # Initial guess from Keich construction
    keich = _keich(N)
    px0 = keich.copy()
    py0 = keich.copy()
    vars0 = np.concatenate([px0, py0])

    # Grids for integration
    s = 40.0  # sharpness of soft step
    G = 400
    # Determine a reasonable bounding box for x and y
    # x ranges over pos + c*i/N + [0,1/N]; c in [0,1], i in [0,N-1]
    # max offset = (N-1)/N = 7/8, plus 1/8 width
    lo = -2.0
    hi = 2.0
    x_grid = np.linspace(lo, hi, G)
    y_grid = np.linspace(lo, hi, G)
    c_grid = np.linspace(0.0, 1.0, 60)

    def obj(v):
        return _objective(v, N, s, c_grid, x_grid, y_grid)

    best_vars = vars0.copy()
    best_val = obj(best_vars)

    # Try optimization with a deadline guard
    try:
        def obj_with_guard(v):
            if time.time() > DEADLINE - 30.0:
                return 1e9
            return _objective(v, N, s, c_grid, x_grid, y_grid)

        res = minimize(
            obj_with_guard,
            vars0,
            method="L-BFGS-B",
            options={"maxiter": 200, "maxfun": 5000, "gtol": 1e-6, "ftol": 1e-8},
        )
        if time.time() < DEADLINE - 30.0:
            val = _objective(res.x, N, s, c_grid, x_grid, y_grid)
            if val < best_val:
                best_vars = res.x.copy()
                best_val = val
    except Exception:
        pass

    # Refine with a second pass if time allows
    if time.time() < DEADLINE - 60.0:
        try:
            s2 = 80.0
            G2 = 600
            x_grid2 = np.linspace(lo, hi, G2)
            y_grid2 = np.linspace(lo, hi, G2)
            c_grid2 = np.linspace(0.0, 1.0, 80)

            def obj2(v):
                if time.time() > DEADLINE - 30.0:
                    return 1e9
                return _objective(v, N, s2, c_grid2, x_grid2, y_grid2)

            res2 = minimize(
                obj2,
                best_vars,
                method="L-BFGS-B",
                options={"maxiter": 200, "maxfun": 5000, "gtol": 1e-7, "ftol": 1e-9},
            )
            if time.time() < DEADLINE - 30.0:
                val2 = _objective(res2.x, N, s2, c_grid2, x_grid2, y_grid2)
                # Compare using same metric; use the refined objective value
                # But best_val was computed with s=40; to be fair, recompute both with s2
                best_val_re = _objective(best_vars, N, s2, c_grid2, x_grid2, y_grid2)
                if val2 < best_val_re:
                    best_vars = res2.x.copy()
        except Exception:
            pass

    px = best_vars[:N]
    py = best_vars[N:]

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = px[i_flat]
    base_y = py[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    # Validate
    if not np.all(np.isfinite(positions)):
        # Fall back to Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
