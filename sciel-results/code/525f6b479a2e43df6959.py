# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).

    The volume is separable: V = Vx * Vy, where Vx (resp. Vy) is the 2-D area of
    the union of N sheared rectangles in the (x, c) (resp. (y, c)) plane. We
    compute that area with a fast, fully-vectorized analytic sweep-line over
    Gauss-Legendre sample heights, and minimize Vx*Vy with multi-start
    basin-hopping plus a tight local polish.
    """
    N = 8
    start_time = time.time()
    deadline = start_time + 585.0  # safety margin before the 630 s kill

    # --- Exact 2-D area of the union of N sheared rectangles in (coord, c) plane ---
    # Rectangle i at height c occupies [base[i] + c*i/N, base[i] + (i+1)/N + c*i/N].
    # We integrate the 1-D union length over c. At each height c the union length
    # is piecewise linear in c, so we sweep the O(N^2) breakpoints exactly and sum
    # the trapezoidal areas. Fully vectorized over the Gauss-Legendre heights.
    c_vals, w_vals = np.polynomial.legendre.leggauss(96)
    c_vals = 0.5 * (c_vals + 1.0)   # map [-1,1] -> [0,1]
    w_vals = 0.5 * w_vals           # scale weights
    idx_arange = np.arange(N, dtype=np.float64)
    interval_widths = (idx_arange + 1.0) / N  # (i+1)/N for i in 0..N-1

    def area_2d(base):
        base = np.asarray(base, dtype=np.float64)
        L = idx_arange / N
        W = interval_widths

        lefts = base[None, :] + c_vals[:, None] * L[None, :]          # (H, N)
        rights = lefts + W[None, :]                                    # (H, N)

        # Per-height union length via a vectorized sweep line.
        # Union length = sum over sorted events of sign * (x_k - x_{k-1}),
        # where sign is the running count of open intervals (>= 1 => covered).
        xs = np.concatenate([lefts, rights], axis=1)   # (H, 2N)
        signs = np.concatenate([np.ones((c_vals.shape[0], N), dtype=np.int64),
                                -np.ones((c_vals.shape[0], N), dtype=np.int64)], axis=1)
        order = np.argsort(xs, axis=1, kind="stable")
        xs_sorted = np.take_along_axis(xs, order, axis=1)
        s_sorted = np.take_along_axis(signs, order, axis=1)
        # Merge coincident x's: net sign change per location.
        diff = np.diff(xs_sorted, axis=1)                       # (H, 2N-1)
        net = np.cumsum(s_sorted, axis=1)                       # (H, 2N)
        # collapse: for each gap between consecutive distinct x, covered if
        # count of open intervals over that gap > 0.
        # count over gap g (between x_g and x_{g+1}) = cumulative sum of signs
        # for events at x_g (i.e. all events whose x == x_g). Use net counts.
        # Build per-location net count by summing s over equal-x runs.
        # Simpler: covered(g) = (running count after processing all events at x_g) > 0
        # where x_g is the left endpoint of gap g.
        # running count after all events at x_g = cumsum(s_sorted) at the last
        # index of the run containing g. We compute coverage per gap.
        # For each gap g, the left boundary x is xs_sorted[:, g]. The open count
        # just to the right of x_g equals the cumulative sum of signs of all
        # events with x <= x_g. That is cumsum up to the last index where
        # xs_sorted == xs_sorted[:, g].
        # Vectorize by grouping equal x's.
        # count_right(g) = sum_{j: xs_sorted[j] <= xs_sorted[g]} s_sorted[j]
        # = for each g, cumulative sum at the rightmost index of its x-run.
        # Compute run right-ends: an index i is a run-end if i is last or
        # xs[i+1] != xs[i].
        is_end = np.concatenate([
            np.diff(xs_sorted, axis=1) != 0.0,   # (H, 2N-1)
            np.ones((c_vals.shape[0], 1), dtype=bool)
        ], axis=1)                                # (H, 2N)
        # For each gap g (0..2N-2), its left x is xs_sorted[:, g]; the run-end
        # index >= g where that x's run ends. We need the cumulative sum there.
        # Since xs are sorted, the run-end for the value xs_sorted[:, g] is the
        # largest index e with xs_sorted[:, e] == xs_sorted[:, g].
        # Build for each index the value; find, per gap, the count.
        # Approach: prefix count array P[:, k] = cumsum(s_sorted)[:, k].
        # For gap g, left value v = xs_sorted[:, g]; find e = rightmost index
        # with xs_sorted == v. Equivalent: e = (last index where xs_sorted
        # equals v). Because sorted, the run of v ends where xs changes.
        # We can compute, for each gap g, the open count = P at index e.
        # e for the value at position g: it is the last position in the same run.
        # Compute run_end_index per position via forward fill of is_end.
        # Let end_idx[i] = smallest e >= i with is_end[e]. Use cumulative trick.
        # Reverse cummax of is_end gives, for each i, the next end.
        rev = is_end[:, ::-1]
        # next end index from the right: we want, for position i, the smallest
        # e >= i with is_end[e]. Compute via cumsum from the right.
        # positions where is_end True are ends. For each i, end = last True at
        # or after i. Use: idx = arange; rev_cumsum of (1-is_end) ... simpler:
        # build array of end positions and searchsorted.
        # Vectorized: for each row, ends = np.flatnonzero(is_end_row). Then for
        # gap g, e = ends[searchsorted(ends, g)].
        # Do it with a loop over rows? H=96, cheap.
        P = np.cumsum(s_sorted, axis=1)                 # (H, 2N)
        H = c_vals.shape[0]
        open_count = np.empty((H, 2 * N - 1), dtype=np.float64)
        for h in range(H):
            ends = np.flatnonzero(is_end[h])            # indices of run ends
            # for each gap g, left value is at index g; its run end = ends[searchsorted(ends, g)]
            e = ends[np.searchsorted(ends, np.arange(2 * N - 1))]
            open_count[h, :] = P[h, e]
        covered = open_count > 0                         # (H, 2N-1)
        # union length = sum over covered gaps of (x_{g+1} - x_g)
        gaps = np.diff(xs_sorted, axis=1)                # (H, 2N-1)
        union_len = np.sum(gaps * covered, axis=1)       # (H,)
        return float(np.dot(w_vals, union_len))

    # --- Objective: minimize Vx * Vy ---
    def objective(params):
        x_opt = params[:N]
        y_opt = params[N:]
        vx = area_2d(x_opt)
        vy = area_2d(y_opt)
        if vx <= 0 or vy <= 0:
            return 1e10
        return vx * vy

    # --- Initial Keich construction ---
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    x_quad = (np.arange(N, dtype=np.float64) / N) ** 2
    x_lin = np.linspace(0, 1, N)

    initial_candidates = [
        np.concatenate([keich, keich]),
        np.concatenate([x_quad, x_quad]),
        np.concatenate([x_lin, x_lin]),
        np.concatenate([keich, x_quad]),
        np.concatenate([x_quad, keich]),
    ]

    best_params = initial_candidates[0].copy()
    best_obj = objective(best_params)
    for candidate in initial_candidates[1:]:
        obj = objective(candidate)
        if obj < best_obj:
            best_obj = obj
            best_params = candidate.copy()

    # --- Phase 1: Differential Evolution global search ---
    if time.time() < deadline - 200.0:
        bounds = [(-0.5, 1.5)] * (2 * N)
        try:
            result_de = minimize(
                objective,
                best_params,
                method='differential_evolution',
                bounds=bounds,
                options={
                    'maxiter': 250,
                    'popsize': 24,
                    'tol': 1e-9,
                    'mutate': (0.5, 1.0),
                    'recombine': 0.7,
                    'seed': 42,
                    'polish': False,
                    'init': 'latinhypercube'
                }
            )
            if result_de.fun < best_obj:
                best_obj = result_de.fun
                best_params = result_de.x.copy()
        except Exception:
            pass

    # --- Phase 2: Multi-start basin hopping (structural global search) ---
    if time.time() < deadline - 80.0:
        rng = np.random.default_rng(7)
        n_starts = 40
        for s in range(n_starts):
            if time.time() > deadline - 80.0:
                break
            # perturb the current best with a random kick
            kick = rng.uniform(-0.25, 0.25, size=2 * N)
            x0 = best_params + kick
            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 1.5)] * (2 * N),
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # --- Phase 3: Tight Nelder-Mead local refinement ---
    if time.time() < deadline - 40.0:
        try:
            result_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-12,
                    'fatol': 1e-15,
                    'adaptive': True,
                    'initial_simplex': 0.05 * np.ones(2 * N)
                }
            )
            if result_nm.fun < best_obj:
                best_obj = result_nm.fun
                best_params = result_nm.x.copy()
        except Exception:
            pass

    # --- Phase 4: Coordinate-descent polish with adaptive step size ---
    if time.time() < deadline - 8.0:
        step = 1e-4
        no_improve = 0
        for _ in range(400):
            if time.time() > deadline - 8.0:
                break
            improved = False
            for idx in range(2 * N):
                old_val = best_params[idx]
                for mult in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                    delta = step * mult
                    for sign in (1, -1):
                        best_params[idx] = old_val + sign * delta
                        obj = objective(best_params)
                        if obj < best_obj - 1e-16:
                            best_obj = obj
                            improved = True
                            break
                        best_params[idx] = old_val
                    if improved:
                        break
                if not improved:
                    best_params[idx] = old_val
            if not improved:
                no_improve += 1
                step *= 0.6
                if step < 1e-9 or no_improve > 40:
                    break
            else:
                no_improve = 0

    # --- Extract final base positions ---
    x_final = best_params[:N]
    y_final = best_params[N:]

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = x_final[i_flat]
    base_y = y_final[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[i_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
