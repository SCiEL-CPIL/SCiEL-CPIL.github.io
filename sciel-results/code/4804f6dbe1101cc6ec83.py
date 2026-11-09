"""Deterministic grid k-NN surrogate with adaptive epsilon-greedy for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-based k-NN surrogate-guided optimizer for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7777)

        best_x = None
        best_f = np.inf
        X_all = []
        Y_all = []

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            X_all.append(x.copy())
            Y_all.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_ok():
            return calls_made < budget and (time.time() - start_time) < (deadline - start_time)

        # ---- Phase 1: Initial exploration with structured + quasi-random points ----
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = int(index)
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_sequence(n, dim):
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]
            pts = []
            for i in range(1, n + 1):
                pts.append(np.array([halton(i, b) for b in bases]))
            return np.array(pts) * 10.0 - 5.0

        structured = [np.zeros(dim)]
        for d in range(dim):
            for v in [-3.0, 0.0, 3.0]:
                p = np.zeros(dim)
                p[d] = v
                structured.append(p)
        for mask in [(1, 0, 1, 0, 1), (0, 1, 0, 1, 0), (1, 1, 0, 0, 1), (0, 0, 1, 1, 0),
                     (1, 0, 0, 1, 0), (0, 1, 1, 0, 1), (1, 1, 1, 0, 0), (0, 0, 1, 1, 1)]:
            p = np.array([(-2.5 if m else 2.5) for m in mask])
            structured.append(p)

        n_halton = 35
        init_pts = halton_sequence(n_halton, dim)

        all_init = np.vstack([np.array(structured), init_pts])
        all_init = np.clip(all_init, -5.0, 5.0)

        unique_init = [all_init[0]]
        for p in all_init[1:]:
            if np.all([np.linalg.norm(p - u) > 0.15 for u in unique_init]):
                unique_init.append(p)
                if len(unique_init) >= 50:
                    break

        for p in unique_init:
            if not time_ok():
                break
            evaluate(p)

        # ---- Grid-based k-NN Surrogate ----
        # Build a coarse grid for fast nearest-neighbor lookup
        GRID_RES = 16
        grid_edges = np.linspace(-5.0, 5.0, GRID_RES + 1)
        grid_centers = (grid_edges[:-1] + grid_edges[1:]) / 2.0
        grid_cell_size = grid_edges[1] - grid_edges[0]

        # Precompute grid cell indices for all evaluated points
        def get_grid_cell(x):
            idx = np.clip(((x + 5.0) / grid_cell_size).astype(int), 0, GRID_RES - 1)
            return idx

        # Maintain a grid-based index: maps cell tuple -> list of point indices
        grid_index = {}

        # Initialize grid index with all points so far
        for i, x in enumerate(X_all):
            cell = tuple(get_grid_cell(x))
            if cell not in grid_index:
                grid_index[cell] = []
            grid_index[cell].append(i)

        K = 6

        def knn_predict_grid(x, k=K):
            """Predict using grid-accelerated k-NN."""
            if len(X_all) == 0:
                return 0.0
            X_arr = np.array(X_all)
            Y_arr = np.array(Y_all)
            n = len(X_arr)

            if n <= k:
                d2 = np.sum((X_arr - x) ** 2, axis=1)
                w = 1.0 / (d2 + 1e-6)
                return float(np.dot(w, Y_arr) / np.sum(w))

            # Find k nearest using grid expansion
            cell = get_grid_cell(x)
            candidates = []
            # Expand grid neighborhood
            for radius in range(0, GRID_RES):
                for offset in np.ndindex(*([2 * radius + 1] * dim)):
                    c = tuple(max(0, min(GRID_RES - 1, cell[d] + offset[d] - radius)) for d in range(dim))
                    if c in grid_index:
                        candidates.extend(grid_index[c])
                candidates = list(set(candidates))
                if len(candidates) >= k:
                    break
                if radius >= GRID_RES - 1:
                    break

            if len(candidates) == 0:
                d2 = np.sum((X_arr - x) ** 2, axis=1)
                w = 1.0 / (d2 + 1e-6)
                return float(np.dot(w, Y_arr) / np.sum(w))

            d2 = np.sum((X_arr[candidates] - x) ** 2, axis=1)
            if len(d2) <= k:
                w = 1.0 / (d2 + 1e-6)
                return float(np.dot(w, Y_arr[candidates]) / np.sum(w))
            idx = np.argpartition(d2, k)[:k]
            d2_k = d2[idx]
            Y_k = Y_arr[candidates][idx]
            w = 1.0 / (d2_k + 1e-6)
            return float(np.dot(w, Y_k) / np.sum(w))

        def knn_predict_batch(Xq, k=K):
            """Predict for multiple query points."""
            preds = np.zeros(len(Xq))
            for i, x in enumerate(Xq):
                preds[i] = knn_predict_grid(x, k)
            return preds

        def max_min_distance(x, k=K):
            """Distance to the k-th nearest neighbor (for exploration)."""
            if len(X_all) == 0:
                return 10.0
            X_arr = np.array(X_all)
            d2 = np.sum((X_arr - x) ** 2, axis=1)
            if len(X_arr) <= k:
                return float(np.sqrt(np.max(d2)))
            idx = np.argpartition(d2, k)[:k]
            return float(np.sqrt(np.max(d2[idx])))

        def add_to_grid(x, idx):
            cell = tuple(get_grid_cell(x))
            if cell not in grid_index:
                grid_index[cell] = []
            grid_index[cell].append(idx)

        # ---- Phase 2: Main optimization loop ----
        x_current = best_x.copy() if best_x is not None else np.zeros(dim)
        f_current = best_f
        es_sigma = 0.8
        n_success = 0
        n_trials = 0
        stall_count = 0
        epsilon = 0.5
        last_best_update = 0

        while time_ok():
            n_trials += 1

            # Adaptive epsilon: decay over time, boost on stall
            progress = min(1.0, n_trials / max(1, (budget - calls_made) * 0.6))
            base_eps = 0.55 * (1.0 - progress) + 0.12 * progress
            if stall_count > 10:
                epsilon = min(0.8, base_eps + 0.15)
            else:
                epsilon = base_eps

            # Generate candidate pool
            candidates = []

            # ES-style candidates around current best
            for _ in range(3):
                c = x_current + es_sigma * rng.standard_normal(dim)
                c = np.clip(c, -5.0, 5.0)
                candidates.append(c)

            # Candidate near global best
            if best_x is not None:
                c = best_x + es_sigma * 0.5 * rng.standard_normal(dim)
                c = np.clip(c, -5.0, 5.0)
                candidates.append(c)

            # Random exploration
            for _ in range(2):
                c = rng.uniform(-5.0, 5.0, size=dim)
                candidates.append(c)

            # Surrogate-guided candidates
            if len(X_all) >= K + 2:
                for _ in range(3):
                    base = best_x if best_x is not None else np.zeros(dim)
                    c = base + es_sigma * 1.5 * rng.standard_normal(dim)
                    c = np.clip(c, -5.0, 5.0)
                    candidates.append(c)

            # Additional exploration: pick point farthest from existing samples
            if len(X_all) >= 5 and rng.random() < epsilon * 0.5:
                # Sample a few random points and pick the one with max min-distance
                far_cands = []
                for _ in range(5):
                    c = rng.uniform(-5.0, 5.0, size=dim)
                    far_cands.append(c)
                best_far = None
                best_far_dist = -1
                for c in far_cands:
                    d = max_min_distance(c)
                    if d > best_far_dist:
                        best_far_dist = d
                        best_far = c
                if best_far is not None:
                    candidates.append(best_far)

            if len(candidates) > 0:
                Xq = np.array(candidates)

                # k-NN prediction
                preds = knn_predict_batch(Xq)

                # Exploration scores
                expl_scores = np.array([max_min_distance(c) for c in Xq])

                # Normalize
                if np.max(preds) - np.min(preds) > 1e-10:
                    preds_norm = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
                else:
                    preds_norm = np.zeros_like(preds)
                if np.max(expl_scores) - np.min(expl_scores) > 1e-10:
                    expl_norm = (expl_scores - np.min(expl_scores)) / (np.max(expl_scores) - np.min(expl_scores))
                else:
                    expl_norm = np.zeros_like(expl_scores)

                if rng.random() < epsilon:
                    idx = int(np.argmax(expl_norm))
                else:
                    idx = int(np.argmin(preds_norm))

                x_candidate = Xq[idx]
            else:
                x_candidate = np.clip(x_current + es_sigma * rng.standard_normal(dim), -5.0, 5.0)

            f_cand = evaluate(x_candidate)
            add_to_grid(x_candidate, len(X_all) - 1)

            # Update current point
            if f_cand < f_current - 1e-10:
                x_current = x_candidate.copy()
                f_current = f_cand
                n_success += 1
                es_sigma = min(es_sigma * 1.1, 2.0)
                stall_count = 0
                last_best_update = n_trials
            else:
                es_sigma = max(es_sigma * 0.92, 0.05)
                stall_count += 1

            # 1/5 success rule
            if n_trials % 5 == 0:
                rate = n_success / 5.0
                if rate > 0.2:
                    es_sigma *= 1.05
                elif rate < 0.1:
                    es_sigma *= 0.95
                es_sigma = max(0.03, min(es_sigma, 2.5))
                n_success = 0

            # Restart if stalled
            if stall_count > 25:
                if len(X_all) >= 10:
                    good_idx = [i for i, y in enumerate(Y_all) if y < best_f + 0.5]
                    if good_idx:
                        gi = rng.choice(good_idx)
                        x_current = X_all[gi].copy()
                        f_current = Y_all[gi]
                        es_sigma = max(es_sigma, 0.4)
                    else:
                        x_current = rng.uniform(-4, 4, size=dim)
                        f_current = np.inf
                        es_sigma = 0.8
                stall_count = 0

            if es_sigma < 0.03:
                break

        # ---- Phase 3: Nelder-Mead final polish ----
        if time_ok() and best_x is not None and calls_made < budget - 10:
            nm_x = best_x.copy()
            nm_f = best_f
            nm_sigma = 0.08

            simplex = [nm_x.copy()]
            for d in range(dim):
                p = nm_x.copy()
                p[d] += nm_sigma
                simplex.append(p)
            simplex = np.array(simplex)
            fvals = [nm_f]
            for i in range(1, dim + 1):
                if not time_ok() or calls_made >= budget:
                    break
                fvals.append(evaluate(simplex[i]))

            for it in range(40):
                if not time_ok() or calls_made >= budget:
                    break
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                if calls_made >= budget:
                    break
                fr = evaluate(xr)

                if fr < fvals[0]:
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    if calls_made >= budget:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        break
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    rho = 0.5
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    if calls_made >= budget:
                        break
                    fc = evaluate(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, dim + 1):
                            if calls_made >= budget:
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])

                if max(fvals) - min(fvals) < 1e-9:
                    break

            # Update best from NM
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

        # ---- Phase 4: Fine-tuning with coordinate descent + random perturbation ----
        if time_ok() and best_x is not None and calls_made < budget:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.03

            for _ in range(25):
                if not time_ok() or calls_made >= budget:
                    break
                # Coordinate descent step
                improved = False
                for d in range(dim):
                    if not time_ok() or calls_made >= budget:
                        break
                    for step in [sigma_p, -sigma_p]:
                        y = x_polish.copy()
                        y[d] += step
                        y = np.clip(y, -5.0, 5.0)
                        fy = evaluate(y)
                        if fy < f_polish - 1e-12:
                            x_polish = y.copy()
                            f_polish = fy
                            improved = True
                            break
                # Random perturbation
                if not improved:
                    y = x_polish + sigma_p * 0.5 * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = evaluate(y)
                    if fy < f_polish:
                        x_polish = y
                        f_polish = fy

                if improved:
                    sigma_p = max(0.003, sigma_p * 0.85)
                else:
                    sigma_p = max(0.002, sigma_p * 0.8)

            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
