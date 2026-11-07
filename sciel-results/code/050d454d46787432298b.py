import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer using Weighted-Sum Decomposition with Nelder-Mead refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Track evaluations carefully
        evals_used = 0

        def eval_and_record(x):
            nonlocal evals_used
            if evals_used >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            evals_used += 1
            return x, f

        # Collect all evaluated points
        all_X = []
        all_F = []

        # Step 1: Generate a set of weight vectors for the 3-objective problem
        # Use a grid of weights that sum to 1
        weight_vectors = []

        # Fine grid for better coverage
        n_grid = 14  # 14^2 = 196 weight vectors, might be too many
        # Let's use a coarser grid and then add some random ones
        # Actually, with budget=200, we need to be careful
        # Let's use n_grid=8, giving 8*7/2 = 28 weight vectors (triangular)
        # Plus some random ones

        # Triangular grid of weights (w1, w2, w3) with w1+w2+w3=1
        n_tri = 8
        for i in range(n_tri + 1):
            for j in range(n_tri + 1 - i):
                k = n_tri - i - j
                w1 = i / n_tri
                w2 = j / n_tri
                w3 = k / n_tri
                if w1 + w2 + w3 > 0:
                    weight_vectors.append([w1, w2, w3])

        # Add some random weight vectors for diversity
        rng = np.random.default_rng(42)
        for _ in range(10):
            w = rng.random(3)
            w = w / w.sum()
            weight_vectors.append(list(w))

        # Remove duplicate-ish weights (optional, keep simple)
        # Sort by first coordinate for consistent ordering
        weight_vectors.sort(key=lambda w: (w[0], w[1], w[2]))

        # Step 2: For each weight vector, run a local optimizer (Nelder-Mead)
        # Budget allocation: distribute evaluations across weight vectors
        n_weights = len(weight_vectors)
        # Reserve some evaluations for final random sampling
        # Each weight vector gets roughly budget/n_weights evaluations
        # But Nelder-Mead can use multiple evals per call

        # Use a wrapper that tracks evaluations
        bounds_scipy = list(zip(lo, hi))

        for w in weight_vectors:
            if time.time() > deadline - 10.0:
                break
            if evals_used >= budget - 5:
                break

            # Initial point: random
            x0 = lo + (hi - lo) * rng.random(dim)

            # Define weighted objective
            def weighted_obj(x, w=w):
                nonlocal evals_used
                if evals_used >= budget:
                    return 1e10
                x = np.clip(np.asarray(x, dtype=float), lo, hi)
                f = np.asarray(problem(x), dtype=float)
                evals_used += 1
                return np.dot(w, f)

            # Determine max function evaluations for this sub-problem
            # Nelder-Mead's maxfev controls function evaluations
            # We want to use at most ~8 evals per weight vector to spread coverage
            max_fev = min(8, budget - evals_used)
            if max_fev < 3:
                break

            try:
                res = minimize(
                    weighted_obj,
                    x0,
                    method='Nelder-Mead',
                    bounds=bounds_scipy,
                    options={
                        'maxiter': max_fev,
                        'maxfev': max_fev,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                if res.x is not None:
                    x_opt, f_opt = eval_and_record(res.x)
                    if f_opt is not None:
                        all_X.append(x_opt)
                        all_F.append(f_opt)
            except Exception:
                pass

            # Also evaluate the initial point if it wasn't already counted
            # (Nelder-Mead may not have evaluated x0 if it failed immediately)
            # Actually, weighted_obj is called by minimize, so x0 should be counted
            # Let's verify by checking if we should add x0 explicitly
            # To be safe, let's evaluate x0 if we haven't used it
            if evals_used < budget:
                x_chk, f_chk = eval_and_record(x0)
                if f_chk is not None:
                    # Check if this is a new point (avoid duplicates)
                    all_X.append(x_chk)
                    all_F.append(f_chk)

        # Step 3: If budget remains, do random sampling to fill gaps
        n_random = budget - evals_used
        if n_random > 0 and time.time() < deadline - 5.0:
            for _ in range(min(n_random, 30)):
                if evals_used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                x = lo + (hi - lo) * rng.random(dim)
                x_rec, f_rec = eval_and_record(x)
                if f_rec is not None:
                    all_X.append(x_rec)
                    all_F.append(f_rec)

        # Step 4: Local refinement of best points using gradient-free method
        if all_F and time.time() < deadline - 15.0 and evals_used < budget - 5:
            # Find non-dominated points approximately
            # Sort by first objective and keep diverse points
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Simple non-dominated filter
            nd_indices = []
            for i in range(len(F_arr)):
                dominated = False
                for j in range(len(F_arr)):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)

            # Take top few non-dominated points for refinement
            top_k = min(5, len(nd_indices))
            if top_k > 0:
                # Pick diverse ones: sort by hypervolume contribution approx
                # Just take first few for simplicity
                candidates = [nd_indices[i] for i in range(top_k)]

                for ci in candidates:
                    if evals_used >= budget - 2:
                        break
                    if time.time() > deadline - 10.0:
                        break

                    x_start = X_arr[ci].copy()
                    f_start = F_arr[ci].copy()

                    # Try to improve each objective individually
                    for obj_idx in range(self.n_obj):
                        if evals_used >= budget - 1:
                            break
                        if time.time() > deadline - 8.0:
                            break

                        # Minimize single objective starting from x_start
                        def single_obj(x, obj_idx=obj_idx):
                            nonlocal evals_used
                            if evals_used >= budget:
                                return 1e10
                            x = np.clip(np.asarray(x, dtype=float), lo, hi)
                            f = np.asarray(problem(x), dtype=float)
                            evals_used += 1
                            return f[obj_idx]

                        max_fev = min(4, budget - evals_used)
                        if max_fev < 2:
                            break

                        try:
                            res = minimize(
                                single_obj,
                                x_start,
                                method='Nelder-Mead',
                                bounds=bounds_scipy,
                                options={
                                    'maxiter': max_fev,
                                    'maxfev': max_fev,
                                    'xatol': 1e-5,
                                    'fatol': 1e-5
                                }
                            )
                            if res.x is not None:
                                x_rec, f_rec = eval_and_record(res.x)
                                if f_rec is not None:
                                    # Check if it's non-dominated
                                    dominated = False
                                    for fj in all_F:
                                        if np.all(fj <= f_rec) and np.any(fj < f_rec):
                                            dominated = True
                                            break
                                    if not dominated:
                                        all_X.append(x_rec)
                                        all_F.append(f_rec)
                        except Exception:
                            pass

        # Final deduplication and return
        if all_X:
            X_final = np.array(all_X)
            F_final = np.array(all_F)

            # Remove points dominated by reference point [1705.0, 12.0, 0.3]
            ref = np.array([1705.0, 12.0, 0.3])
            mask = np.all(F_final < ref, axis=1)
            X_final = X_final[mask]
            F_final = F_final[mask]

            # Remove exact duplicates
            if len(X_final) > 0:
                unique_indices = []
                for i in range(len(X_final)):
                    is_dup = False
                    for j in unique_indices:
                        if np.allclose(X_final[i], X_final[j], atol=1e-10):
                            is_dup = True
                            break
                    if not is_dup:
                        unique_indices.append(i)
                X_final = X_final[unique_indices]

            return X_final

        return np.array([])
