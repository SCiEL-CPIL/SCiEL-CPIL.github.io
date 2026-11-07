"""Multi-objective optimizer on WFG4 using constrained Chebyshev scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Constrained Chebyshev scalarization for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Track all evaluated points and their objective values
        all_X = []
        all_F = []
        evaluations = 0

        def record_point(x, f):
            nonlocal evaluations
            evaluations += 1
            all_X.append(np.array(x, dtype=float))
            all_F.append(np.array(f, dtype=float))

        # Check if we should stop
        def time_ok():
            return time.time() < deadline - 2.0

        # Generate weight vectors for constrained Chebyshev scalarization
        # Use a grid of weight vectors that cover the Pareto front
        # For 3 objectives, we use a simplex grid
        n_weights_per_axis = 5
        weight_vectors = []
        for i in range(n_weights_per_axis):
            for j in range(n_weights_per_axis - i):
                k = n_weights_per_axis - i - j
                w = np.array([i, j, k], dtype=float)
                w /= n_weights_per_axis - 1
                weight_vectors.append(w)
        
        # Also add extreme points (axes)
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            if not any(np.allclose(w, wv) for wv in weight_vectors):
                weight_vectors.append(w)
        
        # Add more diverse weight vectors
        for _ in range(20):
            w = np.random.dirichlet(np.ones(n_obj))
            weight_vectors.append(w)
        
        # Shuffle weight vectors to get diversity early
        rng = np.random.RandomState(42)
        rng.shuffle(weight_vectors)

        # Phase 1: For each weight vector, solve the constrained Chebyshev subproblem
        # Using Nelder-Mead or L-BFGS-B as the local optimizer
        for w in weight_vectors:
            if evaluations >= budget or not time_ok():
                break

            # Initial guess: use a point from the current best set if available,
            # otherwise a random point
            if all_X:
                # Pick a random existing point as starting point for diversity
                idx = rng.randint(len(all_X))
                x0 = all_X[idx].copy()
            else:
                x0 = lo + span * rng.rand(dim)

            # Constrained Chebyshev scalarization:
            # min max_i (w_i * f_i(x) + rho * |w_i * f_i(x) - z_i|)
            # where z is an ideal point estimate
            # We use a simpler approach: minimize sum of weighted objectives
            # with a Chebyshev-like penalty for balance

            # Use multiple starting points for robustness
            n_starts = 3
            best_local_x = None
            best_local_f = None
            best_local_scalar = np.inf

            for start_idx in range(n_starts):
                if evaluations >= budget or not time_ok():
                    break

                # Different starting points
                if start_idx == 0:
                    x_start = x0.copy()
                elif start_idx == 1:
                    x_start = lo + span * rng.rand(dim)
                else:
                    if all_X:
                        idx2 = rng.randint(len(all_X))
                        x_start = all_X[idx2].copy()
                    else:
                        x_start = lo + span * rng.rand(dim)

                # Define the scalarized objective
                def scalarized_obj(x, w=w):
                    f = problem(x)
                    # Constrained Chebyshev: max_i (w_i * f_i)
                    # But we need to track evaluations, so we use a wrapper
                    return np.max(w * f)

                # We need to count evaluations carefully
                # Use a wrapper that counts evaluations
                eval_count = [0]
                def counting_obj(x):
                    if eval_count[0] >= budget - evaluations:
                        return np.inf
                    f = problem(x)
                    eval_count[0] += 1
                    evaluations += 1
                    all_X.append(np.array(x, dtype=float))
                    all_F.append(np.array(f, dtype=float))
                    return np.max(w * f)

                # Use Nelder-Mead for the local optimization
                try:
                    res = minimize(
                        counting_obj,
                        x_start,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4,
                                 'adaptive': True}
                    )
                    if eval_count[0] > 0:
                        # Record the best point found
                        f_best = problem(res.x)
                        evaluations += 1
                        all_X.append(np.array(res.x, dtype=float))
                        all_F.append(np.array(f_best, dtype=float))
                        scalar_val = np.max(w * f_best)
                        if scalar_val < best_local_scalar:
                            best_local_scalar = scalar_val
                            best_local_x = res.x.copy()
                            best_local_f = f_best.copy()
                except Exception:
                    pass

        # Phase 2: Local refinement of the best non-dominated points
        if all_F and evaluations < budget and time_ok():
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            # Find non-dominated points
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[j] = True
            
            # Get non-dominated points
            nd_indices = np.where(~dominated)[0]
            
            # Refine each non-dominated point with local search
            for idx in nd_indices:
                if evaluations >= budget or not time_ok():
                    break
                
                x0 = X_arr[idx].copy()
                
                # Small perturbations
                for step in [0.05, 0.02, 0.01]:
                    if evaluations >= budget or not time_ok():
                        break
                    for _ in range(4):
                        if evaluations >= budget or not time_ok():
                            break
                        perturbation = rng.randn(dim) * step * span
                        x_new = np.clip(x0 + perturbation, lo, hi)
                        f_new = problem(x_new)
                        evaluations += 1
                        all_X.append(x_new)
                        all_F.append(f_new)
                        
                        # Check if this improves any objective
                        f0 = np.array(all_F[-2])
                        if np.any(f_new < f0):
                            x0 = x_new.copy()

        # Return the best points (all recorded points are scored)
        return all_X
# EVOLVE-BLOCK-END
