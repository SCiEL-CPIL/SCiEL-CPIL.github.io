"""Multi-objective optimizer on WFG4 exploiting variable separability."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    WFG4-specific optimizer exploiting the known structure:
    - Deciding variables (x0, x1, x2) shape the Pareto front
    - Positioning variables (x3..x11) add penalty/distance
    
    Strategy:
    1. Systematically sample deciding variable configurations on a 3D grid
    2. For each configuration, optimize positioning variables via coordinate descent
       to minimize the penalty term (which is separable per variable)
    3. Use remaining budget for refinement of promising points
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        X_archive = []
        F_archive = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            x = np.asarray(x, dtype=float)
            fx = np.array(problem(x))
            X_archive.append(x.copy())
            F_archive.append(fx)
            used += 1
            return fx

        # Phase 1: Dense grid on deciding variables (x0, x1, x2)
        # WFG4: first 3 variables determine the front shape
        n0 = 8
        n1 = 8
        n2 = 6
        x0_vals = np.linspace(lo[0], hi[0], n0)
        x1_vals = np.linspace(lo[1], hi[1], n1)
        x2_vals = np.linspace(lo[2], hi[2], n2)

        # Initialize positioning variables to midpoint
        pos_mid = 0.5 * (lo[3:] + hi[3:])

        # Phase 2: For each deciding config, optimize positioning via coordinate descent
        for x0 in x0_vals:
            for x1 in x1_vals:
                for x2 in x2_vals:
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Start with midpoint for positioning variables
                    x = np.zeros(self.dim)
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    x[3:] = pos_mid.copy()
                    fx = evaluate(x)
                    if fx is None:
                        break

                    # Coordinate descent on positioning variables
                    best_x = x.copy()
                    best_f = fx.copy()
                    
                    # Multiple passes of coordinate descent
                    for pass_idx in range(3):
                        if used >= self.budget or time.time() > deadline:
                            break
                        for d in range(3, self.dim):
                            if used >= self.budget or time.time() > deadline:
                                break
                            # Try moving variable d in both directions
                            for direction in [-1, 1]:
                                if used >= self.budget or time.time() > deadline:
                                    break
                                # Try several step sizes
                                for step_frac in [0.3, 0.15, 0.05]:
                                    if used >= self.budget or time.time() > deadline:
                                        break
                                    x_try = best_x.copy()
                                    x_try[d] += direction * step_frac * (hi[d] - lo[d])
                                    x_try[d] = np.clip(x_try[d], lo[d], hi[d])
                                    f_try = evaluate(x_try)
                                    if f_try is None:
                                        break
                                    # Accept if better in at least one objective
                                    # without worsening others too much (use Tchebycheff with uniform weights)
                                    w = np.ones(self.n_obj) / self.n_obj
                                    if np.max(w * (f_try - best_f)) < 0:
                                        best_x = x_try.copy()
                                        best_f = f_try.copy()

        # Phase 3: Refinement - take best points and do local search
        if F_archive and used < self.budget:
            F_arr = np.array(F_archive)
            # Find non-dominated points
            n_pts = len(F_arr)
            dominated = np.zeros(n_pts, dtype=bool)
            for i in range(n_pts):
                if dominated[i]:
                    continue
                for j in range(n_pts):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break

            # Get indices of non-dominated points
            nd_indices = [i for i in range(n_pts) if not dominated[i]]
            
            # Refine top points
            max_refine = min(len(nd_indices), 15)
            for idx in nd_indices[:max_refine]:
                if used >= self.budget or time.time() > deadline:
                    break
                
                best_x = X_archive[idx].copy()
                best_f = F_archive[idx].copy()
                
                # Local search with small perturbations
                for _ in range(8):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_try = best_x.copy()
                    # Perturb positioning variables more than deciding
                    noise = rng.standard_normal(self.dim)
                    noise[:3] *= 0.02
                    noise[3:] *= 0.05
                    x_try += noise * (hi - lo)
                    x_try = np.clip(x_try, lo, hi)
                    f_try = evaluate(x_try)
                    if f_try is None:
                        break
                    w = np.ones(self.n_obj) / self.n_obj
                    if np.max(w * (f_try - best_f)) < 0:
                        best_x = x_try.copy()
                        best_f = f_try.copy()

        return X_archive
# EVOLVE-BLOCK-END
