"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        # Global best tracking
        best_x = np.array([0.0] * dim)
        best_f = np.inf

        # Nelder-Mead implementation
        def nelder_mead(func, x0, fx0, budget_left, dim, max_iter=50):
            """Nelder-Mead with budget tracking. Returns (x_best, f_best, calls_used)."""
            calls = 0
            # Initialize simplex
            alpha = 0.5  # expansion
            gamma = 2.0  # reflection
            rho = 0.5   # contraction
            sigma = 0.5  # shrink
            
            # Build simplex
            simplex = []
            simplex.append((x0.copy(), fx0))
            calls += 1  # fx0 already counted? No, we'll count fresh
            
            # Actually let's be careful: fx0 is given, so 1 call used
            for i in range(dim):
                xi = x0.copy()
                if i == 0:
                    xi[i] += 1.0
                else:
                    xi[i] += 0.5
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls += 1
                simplex.append((xi, fi))
            
            # Sort simplex
            simplex.sort(key=lambda t: t[1])
            x_best_local = simplex[0][0].copy()
            f_best_local = simplex[0][1]
            
            for _ in range(max_iter):
                if calls >= budget_left:
                    break
                if time.time() > deadline:
                    break
                
                # Check convergence: if simplex is small enough
                x_center = np.mean([s[0] for s in simplex], axis=0)
                spread = np.max(np.abs(np.array([s[0] for s in simplex]) - x_center))
                if spread < 1e-4:
                    break
                
                # Get worst, second worst
                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_second_worst = simplex[-2][0]
                f_second_worst = simplex[-2][1]
                
                # Centroid of all but worst
                x_centroid = np.mean([s[0] for s in simplex[:-1]], axis=0)
                
                # Reflection
                x_reflect = x_centroid + gamma * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = func(x_reflect)
                calls += 1
                if calls >= budget_left:
                    break
                
                if f_reflect < simplex[0][1]:
                    # Expansion
                    x_expand = x_centroid + alpha * (x_reflect - x_worst)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = func(x_expand)
                    calls += 1
                    if calls >= budget_left:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = (x_expand, f_expand)
                    else:
                        simplex[-1] = (x_reflect, f_reflect)
                elif f_reflect < f_second_worst:
                    simplex[-1] = (x_reflect, f_reflect)
                else:
                    # Contraction
                    if f_reflect < f_worst:
                        # Outside contraction
                        x_contract = x_centroid + rho * (x_reflect - x_worst)
                    else:
                        # Inside contraction
                        x_contract = x_centroid - rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = func(x_contract)
                    calls += 1
                    if calls >= budget_left:
                        break
                    if f_contract <= min(f_reflect, f_worst):
                        simplex[-1] = (x_contract, f_contract)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = (simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0]),
                                           func(np.clip(simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0]), -5, 5)))
                            calls += 1
                            if calls >= budget_left:
                                break
                        if calls >= budget_left:
                            break
                
                # Sort
                simplex.sort(key=lambda t: t[1])
                if simplex[0][1] < f_best_local:
                    f_best_local = simplex[0][1]
                    x_best_local = simplex[0][0].copy()
            
            return x_best_local, f_best_local, calls

        # Multi-start strategy
        rng = np.random.default_rng(42)
        
        # Initial diverse points
        n_starts = min(10, budget // 50)  # ~50 calls per start initially
        starts = []
        # Include center and random points
        starts.append(np.zeros(dim))
        for i in range(n_starts - 1):
            x = rng.uniform(-4.5, 4.5, dim)
            starts.append(x)
        
        total_calls = 0
        start_idx = 0
        
        while total_calls < budget and start_idx < len(starts):
            if time.time() > deadline:
                break
            
            x_start = starts[start_idx]
            x_start = np.clip(x_start, -5, 5)
            f_start = func(x_start)
            total_calls += 1
            
            if f_start < best_f:
                best_f = f_start
                best_x = x_start.copy()
            
            # Determine budget for this Nelder-Mead run
            remaining = budget - total_calls
            # Give more budget to promising starts
            nm_budget = min(remaining, 40 + int(20 * (1.0 if f_start < best_f else 0.5)))
            nm_budget = max(nm_budget, 10)
            
            x_nm, f_nm, calls_used = nelder_mead(func, x_start, f_start, nm_budget, dim, max_iter=30)
            total_calls += calls_used
            
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
            
            start_idx += 1
        
        # If we have budget left, do more local refinement
        if total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining >= 15:
                # Refine from best point
                x_refine, f_refine, calls_used = nelder_mead(func, best_x.copy(), best_f, remaining, dim, max_iter=20)
                total_calls += calls_used
                if f_refine < best_f:
                    best_f = f_refine
                    best_x = x_refine.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
