"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid: Multi-restart (1+1)-ES with custom Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def check_time():
            return time.time() > deadline

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= total_budget or check_time():
                raise TimeoutError("Budget or time exceeded")
            fx = func(x)
            count_call()
            return fx

        # Phase 1: Multi-restart (1+1)-ES for global exploration
        # Phase 2: Custom Nelder-Mead for local refinement around the best point found

        # Budget allocation:
        # ~65% for ES exploration, ~35% for NM refinement
        # 5 restarts of ES, then NM on the best.
        
        num_restarts = 5
        es_budget = int(total_budget * 0.65)
        budget_per_restart = max(20, es_budget // num_restarts)

        for restart in range(num_restarts):
            if check_time() or calls_made >= es_budget:
                break

            # Diversify starting points using a consistent seed strategy
            # but varying per restart to ensure coverage
            rng = np.random.default_rng(12345 + restart * 7919)

            if restart == 0:
                x = np.zeros(dim)
            elif restart == 1:
                x = rng.uniform(-2, 2, dim)
            elif restart == 2:
                x = rng.uniform(-3, 3, dim)
            elif restart == 3:
                x = rng.uniform(-4, 4, dim)
            else:
                # Final restart: random in full box
                x = rng.uniform(-5, 5, dim)

            try:
                fx = safe_func(x)
            except (TimeoutError, Exception):
                break

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            sigma = 2.0
            window_size = 10
            window = []

            for _ in range(budget_per_restart - 1):
                if check_time() or calls_made >= es_budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = safe_func(y)
                except (TimeoutError, Exception):
                    break

                if fy < fx:
                    x, fx = y, fy
                    window.append(1)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    window.append(0)

                # Adapt sigma
                if len(window) >= window_size:
                    success_rate = sum(window[-window_size:]) / window_size
                    if success_rate > 0.25:
                        sigma *= 1.2
                    elif success_rate < 0.15:
                        sigma *= 0.8
                    window = window[-window_size:]
                elif len(window) > 5:
                    recent_success = sum(window) / len(window)
                    if recent_success > 0.3:
                        sigma *= 1.1
                    elif recent_success < 0.1:
                        sigma *= 0.85

                sigma = np.clip(sigma, 0.05, 5.0)

        # Phase 2: Custom Nelder-Mead refinement around best_x
        # We implement a simple NM here to strictly control the call count and deadline.
        
        if best_x is not None and not check_time() and calls_made < total_budget - 10:
            
            # Initialize simplex
            # Vertices: best_x, best_x + delta*e_i
            delta = 0.5
            simplex = [best_x.copy()]
            for i in range(dim):
                v = best_x.copy()
                v[i] += delta
                simplex.append(v)
            
            # Evaluate initial simplex
            simplex_f = []
            try:
                for v in simplex:
                    simplex_f.append(safe_func(v))
            except (TimeoutError, Exception):
                simplex_f = []

            if len(simplex_f) == dim + 1:
                # Sort simplex by function value
                idx = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idx]
                simplex_f = [simplex_f[i] for i in idx]
                
                # Update best if NM found something better immediately
                if simplex_f[0] < best_fx:
                    best_fx = simplex_f[0]
                    best_x = simplex[0].copy()

                max_nm_iter = 200
                for _ in range(max_nm_iter):
                    if check_time() or calls_made >= total_budget - 5:
                        break

                    # Check for convergence
                    if abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                        break

                    # Centroid of all but the worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    try:
                        fr = safe_func(xr)
                    except (TimeoutError, Exception):
                        break

                    if simplex_f[0] <= fr < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    elif fr < simplex_f[0]:
                        # Expansion
                        xe = centroid + 2 * (centroid - simplex[-1])
                        try:
                            fe = safe_func(xe)
                        except (TimeoutError, Exception):
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    else:
                        # Contraction
                        if fr < simplex_f[-1]:
                            # Outside contraction
                            xc = centroid + 0.5 * (xr - centroid)
                            try:
                                fc = safe_func(xc)
                            except (TimeoutError, Exception):
                                break
                            if fc <= fr:
                                simplex[-1] = xc
                                simplex_f[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    try:
                                        simplex_f[i] = safe_func(simplex[i])
                                    except (TimeoutError, Exception):
                                        break
                                if check_time() or calls_made >= total_budget - 5:
                                    break
                        else:
                            # Inside contraction
                            xc = centroid - 0.5 * (simplex[-1] - centroid)
                            try:
                                fc = safe_func(xc)
                            except (TimeoutError, Exception):
                                break
                            if fc <= simplex_f[-1]:
                                simplex[-1] = xc
                                simplex_f[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    try:
                                        simplex_f[i] = safe_func(simplex[i])
                                    except (TimeoutError, Exception):
                                        break
                                if check_time() or calls_made >= total_budget - 5:
                                    break

                    # Re-sort simplex
                    idx = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in idx]
                    simplex_f = [simplex_f[i] for i in idx]
                    
                    # Update best
                    if simplex_f[0] < best_fx:
                        best_fx = simplex_f[0]
                        best_x = simplex[0].copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_fx = safe_func(best_x)
            except (TimeoutError, Exception):
                best_fx = float('inf')

        return best_x, best_fx
# EVOLVE-BLOCK-END
