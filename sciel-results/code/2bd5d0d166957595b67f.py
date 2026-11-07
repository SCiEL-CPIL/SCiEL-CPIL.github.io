import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        n_calls = 0
        best_x = None
        best_f = float('inf')

        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        rng = np.random.default_rng(12345)

        # 1. Initial Global Exploration - use Sobol/quasi-random for better coverage
        # Use a deterministic set of initial points including center and vertices
        initial_points = [np.zeros(dim)]
        # Add points at various radii along axes
        for axis in range(dim):
            for r in [1.0, 2.0, 3.0, 4.0]:
                p = np.zeros(dim)
                p[axis] = r
                initial_points.append(p.copy())
                p2 = p.copy()
                p2[axis] = -r
                initial_points.append(p2.copy())
        
        # Add random points
        n_init_random = max(10, int(budget * 0.04))
        for _ in range(n_init_random):
            if n_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            timed_call(x)
        
        # Evaluate initial structured points
        for p in initial_points:
            if n_calls >= budget or time.time() > deadline:
                break
            p = np.clip(p, -5.0, 5.0)
            if np.all(p == np.zeros(dim)) and n_calls > 0:
                continue  # Already evaluated center
            timed_call(p)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = timed_call(best_x)

        # 2. Multi-start coordinate descent with adaptive per-dimension steps
        # This exploits the axis-aligned ill-conditioning of Schaffers F7
        # Each walker does coordinate-wise line search with momentum

        # Create multiple walkers from diverse starting points
        n_walkers = 5
        walker_starts = [best_x.copy()]
        for i in range(n_walkers - 1):
            # Perturb best point in different ways
            if i % 2 == 0:
                pert = rng.uniform(-1.5, 1.5, dim)
            else:
                # Larger perturbation
                pert = rng.uniform(-3.0, 3.0, dim)
            x_new = np.clip(best_x + pert, -5.0, 5.0)
            f_new = timed_call(x_new)
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            walker_starts.append(x_new.copy())
        
        # Sort walkers by their fitness
        walker_fits = []
        for ws in walker_starts:
            if np.all(ws == best_x):
                walker_fits.append(best_f)
            else:
                # We need to evaluate each walker start
                # But some may already be evaluated; just re-evaluate to be safe
                f_w = timed_call(ws)
                walker_fits.append(f_w)
        
        walker_order = np.argsort(walker_fits)
        
        # Per-dimension step sizes and momentum
        step_sizes = np.ones(dim) * 0.5
        momentum = np.zeros((n_walkers, dim))
        
        # Coordinate descent phases
        # Each phase: for each walker, try improving along each coordinate
        max_phases = 15
        phase_budget = (budget - n_calls) // max_phases if max_phases > 0 else 0
        
        for phase in range(max_phases):
            if n_calls >= budget or time.time() > deadline:
                break
            
            # Re-sort walkers occasionally
            if phase % 3 == 0 and phase > 0:
                # Re-evaluate current walker positions
                cur_fits = np.array([timed_call(ws) for ws in walker_starts])
                walker_order = np.argsort(cur_fits)
            
            for w_idx in walker_order:
                if n_calls >= budget or time.time() > deadline:
                    break
                
                ws = walker_starts[w_idx]
                ws_f = timed_call(ws)
                
                for coord in range(dim):
                    if n_calls >= budget or time.time() > deadline:
                        break
                    
                    # Adaptive step size: shrink if stuck, expand if making progress
                    step = step_sizes[coord]
                    improved = False
                    prev_f = ws_f
                    
                    # Try positive direction
                    x_try = ws.copy()
                    x_try[coord] = np.clip(x_try[coord] + step, -5.0, 5.0)
                    f_try = timed_call(x_try)
                    
                    if f_try < prev_f - 1e-10:
                        ws = x_try.copy()
                        ws_f = f_try
                        improved = True
                        # Shrink step for next iteration on this coord
                        step_sizes[coord] *= 0.85
                    else:
                        # Try negative direction
                        x_try = ws.copy()
                        x_try[coord] = np.clip(x_try[coord] - step, -5.0, 5.0)
                        f_try = timed_call(x_try)
                        
                        if f_try < prev_f - 1e-10:
                            ws = x_try.copy()
                            ws_f = f_try
                            improved = True
                            step_sizes[coord] *= 0.85
                        else:
                            # No improvement: expand step
                            step_sizes[coord] *= 1.3
                            step_sizes[coord] = min(step_sizes[coord], 2.0)
                    
                    # Update momentum
                    if improved:
                        momentum[w_idx, coord] = 0.7 * momentum[w_idx, coord] + 0.3 * step
                    else:
                        momentum[w_idx, coord] *= 0.5
                    
                    # Periodically recenter step sizes
                    if phase > 0 and phase % 5 == 0:
                        step_sizes = np.clip(step_sizes, 0.01, 3.0)
                
                walker_starts[w_idx] = ws.copy()
                if ws_f < best_f:
                    best_f = ws_f
                    best_x = ws.copy()
        
        # 3. Final local refinement with very small steps
        if n_calls < budget and time.time() < deadline:
            # Nelder-Mead style simplex refinement
            simplex = [best_x.copy() + 0.1 * rng.standard_normal(dim) for _ in range(dim + 1)]
            simplex.append(best_x.copy())
            f_simplex = np.array([timed_call(s) for s in simplex])
            
            max_nm_iter = min(budget - n_calls, 50)
            for _ in range(max_nm_iter):
                if n_calls >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = f_simplex[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = timed_call(x_ref)
                
                if f_ref < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = timed_call(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                elif f_ref < f_simplex[-2]:
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = timed_call(x_con)
                    
                    if f_con < f_simplex[-1]:
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_simplex[i] = timed_call(simplex[i])
                
                if f_simplex[0] < best_f:
                    best_f = f_simplex[0]
                    best_x = simplex[0].copy()
        
        return best_x, best_f
