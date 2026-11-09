import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Estimate the hidden shift by probing the function
        # Büche-Rastrigin has minima at x_i = k_i + shift_i where k_i are integers
        # We probe at half-integer offsets to detect the shift phase
        
        # Strategy: For each dimension, try values around 0, 0.5, 1.0, etc.
        # and use the function's oscillatory nature to estimate the shift.
        
        # First, get a rough best point via random sampling
        n_init = 30
        for _ in range(n_init):
            if calls >= budget or time.time() >= deadline:
                break
            x = np.random.uniform(-5, 5, dim)
            eval_point(x)
        
        # Now estimate shift per dimension using the best point as reference
        # The Büche-Rastrigin function: f(x) = -10 + sum_i [sin^2(pi*x_i) + 10*(1-cos(2*pi*x_i)) + 10*|x_i|]
        # At a local minimum, x_i should be near an integer + shift_i
        
        # For each dimension, probe around the current best to find the phase
        shift_est = np.zeros(dim)
        
        # For each dimension, try a few offsets to find where the function dips
        probe_budget = 50
        probes_per_dim = probe_budget // dim  # ~10 per dim
        
        for d in range(dim):
            if calls >= budget or time.time() >= deadline:
                break
            
            # Probe around best_x[d] with small perturbations
            base = best_x.copy()
            best_probe_f = np.inf
            best_probe_offset = 0.0
            
            # Try offsets: -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5
            offsets = np.linspace(-0.5, 0.5, 11)
            for off in offsets:
                if calls >= budget or time.time() >= deadline:
                    break
                probe = base.copy()
                probe[d] = base[d] + off
                probe = np.clip(probe, -5, 5)
                f_probe = eval_point(probe)
                if f_probe < best_probe_f:
                    best_probe_f = f_probe
                    best_probe_offset = off
            
            # The offset that gives the best value tells us the phase
            # If best_x[d] is already near a minimum, offset ~0 is best
            # The shift is such that (best_x[d] + offset) is near an integer
            # So shift[d] ≈ round(best_x[d] + offset) - (best_x[d] + offset)
            candidate = best_x[d] + best_probe_offset
            nearest_int = np.round(candidate)
            shift_est[d] = nearest_int - candidate
        
        # Phase 2: Generate candidate minima using estimated shift
        # Minima are at x_i = k_i + shift_i where k_i are integers
        # We want k_i such that x_i is in [-5, 5]
        
        candidates = []
        for d in range(dim):
            s = shift_est[d]
            # k + s in [-5, 5] => k in [-5-s, 5-s]
            k_min = int(np.ceil(-5 - s))
            k_max = int(np.floor(5 - s))
            ks = list(range(k_min, k_max + 1))
            candidates.append(ks)
        
        # Generate all combinations (could be large, so limit)
        # For dim=5, if each dim has ~10 candidates, that's 10^5 = 100k - too many
        # Instead, do coordinate-wise: fix all but one dim at best, vary one dim
        
        # Better approach: Start from best_x and snap each coordinate to nearest candidate
        # Then do local search
        
        # First, snap each coordinate independently
        snapped_x = best_x.copy()
        for d in range(dim):
            s = shift_est[d]
            k = np.round(best_x[d] - s)
            snapped_x[d] = k + s
            snapped_x[d] = np.clip(snapped_x[d], -5, 5)
        
        f_snapped = eval_point(snapped_x)
        
        # Phase 3: Aggressive local search around the snapped point
        # Use a shrinking step size local search
        
        local_budget = budget - calls - 10  # Reserve 10 for safety
        if local_budget > 0:
            # Multi-scale local search
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            for step in step_sizes:
                if calls >= budget - 5 or time.time() >= deadline:
                    break
                
                # For each dimension, try +/- step
                improved = True
                while improved and calls < budget - 5 and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        if calls >= budget - 5 or time.time() >= deadline:
                            break
                        
                        for direction in [1, -1]:
                            if calls >= budget - 5 or time.time() >= deadline:
                                break
                            
                            trial = best_x.copy()
                            trial[d] = best_x[d] + direction * step
                            trial = np.clip(trial, -5, 5)
                            f_trial = eval_point(trial)
                            
                            if f_trial < best_f:
                                improved = True
                                # Accept and continue searching from new point
        
        return best_x, best_f
