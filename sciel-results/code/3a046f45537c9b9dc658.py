# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Structured Initialization + Adaptive Multi-Restart (1+1)-ES Optimizer.
    
    This optimizer combines deterministic structured sampling (lattice + Halton) 
    for robust early coverage with an adaptive (1+1)-Evolution Strategy for 
    local refinement. It uses a multi-restart strategy to escape local minima 
    on multimodal functions like Rastrigin and Gallagher.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_exceeded():
            return time.time() > deadline

        # Phase 1: Structured Initialization
        candidates = []

        # Lattice points
        lattice_vals = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        n_lattice_samples = 30
        for _ in range(n_lattice_samples):
            pt = np.array([rng.choice(lattice_vals) for _ in range(dim)])
            candidates.append(pt)

        # Halton Sequence
        def halton(idx, base):
            q = 1.0
            r = 0.0
            while idx > 0:
                q /= base
                r += q * (idx % base)
                idx //= base
            return r

        primes = [2, 3, 5, 7, 11]
        n_halton = 40
        for i in range(1, n_halton + 1):
            pt = np.zeros(dim)
            for j in range(dim):
                pt[j] = halton(i, primes[j])
            pt = pt * 10.0 - 5.0
            candidates.append(np.clip(pt, -5, 5))

        # Random Uniform
        n_random = 20
        for _ in range(n_random):
            pt = rng.uniform(-5, 5, dim)
            candidates.append(pt)

        # High priority points
        high_priority = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 4.0,
            np.ones(dim) * -4.0,
        ]
        
        for hp in high_priority:
            if calls >= budget or time_exceeded():
                break
            pt = np.clip(hp, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        indices = rng.permutation(len(candidates))
        for idx in indices:
            if calls >= budget or time_exceeded():
                break
            pt = np.clip(candidates[idx], -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # Phase 2: Multi-Start Nelder-Mead Portfolio
        # We select a diverse set of starting points from the best found so far
        # and run a short Nelder-Mead optimization from each.
        
        # Collect top candidates
        # We need to store the points we evaluated to pick the best ones.
        # Since we didn't store them all in a list with values, we re-evaluate?
        # No, that wastes budget. 
        # Instead, we'll just use the best_x found so far, and maybe a few random perturbations of it?
        # Or better: during Phase 1, we should have kept track of the best N points.
        
        # Let's refactor Phase 1 to keep track of top N points.
        # But I can't easily refactor the whole loop without a large diff.
        # Let's just use the current best_x and generate a few perturbations.
        
        # Alternative: Use the remaining budget to run Nelder-Mead from the global best.
        # And maybe 1-2 other good points if we can find them.
        
        # Since we don't have a list of top N points from Phase 1, we will:
        # 1. Run Nelder-Mead from best_x.
        # 2. If budget remains, run Nelder-Mead from a random perturbation of best_x.
        # 3. If budget remains, run Nelder-Mead from a random point in the box.
        
        import scipy.optimize
        
        remaining_budget = budget - calls
        if remaining_budget > 10 and not time_exceeded():
            # Run Nelder-Mead from best_x
            max_iter_1 = min(30, remaining_budget)
            
            calls_1 = 0
            def nm_func_1(x):
                nonlocal calls_1, calls
                if calls_1 >= max_iter_1:
                    raise ValueError("Budget exceeded")
                if time_exceeded():
                    raise ValueError("Time exceeded")
                calls_1 += 1
                calls += 1
                return func(x)
            
            try:
                res1 = scipy.optimize.minimize(
                    nm_func_1, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': max_iter_1, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
                )
                if res1.fun < best_f:
                    update_best(res1.x, res1.fun)
            except Exception:
                pass

            remaining_budget = budget - calls
            if remaining_budget > 10 and not time_exceeded():
                # Run Nelder-Mead from a perturbation of best_x
                perturb_scale = 0.5
                start_x2 = np.clip(best_x + perturb_scale * rng.standard_normal(dim), -5, 5)
                
                max_iter_2 = min(20, remaining_budget)
                calls_2 = 0
                def nm_func_2(x):
                    nonlocal calls_2, calls
                    if calls_2 >= max_iter_2:
                        raise ValueError("Budget exceeded")
                    if time_exceeded():
                        raise ValueError("Time exceeded")
                    calls_2 += 1
                    calls += 1
                    return func(x)
                
                try:
                    res2 = scipy.optimize.minimize(
                        nm_func_2, 
                        start_x2, 
                        method='Nelder-Mead', 
                        options={'maxiter': max_iter_2, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res2.fun < best_f:
                        update_best(res2.x, res2.fun)
                except Exception:
                    pass

            remaining_budget = budget - calls
            if remaining_budget > 10 and not time_exceeded():
                # Run Nelder-Mead from a random point
                start_x3 = rng.uniform(-5, 5, dim)
                
                max_iter_3 = min(15, remaining_budget)
                calls_3 = 0
                def nm_func_3(x):
                    nonlocal calls_3, calls
                    if calls_3 >= max_iter_3:
                        raise ValueError("Budget exceeded")
                    if time_exceeded():
                        raise ValueError("Time exceeded")
                    calls_3 += 1
                    calls += 1
                    return func(x)
                
                try:
                    res3 = scipy.optimize.minimize(
                        nm_func_3, 
                        start_x3, 
                        method='Nelder-Mead', 
                        options={'maxiter': max_iter_3, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if res3.fun < best_f:
                        update_best(res3.x, res3.fun)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
