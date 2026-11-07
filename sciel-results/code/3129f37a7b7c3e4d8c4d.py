"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based (mu+lambda)-ES with adaptive step, multi-restart,
    and axis-wise Rastrigin-periodic refinement for f15 (rotated Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        # Global best tracking
        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            call_count += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Rastrigin-aware coordinate descent refinement ---
        def local_refine(x, fx, max_iter=20):
            """Coordinate descent with Rastrigin-snap candidates.
            For each axis, try snapping to nearest 2*pi multiple and small perturbations."""
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return x, fx
            
            for _ in range(max_iter):
                if call_count >= budget or time.time() > deadline:
                    break
                improved = False
                for axis in range(dim):
                    if call_count >= budget or time.time() > deadline:
                        break
                    x0 = x[axis]
                    
                    # Generate candidates: current, nearest 2*pi snap, neighbors, perturbations
                    nearest = 2.0 * np.pi * np.round(x0 / (2.0 * np.pi))
                    candidates = [x0]
                    candidates.append(nearest)
                    candidates.append(nearest + 2.0 * np.pi)
                    candidates.append(nearest - 2.0 * np.pi)
                    for d in [0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5]:
                        candidates.append(x0 + d)
                    
                    # Deduplicate and clip
                    seen = set()
                    unique_candidates = []
                    for c in candidates:
                        c = float(np.clip(c, -5.0, 5.0))
                        key = round(c, 8)
                        if key not in seen:
                            seen.add(key)
                            unique_candidates.append(c)
                    
                    best_cand = x0
                    best_cand_f = fx
                    for c in unique_candidates:
                        if call_count >= budget or time.time() > deadline:
                            break
                        if abs(c - x0) < 1e-10:
                            continue
                        y = x.copy()
                        y[axis] = c
                        fy = eval_func(y)
                        if fy < best_cand_f:
                            best_cand_f = fy
                            best_cand = c
                    
                    if best_cand != x0:
                        x[axis] = best_cand
                        fx = best_cand_f
                        improved = True
                
                if not improved:
                    break
            
            return x, fx

        # --- Main search: (mu+lambda)-ES with adaptive sigma and periodic refinement ---
        mu = 4
        lam = 12
        sigma = 2.5
        
        # Initial population
        pop = np.array([np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(mu)])
        pop_f = np.array([eval_func(p) for p in pop])
        
        # Sort by fitness
        idx = np.argsort(pop_f)
        pop = pop[idx]
        pop_f = pop_f[idx]

        generation = 0
        no_improve_count = 0
        best_f_prev = best_f
        refine_interval = 15  # refine every 15 generations

        while call_count < budget and time.time() < deadline:
            generation += 1

            # Adaptive sigma: expand if improving, shrink if stuck
            if generation % 10 == 0:
                if best_f < best_f_prev - 1e-6:
                    sigma = min(sigma * 1.3, 8.0)
                    best_f_prev = best_f
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= 3:
                        sigma = max(sigma * 0.7, 0.1)
                        no_improve_count = 0

            # Generate offspring
            offspring = []
            offspring_f = []
            for i in range(lam):
                if call_count >= budget or time.time() > deadline:
                    break
                # Select parent: tournament of size 2 from population
                i1 = rng.integers(0, mu)
                i2 = rng.integers(0, mu)
                parent_idx = i1 if pop_f[i1] < pop_f[i2] else i2
                parent = pop[parent_idx]
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                child_f = eval_func(child)
                offspring.append(child)
                offspring_f.append(child_f)

            if not offspring:
                break

            offspring = np.array(offspring)
            offspring_f = np.array(offspring_f)

            # Combine parents and offspring, select best mu
            all_x = np.vstack([pop, offspring])
            all_f = np.concatenate([pop_f, offspring_f])
            idx = np.argsort(all_f)[:mu]
            pop = all_x[idx]
            pop_f = all_f[idx]

            # Periodic local refinement on the best point
            if generation % refine_interval == 0:
                x_ref, f_ref = local_refine(pop[0].copy(), pop_f[0], max_iter=10)
                if f_ref < pop_f[0]:
                    pop[0] = x_ref
                    pop_f[0] = f_ref

            # Restart if stuck: if sigma is very small and no improvement
            if sigma < 0.15 and generation % 30 == 0:
                # Inject a few new random points
                for _ in range(2):
                    if call_count >= budget or time.time() > deadline:
                        break
                    new_x = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    new_f = eval_func(new_x)
                    worst_idx = np.argmax(pop_f)
                    if new_f < pop_f[worst_idx]:
                        pop[worst_idx] = new_x
                        pop_f[worst_idx] = new_f
                sigma = 2.0
                no_improve_count = 0

        # Final phase: intensive local refinement on best point
        if best_x is not None and call_count < budget and time.time() < deadline:
            x_final = best_x.copy()
            f_final = best_f
            
            # Multiple rounds of coordinate descent with decreasing max_iter
            for round_num in range(3):
                if call_count >= budget or time.time() > deadline:
                    break
                x_final, f_final = local_refine(x_final, f_final, max_iter=15)
                if f_final >= best_f:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
