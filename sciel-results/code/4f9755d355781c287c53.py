import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3000.0, 0.06])
        
        # Archive of points that contribute to hypervolume
        # We maintain a list of (x, f) pairs
        archive = []
        
        def is_dominated(f_new, archive_f):
            """Check if f_new is dominated by any point in archive_f."""
            for f in archive_f:
                if f[0] <= f_new[0] and f[1] <= f_new[1] and (f[0] < f_new[0] or f[1] < f_new[1]):
                    return True
            return False

        def dominates(f_new, f_old):
            """Check if f_new strictly dominates f_old."""
            return f_new[0] <= f_old[0] and f_new[1] <= f_old[1] and (f_new[0] < f_old[0] or f_new[1] < f_old[1])

        def add_to_archive(x, f):
            """
            Add point to archive if it is not dominated and not too close to existing points.
            Returns True if added, False otherwise.
            """
            if f[0] >= ref[0] or f[1] >= ref[1]:
                return False
            
            # Check if dominated
            if is_dominated(f, [af for _, af in archive]):
                return False
            
            # Remove points dominated by new point
            new_archive = []
            for ax, af in archive:
                if dominates(f, af):
                    continue
                new_archive.append((ax, af))
            
            # Check proximity to avoid cluttering (simple distance check in objective space)
            # If the new point is very close to an existing point in objective space, 
            # it might not add significant HV. 
            # For 2D, we can check if it falls into the "shadow" of the staircase.
            # A simple heuristic: if it's within a small epsilon of an existing point, skip.
            # But since we want to maximize HV, even small improvements count if they extend the front.
            # However, to keep the archive manageable and useful for DE, we prune if archive is large.
            
            new_archive.append((x, f))
            
            # Prune if archive is too large? 
            # With budget 200, archive shouldn't get huge. 
            # But if it does, we keep the best 50 by HV contribution or just random.
            # For now, just keep all non-dominated.
            
            archive.clear()
            archive.extend(new_archive)
            return True

        def hv_contribution(f_new, archive_f):
            """
            Calculate the hypervolume contribution of f_new given the current archive.
            Only returns positive if f_new improves the total HV.
            """
            if f_new[0] >= ref[0] or f_new[1] >= ref[1]:
                return 0.0
            
            # Filter valid points
            valid_f = [f for f in archive_f if f[0] < ref[0] and f[1] < ref[1]]
            
            # If dominated by archive, contribution is 0
            for f in valid_f:
                if f[0] <= f_new[0] and f[1] <= f_new[1] and (f[0] < f_new[0] or f[1] < f_new[1]):
                    return 0.0
            
            # Calculate HV of current archive
            def calc_hv(points):
                if not points:
                    return 0.0
                points = sorted(points, key=lambda x: x[0])
                hv = 0.0
                prev_y = ref[1]
                for i, p in enumerate(points):
                    x_i = p[0]
                    y_i = p[1]
                    x_next = points[i+1][0] if i + 1 < len(points) else ref[0]
                    # The height of the rectangle is from y_i to the "upper bound" of the previous step?
                    # No, in 2D minimization, the HV is the area under the Pareto front curve from (0,0) to ref.
                    # If points are sorted by x ascending, y is descending (non-dominated).
                    # The area is sum of rectangles: (x_next - x_i) * (ref_y - y_i) is WRONG.
                    # Correct 2D HV:
                    # Sort by x ascending.
                    # For each point i, the width is x_{i+1} - x_i (or ref_x - x_last).
                    # The height is ref_y - y_i.
                    # Wait, the "staircase" goes down and to the right.
                    # The area is the union of rectangles [x_i, ref_x] x [y_i, ref_y].
                    # This union is equivalent to summing (x_{i+1} - x_i) * (ref_y - y_i) ONLY IF the points form a staircase where y decreases as x increases.
                    # Let's verify:
                    # Point A: (1, 9), Point B: (9, 1). Ref: (10, 10).
                    # Sort by x: A(1,9), B(9,1).
                    # Rect 1: x in [1, 9], height 10-9=1. Area = 8*1 = 8.
                    # Rect 2: x in [9, 10], height 10-1=9. Area = 1*9 = 9.
                    # Total = 17.
                    # Actual HV: Union of [1,10]x[9,10] and [9,10]x[1,10].
                    # [1,10]x[9,10] is 9*1 = 9.
                    # [9,10]x[1,10] is 1*9 = 9.
                    # Overlap [9,10]x[9,10] is 1*1 = 1.
                    # Union = 9 + 9 - 1 = 17. Correct.
                    
                    # So the formula is:
                    # Sum_{i} (x_{i+1} - x_i) * (ref_y - y_i)
                    # where x_{n+1} = ref_x.
                    
                    w = x_next - x_i
                    h = ref[1] - y_i
                    hv += w * h
                return hv

            hv_before = calc_hv(valid_f)
            hv_after = calc_hv(valid_f + [f_new])
            return hv_after - hv_before

        # 1. Initial Sampling: Latin Hypercube or Random
        N_init = 30
        X = lo + (hi - lo) * np.random.random((N_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = N_init
        
        for i in range(N_init):
            add_to_archive(X[i], F[i])
        
        # 2. Differential Evolution Loop
        # We maintain a population. 
        # Strategy: 
        # - Keep a population of size N_pop = 20.
        # - Initialize from archive + random.
        # - For each generation, generate offspring using DE formula: x_trial = x_base + F * (x_rand1 - x_rand2)
        # - Evaluate offspring.
        # - If offspring improves HV, add to archive and replace the worst member of the population (or the one it dominates).
        
        N_pop = 20
        # Initialize population
        X_pop = np.zeros((N_pop, self.dim))
        F_pop = np.zeros((N_pop, self.n_obj))
        
        # Fill population from archive
        if len(archive) >= N_pop:
            indices = np.random.choice(len(archive), N_pop, replace=False)
            for i, idx in enumerate(indices):
                X_pop[i] = archive[idx][0]
                F_pop[i] = archive[idx][1]
        else:
            for i in range(len(archive)):
                X_pop[i] = archive[i][0]
                F_pop[i] = archive[i][1]
            for i in range(len(archive), N_pop):
                X_pop[i] = lo + (hi - lo) * np.random.random(self.dim)
                F_pop[i] = problem(X_pop[i])
                used += 1
                add_to_archive(X_pop[i], F_pop[i])
        
        # DE Parameters
        F_factor = 0.8  # Scale factor
        CR = 0.9  # Crossover rate
        max_generations = (self.budget - used) // N_pop
        
        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring for each member of the population
            for i in range(N_pop):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select base, rand1, rand2
                base = i
                rand1 = np.random.randint(N_pop)
                rand2 = np.random.randint(N_pop)
                while rand1 == base or rand2 == base or rand1 == rand2:
                    rand1 = np.random.randint(N_pop)
                    rand2 = np.random.randint(N_pop)
                
                # DE formula
                x_trial = X_pop[base] + F_factor * (X_pop[rand1] - X_pop[rand2])
                
                # Crossover (binomial)
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.random() < CR or j == j_rand:
                        x_trial[j] = x_trial[j]
                    else:
                        x_trial[j] = X_pop[base][j]
                
                x_trial = np.clip(x_trial, lo, hi)
                
                f_trial = problem(x_trial)
                used += 1
                
                # Check if trial improves hypervolume
                # We only add to archive if it's non-dominated and contributes to HV
                # To be efficient, we can just check if it's non-dominated in the current archive
                # and if it dominates the current population member or improves HV.
                
                # Simple greedy: If it's non-dominated in archive, add it.
                # If it dominates the current pop member, replace pop member.
                
                # Check domination against archive
                dominated = False
                for _, af in archive:
                    if af[0] <= f_trial[0] and af[1] <= f_trial[1] and (af[0] < f_trial[0] or af[1] < f_trial[1]):
                        dominated = True
                        break
                
                if not dominated:
                    # It is non-dominated. Check if it improves HV.
                    # For speed, we can just add it to archive and let the archive handle pruning.
                    # But we want to be careful not to bloat the archive with points that don't add HV.
                    # Let's calculate HV contribution.
                    # If contribution > 0, it's a good point.
                    
                    # Optimization: Only calculate HV contribution if archive is small or if it dominates pop member.
                    # If it dominates pop member, it's likely good.
                    if dominates(f_trial, F_pop[i]):
                        add_to_archive(x_trial, f_trial)
                        X_pop[i] = x_trial
                        F_pop[i] = f_trial
                    else:
                        # Check HV contribution
                        # To save time, we can skip HV calc for every single point if archive is large.
                        # But with budget 200, archive is small.
                        contrib = hv_contribution(f_trial, [af for _, af in archive])
                        if contrib > 1e-9: # Small threshold
                            add_to_archive(x_trial, f_trial)
                            # Replace the worst member in population?
                            # Or just replace random?
                            # Replacing the member it "competes" with is better for diversity.
                            # Let's replace the member with the lowest HV contribution in the population?
                            # That's expensive.
                            # Let's replace a random member if the new point is "better" in some sense.
                            # Or just replace the base member if it's not dominating but is non-dominated.
                            # Actually, if it doesn't dominate the base, but is non-dominated, 
                            # it might be a good addition to the population for future DE operations.
                            # Let's replace the base member only if it dominates it, otherwise replace a random one 
                            # if the population is "stuck" or just keep diversity.
                            
                            # Simple strategy: Replace a random member to maintain diversity, 
                            # but prefer replacing members that are dominated by the new point (already handled).
                            # If it doesn't dominate anyone, replace the one with the lowest "crowding" or random.
                            k = np.random.randint(N_pop)
                            X_pop[k] = x_trial
                            F_pop[k] = f_trial

        # 3. Local Refinement Phase
        # Use remaining budget to refine the top HV contributors.
        # Sort archive by HV contribution (approximate) or just take the top few non-dominated points.
        # We can identify the "corners" of the Pareto front.
        
        remaining_budget = self.budget - used
        
        if remaining_budget > 0 and archive:
            # Select top candidates for local search.
            # We want points that are "extreme" in the objective space.
            # Sort archive by f[0] and f[1].
            sorted_by_f0 = sorted(archive, key=lambda x: x[1][0])
            sorted_by_f1 = sorted(archive, key=lambda x: x[1][1])
            
            # Pick the best in f0, best in f1, and a middle one
            candidates = []
            candidates.append(sorted_by_f0[0])
            candidates.append(sorted_by_f1[0])
            
            # Add a middle point if available
            if len(archive) > 2:
                mid_idx = len(archive) // 2
                # Find a point that is balanced
                # Simple heuristic: point with min(f0/f1 ratio) or similar
                # Just pick the middle of the sorted list by f0
                candidates.append(sorted_by_f0[mid_idx])
            
            # Remove duplicates
            unique_candidates = []
            for c in candidates:
                if not any(np.allclose(c[0], uc[0]) for uc in unique_candidates):
                    unique_candidates.append(c)
            
            # Allocate budget to local search
            # Use Nelder-Mead or similar local minimization on a scalarized objective?
            # Or just perturb and accept if HV improves.
            
            # Simple local search: Gaussian perturbation
            # For each candidate, try to improve it.
            n_local = len(unique_candidates)
            budget_per_local = max(1, remaining_budget // n_local)
            
            for cand in unique_candidates:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x_best = cand[0].copy()
                f_best = cand[1].copy()
                
                # Try to improve x_best
                step_size = 0.1 * (hi - lo)
                
                for _ in range(budget_per_local):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Random direction
                    direction = np.random.randn(self.dim)
                    direction /= np.linalg.norm(direction)
                    
                    x_new = x_best + step_size * direction
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    # Check if f_new is better than f_best in terms of HV contribution
                    # Or simply if it dominates f_best
                    if dominates(f_new, f_best):
                        x_best = x_new
                        f_best = f_new
                        add_to_archive(x_best, f_best)
                        # Reduce step size if improved?
                        step_size *= 0.5
                    else:
                        # If not improved, maybe increase step size if stuck?
                        # Or just keep same.
                        pass

        return X_pop
