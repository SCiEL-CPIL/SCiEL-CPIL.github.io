"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using Multi-start Nelder-Mead Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Strategy: Multi-start Nelder-Mead
        # 1. Generate a diverse set of initial points (seeds)
        # 2. For each seed, run a local Nelder-Mead optimization using a scalarized objective
        #    (Tchebycheff or Weighted Sum) to drive the point towards the Pareto front.
        # 3. Record all evaluated points.
        
        # Budget allocation:
        # - Initial seeds: ~40 points (20% of budget)
        # - Local search: ~160 points (80% of budget)
        # Nelder-Mead in D dimensions typically takes ~D^2 to D^3 evaluations to converge.
        # D=12, so ~144-1728 evals. This is too many for full convergence per seed.
        # We will use a limited number of steps or early stopping.
        
        # Let's use 20 seeds.
        # 20 seeds * 12 init = 240? No, we evaluate seeds once.
        # 20 seeds = 20 evals.
        # Remaining 180 evals for local search.
        # 180 / 20 = 9 evals per seed.
        # This is very few for NM. 
        # Maybe fewer seeds with more budget?
        # 10 seeds = 10 evals. 190 / 10 = 19 evals per seed.
        # 5 seeds = 5 evals. 195 / 5 = 39 evals per seed.
        
        # Let's try 10 seeds.
        n_seeds = 10
        
        # 1. Initialize Seeds
        # Use Latin Hypercube or simple random for diversity.
        X_seeds = lo + range_ * np.random.random((n_seeds, self.dim))
        F_seeds = np.array([problem(x) for x in X_seeds])
        used = n_seeds
        
        # Store all points
        all_points = list(zip(X_seeds, F_seeds))
        
        # 2. Local Search Loop
        # For each seed, we will perform local optimization.
        # We need a scalar objective. We can use a weighted sum with random weights,
        # or Tchebycheff with the current ideal point.
        # Since we want to cover the front, we should use different weights for different seeds.
        
        # Pre-define weights for each seed to target different parts of the front.
        # For 3 objectives, we can use Dirichlet distribution or random normalized weights.
        weights = np.random.dirichlet(np.ones(self.n_obj), size=n_seeds)
        
        # Ideal point for Tchebycheff (dynamic)
        # But for simplicity and stability in a fixed budget, let's use a static reference or 
        # update it slowly. Actually, Weighted Sum is easier to implement robustly.
        # However, Weighted Sum fails on non-convex fronts. WFG7 is non-convex.
        # Tchebycheff is better.
        
        # Let's use Tchebycheff with a fixed reference point (ideal point estimate).
        # We can estimate the ideal point from the initial seeds.
        ref_point = np.min(F_seeds, axis=0)
        
        # We will run a simplified Nelder-Mead for each seed.
        # Standard NM maintains a simplex of D+1 points.
        # D=12 -> 13 points.
        # Initial simplex: Seed + D perturbations.
        # This costs 13 evals per seed just for initialization.
        # 10 seeds * 13 = 130 evals.
        # Remaining 70 evals for iterations.
        # This is feasible.
        
        # NM Parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        # We will implement a lightweight NM that stops when budget is low or improvement is small.
        
        for s in range(n_seeds):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a basic simplex (D+1 points)
            # We already have the seed point. We need D more.
            if used + self.dim > self.budget:
                # Not enough for full simplex, skip or do single-step perturbations
                # Let's just do random local perturbations if budget is tight
                while used < self.budget and time.time() < deadline:
                    # Random perturbation around current best in this seed's local region
                    # For simplicity, just pick a random point near the seed
                    x_cur = X_seeds[s]
                    # Perturb
                    pert = np.random.normal(0, 0.1, self.dim) * range_
                    x_new = np.clip(x_cur + pert, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    all_points.append((x_new, f_new))
                    # Update seed if better (using Tchebycheff with fixed ref)
                    # We need a scalar metric.
                    # Let's just keep the seed if it dominates? No, Tchebycheff is better for non-convex.
                    # But for single point update, we can just compare Tchebycheff distance to ref.
                    w = weights[s]
                    dist_old = np.max(w * np.abs(F_seeds[s] - ref_point))
                    dist_new = np.max(w * np.abs(f_new - ref_point))
                    if dist_new < dist_old:
                        X_seeds[s] = x_new
                        F_seeds[s] = f_new
                continue
            
            # Initialize Simplex
            # Point 0 is the seed
            simplex = [X_seeds[s].copy()]
            f_simplex = [F_seeds[s].copy()]
            
            # Generate D perturbations
            # Use a small step size, e.g., 0.1 * range
            step = 0.1 * range_
            for j in range(self.dim):
                x_pert = X_seeds[s].copy()
                x_pert[j] += step[j]
                x_pert = np.clip(x_pert, lo, hi)
                f_pert = problem(x_pert)
                used += 1
                simplex.append(x_pert)
                f_simplex.append(f_pert)
                all_points.append((x_pert, f_pert))
                
                if used >= self.budget:
                    break
            
            if used >= self.budget:
                break
                
            # Run NM iterations
            # We have D+1 points in simplex.
            # We will iterate until budget is low or no improvement.
            # Max iterations per seed: (budget - used) // (D+1) roughly, but we add 1 point per iter.
            
            w = weights[s]
            
            while used < self.budget and time.time() < deadline:
                # Sort simplex by Tchebycheff fitness
                # Fitness: max_k ( w_k * |f_k - ref_point_k| )
                # Note: ref_point is fixed for this seed's search to keep it simple.
                # If we want to be smarter, we could update ref_point, but that complicates NM.
                
                fitnesses = []
                for i in range(len(simplex)):
                    diff = np.abs(f_simplex[i] - ref_point)
                    fit = np.max(w * diff)
                    fitnesses.append(fit)
                
                # Indices sorted by fitness
                idx = np.argsort(fitnesses)
                
                # Reorder simplex and fitnesses
                simplex = [simplex[i] for i in idx]
                f_simplex = [f_simplex[i] for i in idx]
                fitnesses = [fitnesses[i] for i in idx]
                
                # Worst point is last
                worst_idx = len(simplex) - 1
                best_idx = 0
                second_best_idx = 1
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[worst_idx])
                x_r = np.clip(x_r, lo, hi)
                f_r = problem(x_r)
                used += 1
                all_points.append((x_r, f_r))
                
                # Calculate fitness of reflection
                diff_r = np.abs(f_r - ref_point)
                fit_r = np.max(w * diff_r)
                
                if used >= self.budget:
                    break
                
                if fit_r < fitnesses[best_idx]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = problem(x_e)
                    used += 1
                    all_points.append((x_e, f_e))
                    
                    diff_e = np.abs(f_e - ref_point)
                    fit_e = np.max(w * diff_e)
                    
                    if fit_e < fit_r:
                        simplex[worst_idx] = x_e
                        f_simplex[worst_idx] = f_e
                    else:
                        simplex[worst_idx] = x_r
                        f_simplex[worst_idx] = f_r
                        
                elif fit_r < fitnesses[second_best_idx]:
                    # Accept Reflection
                    simplex[worst_idx] = x_r
                    f_simplex[worst_idx] = f_r
                else:
                    # Contraction
                    # Outside or Inside contraction?
                    # If worst is better than reflection, inside contraction.
                    # If worst is worse, outside contraction.
                    # Standard NM:
                    # if fit_r < fit_worst: inside contraction
                    # else: outside contraction
                    
                    if fit_r < fitnesses[worst_idx]:
                        # Inside Contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, lo, hi)
                    else:
                        # Outside Contraction
                        x_c = centroid - rho * (simplex[worst_idx] - centroid)
                        x_c = np.clip(x_c, lo, hi)
                        
                    f_c = problem(x_c)
                    used += 1
                    all_points.append((x_c, f_c))
                    
                    diff_c = np.abs(f_c - ref_point)
                    fit_c = np.max(w * diff_c)
                    
                    if used >= self.budget:
                        break
                        
                    if fit_c < fitnesses[worst_idx]:
                        simplex[worst_idx] = x_c
                        f_simplex[worst_idx] = f_c
                    else:
                        # Shrink
                        # Shrink all points except the best towards the best
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[best_idx] + sigma * (simplex[i] - simplex[best_idx])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_simplex[i] = problem(simplex[i])
                            used += 1
                            all_points.append((simplex[i], f_simplex[i]))
                            
                            if used >= self.budget:
                                break
                            
                        if used >= self.budget:
                            break

            # Update the seed to the best point found in this local search
            # Find the best point in the final simplex
            final_fitnesses = []
            for i in range(len(simplex)):
                diff = np.abs(f_simplex[i] - ref_point)
                fit = np.max(w * diff)
                final_fitnesses.append(fit)
            
            best_local_idx = np.argmin(final_fitnesses)
            X_seeds[s] = simplex[best_local_idx]
            F_seeds[s] = f_simplex[best_local_idx]

        return X_seeds
# EVOLVE-BLOCK-END
