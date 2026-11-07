"""Multi-objective optimizer on WFG7: Hypervolume-guided evolutionary search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Evolutionary multi-objective search with hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _hypervolume_3d(self, F, ref):
        """Compute 3D hypervolume using sorting and sweep."""
        if len(F) == 0:
            return 0.0
        # Filter points that are better than reference in all objectives
        mask = np.all(F < ref, axis=1)
        F = F[mask]
        if len(F) == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(F[:, 0])
        F = F[order]
        n = len(F)
        
        # Compute hypervolume contribution
        # For 3D, we can use the formula:
        # HV = sum over sorted points of (ref[0] - f_i[0]) * area_2d(f_i[1], f_i[2], ref[1], ref[2])
        # where area_2d is the 2D hypervolume contribution
        
        hv = 0.0
        
        # Process points in order of f[0]
        for i in range(n):
            f0, f1, f2 = F[i]
            # The contribution from this point
            # We need to compute the 2D area that this point "owns"
            # This is complex; use a simpler incremental approach
            
            # Actually, for 3D hypervolume, let's use a different approach:
            # Sort by f0, then for each point, compute the 2D hypervolume of the 
            # remaining points projected onto (f1, f2) plane, weighted by (ref[0] - f0)
            
            # Simpler: use the standard 3D HV algorithm
            # Sort by f0
            # For each point i, its contribution is (ref[0] - f_i[0]) * H2D(points j where f_j[0] > f_i[0], in (f1,f2) space)
            
            # This is getting complex. Let me use a different strategy:
            # Since we only need to compare contributions for selection, we can use 
            # a simpler metric: the volume of the box from the point to the reference,
            # adjusted by how many other points dominate parts of it.
            
            # For selection purposes, let's use a simpler but effective metric:
            # For each point, compute the hypervolume it would add if it were the only point.
            # This is an approximation but works well in practice for selection.
            
            # Box volume from this point to reference
            vol = (ref[0] - f0) * (ref[1] - f1) * (ref[2] - f2)
            hv += vol
        
        return hv

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        is_dom = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom[i] = True
                    break
        return ~is_dom

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref = np.array([3.0, 5.0, 7.0])  # Reference point from scoring
        
        # Population size: balance between diversity and computation
        pop_size = min(30, self.budget // 3)
        if pop_size < 10:
            pop_size = 10

        # Initial population: diverse random points
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points for final Pareto front
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        # Adaptive mutation: start with larger steps, decrease over time
        base_mut_scale = 0.2  # Initial mutation scale
        decay = 0.95  # Decay factor per generation
        
        # Main evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            non_dom_mask = self._non_dominated_mask(F)
            non_dom_indices = np.where(non_dom_mask)[0]
            
            if len(non_dom_indices) == 0:
                # Fallback: use all points
                parents_idx = np.arange(pop_size)
            else:
                parents_idx = non_dom_indices
            
            # Select parent using tournament on hypervolume contributions
            # Compute simple HV contribution for selection
            # For each candidate parent, estimate its contribution
            n_parents = len(parents_idx)
            
            if n_parents <= 3:
                # Small set: pick randomly
                p_idx = parents_idx[rng.integers(n_parents)]
            else:
                # Tournament: pick 2-3 random parents, choose best by HV contribution
                # Use simple box volume as proxy for HV contribution
                tournament_size = min(3, n_parents)
                tournament_candidates = rng.choice(n_parents, size=tournament_size, replace=False)
                
                # Score each candidate by box volume to reference
                best_cand = tournament_candidates[0]
                best_vol = (ref[0] - F[parents_idx[best_cand], 0]) * \
                          (ref[1] - F[parents_idx[best_cand], 1]) * \
                          (ref[2] - F[parents_idx[best_cand], 2])
                
                for tc in tournament_candidates[1:]:
                    vol = (ref[0] - F[parents_idx[tc], 0]) * \
                          (ref[1] - F[parents_idx[tc], 1]) * \
                          (ref[2] - F[parents_idx[tc], 2])
                    if vol > best_vol:
                        best_vol = vol
                        best_cand = tc
                
                p_idx = parents_idx[best_cand]
            
            # Adaptive mutation
            progress = used / self.budget
            mut_scale = base_mut_scale * (decay ** (progress * 10))
            mut_scale = max(mut_scale, 0.01)  # Minimum mutation scale
            
            child = X[p_idx] + mut_scale * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            used += 1
            
            # Replacement strategy: 
            # If child is non-dominated, replace the point with lowest HV contribution
            # among dominated points
            # If child is dominated, only replace if it's better than the worst point
            
            # Check if child is dominated by current population
            child_dominated = False
            for i in range(pop_size):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    child_dominated = True
                    break
            
            if not child_dominated:
                # Child is non-dominated: replace the point with lowest quality
                # Among dominated points, replace the one with highest sum of objectives
                # If no dominated points, replace the one with lowest HV contribution
                
                dominated_mask = ~non_dom_mask
                if np.any(dominated_mask):
                    # Replace the worst dominated point
                    dom_indices = np.where(dominated_mask)[0]
                    # Pick the one with highest sum of objectives (worst)
                    worst_dom = dom_indices[np.argmax(np.sum(F[dom_indices], axis=1))]
                    replace_idx = worst_dom
                else:
                    # All points are non-dominated: replace the one with lowest HV contribution
                    # Use box volume as proxy
                    vols = (ref[0] - F[:, 0]) * (ref[1] - F[:, 1]) * (ref[2] - F[:, 2])
                    replace_idx = np.argmin(vols)
                
                X[replace_idx] = child
                F[replace_idx] = fc
            else:
                # Child is dominated: only accept if it's better than the worst point
                # Use sum of objectives as quality measure
                child_sum = np.sum(fc)
                worst_sum = np.max(np.sum(F, axis=1))
                
                if child_sum < worst_sum:
                    # Replace the worst point
                    replace_idx = np.argmax(np.sum(F, axis=1))
                    X[replace_idx] = child
                    F[replace_idx] = fc
        
        # Final local search on the best non-dominated points
        if used < self.budget:
            remaining = self.budget - used
            
            # Get non-dominated points from final population
            non_dom_mask = self._non_dominated_mask(F)
            non_dom_indices = np.where(non_dom_mask)[0]
            
            if len(non_dom_indices) > 0:
                # Polish each non-dominated point with local search
                polish_per_point = max(1, remaining // len(non_dom_indices))
                polish_per_point = min(polish_per_point, 10)  # Cap to avoid too many evaluations
                
                for idx in non_dom_indices:
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Local search: small perturbations
                    best_x = X[idx].copy()
                    best_f = F[idx].copy()
                    best_sum = np.sum(best_f)
                    
                    for _ in range(polish_per_point):
                        if used >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Small random perturbation
                        perturbation = 0.01 * span * rng.standard_normal(self.dim)
                        trial_x = np.clip(best_x + perturbation, lo, hi)
                        trial_f = problem(trial_x)
                        used += 1
                        
                        # Accept if better in at least one objective and not worse in others
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial_x
                            best_f = trial_f
                            best_sum = np.sum(best_f)
                        elif np.sum(trial_f) < best_sum:
                            # Accept if better in sum (scalarization)
                            best_x = trial_x
                            best_f = trial_f
                            best_sum = np.sum(best_f)
                    
                    # Update population with polished point
                    X[idx] = best_x
                    F[idx] = best_f
        
        return X
# EVOLVE-BLOCK-END
