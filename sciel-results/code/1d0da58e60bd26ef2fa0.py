"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic grid search + multi-start Nelder-Mead for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        calls = 0

        def eval_point(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            calls += 1
            return func(x)

        # Phase 1: Deterministic grid search for consistent early progress
        # Evaluate a set of well-spread points to find promising basins
        grid_points = []
        
        # Origin
        grid_points.append(np.zeros(dim))
        
        # Axis-aligned points at various distances
        for dist in [4.5, 3.0, 1.5, 0.5]:
            for d in range(dim):
                for s in [-1, 1]:
                    pt = np.zeros(dim)
                    pt[d] = s * dist
                    grid_points.append(pt)
        
        # Corner-like points (but clipped to box)
        for d1 in range(dim):
            for d2 in range(dim):
                if d1 != d2:
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            pt = np.zeros(dim)
                            pt[d1] = s1 * 4.0
                            pt[d2] = s2 * 4.0
                            grid_points.append(pt)
        
        # Some intermediate points
        for d in range(dim):
            for s in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = s * 2.0
                grid_points.append(pt)
                pt2 = np.zeros(dim)
                pt2[d] = s * 3.5
                grid_points.append(pt2)
        
        # Evaluate grid points
        evaluated = []
        for pt in grid_points:
            if calls >= budget or time_left() < 5:
                break
            pt = np.clip(pt, -5, 5)
            # Check if already evaluated
            already = False
            for ep, ef in evaluated:
                if np.allclose(ep, pt, atol=1e-8):
                    already = True
                    break
            if not already:
                f = eval_point(pt)
                evaluated.append((pt.copy(), f))
        
        if not evaluated:
            # Fallback: just evaluate origin
            f = eval_point(np.zeros(dim))
            evaluated.append((np.zeros(dim).copy(), f))
        
        # Sort by function value
        evaluated.sort(key=lambda x: x[1])
        
        best_x = evaluated[0][0].copy()
        best_f = evaluated[0][1]
        
        # Phase 2: Multi-start Nelder-Mead from top candidates
        # Use top K starting points for local refinement
        K = min(5, len(evaluated), max(1, (budget - calls) // 80))
        
        for i in range(K):
            if calls >= budget - 20 or time_left() < 15:
                break
            
            x0 = evaluated[i][0].copy()
            f0 = evaluated[i][1]
            
            # Build initial simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            # Add dim additional points with small perturbation
            for d in range(dim):
                if calls >= budget or time_left() < 2:
                    break
                new_pt = x0.copy()
                new_pt[d] += 0.1
                new_pt = np.clip(new_pt, -5, 5)
                new_f = eval_point(new_pt)
                simplex.append(new_pt)
                simplex_f.append(new_f)
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_pt.copy()
            
            if len(simplex) < 2:
                continue
            
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Run Nelder-Mead for a limited number of iterations
            max_iter = min(30, max(10, (budget - calls) // (K - i + 1)))
            
            for _ in range(max_iter):
                if calls >= budget - 10 or time_left() < 3:
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[j].copy() for j in order]
                simplex_f = [simplex_f[j] for j in order]
                
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                
                if calls >= budget or time_left() < 1:
                    break
                fr = eval_point(xr)
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget or time_left() < 1:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                        break
                    fe = eval_point(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    if fr < simplex_f[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    
                    if calls >= budget or time_left() < 1:
                        break
                    fc = eval_point(xc)
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                    else:
                        # Shrink
                        for j in range(1, len(simplex)):
                            if calls >= budget or time_left() < 1:
                                break
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            simplex_f[j] = eval_point(simplex[j])
                            if simplex_f[j] < best_f:
                                best_f = simplex_f[j]
                                best_x = simplex[j].copy()
                        break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
