"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble with LHS initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        
        # Latin Hypercube sampling for diverse initial points
        n_starts = 8
        dim = self.dim
        budget = self.budget
        
        # Generate LHS points in [-5, 5]^dim
        lhs_points = []
        for i in range(n_starts):
            point = np.zeros(dim)
            for d in range(dim):
                # Each dimension gets a unique stratum
                stratum = (i + d) % n_starts
                # Random within stratum
                u = rng.random()
                point[d] = -5 + (stratum + u) * (10.0 / n_starts)
            lhs_points.append(point)
        
        # Evaluate initial points
        best_x = None
        best_f = np.inf
        call_count = 0
        
        simplex_starts = []
        for p in lhs_points:
            if call_count >= budget:
                break
            fp = func(p)
            call_count += 1
            simplex_starts.append((p, fp))
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
        
        # Nelder-Mead for each simplex
        for i, (x0, f0) in enumerate(simplex_starts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            
            # Initialize simplex: x0 plus dim perturbed points
            simplex = [x0.copy()]
            fvals = [f0]
            
            # Add perturbed points
            for d in range(dim):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                perturbed = x0.copy()
                perturbed[d] += 0.5
                perturbed = np.clip(perturbed, -5, 5)
                fp = func(perturbed)
                call_count += 1
                simplex.append(perturbed)
                fvals.append(fp)
                if fp < best_f:
                    best_f = fp
                    best_x = perturbed.copy()
            
            # Nelder-Mead iterations
            max_iter = 200
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Sort simplex by function values
                order = np.argsort(fvals)
                simplex = [simplex[j] for j in order]
                fvals = [fvals[j] for j in order]
                
                # Check convergence
                if fvals[-1] - fvals[0] < 1e-10:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                call_count += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    call_count += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    call_count += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for j in range(1, len(simplex)):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            fj = func(simplex[j])
                            call_count += 1
                            fvals[j] = fj
                            if fj < best_f:
                                best_f = fj
                                best_x = simplex[j].copy()
                        if call_count >= budget:
                            break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
