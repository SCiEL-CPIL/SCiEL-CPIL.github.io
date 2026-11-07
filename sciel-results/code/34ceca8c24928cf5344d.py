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
        deadline = t0 + 270  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        calls = 0

        def eval_point(x):
            nonlocal calls
            calls += 1
            return func(x)

        # Phase 1: Deterministic grid search for consistent early progress
        # 32 vertices of the hypercube + origin + face centers = 32 + 1 + 10 = 43 points
        # This gives good coverage of [-5,5]^5 with predictable early progress
        
        # Generate 32 vertices
        vertices = []
        for i in range(32):
            v = np.array([(i >> d) & 1 for d in range(dim)], dtype=float)
            v = v * 10 - 5  # map {0,1} to {-5,5}
            vertices.append(v)
        
        # Add origin
        vertices.append(np.zeros(dim))
        
        # Add face centers (midpoints of edges)
        for d in range(dim):
            for s in [-1, 1]:
                fc = np.full(dim, 0.0)
                fc[d] = s * 2.5
                vertices.append(fc)
        
        # Evaluate all grid points
        best_x = None
        best_f = np.inf
        for v in vertices:
            if calls >= budget or time_left() < 0.5:
                break
            f = eval_point(v)
            if f < best_f:
                best_f = f
                best_x = v.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_point(best_x)

        # Phase 2: Multi-start Nelder-Mead for local convergence
        # Use the top K grid points as starting points for Nelder-Mead
        # Sort vertices by function value
        vertex_fvals = []
        for v in vertices:
            if calls >= budget or time_left() < 0.5:
                break
            # Check if already evaluated
            already_evaluated = False
            for existing_v, existing_f in vertex_fvals:
                if np.allclose(existing_v, v, atol=1e-10):
                    already_evaluated = True
                    break
            if not already_evaluated:
                f = eval_point(v)
                vertex_fvals.append((v.copy(), f))
        
        # Sort by function value
        vertex_fvals.sort(key=lambda x: x[1])
        
        # Select top K starts for Nelder-Mead
        K = min(8, len(vertex_fvals), max(1, (budget - calls) // 50))
        
        for i in range(K):
            if calls >= budget - 10 or time_left() < 5:
                break
            
            x0 = vertex_fvals[i][0].copy()
            f0 = vertex_fvals[i][1]
            
            # Nelder-Mead simplex method
            def nelder_mead(x0, f0, max_iter=100):
                nonlocal calls, best_f, best_x
                dim = self.dim
                
                # Initialize simplex
                simplex = [x0.copy()]
                simplex_f = [f0]
                
                # Add dim additional points with small perturbations
                for d in range(dim):
                    new_pt = x0.copy()
                    new_pt[d] += 0.5
                    new_pt = np.clip(new_pt, -5, 5)
                    if calls < budget and time_left() > 1:
                        new_f = eval_point(new_pt)
                        simplex.append(new_pt)
                        simplex_f.append(new_f)
                        if new_f < best_f:
                            best_f = new_f
                            best_x = new_pt.copy()
                    else:
                        # If we can't evaluate, use a tiny perturbation
                        new_pt = x0.copy()
                        new_pt[d] += 0.01
                        new_pt = np.clip(new_pt, -5, 5)
                        simplex.append(new_pt)
                        simplex_f.append(f0)  # approximate
                
                # Nelder-Mead iterations
                alpha = 1.0   # reflection
                gamma = 2.0   # expansion
                rho = 0.5     # contraction
                sigma = 0.5   # shrink
                
                for _ in range(max_iter):
                    if calls >= budget - 5 or time_left() < 1:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i].copy() for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    
                    if calls >= budget or time_left() < 0.5:
                        break
                    fr = eval_point(xr)
                    
                    if fr < simplex_f[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        if calls >= budget or time_left() < 0.5:
                            break
                        fe = eval_point(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
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
                            # Outside contraction
                            xc = centroid + rho * (xr - centroid)
                        else:
                            # Inside contraction
                            xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        
                        if calls >= budget or time_left() < 0.5:
                            break
                        fc = eval_point(xc)
                        
                        if fc < min(fr, simplex_f[-1]):
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if calls >= budget or time_left() < 0.5:
                                    break
                                simplex_f[i] = eval_point(simplex[i])
                            if calls >= budget or time_left() < 0.5:
                                break
                    
                    # Update best
                    min_idx = np.argmin(simplex_f)
                    if simplex_f[min_idx] < best_f:
                        best_f = simplex_f[min_idx]
                        best_x = simplex[min_idx].copy()
                
                return best_x, best_f
            
            nelder_mead(x0, f0)

        return best_x, best_f
# EVOLVE-BLOCK-END
