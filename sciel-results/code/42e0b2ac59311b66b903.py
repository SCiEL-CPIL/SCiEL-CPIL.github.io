# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Manifold-restricted optimizer using SVD-based subspace detection for ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget

        state = {'calls': 0, 'best_f': np.inf, 'best_x': np.zeros(n)}

        def wrapped_func(x):
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if state['calls'] >= budget:
                raise Exception("Budget exceeded")
            
            x_clipped = np.clip(x, -5, 5)
            fx = float(func(x_clipped))
            state['calls'] += 1
            
            if fx < state['best_f']:
                state['best_f'] = fx
                state['best_x'] = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)

        def finite_diff_jacobian(x, h=1e-4):
            """Compute approximate Jacobian using forward differences."""
            J = np.zeros((1, n))
            f0 = wrapped_func(x)
            for i in range(n):
                x_pert = x.copy()
                x_pert[i] += h
                x_pert = np.clip(x_pert, -5, 5)
                f_pert = wrapped_func(x_pert)
                J[0, i] = (f_pert - f0) / h
            return J

        def estimate_manifold(x_center, h=1e-2):
            """Estimate low-dimensional manifold via SVD of finite-difference Jacobian."""
            # Sample points around center to build local Jacobian
            samples = [x_center]
            for i in range(n):
                for sign in [1, -1]:
                    pt = x_center.copy()
                    pt[i] += sign * h
                    pt = np.clip(pt, -5, 5)
                    samples.append(pt)
            
            # Build local Jacobian by evaluating function at perturbed points
            # Use central differences for better accuracy
            J_rows = []
            for i in range(n):
                pt_plus = x_center.copy()
                pt_plus[i] += h
                pt_plus = np.clip(pt_plus, -5, 5)
                
                pt_minus = x_center.copy()
                pt_minus[i] -= h
                pt_minus = np.clip(pt_minus, -5, 5)
                
                f_plus = wrapped_func(pt_plus)
                f_minus = wrapped_func(pt_minus)
                J_rows.append((f_plus - f_minus) / (2 * h))
            
            J = np.array(J_rows).reshape(1, n)
            
            # SVD to find dominant directions
            # For a ridge function, we expect one large singular value and rest small
            try:
                U, S, Vt = np.linalg.svd(J, full_matrices=False)
            except np.linalg.LinAlgError:
                # Fall back to full space if SVD fails
                return np.eye(n), n
            
            # Determine rank based on singular value threshold
            if len(S) == 0:
                return np.eye(n), n
            
            threshold = max(S) * 1e-3
            rank = np.sum(S > threshold)
            rank = max(1, min(rank, n))
            
            # The manifold is spanned by the top-rank right singular vectors
            # For a ridge, the sharp direction is the first one, flat directions are the rest
            # We want to search primarily along the sharp direction
            V = Vt.T
            manifold_basis = V[:, :rank]
            
            return manifold_basis, rank

        def run_manifold_optimization(x_start, max_iter=200):
            """Optimize using manifold-restricted search."""
            nonlocal state, deadline
            
            if state['calls'] >= budget or time.time() > deadline:
                return
            
            # Estimate initial manifold
            basis, rank = estimate_manifold(x_start)
            
            # Project current point onto manifold for optimization
            # Parameterize: x = x_start + basis @ coeffs
            coeffs = np.zeros(rank)
            
            best_coeffs = coeffs.copy()
            best_f_local = state['best_f']
            
            # Use Nelder-Mead in the reduced space
            # Initialize simplex in coefficient space
            simplex = [coeffs.copy()]
            for i in range(rank):
                pt = coeffs.copy()
                pt[i] += 0.5
                simplex.append(pt)
            
            # Evaluate initial simplex
            f_values = []
            for c in simplex:
                x_pt = x_start + basis @ c
                x_pt = np.clip(x_pt, -5, 5)
                f_pt = wrapped_func(x_pt)
                f_values.append(f_pt)
                
                if state['calls'] >= budget:
                    return
            
            if len(f_values) < rank + 1:
                return
            
            # Nelder-Mead in reduced space
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            iter_count = 0
            stall_count = 0
            prev_best = min(f_values)
            
            while state['calls'] < budget and iter_count < max_iter:
                if time.time() > deadline:
                    break
                
                # Sort by fitness
                sorted_idx = np.argsort(f_values)
                simplex = [simplex[i] for i in sorted_idx]
                f_values = [f_values[i] for i in sorted_idx]
                
                # Check stall
                if f_values[0] < prev_best - 1e-10:
                    prev_best = f_values[0]
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 15:
                        break
                
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_best = simplex[0]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                c_r = centroid + alpha * (centroid - x_worst)
                x_r = x_start + basis @ c_r
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                
                if state['calls'] >= budget:
                    break
                
                if f_r < f_values[0]:
                    # Expansion
                    c_e = centroid + gamma * (centroid - x_worst)
                    x_e = x_start + basis @ c_e
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    
                    if state['calls'] >= budget:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = c_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = c_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = c_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        c_c = centroid + rho * (c_r - centroid)
                    else:
                        c_c = centroid - rho * (c_r - centroid)
                    
                    x_c = x_start + basis @ c_c
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)
                    
                    if state['calls'] >= budget:
                        break
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = c_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, rank + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            x_shrink = x_start + basis @ simplex[i]
                            x_shrink = np.clip(x_shrink, -5, 5)
                            f_shrink = wrapped_func(x_shrink)
                            f_values[i] = f_shrink
                            
                            if state['calls'] >= budget:
                                break
                        
                        if state['calls'] >= budget:
                            break
                        
                        # Check convergence
                        simplex_size = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, rank+1)])
                        if simplex_size < 1e-6:
                            break
                
                iter_count += 1
            
            # After manifold optimization, do a full-space Nelder-Mead polish
            if state['calls'] < budget and time.time() < deadline:
                run_full_nelder_mead(state['best_x'], max_iter=50)

        def run_full_nelder_mead(start_x, max_iter=50):
            """Standard Nelder-Mead in full space for final polish."""
            nonlocal state, deadline
            
            if state['calls'] >= budget or time.time() > deadline:
                return
            
            init_sigma = 0.5
            simplex = [start_x.copy()]
            for i in range(n):
                pt = start_x.copy()
                step = init_sigma
                if start_x[i] + step > 5:
                    pt[i] = start_x[i] - step
                elif start_x[i] - step < -5:
                    pt[i] = start_x[i] + step
                else:
                    pt[i] = start_x[i] + step
                simplex.append(pt)
            
            f_values = []
            for pt in simplex:
                f_pt = wrapped_func(pt)
                f_values.append(f_pt)
                
                if state['calls'] >= budget:
                    return
            
            if len(f_values) < n + 1:
                return
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            iter_count = 0
            stall_count = 0
            prev_best = min(f_values)
            
            while state['calls'] < budget and iter_count < max_iter:
                if time.time() > deadline:
                    break
                
                sorted_idx = np.argsort(f_values)
                simplex = [simplex[i] for i in sorted_idx]
                f_values = [f_values[i] for i in sorted_idx]
                
                if f_values[0] < prev_best - 1e-10:
                    prev_best = f_values[0]
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 10:
                        break
                
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_best = simplex[0]
                centroid = np.mean(simplex[:-1], axis=0)
                
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                
                if state['calls'] >= budget:
                    break
                
                if f_r < f_values[0]:
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    
                    if state['calls'] >= budget:
                        break
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)
                    
                    if state['calls'] >= budget:
                        break
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = wrapped_func(simplex[i])
                            
                            if state['calls'] >= budget:
                                break
                        
                        if state['calls'] >= budget:
                            break
                        
                        simplex_size = np.max([np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n+1)])
                        if simplex_size < 1e-6:
                            break
                
                iter_count += 1

        # Phase 1: Exploration - evaluate random points
        num_explore = 15
        explore_points = []
        for _ in range(num_explore):
            pt = rng.uniform(-5, 5, n)
            fx = wrapped_func(pt)
            explore_points.append((fx, pt))
        
        if state['calls'] >= budget:
            return state['best_x'], state['best_f']
        
        explore_points.sort(key=lambda x: x[0])
        
        # Phase 2: Manifold-restricted optimization from top points
        top_k = 3
        for i in range(top_k):
            if state['calls'] >= budget or time.time() > deadline:
                break
            
            start_x = explore_points[i][1]
            run_manifold_optimization(start_x, max_iter=150)
        
        # Phase 3: Final polish with full-space Nelder-Mead if budget remains
        if state['calls'] < budget and time.time() < deadline:
            run_full_nelder_mead(state['best_x'], max_iter=30)

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
