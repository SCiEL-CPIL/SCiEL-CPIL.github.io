"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """Construct a good packing of 200 equal circles in the unit square [-1/2, 1/2]^2."""
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    def compute_radius_and_subgrad(centers):
        """
        Computes the radius r, the active pair (i,j) or wall (i, side), 
        and the subgradient of the objective function f(x) = min(distances).
        f(x) is the min of all pairwise distances and wall distances.
        """
        n = len(centers)
        # Wall distances
        # d_i^L = x_i[0] + 0.5, d_i^R = 0.5 - x_i[0]
        # d_i^D = x_i[1] + 0.5, d_i^U = 0.5 - x_i[1]
        
        min_val = float('inf')
        active_type = None # 'pair' or 'wall'
        active_idx = None
        active_side = None # for wall
        subgrad = np.zeros_like(centers)
        
        # Check walls
        # Left
        vals = centers[:, 0] + 0.5
        idx = np.argmin(vals)
        if vals[idx] < min_val:
            min_val = vals[idx]
            active_type = 'wall'
            active_idx = idx
            active_side = 'L'
            subgrad = np.zeros_like(centers)
            subgrad[idx, 0] = 1.0
        # Right
        vals = 0.5 - centers[:, 0]
        idx = np.argmin(vals)
        if vals[idx] < min_val:
            min_val = vals[idx]
            active_type = 'wall'
            active_idx = idx
            active_side = 'R'
            subgrad = np.zeros_like(centers)
            subgrad[idx, 0] = -1.0
        # Down
        vals = centers[:, 1] + 0.5
        idx = np.argmin(vals)
        if vals[idx] < min_val:
            min_val = vals[idx]
            active_type = 'wall'
            active_idx = idx
            active_side = 'D'
            subgrad = np.zeros_like(centers)
            subgrad[idx, 1] = 1.0
        # Up
        vals = 0.5 - centers[:, 1]
        idx = np.argmin(vals)
        if vals[idx] < min_val:
            min_val = vals[idx]
            active_type = 'wall'
            active_idx = idx
            active_side = 'U'
            subgrad = np.zeros_like(centers)
            subgrad[idx, 1] = -1.0
            
        # Check pairs
        # Efficiently find min pair distance
        # For N=200, O(N^2) is 40k ops, which is fast in numpy if vectorized, 
        # but we need the specific pair.
        # We can use a simple loop or vectorized distance matrix.
        # Given N=200, computing full dist matrix is 200x200 = 40k floats. Very fast.
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        # Flatten to find min
        flat_idx = np.argmin(dists)
        i = flat_idx // n
        j = flat_idx % n
        
        d = dists[i, j]
        if d < min_val:
            min_val = d
            active_type = 'pair'
            active_idx = i
            active_side = j # reuse for j
            subgrad = np.zeros_like(centers)
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                subgrad[i] = diff / norm
                subgrad[j] = -diff / norm
            else:
                # Degenerate case: random subgradient
                subgrad[i] = np.random.normal(0, 1, 2)
                subgrad[j] = -subgrad[i]
        
        return min_val, active_type, active_idx, active_side, subgrad

    def bundle_method(centers, max_iters=500, deadline=None):
        """
        Bundle Method for maximizing f(x) = min_k f_k(x).
        We minimize -f(x).
        Model: m(x) = max_{i in B} (f_i + g_i^T (x - x_i))
        Wait, standard bundle for max-min:
        We want to maximize min_k f_k(x).
        Let g_k(x) = f_k(x).
        Subgradient of min_k g_k at x is the convex hull of subgradients of active g_k.
        Simplest approach: Cutting Plane with one subgradient per iteration.
        
        Algorithm:
        1. Start with x0.
        2. Loop:
           a. Solve QP: min_x 0.5 ||x - x_k||^2 + max_{i in B_k} (f_i + g_i^T (x - x_k))
              This is equivalent to: min_x 0.5 ||x - x_k||^2 + t
              s.t. t >= f_i + g_i^T (x - x_k) for all i in B_k
           b. Let x_{k+1} be the solution.
           c. Evaluate f(x_{k+1}), get subgradient g_{k+1}.
           d. If f(x_{k+1}) < model_value(x_{k+1}) - epsilon, add (f_{k+1}, g_{k+1}, x_{k+1}) to bundle.
           e. If ||x_{k+1} - x_k|| < tol, stop.
        """
        x = centers.copy().flatten()
        n = len(centers)
        dim = 2 * n
        
        # Bundle storage: list of (f_val, grad, x_point)
        bundle = []
        
        # Initial point
        f_curr, _, _, _, g_curr = compute_radius_and_subgrad(centers)
        bundle.append((f_curr, g_curr.flatten(), x.copy()))
        
        x_curr = x
        f_curr_val = f_curr
        
        # QP Solver using SLSQP or L-BFGS-B?
        # The QP is: min 0.5 * ||x - x_curr||^2 + t
        # subject to t >= f_i + g_i^T (x - x_curr) for all i in bundle
        # This can be solved by scipy.optimize.minimize with constraints.
        
        def qp_objective(x_flat, t):
            # Objective: 0.5 * ||x - x_curr||^2 + t
            return 0.5 * np.sum((x_flat - x_curr)**2) + t
            
        def qp_grad(x_flat, t):
            # Grad wrt x: x - x_curr
            # Grad wrt t: 1
            return np.concatenate([x_flat - x_curr, [1.0]])
            
        # Constraints: t - f_i - g_i^T (x - x_curr) >= 0
        # In SLSQP, constraints are >= 0 for 'ineq'.
        def make_constraints(bundle_list):
            cons = []
            for f_i, g_i, x_i in bundle_list:
                # Note: x_i is the point where subgradient was taken.
                # The linearization is at x_i.
                # f(x) >= f_i + g_i^T (x - x_i)
                # So constraint: t >= f_i + g_i^T (x - x_i)
                # t - f_i - g_i^T (x - x_i) >= 0
                def con(x_t, fi=f_i, gi=g_i, xi=x_i):
                    x_part = x_t[:dim]
                    t_val = x_t[dim]
                    return t_val - fi - np.dot(gi, x_part - xi)
                cons.append({'type': 'ineq', 'fun': con})
            return cons

        # To speed up QP, we can limit bundle size or use a simpler solver.
        # Given time limits, SLSQP on 400 vars + 1 with ~50 constraints is feasible but might be slow.
        # Alternative: Use the fact that the QP is convex.
        # Let's try a simpler "Proximal" step if QP is too slow.
        # Actually, for N=200, dim=400. SLSQP might struggle.
        # Let's use a simplified Bundle: only keep the last K subgradients (e.g., 10).
        
        max_bundle_size = 20
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
                
            # Solve QP
            # Variable: [x (400), t (1)]
            x0 = np.concatenate([x_curr, [f_curr_val]])
            
            # We use L-BFGS-B for the QP? No, it has constraints.
            # SLSQP is the standard choice.
            try:
                res = minimize(
                    lambda x_t: qp_objective(x_t[:dim], x_t[dim]),
                    x0,
                    jac=lambda x_t: qp_grad(x_t[:dim], x_t[dim]),
                    method='SLSQP',
                    constraints=make_constraints(bundle),
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                x_new_flat = res.x[:dim]
                t_new = res.x[dim]
            except Exception:
                # Fallback: simple subgradient step if QP fails
                x_new_flat = x_curr - 0.01 * g_curr.flatten()
                t_new = f_curr_val

            # Evaluate new point
            centers_new = x_new_flat.reshape(n, 2)
            # Ensure inside square (soft projection or reject)
            # Bundle method assumes feasible domain. If we go out, we project.
            centers_new[:, 0] = np.clip(centers_new[:, 0], -0.5, 0.5)
            centers_new[:, 1] = np.clip(centers_new[:, 1], -0.5, 0.5)
            
            f_new, _, _, _, g_new = compute_radius_and_subgrad(centers_new)
            g_new_flat = g_new.flatten()
            
            # Update bundle
            # Add new subgradient
            bundle.append((f_new, g_new_flat, x_new_flat.copy()))
            if len(bundle) > max_bundle_size:
                # Remove oldest or least relevant
                bundle.pop(0)
            
            # Check convergence
            step_size = np.linalg.norm(x_new_flat - x_curr)
            if step_size < 1e-5:
                # Converged
                if f_new > f_curr_val:
                    f_curr_val = f_new
                    x_curr = x_new_flat
                break
            
            x_curr = x_new_flat
            f_curr_val = f_new
            
        return x_curr.reshape(n, 2), f_curr_val

    def generate_initial_grid(n):
        """Generate a grid-like initial configuration."""
        # 200 circles. 14x14 = 196. 15x15 = 225.
        # Let's try 15x15 grid and remove some, or 14x14 plus 4.
        # Better: Hexagonal packing initial guess.
        rows = 15
        cols = 14
        centers = []
        dx = 1.0 / (cols + 0.5)
        dy = 1.0 / (rows + 0.5)
        
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5) * dx - 0.5
                y = (i + 0.5) * dy - 0.5
                # Offset odd rows for hexagonal
                if i % 2 == 1:
                    x += dx / 2
                centers.append([x, y])
        
        # We have 210 points. We need 200.
        # Remove the last 10.
        centers = centers[:n]
        return np.array(centers)

    # Main execution
    initial_centers = generate_initial_grid(N)
    r_init, _, _, _, _ = compute_radius_and_subgrad(initial_centers)
    
    best_centers = initial_centers.copy()
    best_r = r_init
    
    # Run Bundle Method
    # Bundle method is slow per iteration due to QP.
    # We need to be careful with time.
    # Let's run it for a few seconds.
    
    try:
        polished_centers, polished_r = bundle_method(initial_centers, max_iters=50, deadline=deadline)
        if polished_r > best_r:
            best_r = polished_r
            best_centers = polished_centers
    except Exception:
        pass
    
    # Fallback / Additional Polish: Simple local search if bundle didn't improve much
    # Or just return the best found.
    
    return best_centers
# EVOLVE-BLOCK-END
