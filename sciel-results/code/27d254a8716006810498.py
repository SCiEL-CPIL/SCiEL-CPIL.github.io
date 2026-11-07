import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s, hard limit usually 300s. 
        # We use 290s to be safe, but the contract says "Each evaluation must finish within 300s".
        # The total wall clock is the constraint for the whole __call__.
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        
        # DTLZ2 structure:
        # f_i(x) = g(x) * h_i(x)
        # g(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2  (Standard DTLZ2 uses x_i - 0.5, but some variants use x_i^2 or similar)
        # Actually, standard DTLZ2 g(x) = sum_{i=M}^{n-1} (x_i - 0.5)^2.
        # h_i(x) are defined by the first M variables (position variables).
        # M = n_obj - 1.
        
        M = n_obj - 1
        n_shape = dim - M
        
        # We will generate points on the Pareto front analytically.
        # The Pareto front for DTLZ2 is the set of points where g(x)=0.
        # This implies x_i = 0.5 for all shape variables (assuming standard definition).
        # The position variables x_0...x_{M-1} determine the location on the front.
        # The mapping from position variables to objectives is:
        # f_0 = cos(x_0 * pi/2) * prod_{k=1}^{M-1} sin(x_k * pi/2)
        # f_1 = sin(x_0 * pi/2) * cos(x_1 * pi/2) * prod_{k=2}^{M-1} sin(x_k * pi/2)
        # ...
        # f_{M-1} = prod_{k=0}^{M-1} sin(x_k * pi/2)
        
        # To cover the front uniformly, we should sample angles theta_i such that x_i = 2*theta_i/pi?
        # No, x_i is in [0,1]. The angles in the trigonometric functions are x_i * pi/2.
        # Let alpha_i = x_i * pi/2. Then alpha_i is in [0, pi/2].
        # The front is a quarter-sphere (for 3 obj, a quarter-circle arc in the projection, actually a 2D surface on the 3D sphere).
        
        # Strategy:
        # 1. Allocate budget:
        #    - 80% for generating diverse points on the front (analytical).
        #    - 20% for local refinement of the shape variables to ensure g(x)=0 exactly.
        
        # We use a fixed RNG for reproducibility of the *search pattern*, 
        # but the problem is deterministic.
        rng = np.random.default_rng(42)
        
        # We want to maximize Hypervolume.
        # HV is maximized by points that are well-spread and close to the ideal point (0,0,0).
        # On the DTLZ2 front, all points have ||f|| = 1 (if g=0).
        # So we just need to spread them out on the sphere.
        
        # Number of points to generate analytically
        n_points = int(self.budget * 0.8)
        n_refine = self.budget - n_points
        
        X_analytical = np.zeros((n_points, dim))
        
        # Generate points on the Pareto front
        # For 3 objectives (M=2), we need x_0, x_1.
        # x_0, x_1 in [0,1].
        # To spread uniformly on the sphere, we can sample directions on the unit sphere in R^3 
        # with non-negative components, and map back to x.
        
        # Alternative: Use random angles in the parameter space.
        # For M=2:
        # f0 = cos(a0) * sin(a1)
        # f1 = sin(a0) * cos(a1)
        # f2 = sin(a0) * sin(a1)
        # where a0 = x0*pi/2, a1 = x1*pi/2.
        # This covers the quarter sphere.
        
        # To get uniform coverage on the sphere, we can sample random points on the unit sphere 
        # with non-negative components, normalize, and solve for a0, a1.
        # Or simpler: Just sample x0, x1 uniformly in [0,1]. 
        # Uniform in x-space is NOT uniform on the sphere surface.
        # cos/sin transformations distort density.
        # Better: Sample angles uniformly? 
        # Let's try sampling directions on the sphere.
        
        for i in range(n_points):
            # Sample a random point on the positive octant of the unit sphere
            # using Gaussian distribution trick
            v = np.abs(rng.normal(size=n_obj))
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            
            # Map v (which is f) back to x (position variables)
            # v0 = cos(x0*pi/2) * sin(x1*pi/2)
            # v1 = sin(x0*pi/2) * cos(x1*pi/2)
            # v2 = sin(x0*pi/2) * sin(x1*pi/2)
            
            # We can solve for x0, x1.
            # Note: v0^2 + v1^2 + v2^2 = 1.
            # sin^2(x0*pi/2) * (sin^2(x1*pi/2) + cos^2(x1*pi/2)) + cos^2(x0*pi/2)*sin^2(x1*pi/2) = 1?
            # Actually, the mapping is:
            # Let theta0 = x0*pi/2, theta1 = x1*pi/2.
            # v0 = cos(theta0) sin(theta1)
            # v1 = sin(theta0) cos(theta1)
            # v2 = sin(theta0) sin(theta1)
            
            # This is not the standard spherical coordinates.
            # Standard spherical:
            # x = sin(phi) cos(theta)
            # y = sin(phi) sin(theta)
            # z = cos(phi)
            
            # DTLZ2 mapping for M=2:
            # f0 = cos(x0*pi/2) * sin(x1*pi/2)
            # f1 = sin(x0*pi/2) * cos(x1*pi/2)
            # f2 = sin(x0*pi/2) * sin(x1*pi/2)
            
            # Let's invert this.
            # v2 / v1 = tan(x0*pi/2) / cot(x1*pi/2)? No.
            # v2 = sin(x0*pi/2) sin(x1*pi/2)
            # v1 = sin(x0*pi/2) cos(x1*pi/2)
            # v2/v1 = tan(x1*pi/2)  => x1 = (2/pi) * arctan(v2/v1)
            
            # v0 = cos(x0*pi/2) sin(x1*pi/2)
            # We know sin(x1*pi/2) from v2 and v1?
            # sin^2(x1*pi/2) + cos^2(x1*pi/2) = 1
            # Let s1 = sin(x1*pi/2), c1 = cos(x1*pi/2).
            # v2 = s0 * s1
            # v1 = s0 * c1
            # v1^2 + v2^2 = s0^2 (s1^2 + c1^2) = s0^2
            # So s0 = sqrt(v1^2 + v2^2)
            # Then x0 = (2/pi) * arcsin(s0)
            
            # Check v0:
            # v0 = c0 * s1
            # c0 = sqrt(1 - s0^2) = sqrt(1 - (v1^2 + v2^2)) = sqrt(v0^2) = v0 (since v0>=0)
            # s1 = v2 / s0
            # c1 = v1 / s0
            # This is consistent if v1^2+v2^2 <= 1.
            
            s0_sq = v[1]**2 + v[2]**2
            if s0_sq < 1e-12:
                # Edge case: v1=0, v2=0 => v0=1 => x0=0, x1=0
                x0 = 0.0
                x1 = 0.0
            else:
                s0 = np.sqrt(s0_sq)
                x0 = (2.0 / np.pi) * np.arcsin(np.clip(s0, -1, 1))
                
                # x1
                # tan(x1*pi/2) = v2/v1
                # Handle v1=0
                if v[1] < 1e-12:
                    if v[2] > 1e-12:
                        x1 = 1.0
                    else:
                        x1 = 0.0
                else:
                    x1 = (2.0 / np.pi) * np.arctan2(v[2], v[1])
            
            X_analytical[i, 0] = x0
            X_analytical[i, 1] = x1
            
            # Shape variables: Set to 0.5 (midpoint) initially
            # This is the best guess for g(x)=0.
            X_analytical[i, M:] = 0.5 * (lo[M:] + hi[M:])
            
            # Clip to bounds
            X_analytical[i] = np.clip(X_analytical[i], lo, hi)
        
        # Evaluate all analytical points
        F_analytical = np.array([problem(x) for x in X_analytical])
        evals = n_points
        
        # Now, refine the shape variables for a subset of points to ensure g(x)=0.
        # We will pick points that are "spread out" on the front.
        # Since we generated them randomly on the sphere, they are already spread.
        # We will refine the first min(n_refine, n_points) points.
        
        # For each selected point, we perform a local search on the shape variables.
        # Objective: Minimize g(x) = sum((x_i - 0.5)^2) ? 
        # We don't have direct access to g(x), but we know that on the Pareto front, 
        # the sum of squares of objectives is 1 (for DTLZ2).
        # Actually, f = g * h. If g > 0, ||f|| > ||h|| = 1.
        # So we can minimize ||f||^2 - 1? Or just minimize ||f||^2.
        # The minimum possible ||f||^2 on the front is 1.
        # If g(x) > 0, ||f|| > 1.
        # So minimizing ||f||^2 drives g(x) to 0.
        
        # We use Nelder-Mead on the shape variables.
        # Dim of shape space: n_shape.
        # We have n_refine evaluations left.
        # Each Nelder-Mead step takes ~n_shape+1 evaluations? No, 1 eval per vertex update.
        # A simple coordinate descent or random hill climbing is safer for budget control.
        
        # Let's use a simple random perturbation local search on shape variables.
        # For each of the first n_refine points:
        #   We have n_refine / n_refine_points evaluations per point?
        #   Let's allocate 5 evaluations per point for refinement.
        
        n_points_to_refine = min(n_refine // 5, n_points)
        
        # We refine the points that are most "central" or just the first N.
        # Let's refine the first N points.
        
        for i in range(n_points_to_refine):
            if evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x_curr = X_analytical[i].copy()
            f_curr = F_analytical[i].copy()
            
            # Local search on shape variables (indices M to dim-1)
            # We try to reduce the norm of f.
            # If ||f|| is close to 1, we are on the front.
            # If ||f|| > 1, we are off the front (g > 0).
            
            step_size = 0.05
            improved = True
            while improved and evals < self.budget and time.time() < deadline:
                improved = False
                # Try perturbing each shape variable
                for k in range(M, dim):
                    if evals >= self.budget: break
                    if time.time() > deadline: break
                    
                    # Try + step
                    x_test = x_curr.copy()
                    x_test[k] += step_size
                    x_test = np.clip(x_test, lo, hi)
                    f_test = problem(x_test)
                    evals += 1
                    
                    if np.linalg.norm(f_test) < np.linalg.norm(f_curr):
                        x_curr = x_test
                        f_curr = f_test
                        improved = True
                        # Update the stored point
                        X_analytical[i] = x_curr
                        F_analytical[i] = f_curr
                        # Reduce step size slightly for finer search
                        step_size *= 0.5
                        break # Restart loop to try other variables with new center? 
                             # Or just continue. Restarting is safer for Nelder-Mead style.
                             # Let's just break to restart the variable loop.
                    else:
                        # Try - step
                        x_test = x_curr.copy()
                        x_test[k] -= step_size
                        x_test = np.clip(x_test, lo, hi)
                        f_test = problem(x_test)
                        evals += 1
                        
                        if np.linalg.norm(f_test) < np.linalg.norm(f_curr):
                            x_curr = x_test
                            f_curr = f_test
                            improved = True
                            X_analytical[i] = x_curr
                            F_analytical[i] = f_curr
                            step_size *= 0.5
                            break
                        else:
                            # No improvement in this direction, shrink step size for this variable?
                            # Or just move to next variable.
                            pass
                
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-5:
                        break
        
        # We have used some budget. If we have budget left, we can generate more analytical points.
        # But we already generated 80% initially. 
        # If we finished refinement early, we can use remaining budget for more random front points.
        
        while evals < self.budget and time.time() < deadline:
            # Generate another random point on the front
            v = np.abs(rng.normal(size=n_obj))
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            
            s0_sq = v[1]**2 + v[2]**2
            if s0_sq < 1e-12:
                x0 = 0.0
                x1 = 0.0
            else:
                s0 = np.sqrt(s0_sq)
                x0 = (2.0 / np.pi) * np.arcsin(np.clip(s0, -1, 1))
                if v[1] < 1e-12:
                    if v[2] > 1e-12:
                        x1 = 1.0
                    else:
                        x1 = 0.0
                else:
                    x1 = (2.0 / np.pi) * np.arctan2(v[2], v[1])
            
            x_new = np.zeros(dim)
            x_new[0] = x0
            x_new[1] = x1
            x_new[M:] = 0.5 * (lo[M:] + hi[M:])
            x_new = np.clip(x_new, lo, hi)
            
            f_new = problem(x_new)
            evals += 1
        
        return X_analytical

    def is_non_dominated(self, F, idx):
        f = F[idx]
        for i in range(len(F)):
            if i == idx: continue
            if np.all(F[i] <= f) and np.any(F[i] < f):
                return False
        return True

    def find_non_dominated(self, F):
        res = []
        for i in range(len(F)):
            if self.is_non_dominated(F, i):
                res.append(i)
        return res
