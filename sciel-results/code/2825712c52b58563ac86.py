"""
Strategy: L-BFGS-B Optimization of a Differentiable Soft-Max Surrogate.

The current program uses manual gradient steps and heuristic facet repair.
This strategy reformulates the problem as a smooth optimization task:
1. Objective: Minimize a soft-max approximation of the maximum circumradius.
   The circumradius of each Delaunay facet is computed. The soft-max function 
   approximates the max function, making the objective differentiable.
2. Constraint/Regularization: Add a term that penalizes close pairwise distances 
   (soft-min of dot products) to ensure points remain spread out and don't collapse.
3. Optimizer: Use scipy.optimize.minimize with L-BFGS-B. This method uses 
   quasi-Newton updates and is efficient for medium-sized problems (N=130, dim=390).
4. Projection: Since L-BFGS-B works in unconstrained space (or with box constraints),
   we optimize on the sphere by using a parameterization or by projecting after each step.
   However, L-BFGS-B doesn't support non-linear constraints like ||x||=1 directly in a simple way 
   without reformulation. 
   
   Alternative: Optimize in Cartesian coordinates but use a penalty method or 
   simply project back to the sphere after each internal step? 
   Actually, scipy's L-BFGS-B expects a vector x. If we optimize x in R^390 and 
   project to sphere, the gradient of the objective w.r.t x must account for the projection.
   
   Better Approach for L-BFGS-B on Sphere:
   Use the fact that we can optimize the angles (spherical coordinates) or use 
   a retraction-based optimization. But scipy.optimize.minimize doesn't natively support retraction.
   
   Let's use a simpler trick: 
   We can optimize the points on the sphere by defining the objective on the unit sphere.
   To use L-BFGS-B, we can optimize the raw coordinates and use a penalty for ||p_i||^2 - 1,
   OR we can use the fact that for small steps, we can just project.
   
   Actually, a common robust approach with scipy for sphere constraints is to use 
   `scipy.optimize.minimize` with method 'L-BFGS-B' on the raw coordinates, 
   but the gradient must be projected onto the tangent space.
   
   Let's implement a custom objective function that takes raw coordinates, 
   projects them to the sphere, computes the soft-max covering radius + repulsion penalty, 
   and returns the value and gradient (projected).
   
   Wait, if we project inside the objective, the gradient returned by finite differences 
   (if we don't provide analytic grad) will be wrong because the projection is non-smooth at 0 
   and the mapping is non-linear.
   
   Better: Provide analytic gradient.
   Let P be the matrix of points (N, 3).
   Objective f(P) = SoftMax(Circumradii(P)) + lambda * SoftMin(PairwiseDots(P))
   
   We need df/dP.
   This is complex.
   
   Alternative Strategy: 
   Use the "Spherical L-BFGS" approach by parameterizing points using 2D coordinates 
   (e.g., stereographic projection) or simply using the fact that N=130 is small enough 
   that we can use a simpler optimizer like `scipy.optimize.minimize` with method 'BFGS' 
   or 'L-BFGS-B' on the raw coordinates, but we must ensure the gradient is correct.
   
   Let's stick to a simpler, more robust method that fits the "Differentiable Surrogate" directive:
   Use `scipy.optimize.minimize` with method 'L-BFGS-B' on the raw 3D coordinates.
   To handle the sphere constraint, we will use a **penalty method**:
   f_total = f_covering + mu * sum(||p_i||^2 - 1)^2
   
   If we provide the analytic gradient for f_covering (via chain rule) and the penalty gradient,
   L-BFGS-B will find a local minimum where points are near the sphere and covering radius is minimized.
   
   However, implementing the analytic gradient for the soft-max of circumradii is non-trivial 
   and error-prone.
   
   Let's look at the "Inspiration 2" which used projected gradient ascent. 
   The current program (score 1.149) uses facet repair.
   The "Inspiration 2" (score 1.082) used QP/Projected Gradient.
   
   Let's try a **Soft-Max Objective with L-BFGS-B** but using **finite difference gradients** 
   initially? No, N=130, dim=390. Finite difference is 390 evaluations per gradient step. 
   L-BFGS-B might take 100-500 iterations. 500 * 390 * cost_of_eval. 
   Cost of eval: ConvexHull is O(N^2 log N) or similar. For N=130, it's fast (~ms).
   500 * 390 * 5ms = ~1000 seconds. Too slow.
   
   We MUST use analytic gradients or a faster optimizer.
   
   Let's use **scipy.optimize.minimize** with method 'L-BFGS-B' and **analytic gradients**.
   
   Objective:
   1. Compute ConvexHull.
   2. For each facet, compute circumradius R_i.
   3. SoftMax: S = (1/tau) * log(sum(exp(tau * R_i))).
   4. Gradient of S w.r.t R_i is exp(tau * R_i) / sum(exp(tau * R_j)).
   5. Gradient of R_i w.r.t points p1, p2, p3.
      R_i = arccos(p1 . c_i). c_i is circumcenter.
      This is messy.
      
   Alternative Objective: 
   Instead of circumradius, use the **spherical Voronoi cell area** or **distance to nearest neighbor**?
   No, the task is specifically covering radius.
   
   Let's try a different differentiable surrogate:
   **Maximize the minimum distance to the nearest point**? No, that's packing.
   
   Let's use the **soft-min of the angular distances to the nearest point** for a dense grid?
   No, grid is too slow.
   
   Let's go back to the **Facet Circumradius** but simplify the gradient.
   Actually, there is a simpler differentiable proxy for covering radius:
   The covering radius is the maximum over all points y of dist(y, P).
   This is equal to the maximum circumradius of the Delaunay triangulation.
   
   If analytic gradient is too hard, can we use **SLSQP** or **COBYLA**? 
   COBYLA is derivative-free but very slow for 390 dims.
   
   What if we use **scipy.optimize.minimize** with method 'L-BFGS-B' on the **spherical coordinates**?
   Still need gradients.
   
   Let's look at the "Inspiration 1" (Lloyd's). 
   The current program is a hybrid.
   
   Let's try **Basinhopping** or **Differential Evolution**?
   DE is global but slow.
   
   Let's try **Simulated Annealing** with a soft-max objective?
   
   Actually, the most promising "Structural Change" that is different from manual gradient steps 
   is using a **Black-Box Optimizer** that doesn't require analytic gradients, but is efficient.
   `scipy.optimize.differential_evolution` is too slow.
   `scipy.optimize.shgo` (Simulated Annealing + Global Optimization) might work but is risky.
   
   Let's reconsider the **Projected Gradient** but with a **better step size strategy** and **soft-max objective**.
   The current program uses hard facet repair.
   Inspiration 2 used QP/Projected Gradient and got 1.082.
   Current is 1.149.
   
   Why is Current better? It uses facet repair which is more targeted.
   
   Let's combine: **L-BFGS-B on a Soft-Max Surrogate** with **Analytic Gradients**.
   I will implement the analytic gradient for the soft-max of circumradii.
   
   Math:
   R_i = arccos( cos_theta_i ) where cos_theta_i = p1_i . c_i (normalized).
   c_i is the normalized circumcenter of facet i.
   c_i = (p1 - p2) x (p1 - p3) / ||...||
   
   This is very complex to differentiate manually without mistakes.
   
   Alternative: Use **JAX** or **PyTorch**? Not allowed (only NumPy, SciPy, StdLib).
   
   Let's use **NumPy's autograd**? Not available.
   
   Okay, let's use a **Finite Difference Gradient** but **reduce the dimensionality** or **frequency**.
   Or, use **scipy.optimize.minimize** with method 'L-BFGS-B' and `approx_grad=True`?
   It uses forward differences. 390 evals.
   If we limit iterations to 50, and each eval is 2ms, 50 * 390 * 2ms = 39s.
   This is feasible!
   
   Strategy:
   1. Initialize with Fibonacci.
   2. Define objective: SoftMax(Circumradii) + Penalty(Sphere Constraint).
   3. Use L-BFGS-B with `approx_grad=True`.
   4. Set `maxiter` to a value that fits in time (e.g., 100-200).
   5. Project points to sphere inside the objective function to ensure validity during optimization, 
      but this breaks the gradient consistency.
      
   Correction: If we project inside the objective, the finite difference gradient will be 
   calculated on the projected points, which is effectively optimizing on the manifold 
   using a "pullback" approach. It is not strictly correct but often works in practice 
   for sphere constraints if the penalty is strong or projection is tight.
   
   However, a better way: 
   Optimize the raw coordinates. 
   Objective(x):
      P = x.reshape(N, 3)
      P_norm = P / ||P||
      Compute SoftMax(Circumradii(P_norm))
      Add penalty: sum(||P||^2 - 1)^2
      
   This way, the finite difference is taken on the raw x, which is correct for the 
   composite function (projection + objective + penalty).
   
   Let's implement this.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p1 - p2
        v2 = p1 - p3
        
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        return np.max(theta) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def _objective_and_penalty(x_flat):
    """
    Objective function for L-BFGS-B.
    x_flat: 1D array of shape (3*N,).
    Returns: scalar value.
    """
    points = x_flat.reshape(N, 3)
    
    # Penalty for being off the sphere
    norms = np.linalg.norm(points, axis=1)
    sphere_penalty = np.sum((norms - 1.0) ** 2) * 100.0
    
    # Project to sphere for geometric calculations
    valid_norms = norms > 1e-10
    points_safe = points.copy()
    points_safe[valid_norms] /= norms[valid_norms, np.newaxis]
    
    # Calculate Soft-Max of Circumradii
    try:
        hull = ConvexHull(points_safe)
        facets = hull.simplices
        p1 = points_safe[facets[:, 0]]
        p2 = points_safe[facets[:, 1]]
        p3 = points_safe[facets[:, 2]]
        
        v1 = p1 - p2
        v2 = p1 - p3
        
        c_dir = np.cross(v1, v2)
        norms_c = np.linalg.norm(c_dir, axis=1)
        
        valid = norms_c > 1e-10
        if not np.any(valid):
            return 180.0 + sphere_penalty
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms_c[valid, np.newaxis]
        
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        
        # Soft-Max approximation
        tau = 50.0
        # Shift for numerical stability
        max_theta = np.max(theta)
        exp_terms = np.exp(tau * (theta - max_theta))
        soft_max = (1.0 / tau) * (max_theta + np.log(np.sum(exp_terms)))
        
        # Convert to degrees for consistency with score
        soft_max_deg = soft_max * 180.0 / np.pi
        
        return soft_max_deg + sphere_penalty
        
    except Exception:
        return 180.0 + sphere_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Flatten for optimizer
    x0 = points.flatten()
    
    # Use L-BFGS-B with finite difference gradients
    # maxiter is critical. Each iteration takes ~N_evals * cost.
    # With approx_grad, it's ~390 evals per iteration.
    # Cost of one eval: ConvexHull + SoftMax. ~1-2 ms.
    # 390 * 2ms = 0.78s per iteration.
    # In 140s, we can do ~180 iterations.
    
    try:
        result = minimize(
            _objective_and_penalty, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'disp': False, 'gtol': 1e-5}
        )
        
        if time.time() < deadline:
            x_opt = result.x
            points_opt = x_opt.reshape(N, 3)
            
            # Final projection to ensure unit sphere
            norms = np.linalg.norm(points_opt, axis=1, keepdims=True)
            valid = norms > 1e-10
            points_opt[valid] /= norms[valid]
            
            current_radius = _covering_radius(points_opt)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points_opt.copy()
                
    except Exception as e:
        pass
        
    # If time remains, do a quick local refinement with the old facet repair logic
    # to polish the L-BFGS-B result.
    if time.time() < deadline - 10.0:
        points = best_points.copy()
        lr_repair = 0.01
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            # Simple worst facet repair
            try:
                hull = ConvexHull(points)
                facets = hull.simplices
                p1 = points[facets[:, 0]]
                p2 = points[facets[:, 1]]
                p3 = points[facets[:, 2]]
                v1 = p1 - p2
                v2 = p1 - p3
                c_dir = np.cross(v1, v2)
                norms_c = np.linalg.norm(c_dir, axis=1)
                valid = norms_c > 1e-10
                if not np.any(valid): break
                c = np.zeros_like(c_dir)
                c[valid] = c_dir[valid] / norms_c[valid, np.newaxis]
                centroids = (p1 + p2 + p3) / 3.0
                c_norms = np.linalg.norm(centroids, axis=1)
                valid_centroids = c_norms > 1e-10
                centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
                dots = np.sum(c * centroids, axis=1)
                c[dots < 0] *= -1
                cos_theta = np.sum(c * p1, axis=1)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                theta[~valid] = 0.0
                max_idx = np.argmax(theta)
                if not valid[max_idx]: break
                f_idx = facets[max_idx]
                cc = c[max_idx]
                for v in f_idx:
                    d = cc - points[v]
                    n = np.linalg.norm(d)
                    if n > 1e-10:
                        points[v] += lr_repair * (d / n)
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                points = points / norms
            except Exception:
                break
            
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
