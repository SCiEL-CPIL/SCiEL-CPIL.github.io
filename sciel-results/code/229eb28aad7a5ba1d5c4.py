"""
Program for covering the sphere with 130 equal caps.

Strategy:
1.  Start with multiple diverse initializations (Fibonacci, random).
2.  Use a "Variational Density" approach: Model the points as a sum of Gaussians on the sphere.
    Minimize the KL divergence between the empirical density (sum of Gaussians) and the uniform
    density. This is equivalent to maximizing the log-likelihood of the uniform distribution
    under the model, or minimizing the negative log-likelihood.
    Specifically, we want the density $P(y) = \sum_{i=1}^N w_i K(y, p_i)$ to be close to uniform.
    Since the total mass is fixed, minimizing the "energy" or maximizing the "entropy" of the
    empirical distribution relative to the uniform one drives points apart.
    
    A more direct gradient-based objective for sphere covering is the "soft covering radius".
    The current program uses a two-stage soft-min/soft-max. 
    The inspiration suggests a "Variational Density" / KL divergence approach.
    
    Let's combine:
    - Use the robust hull-based local refinement from the current program (it's very effective for the final polish).
    - Replace the initial coarse optimization (Lloyd + simple soft-min) with a gradient-based
      optimization of a "Potential Energy" or "Density" objective that is smoother and potentially
      finds better basins.
    
    Objective: Minimize $\sum_{y \in Grid} \max_i d(y, p_i)$ is non-smooth.
    Smooth approximation: $\sum_{y \in Grid} \log(\sum_i \exp(-d(y, p_i)/\sigma))$.
    This is the "LogSumExp" of the negative distances. Minimizing this pushes points to cover
    the space uniformly. This is effectively the same as the soft-min approach but framed as
    density estimation (maximizing the probability of grid points under a mixture model).
    
    Actually, the current program's `soft_covering_radius_and_grad` is already doing this:
    It computes a soft-min of distances (which is $-\log \sum \exp(-d/\sigma)$ up to constants)
    and then a soft-max over the grid (to get the "radius").
    
    The key structural change requested is "Variational Density Estimation".
    Let's interpret this as: Instead of optimizing the covering radius directly with a soft-min,
    we optimize the positions to minimize the "Voronoi Cell Area Variance" or similar, or simply
    use a different potential.
    
    However, the most robust improvement often comes from:
    1. Better initialization.
    2. More aggressive local search using the exact hull geometry.
    3. A better global search strategy.
    
    Let's look at the inspiration code. It has `_exact_local_optimize` which moves points based on
    hull facets. The current program also has `exact_local_optimize`.
    
    The main difference in the inspiration is the `_lloyd_step` which is simpler (centroid of grid points
    in Voronoi cell) vs the current program's weighted Lloyd.
    
    Let's try a hybrid:
    1.  Use the current program's structure.
    2.  Replace the `optimize_with_annealing` (which uses L-BFGS-B on a grid) with a more direct
        gradient ascent on the "Density" objective, or keep it but improve the objective.
    3.  Crucially, the "Variational Density" idea implies treating the points as particles repelling
        each other. The force on point $p_i$ is the gradient of the log-density at $p_i$.
        $L(p) = \sum_{j \neq i} \log K(p_i, p_j)$. This is like a Coulomb gas on the sphere.
        Minimizing the covering radius is related to maximizing the minimum distance, which is
        related to the Riesz s-energy for large s.
        
    Let's implement a "Repulsive Force" based optimizer as the global phase.
    $U = \sum_{i < j} \frac{1}{\sin^{s}(d_{ij})}$ or similar.
    But this is computationally expensive ($O(N^2)$). $N=130$, so $130^2 \approx 17000$ pairs. Very fast.
    
    Proposed Plan:
    1.  Initialize with Fibonacci and Random.
    2.  Phase 1: "Coulomb/Repulsive" optimization. Minimize $E = \sum_{i<j} f(d_{ij})$ where $f$ is a
        repulsive potential (e.g., $1/\sin^2(d_{ij})$ or $-\log(\sin(d_{ij}))$). This spreads points out.
        Use L-BFGS-B.
    3.  Phase 2: "Soft Covering" optimization (from current program) to refine the coverage.
    4.  Phase 3: Exact Hull-based local search (from current program) for final precision.
    
    This combines the "Density/Repulsion" global idea with the "Hull" local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s hard limit

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Grids for evaluation and gradient
    grid_main = generate_grid(150, 300)
    grid_coarse = generate_grid(60, 120)

    def covering_radius(pts, grid=None):
        if grid is None:
            grid = grid_main
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
        """
        Vectorized soft covering radius and its gradient.
        """
        dots = grid_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # (M, N)
        
        neg_scaled = -angles / sigma  # (M, N)
        max_neg = np.max(neg_scaled, axis=1, keepdims=True)
        exp_neg = np.exp(neg_scaled - max_neg)
        weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
        
        soft_min_vals = np.sum(weights * angles, axis=1)
        
        scaled_max = soft_min_vals / sigma
        max_scaled = np.max(scaled_max)
        exp_max = np.exp(scaled_max - max_scaled)
        max_weights = exp_max / np.sum(exp_max)
        soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
        
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))
        coeff = max_weights[:, np.newaxis] * weights
        alpha = coeff / denom
        term1 = -(alpha.T @ grid_points)
        beta = alpha * dots
        beta_sum = np.sum(beta, axis=0)
        term2 = beta_sum[:, np.newaxis] * points
        grad = term1 + term2
        
        return soft_max_val, grad

    def repulsive_energy_and_grad(points, power=2.0):
        """
        Compute repulsive energy (Riesz s-energy like) and gradient.
        $E = \sum_{i < j} \frac{1}{\sin^{power}(d_{ij})}$
        This encourages points to be far apart.
        """
        # Compute all pairwise dot products
        dots = points @ points.T
        # Zero out diagonal
        dots[np.diag_indices(N)] = 0.0
        dots = np.clip(dots, -1.0, 1.0)
        
        # sin^2(d) = 1 - cos^2(d)
        sin2_d = 1.0 - dots * dots
        sin2_d = np.maximum(sin2_d, 1e-15) # Avoid division by zero
        
        # We want to minimize sum 1/sin^p(d)
        # Let's use p=2 for stability, or p=1.
        # Actually, minimizing covering radius is better served by a potential that goes to infinity
        # as d -> 0, but is flat for large d.
        # $1/\sin^2(d)$ works well.
        
        # Energy: sum over i<j of (sin2_d_ij)^(-power/2)
        # Let's stick to power=2 for simplicity: 1/sin2_d
        energy = np.sum(1.0 / sin2_d)
        
        # Gradient:
        # d/dp_i [ (1 - (p_i . p_j)^2)^(-1) ]
        # Let u = 1 - (p_i . p_j)^2
        # d u / d p_i = -2 (p_i . p_j) p_j
        # d (u^-1) / d p_i = -u^-2 * d u / d p_i = -u^-2 * (-2 (p_i . p_j) p_j) = 2 (p_i . p_j) p_j / u^2
        # Wait, we are minimizing. So grad = + 2 (p_i . p_j) p_j / u^2 ?
        # Let's check signs.
        # If p_i and p_j are close, dot ~ 1, u ~ 0. Energy is high.
        # Moving p_i away from p_j should decrease energy.
        # Gradient of energy w.r.t p_i should point towards p_j (increasing energy) or away?
        # Minimization moves opposite to gradient.
        # If grad points to p_j, we move away from p_j. Correct.
        
        # Term for i from j: 2 * dots[i,j] * p_j / (sin2_d[i,j])^2
        # Note: sin2_d is squared in denominator for power=2?
        # Energy term: 1 / sin(d)^2 = 1 / sin2_d.
        # Deriv of (1/u) is -1/u^2 * du.
        # du = d(1 - dot^2) = -2 dot p_j.
        # d(1/u) = -1/u^2 * (-2 dot p_j) = 2 dot p_j / u^2.
        
        # So grad_i = sum_{j != i} 2 * dots[i,j] * p_j / (sin2_d[i,j])^2
        
        # Vectorized:
        # coeff[i,j] = 2 * dots[i,j] / (sin2_d[i,j])^2
        coeff = 2.0 * dots / (sin2_d * sin2_d)
        coeff[np.diag_indices(N)] = 0.0
        
        # grad_i = sum_j coeff[i,j] * p_j
        grad = coeff @ points
        
        # Project gradient to tangent space: grad - (grad . p_i) p_i
        proj = np.einsum('ij,ij->i', grad, points)
        grad = grad - proj[:, np.newaxis] * points
        
        return energy, grad

    def optimize_repulsive(pts, max_iter=100):
        """
        Optimize using repulsive energy.
        """
        if time.time() > deadline - 30.0:
            return pts
        
        current = pts.copy()
        
        def obj_and_grad(x_flat):
            p = x_flat.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            p = p / norms
            val, grad = repulsive_energy_and_grad(p, power=2.0)
            return val, grad.flatten()
        
        x0 = current.flatten()
        try:
            res = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-5}
            )
            if res.success or res.fun < obj_and_grad(x0)[0]:
                p = res.x.reshape(N, 3)
                p /= np.linalg.norm(p, axis=1, keepdims=True)
                current = p
        except Exception:
            pass
            
        return current

    def soft_optimize(pts, sigmas, max_iter_per_stage=20):
        """
        Multi-stage soft covering radius optimization.
        """
        if time.time() > deadline - 20.0:
            return pts
        
        current = pts.copy()
        
        for sigma in sigmas:
            if time.time() > deadline - 15.0:
                break
            
            def objective_and_grad(x_flat, sigma=sigma):
                pts_local = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts_local, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                pts_local = pts_local / norms
                val, grad = soft_covering_radius_and_grad(pts_local, grid_coarse, sigma=sigma)
                return val, grad.flatten()
            
            x0 = current.flatten()
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter_per_stage, 'ftol': 1e-8, 'gtol': 1e-4}
                )
                val_new, _ = objective_and_grad(res.x)
                val_old, _ = objective_and_grad(x0)
                if val_new < val_old:
                    p = res.x.reshape(N, 3)
                    p /= np.linalg.norm(p, axis=1, keepdims=True)
                    current = p
            except Exception:
                pass
        
        return current

    def compute_hull_and_facets(p):
        """Vectorized computation of hull facets and their radii."""
        try:
            h = ConvexHull(p)
        except Exception:
            return None, None
            
        simplices = h.simplices
        verts = p[simplices] # (F, 3, 3)
        if verts.shape[0] == 0:
            return h, None
            
        a = verts[:, 0, :]
        b = verts[:, 1, :]
        c = verts[:, 2, :]
        
        normals = np.cross(b - a, c - a)
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return h, None
            
        normals = normals[valid]
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        
        centroids = np.mean(verts[valid], axis=1)
        dots_cent = np.einsum('ij,ij->i', normals, centroids)
        normals[dots_cent < 0] *= -1
        
        v_valid = verts[valid]
        a_v = v_valid[:, 0, :]
        b_v = v_valid[:, 1, :]
        c_v = v_valid[:, 2, :]
        
        dots_a = np.einsum('ij,ij->i', normals, a_v)
        dots_b = np.einsum('ij,ij->i', normals, b_v)
        dots_c = np.einsum('ij,ij->i', normals, c_v)
        
        angles_a = np.arccos(np.clip(dots_a, -1, 1))
        angles_b = np.arccos(np.clip(dots_b, -1, 1))
        angles_c = np.arccos(np.clip(dots_c, -1, 1))
        
        facet_radii = np.max(np.stack([angles_a, angles_b, angles_c]), axis=0)
        simplices_valid = simplices[valid]
        
        return h, (simplices_valid, normals, facet_radii)

    def exact_local_optimize(pts, max_iters=300, initial_step=0.05):
        """
        Refined local optimization using vectorized facet checks and line search.
        """
        pts = pts.copy()
        
        _, facet_data = compute_hull_and_facets(pts)
        if facet_data is None:
            return pts
            
        worst_radius = np.max(facet_data[2])
        step_size = initial_step
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Find the worst facet
            worst_idx = np.argmax(facet_data[2])
            worst_simplex = facet_data[0][worst_idx]
            worst_normal = facet_data[1][worst_idx]
            worst_radius = facet_data[2][worst_idx]
            
            improved = False
            # Try to improve the worst facet by moving its vertices
            for j in worst_simplex:
                p_j = pts[j]
                # Direction toward circumcenter (normal)
                move_dir = worst_normal - np.dot(p_j, worst_normal) * p_j
                move_norm = np.linalg.norm(move_dir)
                if move_norm < 1e-10:
                    continue
                move_dir /= move_norm
                
                # Line search for step size
                best_ss = 0
                best_new_radius = worst_radius
                best_new_pts = None
                best_facet_data = None
                
                # Try a range of step sizes
                for ss in [step_size, step_size * 0.5, step_size * 0.25, step_size * 0.1]:
                    new_p = p_j + ss * move_dir
                    new_p /= np.linalg.norm(new_p)
                    
                    new_pts = pts.copy()
                    new_pts[j] = new_p
                    
                    _, new_facet_data = compute_hull_and_facets(new_pts)
                    if new_facet_data is None:
                        continue
                        
                    new_worst_radius = np.max(new_facet_data[2])
                    
                    if new_worst_radius < best_new_radius - 1e-12:
                        best_new_radius = new_worst_radius
                        best_ss = ss
                        best_new_pts = new_pts
                        best_facet_data = new_facet_data
                
                if best_ss > 0:
                    pts = best_new_pts
                    facet_data = best_facet_data
                    worst_radius = best_new_radius
                    improved = True
                    step_size = best_ss * 1.5
                    break # Move to next iteration (re-evaluate worst facet)
                
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
                    
        return pts

    best_pts = None
    best_rad = np.inf

    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 2:
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 3:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r >= 4 and best_pts is not None:
            rng = np.random.RandomState(1000 + r)
            scale = max(0.01, 0.03 - 0.003 * (r - 4))
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)
        
        # Phase 1: Repulsive Energy Optimization (Global Spread)
        # This acts as the "Variational Density" step, pushing points apart.
        cand = optimize_repulsive(pts, max_iter=100)
        
        # Phase 2: Soft Covering Optimization (Refinement)
        sigmas = [0.15, 0.08, 0.04, 0.02]
        cand = soft_optimize(cand, sigmas, max_iter_per_stage=20)
        
        # Phase 3: Exact Hull-based Local Optimization
        if time.time() < deadline - 20.0:
            cand = exact_local_optimize(cand, max_iters=100, initial_step=0.03)
        
        # Phase 4: Final fine-tuning
        if time.time() < deadline - 15.0:
            cand = soft_optimize(cand, [0.01, 0.005], max_iter_per_stage=10)
            cand = exact_local_optimize(cand, max_iters=200, initial_step=0.01)
        
        rad = covering_radius(cand)
        if rad < best_rad:
            best_rad = rad
            best_pts = cand.copy()
            print(f"Restart {r}: {np.degrees(best_rad):.6f} deg")

    # Final refinement on best solution
    if best_pts is not None and time.time() < deadline - 10.0:
        # Small repulsive step to ensure no points are too close
        fp = optimize_repulsive(best_pts, max_iter=20)
        fp = soft_optimize(fp, [0.005, 0.002], max_iter_per_stage=10)
        fp = exact_local_optimize(fp, max_iters=300, initial_step=0.005)
        fr = covering_radius(fp)
        if fr < best_rad:
            best_rad = fr
            best_pts = fp

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
