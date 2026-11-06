"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return pts / norms

    def compute_covering_radius(points):
        """Compute covering radius using convex hull facets."""
        try:
            hull = ConvexHull(points)
            max_radius = 0.0
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm

                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue

                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)

                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)

                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm

                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                max_radius = max(max_radius, facet_radius)
            return max_radius
        except Exception:
            return np.pi

    def compute_radius_and_grads(points):
        """
        Compute covering radius and approximate gradient.
        Gradient direction: Sum of tangent vectors towards circumcenters of active facets.
        """
        try:
            hull = ConvexHull(points)
            n_faces = len(hull.simplices)
            
            # First pass: get all radii to find max
            radii = np.zeros(n_faces)
            centers = np.zeros((n_faces, 3))
            valid_faces = []
            
            for i, simplex in enumerate(hull.simplices):
                face_points = points[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm

                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue

                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)

                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)

                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm
                
                centers[i] = circumcenter
                valid_faces.append(i)
                
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                radii[i] = np.max(angles)

            if not valid_faces:
                return np.pi, np.zeros_like(points)

            max_radius = np.max(radii)
            
            # Identify active facets
            # Use a small threshold to include facets close to max
            threshold = max_radius - 0.001 
            active_indices = [i for i in valid_faces if radii[i] > threshold]
            
            if not active_indices:
                return max_radius, np.zeros_like(points)

            grads = np.zeros_like(points)
            
            for i in active_indices:
                simplex = hull.simplices[i]
                center = centers[i]
                
                for idx in simplex:
                    v = points[idx]
                    # Tangent vector towards center
                    # proj = center - (v . center) * v
                    dot_vc = np.dot(v, center)
                    tangent = center - dot_vc * v
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > 1e-10:
                        # We want to minimize the angle, so move towards center
                        # Gradient of distance is -tangent/sin(angle)
                        # We just use the direction
                        grads[idx] += tangent / norm_t
                        
            return max_radius, grads

        except Exception:
            return np.pi, np.zeros_like(points)

    # Initialize with Fibonacci spiral points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Phase 0: Thomson-like energy minimization to maximize minimum pairwise distance
    def thomson_energy(x):
        """Energy that penalizes close points (inverse power law)."""
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - dots)))
        np.fill_diagonal(dists, 1.0)
        energy = np.sum(1.0 / (dists ** 4))
        return energy

    # Run Thomson minimization
    if time.time() - start_time < time_limit - 80:
        x0 = pts.flatten()
        best_energy = np.inf
        best_x = x0.copy()

        for restart in range(3):
            if time.time() - start_time > time_limit - 80:
                break
            if restart == 0:
                x_start = x0.copy()
            else:
                x_start = x0 + 0.1 * np.random.randn(N * 3)
                x_start = x_start.reshape(N, 3)
                x_start = x_start / np.linalg.norm(x_start, axis=1, keepdims=True)
                x_start = x_start.flatten()

            result = minimize(
                thomson_energy,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-8}
            )

            if result.fun < best_energy:
                best_energy = result.fun
                best_x = result.x.copy()

        pts = best_x.reshape(N, 3)
        pts = normalize(pts)

    # Phase 0.5: Spectral pair-correlation variance minimization
    def spectral_energy(x):
        """
        Smooth energy based on pair correlation variance.
        """
        if time.time() - start_time > time_limit:
            return np.pi
        p = x.reshape(N, 3)
        p = normalize(p)

        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, 0.0)

        iu = np.triu_indices(N, k=1)
        d = dists[iu]

        mean_d = np.mean(d)
        var_d = np.var(d)

        target_min = 0.2  # ~11.5 degrees
        penalty = np.sum(np.maximum(0, target_min - d) ** 2)

        energy = var_d + 0.1 * penalty
        return energy

    if time.time() - start_time < time_limit - 60:
        x0 = pts.flatten()
        result = minimize(
            spectral_energy,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-10}
        )
        new_pts = result.x.reshape(N, 3)
        new_pts = normalize(new_pts)

        current_radius = compute_covering_radius(pts)
        new_radius = compute_covering_radius(new_pts)
        if new_radius < current_radius:
            pts = new_pts

    best_pts = pts.copy()
    best_radius = compute_covering_radius(best_pts)

    # Phase 1: L-BFGS optimization on smooth covering radius approximation
    def smooth_covering_radius(x):
        """
        Smooth approximation of covering radius for L-BFGS.
        """
        if time.time() - start_time > time_limit:
            return np.pi
        p = x.reshape(N, 3)
        p = normalize(p)

        try:
            hull = ConvexHull(p)
            facet_radii = []
            for simplex in hull.simplices:
                face_points = p[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue

                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue

                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)

                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)

                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm

                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                facet_radii.append(facet_radius)

            if not facet_radii:
                return np.pi

            facet_radii = np.array(facet_radii)
            beta = 50.0
            max_r = np.max(facet_radii)
            soft_max = max_r + np.log(np.sum(np.exp(beta * (facet_radii - max_r)))) / beta
            return soft_max
        except Exception:
            return np.pi

    if time.time() - start_time < time_limit - 50:
        x0 = pts.flatten()
        result = minimize(
            smooth_covering_radius,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-10}
        )

        if result.fun < best_radius:
            new_pts = result.x.reshape(N, 3)
            new_pts = normalize(new_pts)
            new_radius = compute_covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts
                pts = new_pts

    # Phase 2: Gradient-based local search with simulated annealing
    # This replaces the simple force-based loops with a more robust optimizer
    
    current_pts = best_pts.copy()
    current_radius = best_radius
    
    # Parameters for annealing
    step_size = 0.02
    temperature = 0.005
    cool_rate = 0.99
    rng = np.random.default_rng(42)
    
    max_iters = 2000
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit - 5:
            break
        if current_radius < 0.19705:
            break
            
        radius, grads = compute_radius_and_grads(current_pts)
        
        if radius < current_radius:
            current_radius = radius
            best_radius = radius
            best_pts = current_pts.copy()
            
        # Normalize gradients to have unit norm per point? Or global?
        # Let's normalize the total step
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-10:
            # If gradient is zero, we are at a local min or stuck
            # Add noise to escape
            noise = rng.normal(0, 1, (N, 3))
            # Project to tangent space
            noise = noise - np.sum(noise * current_pts, axis=1, keepdims=True) * current_pts
            step = noise
        else:
            # Scale gradient to step size
            # Add noise
            noise_scale = temperature * 0.1
            noise = rng.normal(0, noise_scale, (N, 3))
            # Project noise to tangent space
            noise = noise - np.sum(noise * current_pts, axis=1, keepdims=True) * current_pts
            step = grads / grad_norm + noise
            
        new_pts = current_pts + step_size * step
        new_pts = normalize(new_pts)
        
        new_radius, _ = compute_radius_and_grads(new_pts)
        
        # Metropolis acceptance
        delta_r = new_radius - current_radius
        if delta_r < 0:
            current_pts = new_pts
            current_radius = new_radius
            # Cool down
            temperature *= cool_rate
        else:
            # Accept with probability exp(-delta_r / temperature)
            if rng.random() < math.exp(-delta_r / temperature):
                current_pts = new_pts
                current_radius = new_radius
            # Do not cool if rejected? Or cool slightly?
            # Usually cool regardless to ensure convergence
            temperature *= cool_rate
            
        # Adjust step size if stuck
        if it % 100 == 0:
             step_size *= 0.95

    # Phase 3: Final high-precision polish
    # Use very small steps and no noise
    for it in range(100):
        if time.time() - start_time > time_limit - 2:
            break
            
        radius, grads = compute_radius_and_grads(current_pts)
        if radius < best_radius:
            best_radius = radius
            best_pts = current_pts.copy()
            
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-10:
            break
            
        step = grads / grad_norm
        new_pts = current_pts + 0.001 * step
        new_pts = normalize(new_pts)
        
        new_radius, _ = compute_radius_and_grads(new_pts)
        
        if new_radius < current_radius:
            current_pts = new_pts
            current_radius = new_radius
        else:
            break

    return best_pts
# EVOLVE-BLOCK-END
