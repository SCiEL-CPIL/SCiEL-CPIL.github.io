"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _make_grid(n_theta=400, n_phi=800):
    """Create a dense spherical grid for evaluating covering radius."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return grid

_GRID = _make_grid()

def _covering_radius_grid(pts, n_theta=180, n_phi=360):
    """Compute covering radius in degrees using a grid approximation."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)
    max_radius_rad = np.arccos(np.min(min_dots))
    return np.degrees(max_radius_rad)

def _facet_circumradius(v1, v2, v3):
    """Compute the angular circumradius of a spherical triangle formed by three unit vectors."""
    # The circumcenter of a spherical triangle lies along the normal to the plane
    # containing the three points. The distance from origin to this plane gives
    # cos(angle) where angle is the circumradius.
    e1 = v2 - v1
    e2 = v3 - v1
    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n)
    
    if n_norm < 1e-15:
        return 0.0  # Degenerate triangle
    
    d = np.dot(n, v1)
    dist = abs(d) / n_norm
    dist = min(max(dist, -1.0), 1.0)
    angle = np.arccos(dist)
    return angle

def _covering_radius(pts):
    """Compute the exact covering radius using convex hull facets."""
    if len(pts) < 4:
        return _covering_radius_grid(pts, 360, 720)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_grid(pts, 360, 720)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        angle = _facet_circumradius(v1, v2, v3)
        if angle > max_radius:
            max_radius = angle
    
    return np.degrees(max_radius)

def _initial_fib():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_iteration(pts, n_theta=360, n_phi=720):
    """One Lloyd iteration: move each point to the centroid of its Voronoi cell."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    dots = grid @ pts.T
    assignments = dots.argmax(axis=1)
    
    new_pts = pts.copy()
    for i in range(len(pts)):
        mask = assignments == i
        if np.sum(mask) > 0:
            centroid = grid[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_pts[i] = centroid / norm
    
    return new_pts

def _refine_lloyd(pts, start_time, max_time=120.0, max_iters=30):
    """Lloyd-based refinement with annealing."""
    current = pts.copy()
    best = pts.copy()
    best_radius = _covering_radius_grid(current, 180, 360)
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
        
        new_pts = _lloyd_iteration(current)
        
        # Check for coincident points and nudge apart
        for i in range(N):
            for j in range(i + 1, N):
                dot = np.dot(new_pts[i], new_pts[j])
                if dot > 0.999999:
                    diff = new_pts[i] - new_pts[j]
                    norm = np.linalg.norm(diff)
                    if norm < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    diff /= norm
                    new_pts[i] = new_pts[i] + 0.01 * diff
                    new_pts[j] = new_pts[j] - 0.01 * diff
                    new_pts[i] /= np.linalg.norm(new_pts[i])
                    new_pts[j] /= np.linalg.norm(new_pts[j])
        
        new_radius = _covering_radius_grid(new_pts, 180, 360)
        if new_radius < best_radius:
            current = new_pts
            best = new_pts.copy()
            best_radius = new_radius
        else:
            # Try blending
            alpha = 0.5
            blended = (1 - alpha) * current + alpha * new_pts
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            blended = blended / norms
            blended_radius = _covering_radius_grid(blended, 180, 360)
            if blended_radius < best_radius:
                current = blended
                best = blended.copy()
                best_radius = blended_radius
            else:
                break
    
    return best, best_radius

def _gradient_descent_facets(pts, start_time, max_time=120.0, max_iters=500):
    """
    Gradient descent targeting the worst facets.
    For each facet, compute the circumcenter (on the sphere) and move the 
    three vertices toward it to shrink the circumradius.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    step_size = 0.02
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 15.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        # Find all facets and their circumradii
        facet_data = []
        for simplex in hull.simplices:
            v1 = best_pts[simplex[0]]
            v2 = best_pts[simplex[1]]
            v3 = best_pts[simplex[2]]
            angle = _facet_circumradius(v1, v2, v3)
            facet_data.append((angle, simplex))
        
        if not facet_data:
            break
        
        # Sort by circumradius descending, work on the worst few
        facet_data.sort(key=lambda x: -x[0])
        
        # Get the worst K facets
        K = min(10, len(facet_data))
        worst_facets = facet_data[:K]
        
        if worst_facets[0][0] < np.radians(11.3):
            break  # Already at or below known best
        
        # Compute gradient: for each vertex in a worst facet, move toward circumcenter
        grad = np.zeros_like(best_pts)
        
        for angle, simplex in worst_facets:
            v1 = best_pts[simplex[0]]
            v2 = best_pts[simplex[1]]
            v3 = best_pts[simplex[2]]
            
            # Compute circumcenter of the spherical triangle
            # The circumcenter lies along the normal to the plane of the triangle
            e1 = v2 - v1
            e2 = v3 - v1
            n = np.cross(e1, e2)
            n_norm = np.linalg.norm(n)
            
            if n_norm < 1e-15:
                continue
            
            # The circumcenter on the unit sphere is in the direction of the
            # normal, at the point where the plane intersects the sphere
            # Plane: n . x = d where d = n . v1
            d = np.dot(n, v1)
            
            # The circumcenter is at n * (d / n_norm^2) normalized to unit length
            # Actually, the foot of the perpendicular from origin to the plane is:
            # c = (d / n_norm^2) * n
            c = (d / (n_norm * n_norm)) * n
            c_norm = np.linalg.norm(c)
            
            if c_norm < 1e-15:
                continue
            
            c = c / c_norm  # Unit circumcenter
            
            # Move each vertex toward the circumcenter
            weight = 1.0 / K  # Distribute effort among worst facets
            
            for idx in simplex:
                v = best_pts[idx]
                # Direction toward circumcenter (tangent to sphere)
                direction = c - np.dot(v, c) * v
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    grad[idx] += weight * direction / dir_norm
        
        # Apply gradient step
        if np.linalg.norm(grad) > 1e-15:
            grad_norm = np.linalg.norm(grad)
            grad = grad / grad_norm * step_size
            
            new_pts = best_pts + grad
            # Project back to unit sphere
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-10:
                best_pts = new_pts
                best_radius = new_radius
                step_size = min(step_size * 1.05, 0.1)
            else:
                step_size *= 0.7
                if step_size < 0.001:
                    break
    
    return best_pts, best_radius

def _polish_step(pts, start_time, max_time=120.0, max_iters=100):
    """
    Final polish: combine repulsion (to prevent clustering) with targeted
    facet shrinking. Uses a smooth surrogate for the max circumradius.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Temperature for soft-max
    beta = 50.0  # Higher = sharper max approximation
    
    for it in range(max_iters):
        if time.time() - start_time > max_time - 10.0:
            break
        
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        # Compute all facet circumradii
        angles = []
        simplices_list = []
        for simplex in hull.simplices:
            v1 = best_pts[simplex[0]]
            v2 = best_pts[simplex[1]]
            v3 = best_pts[simplex[2]]
            angle = _facet_circumradius(v1, v2, v3)
            angles.append(angle)
            simplices_list.append(simplex)
        
        if not angles:
            break
        
        angles = np.array(angles)
        
        # Soft-max: approximate max with log-sum-exp
        # This gives smooth gradients
        max_angle = np.max(angles)
        if max_angle < 1e-15:
            break
        
        # exp(beta * (angle - max_angle)) for numerical stability
        exp_angles = np.exp(beta * (angles - max_angle))
        weights = exp_angles / np.sum(exp_angles)
        
        # Compute gradient: weighted sum of "move toward circumcenter" directions
        grad = np.zeros_like(best_pts)
        
        for w, simplex in zip(weights, simplices_list):
            if w < 1e-6:
                continue
            
            v1 = best_pts[simplex[0]]
            v2 = best_pts[simplex[1]]
            v3 = best_pts[simplex[2]]
            
            e1 = v2 - v1
            e2 = v3 - v1
            n = np.cross(e1, e2)
            n_norm = np.linalg.norm(n)
            
            if n_norm < 1e-15:
                continue
            
            d = np.dot(n, v1)
            c = (d / (n_norm * n_norm)) * n
            c_norm = np.linalg.norm(c)
            
            if c_norm < 1e-15:
                continue
            
            c = c / c_norm
            
            for idx in simplex:
                v = best_pts[idx]
                direction = c - np.dot(v, c) * v
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    grad[idx] += w * direction / dir_norm
        
        # Add small repulsion to prevent coincident points
        repulsion = np.zeros_like(best_pts)
        for i in range(N):
            for j in range(i + 1, N):
                diff = best_pts[i] - best_pts[j]
                dist = np.linalg.norm(diff)
                if dist < 0.05:
                    force = 0.01 / (dist + 1e-10)
                    repulsion[i] += force * diff
                    repulsion[j] -= force * diff
        
        total_grad = grad + 0.1 * repulsion
        
        if np.linalg.norm(total_grad) < 1e-15:
            break
        
        # Adaptive step size
        step = 0.01
        for _ in range(5):
            new_pts = best_pts + step * total_grad
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_radius = _covering_radius(new_pts)
            if new_radius < best_radius - 1e-12:
                best_pts = new_pts
                best_radius = new_radius
                break
            step *= 0.5
        
        if step < 1e-6:
            break
    
    return best_pts, best_radius

def construct():
    """Construct 130 points on the unit sphere to minimize the covering radius."""
    start_time = time.time()
    max_time = 120.0
    
    # Start with Fibonacci spiral
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # Phase 1: Lloyd refinement (coarse)
    lloyd_pts, lloyd_radius = _refine_lloyd(pts, start_time, max_time, max_iters=20)
    if lloyd_radius < best_radius:
        best_pts = lloyd_pts
        best_radius = lloyd_radius
    
    # Phase 2: Gradient descent on worst facets
    remaining = max_time - (time.time() - start_time)
    if remaining > 30.0:
        gd_pts, gd_radius = _gradient_descent_facets(best_pts, start_time, max_time)
        if gd_radius < best_radius:
            best_pts = gd_pts
            best_radius = gd_radius
    
    # Phase 3: Polish with soft-max surrogate
    remaining = max_time - (time.time() - start_time)
    if remaining > 15.0:
        polish_pts, polish_radius = _polish_step(best_pts, start_time, max_time)
        if polish_radius < best_radius:
            best_pts = polish_pts
            best_radius = polish_radius
    
    # Phase 4: Final Lloyd refinement (fine)
    remaining = max_time - (time.time() - start_time)
    if remaining > 10.0:
        final_pts, final_radius = _refine_lloyd(best_pts, start_time, max_time, max_iters=15)
        if final_radius < best_radius:
            best_pts = final_pts
            best_radius = final_radius
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-10:
                best_pts[j] += np.random.randn(3) * 0.001
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Final normalization
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
