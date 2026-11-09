import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf

    # Helper to compute covering radius from points (High Precision, used for final scoring)
    def compute_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                c_s = normal / norm
                
                # Check if circumcenter is inside the spherical triangle
                # If the triangle is acute, the circumcenter is inside.
                # If obtuse, the circumcenter is outside, and the worst point is the midpoint of the longest edge.
                
                # Calculate angular distances from circumcenter to vertices
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                
                max_d_circ = max(d1, d2, d3)
                
                # Check edges
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0

                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0

                max_d_edge = max(d_e12, d_e23, d_e31)

                # If the circumcenter is "inside" (acute triangle), the radius is max_d_circ.
                # If it's "outside" (obtuse), the radius is max_d_edge.
                # A robust check: if max_d_circ > pi/2, it's likely obtuse in a way that puts the center outside the "empty cap" region relative to the vertices?
                # Actually, for a spherical triangle, if it's acute, the circumcenter is inside. If obtuse, it's outside.
                # The covering radius contribution is the maximum distance to the nearest point.
                # If circumcenter is inside, it's the center of the largest empty circle for this facet.
                # If circumcenter is outside, the center of the largest empty circle is the midpoint of the longest edge.
                
                # Heuristic: If max_d_circ <= pi/2, use it. Else use max_d_edge.
                # Note: For an equilateral triangle with side < pi/2, circumcenter is inside.
                # If side > pi/2, circumcenter might be "far".
                
                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                else:
                    local_rad = max_d_edge

                if local_rad > max_rad:
                    max_rad = local_rad
            return max_rad
        except Exception:
            return np.inf

    # Helper to compute a fast approximate energy (for SA)
    # Using a fixed set of probe points on the sphere
    # Generate a dense grid of probe points (e.g., Fibonacci sphere with M points)
    M = 2000
    i_probe = np.arange(M) + 0.5
    z_probe = 1.0 - 2.0 * i_probe / M
    phi_probe = np.pi * (1.0 + np.sqrt(5.0)) * i_probe
    r_probe = np.sqrt(1.0 - z_probe * z_probe)
    probe_points = np.column_stack([r_probe * np.cos(phi_probe), r_probe * np.sin(phi_probe), z_probe])
    
    def compute_energy(pts):
        # Compute angular distance from each probe point to its nearest configuration point
        # dist^2 = 2 - 2 * dot
        # We want to minimize max(dist). Using sum of squares or max of squares.
        # Max is better for covering radius, but non-smooth. 
        # Let's use a soft-max or just max for SA.
        # Dot product matrix: (M, N)
        dots = probe_points @ pts.T
        # Clip for numerical stability
        dots = np.clip(dots, -1.0, 1.0)
        # Max dot for each probe point
        max_dots = np.max(dots, axis=1)
        # Min angular distance for each probe point
        min_dists = np.arccos(max_dots)
        # The covering radius is the max of these min_dists
        return np.max(min_dists)

    # Simulated Annealing
    current_points = points.copy()
    current_energy = compute_energy(current_points)
    
    # SA Parameters
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.999
    step_size = 0.1
    
    # We will run SA for a fixed number of steps or until time limit
    # Each step: pick a random point, move it randomly, compute energy.
    
    # To speed up, we can update energy incrementally? 
    # No, recomputing MxN dot product is fast enough for M=2000, N=50.
    # 2000 * 50 = 100k ops. Very fast.
    
    n_iter = 0
    while time.time() - start_time < time_limit - 10.0: # Leave 10s for final refinement
        # Pick a random point
        idx = np.random.randint(N)
        
        # Generate a random perturbation
        # Move on the sphere: add random vector and normalize
        perturbation = np.random.normal(0, step_size, 3)
        new_point = current_points[idx] + perturbation
        new_point /= np.linalg.norm(new_point)
        
        # Save old point
        old_point = current_points[idx].copy()
        current_points[idx] = new_point
        
        # Compute new energy
        new_energy = compute_energy(current_points)
        
        # Acceptance criterion
        if new_energy < current_energy:
            current_energy = new_energy
            # Update best
            best_radius_approx = new_energy
            # We don't update best_points yet, we will refine at the end
            # But we can keep track of the best SA state
            # Actually, we should update best_points if the *approximate* radius is better?
            # The approximate radius is an upper bound on the true covering radius (since probe points are finite).
            # So if approx_radius decreases, true radius likely decreases.
            
            # Let's store the best SA points
            if best_radius_approx < best_radius:
                 best_radius = best_radius_approx
                 best_points = current_points.copy()
        else:
            # Metropolis criterion
            delta_E = new_energy - current_energy
            if delta_E > 0:
                if np.random.rand() < np.exp(-delta_E / T):
                    current_energy = new_energy
                else:
                    # Reject
                    current_points[idx] = old_point
            
        # Cool down
        T *= cooling_rate
        
        # Adaptive step size? 
        # If T is low, we might want smaller steps.
        # For now, fixed step size with random direction.
        
        # Check time
        if time.time() - start_time > time_limit - 10.0:
            break
            
    # Final Refinement Phase
    # Use the best_points found by SA
    # Perform a local optimization using the exact convex hull method
    # We will do a few iterations of "move worst facet vertices towards center"
    
    # First, compute the exact radius of the SA best point
    exact_rad = compute_covering_radius(best_points)
    
    # If the SA didn't find anything better than Fibonacci, use Fibonacci
    if exact_rad > compute_covering_radius(points):
         best_points = points.copy()
         exact_rad = compute_covering_radius(best_points)
         
    # Local Optimization Loop
    # We use the exact covering radius to guide the moves.
    # This is the same as the original program's loop, but starting from a better initial state (SA result).
    # Also, we can use multiple restarts or different step sizes.
    
    # To make it more robust, we can do a "shake and refine"
    # 1. Take best_points
    # 2. Perturb slightly
    # 3. Refine using convex hull gradient
    # 4. If better, keep.
    
    # Let's implement a simple local refinement that runs for the remaining time.
    
    def local_refine(pts, max_time):
        pts = pts.copy()
        best_local_rad = compute_covering_radius(pts)
        best_local_pts = pts.copy()
        
        step_size = 0.05
        t_start = time.time()
        
        while time.time() - t_start < max_time:
            # Compute exact radius and worst facet
            try:
                hull = ConvexHull(pts)
                max_rad = 0.0
                worst_simplex = None
                worst_center = None
                
                for simplex in hull.simplices:
                    v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                    
                    normal = np.cross(v2 - v1, v3 - v1)
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    c_s = normal / norm
                    
                    d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                    d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                    d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                    
                    max_d_circ = max(d1, d2, d3)
                    
                    e12 = v1 + v2
                    e12_norm = np.linalg.norm(e12)
                    if e12_norm > 1e-10:
                        m12 = e12 / e12_norm
                        d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                    else:
                        d_e12 = 0.0
                    
                    e23 = v2 + v3
                    e23_norm = np.linalg.norm(e23)
                    if e23_norm > 1e-10:
                        m23 = e23 / e23_norm
                        d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                    else:
                        d_e23 = 0.0

                    e31 = v3 + v1
                    e31_norm = np.linalg.norm(e31)
                    if e31_norm > 1e-10:
                        m31 = e31 / e31_norm
                        d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                    else:
                        d_e31 = 0.0

                    max_d_edge = max(d_e12, d_e23, d_e31)

                    if max_d_circ <= np.pi / 2.0:
                        local_rad = max_d_circ
                        center = c_s
                    else:
                        local_rad = max_d_edge
                        if d_e12 == max_d_edge:
                            center = m12
                        elif d_e23 == max_d_edge:
                            center = m23
                        else:
                            center = m31

                    if local_rad > max_rad:
                        max_rad = local_rad
                        worst_simplex = simplex
                        worst_center = center
                        
                if worst_simplex is None:
                    break
                    
                # Move vertices towards center
                v1, v2, v3 = pts[worst_simplex[0]], pts[worst_simplex[1]], pts[worst_simplex[2]]
                
                # Adaptive step size based on current radius?
                # Smaller radius -> smaller step?
                # Or just fixed small step.
                ss = step_size * 0.5 # Start smaller
                
                new_v1 = v1 + ss * (worst_center - v1)
                new_v2 = v2 + ss * (worst_center - v2)
                new_v3 = v3 + ss * (worst_center - v3)
                
                new_v1 /= np.linalg.norm(new_v1)
                new_v2 /= np.linalg.norm(new_v2)
                new_v3 /= np.linalg.norm(new_v3)
                
                pts[worst_simplex[0]] = new_v1
                pts[worst_simplex[1]] = new_v2
                pts[worst_simplex[2]] = new_v3
                
                # Check if improved
                new_rad = compute_covering_radius(pts)
                if new_rad < best_local_rad:
                    best_local_rad = new_rad
                    best_local_pts = pts.copy()
                else:
                    # Revert and try a different move or smaller step
                    # For simplicity, revert and break if stuck?
                    # Or just keep trying with smaller step?
                    # Let's revert to best_local_pts and perturb
                    pts = best_local_pts.copy()
                    # Add small random perturbation to escape
                    pts += np.random.normal(0, 0.005, pts.shape)
                    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                    
            except Exception:
                break
                
        return best_local_pts, best_local_rad

    # Run local refinement on the best SA result
    remaining_time = time_limit - (time.time() - start_time) - 2.0 # 2s buffer
    if remaining_time > 5.0:
        refined_points, refined_rad = local_refine(best_points, remaining_time)
        if refined_rad < exact_rad:
            best_points = refined_points
            exact_rad = refined_rad

    return best_points

# EVOLVE-BLOCK-END
