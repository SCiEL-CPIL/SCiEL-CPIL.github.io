"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius_and_worst_indices(pts):
        """
        Computes the exact covering radius and identifies the facet responsible for it.
        Returns (radius_degrees, worst_face_indices, circumcenter).
        """
        try:
            hull = ConvexHull(pts)
            max_r_rad = 0.0
            worst_face = None
            worst_center = None
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of the spherical triangle
                v1 = p0 - p1
                v2 = p0 - p2
                c_vec = np.cross(v1, v2)
                
                if np.linalg.norm(c_vec) < 1e-10:
                    continue
                    
                c_vec /= np.linalg.norm(c_vec)
                
                # Determine correct hemisphere for circumcenter
                centroid = (p0 + p1 + p2) / 3.0
                centroid /= np.linalg.norm(centroid)
                
                if np.dot(c_vec, centroid) < 0:
                    c_vec = -c_vec
                    
                # Check if circumcenter is inside the spherical triangle
                n1 = np.cross(p1, p2)
                n2 = np.cross(p2, p0)
                n3 = np.cross(p0, p1)
                
                # Orient normals towards centroid
                if np.dot(centroid, n1) < 0: n1 = -n1
                if np.dot(centroid, n2) < 0: n2 = -n2
                if np.dot(centroid, n3) < 0: n3 = -n3
                
                if (np.dot(c_vec, n1) >= -1e-8 and 
                    np.dot(c_vec, n2) >= -1e-8 and 
                    np.dot(c_vec, n3) >= -1e-8):
                    
                    dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                    if dist > max_r_rad:
                        max_r_rad = dist
                        worst_face = face
                        worst_center = c_vec
                        
            return np.degrees(max_r_rad), worst_face, worst_center
        except Exception:
            return 180.0, None, None

    def lloyd_relax(pts, steps=50):
        """
        Perform Lloyd's algorithm iterations on the sphere.
        For each point, find the centroid of its Voronoi cell (approximated by convex hull facets 
        or by assigning nearest neighbors from a dense grid).
        Here, we use a high-resolution grid to approximate the Voronoi centroids for speed and robustness.
        """
        n_grid = 50000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        grid_pts = normalize(grid_pts)
        
        current_pts = pts.copy()
        
        for _ in range(steps):
            # Assign each grid point to its nearest configuration point
            dots = grid_pts @ current_pts.T
            nearest_idx = np.argmax(dots, axis=1)
            
            # Compute new positions as the normalized centroid of assigned grid points
            new_pts = np.zeros_like(current_pts)
            counts = np.zeros(N)
            
            # Efficient accumulation
            for k in range(N):
                mask = (nearest_idx == k)
                if np.any(mask):
                    new_pts[k] = np.sum(grid_pts[mask], axis=0)
                    counts[k] = np.sum(mask)
                else:
                    # If no points assigned, keep old position (or randomize slightly?)
                    new_pts[k] = current_pts[k]
            
            # Normalize
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            new_pts /= norms
            
            # Check for convergence
            diff = np.linalg.norm(new_pts - current_pts)
            current_pts = new_pts
            if diff < 1e-6:
                break
                
        return current_pts

    def refine_with_hull(pts, iterations=20):
        """
        Specific refinement step: Identify the largest facet (covering radius bottleneck)
        and move the vertices of that facet towards the circumcenter to shrink the facet.
        This is a targeted local search that directly minimizes the covering radius.
        """
        current_pts = pts.copy()
        best_cr, _, _ = compute_covering_radius_and_worst_indices(current_pts)
        
        for _ in range(iterations):
            if time.time() > deadline - 10:
                break
                
            cr, face, center = compute_covering_radius_and_worst_indices(current_pts)
            if face is None or center is None:
                break
                
            if cr < best_cr:
                best_cr = cr
                # Update best if we improved
                # We might want to store the best configuration found during this phase
                
            # Move the vertices of the worst face towards the circumcenter
            # We use a small step size to avoid oscillation
            step_size = 0.1
            for idx in face:
                # Move point idx towards center
                # current_pts[idx] = normalize(current_pts[idx] + step_size * (center - current_pts[idx]))
                # Better: Spherical interpolation or just linear in tangent space
                # Simple linear update followed by normalization
                new_pos = current_pts[idx] + step_size * (center - current_pts[idx])
                current_pts[idx] = new_pos / np.linalg.norm(new_pos)
                
            # Also, to prevent points from clustering too much, we can add a small repulsive force
            # but Lloyd's relaxation usually handles the global distribution.
            # Let's alternate: one step of targeted shrink, then a few steps of Lloyd to relax stresses.
            
            # Relax slightly to allow neighbors to adjust
            current_pts = lloyd_relax(current_pts, steps=2)
            
            # Check if we actually improved the global covering radius
            new_cr, _, _ = compute_covering_radius_and_worst_indices(current_pts)
            if new_cr > best_cr + 1e-4:
                # If we got worse, revert? Or just keep the best found so far.
                # For simplicity, we track the best configuration separately.
                pass
                
        # Return the configuration that had the best covering radius during this process?
        # The function returns the current state. We should track the best state inside the loop.
        # Let's re-implement to track best state explicitly.
        
        # Re-doing logic to track best state properly
        current_pts = pts.copy()
        best_pts_so_far = current_pts.copy()
        best_cr, _, _ = compute_covering_radius_and_worst_indices(current_pts)
        
        for _ in range(iterations):
            if time.time() > deadline - 10:
                break
                
            cr, face, center = compute_covering_radius_and_worst_indices(current_pts)
            if face is None or center is None:
                break
                
            if cr < best_cr:
                best_cr = cr
                best_pts_so_far = current_pts.copy()
                
            # Move vertices of the worst face towards circumcenter
            step_size = 0.05
            for idx in face:
                new_pos = current_pts[idx] + step_size * (center - current_pts[idx])
                current_pts[idx] = new_pos / np.linalg.norm(new_pos)
                
            # Relax with Lloyd to stabilize
            current_pts = lloyd_relax(current_pts, steps=3)
            
        return best_pts_so_far

    # Main Strategy:
    # 1. Initialize with Fibonacci spiral (good global start)
    # 2. Run Lloyd's algorithm to get to a local minimum of the mean squared error (uniformity)
    # 3. Run the hull-based refinement to specifically minimize the covering radius
    # 4. Repeat with different random perturbations or seeds if time permits
    
    # Initial Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = fib_pts.copy()
    best_cr, _, _ = compute_covering_radius_and_worst_indices(best_pts)
    
    # Phase 1: Lloyd Relaxation from Fibonacci
    relaxed_pts = lloyd_relax(best_pts, steps=100)
    cr_relaxed, _, _ = compute_covering_radius_and_worst_indices(relaxed_pts)
    if cr_relaxed < best_cr:
        best_cr = cr_relaxed
        best_pts = relaxed_pts
        
    # Phase 2: Hull-based Refinement
    refined_pts = refine_with_hull(best_pts, iterations=30)
    cr_refined, _, _ = compute_covering_radius_and_worst_indices(refined_pts)
    if cr_refined < best_cr:
        best_cr = cr_refined
        best_pts = refined_pts
        
    # Phase 3: Multi-start with Random Perturbations + Refinement
    rng = np.random.default_rng(42)
    
    # We have time budget. Let's try a few random restarts.
    # Each restart: Random init -> Lloyd -> Refine
    # Lloyd is expensive (50k grid * 50 points). 100 steps might take a while.
    # Let's estimate: 50k*50 dots = 2.5M ops. 100 steps = 250M ops. ~0.1-0.5s per Lloyd phase.
    # Refine is cheap (hull of 50 points).
    
    max_restarts = 10
    
    for k in range(max_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random initialization
        init_pts = rng.normal(size=(N, 3))
        init_pts = normalize(init_pts)
        
        # Short Lloyd relaxation to get out of bad local minima
        # Fewer steps for random starts to save time
        relaxed = lloyd_relax(init_pts, steps=30)
        
        # Refinement
        refined = refine_with_hull(relaxed, iterations=20)
        
        cr, _, _ = compute_covering_radius_and_worst_indices(refined)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = refined
            
    # Final check
    final_cr, _, _ = compute_covering_radius_and_worst_indices(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
