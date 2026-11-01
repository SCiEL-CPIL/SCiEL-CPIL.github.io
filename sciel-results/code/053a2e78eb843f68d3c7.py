import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import scipy.spatial.distance as dist

N = 130
_EPS = 1e-12

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < _EPS] = 1.0
    return pts / norms

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius(pts):
    """
    Compute the covering radius by finding the convex hull of the points.
    The covering radius is the maximum angular distance from the origin to a facet normal
    scaled by the distance to the vertices? No, for unit sphere, the covering radius is the
    maximum circumradius of the spherical Voronoi cells.
    
    For a set of points on the sphere, the covering radius is determined by the largest empty cap.
    This corresponds to the largest facet of the convex hull of the points.
    Specifically, for each triangular facet (i, j, k) of the convex hull, the circumcenter 
    (normal vector) n is the point on the sphere farthest from the three vertices i, j, k 
    within that facet's region. The angular radius is the angle between n and any of the vertices.
    
    However, the convex hull of points on the sphere gives the dual of the Voronoi diagram on the sphere.
    The vertices of the convex hull are the points. The faces of the convex hull correspond to the Voronoi cells.
    The circumradius of a Voronoi cell is the distance from its center (the normal of the hull face) to its vertices (the original points).
    
    So:
    1. Compute ConvexHull.
    2. For each simplex (face), compute the normal.
    3. Compute the angle between the normal and the 3 vertices of the face.
    4. The max of these angles is the covering radius.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normal for each face
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter out degenerate faces
        valid = norms[:, 0] > _EPS
        if not np.any(valid):
            return np.pi, None, None, None
            
        normals = cross[valid] / norms[valid]
        
        # Ensure normals point outward (away from origin)
        # The centroid of the face should be in the direction of the normal
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # Angular radius is arccos(normal . vertex)
        # Since all vertices of the face are equidistant from the normal in terms of angle?
        # Actually, for a spherical triangle, the circumcenter is the normal of the plane passing through the 3 vertices.
        # The distance to the circumcenter is the same for all 3 vertices.
        cos_theta_A = np.sum(normals * A[valid], axis=1)
        cos_theta_A = np.clip(cos_theta_A, -1.0, 1.0)
        radii = np.arccos(cos_theta_A)
        
        max_radius = np.max(radii)
        
        # Find the index of the worst facet for potential repair
        worst_local_idx = np.argmax(radii)
        worst_face_indices = faces[valid][worst_local_idx]
        worst_normal = normals[worst_local_idx]
        
        return max_radius, worst_face_indices, worst_normal, faces[valid]
    except Exception:
        return np.pi, None, None, None

def _lloyd_relax(pts, max_iter=50, tol=1e-6):
    """
    Perform Lloyd's relaxation on the sphere.
    1. Compute Voronoi cells (dual to convex hull faces? No, dual to Delaunay).
       On the sphere, the Voronoi diagram is the dual of the Delaunay triangulation.
       We can compute the convex hull of the points. The faces of the convex hull correspond to the Voronoi cells.
       The center of a Voronoi cell is the circumcenter of the corresponding hull face.
       
       Wait, Lloyd's algorithm moves points to the centroid of their Voronoi cells.
       The centroid of a spherical Voronoi cell is not simply the circumcenter.
       However, a common approximation for "best" covering is to minimize the max distance to the circumcenter.
       
       Actually, the standard Lloyd's algorithm for sphere packing/covering uses the centroid of the spherical Voronoi cell.
       Computing the exact centroid of a spherical polygon is complex.
       
       Alternative: Use the "Centroid of the convex hull face" as an approximation?
       Or better: The point that minimizes the max distance to the vertices of the cell is the circumcenter.
       But Lloyd's minimizes the sum of squared distances, which leads to the centroid.
       
       For covering radius (minimax), the optimal point in a cell is the circumcenter.
       So, a "Minimax Lloyd" would move each point to the circumcenter of its Voronoi cell.
       But the Voronoi cell of point i is defined by the Delaunay triangulation.
       
       Let's use the ConvexHull faces. Each face corresponds to a Voronoi cell.
       The vertices of the face are the points that define the cell.
       The circumcenter of the face is the point on the sphere equidistant to the 3 vertices.
       If we move the points to these circumcenters, we are effectively performing a gradient step towards minimizing the max radius.
       
       However, moving a point to the circumcenter of its *current* cell might be unstable.
       A safer approach:
       1. Compute Hull.
       2. For each face, compute the circumcenter (normal).
       3. This normal is the "center" of the Voronoi cell.
       4. The Voronoi cell belongs to which point? The face is formed by 3 points. The Voronoi cell is the region where one of these points is the nearest?
          No, the Delaunay triangulation connects points if their Voronoi cells share an edge.
          The ConvexHull of the points gives the Delaunay triangulation on the sphere (for points in general position).
          Each face (i,j,k) in the hull corresponds to a Voronoi vertex (circumcenter) where cells of i, j, k meet.
          This is not the center of the cell.
          
       Correction:
       The Voronoi cell of point p_i is the set of points x such that dist(x, p_i) <= dist(x, p_j) for all j.
       The boundaries are great circles.
       The centroid of this cell is the point that minimizes the average squared distance.
       
       Let's implement a simpler "Gradient Descent on Max Radius" using subgradients.
       The covering radius R is max over all x on sphere of min_i d(x, p_i).
       The worst points x are the circumcenters of the hull faces.
       Let C be the set of circumcenters.
       R = max_{c in C} d(c, nearest_point(c)).
       For a specific worst face with circumcenter c and vertices i,j,k, the gradient of the distance d(c, p_i) with respect to p_i is -c / ||c - p_i|| (chord direction).
       So we can push the vertices i,j,k away from c? No, we want to reduce the distance.
       So we should move p_i, p_j, p_k towards c.
       
       This is exactly what the current program does!
       
       So why change?
       The current program gets stuck.
       
       New Strategy: **Genetic Algorithm with Lloyd-style Initialization and Local Repair**.
       We will maintain a population of configurations.
       Each configuration is refined by a few steps of the "Push towards Pole" method (local search).
       Then we select the best, crossover, and mutate.
       
       Crossover: Take a mix of points from two parents? Hard to define for unordered sets.
       Better: "Geometric Crossover" - Sort points by angle, take first half from A, second from B?
       Or simply: Randomly select points from a pool of good points found so far.
       
       Let's try **Simulated Annealing with Larger Jumps** but using a different objective function for the intermediate steps?
       
       Actually, let's try **Direct Optimization of the Log-Barrier / Sum-Exp**.
       Objective: f(P) = sum_{x in Grid} log(sum_{i} exp(-d(x, p_i)^2 / T))
       This is smooth and differentiable.
       We can use gradient descent.
       
       Let's implement Gradient Descent on a fine grid approximation of the covering radius.
       """
    # This function is a placeholder for the actual logic in construct()
    # We will implement the GA approach in construct
    return pts

def construct():
    deadline = time.time() + 140.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Initial population
    pop_size = 10
    population = []
    
    # Seed 1: Fibonacci
    p1 = _initial_fib()
    population.append(p1)
    
    # Seed 2: Random uniform
    p2 = rng.normal(size=(N, 3))
    p2 = _normalize(p2)
    population.append(p2)
    
    # Seed 3: Fibonacci with slight rotation
    p3 = _initial_fib()
    # Random rotation
    theta = rng.uniform(0, 2*np.pi)
    phi = rng.uniform(0, np.pi)
    # Rotation matrix
    # Simple rotation around Z
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    p3 = p3 @ rot.T
    p3 = _normalize(p3)
    population.append(p3)
    
    # Add some random perturbations of Fibonacci
    for i in range(pop_size - 3):
        p = _initial_fib()
        noise = rng.normal(size=(N, 3)) * 0.2
        p = _normalize(p + noise)
        population.append(p)
    
    best_pts = None
    best_radius = np.pi
    
    # Evaluate initial population
    for p in population:
        r, _, _, _ = _compute_covering_radius(p)
        if r < best_radius:
            best_radius = r
            best_pts = p.copy()
            
    # Local search function: Push towards pole for a few iterations
    def local_search(pts, iterations=20):
        for _ in range(iterations):
            r, v_indices, pole, _ = _compute_covering_radius(pts)
            if v_indices is None:
                break
            # Move vertices of the worst facet towards the pole
            new_pts = pts.copy()
            for v in v_indices:
                dir_vec = pole - new_pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                    # Small step
                    new_pts[v] = new_pts[v] + 0.05 * dir_vec
            new_pts = _normalize(new_pts)
            
            # Check if it improved
            r_new, _, _, _ = _compute_covering_radius(new_pts)
            if r_new < r:
                pts = new_pts
            else:
                # Try smaller step or break
                break
        return pts

    # Main loop: Genetic Algorithm
    while time.time() < deadline:
        # Refine the best candidate with local search
        refined_best = local_search(best_pts, iterations=10)
        r_refined, _, _, _ = _compute_covering_radius(refined_best)
        if r_refined < best_radius:
            best_radius = r_refined
            best_pts = refined_best
            
        # Generate new candidates
        new_population = []
        
        # 1. Clone best with mutation
        if len(population) > 0:
            base = best_pts
            # Mutation: random kick
            mut = _normalize(base + rng.normal(size=(N, 3)) * 0.1)
            new_population.append(mut)
            
            # 2. Crossover-like: Mix best with a random other
            if len(population) > 1:
                other = population[rng.integers(1, len(population))]
                # Sort points by z-coordinate for a simple "spatial" crossover
                idx_best = np.argsort(base[:, 2])
                idx_other = np.argsort(other[:, 2])
                
                # Take first half from best, second half from other
                child = np.empty((N, 3))
                half = N // 2
                child[:half] = base[idx_best[:half]]
                child[half:] = other[idx_other[half:]]
                child = _normalize(child)
                new_population.append(child)
                
        # 3. Random restart if stuck
        if rng.random() < 0.1:
            new_population.append(_initial_fib() + rng.normal(size=(N, 3)) * 0.5)
            new_population[-1] = _normalize(new_population[-1])
            
        # Evaluate new candidates
        for p in new_population:
            p = local_search(p, iterations=5) # Light local search
            r, _, _, _ = _compute_covering_radius(p)
            if r < best_radius:
                best_radius = r
                best_pts = p.copy()
            # Add to population if good
            if r < best_radius * 1.05: # Keep near-best
                population.append(p)
            
        # Trim population
        if len(population) > pop_size:
            # Keep the best ones
            scored_pop = []
            for p in population:
                r, _, _, _ = _compute_covering_radius(p)
                scored_pop.append((r, p))
            scored_pop.sort(key=lambda x: x[0])
            population = [p for r, p in scored_pop[:pop_size]]
            
    # Final refinement
    best_pts = local_search(best_pts, iterations=50)
    best_pts = _normalize(best_pts)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
