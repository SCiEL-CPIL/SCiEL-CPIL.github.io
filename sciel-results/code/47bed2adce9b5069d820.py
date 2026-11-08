<<<<<<< SEARCH
def _project_points(points):
    """
    Project all points to the triangle.
    """
    projected = np.array([_project_to_triangle(p) for p in points])
    return projected
=======
def _project_points(points):
    """
    Project all points to the triangle.
    """
    projected = np.array([_project_to_triangle(p) for p in points])
    return projected

def _sdp_init(time_budget=5.0):
    """
    Use an SDP-style approach to maximize minimum pairwise distance squared.
    This is a convex surrogate that gives a better global initial point.
    We use an iterative approach: start from a random feasible point,
    then use coordinate descent / projected gradient to maximize min pairwise dist sq.
    
    Returns: array of shape (N, 2)
    """
    rng = random.Random(123)
    best_pts = None
    best_score = -1.0
    start = time.time()
    
    # Multiple random restarts for the SDP-like optimization
    for restart in range(20):
        if time.time() - start > time_budget:
            break
        
        # Random initialization inside triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            u, v = rng.random(), rng.random()
            while u + v >= 1.0:
                u, v = rng.random(), rng.random()
            pts[i] = [u + 0.5 * v, H * v]
        
        # Coordinate descent to maximize min pairwise distance squared
        # This is like solving the Tammes problem on a triangle
        for iteration in range(2000):
            if time.time() - start > time_budget:
                break
            
            improved = False
            for i in range(N):
                # Try to move point i to maximize the minimum distance to all other points
                # Use a simple hill-climbing step
                best_pos = pts[i].copy()
                best_min_d = -1.0
                
                # Try several candidate positions around current
                for dx in [-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05]:
                    for dy in [-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05]:
                        candidate = pts[i] + np.array([dx, dy])
                        candidate = _project_to_triangle(candidate)
                        
                        # Compute min distance to all other points
                        other_pts = np.delete(pts, i, axis=0)
                        diff = candidate - other_pts
                        dist_sq = np.sum(diff * diff, axis=1)
                        min_d = np.min(dist_sq)
                        
                        if min_d > best_min_d:
                            best_min_d = min_d
                            best_pos = candidate
                
                if np.any(best_pos != pts[i]):
                    pts[i] = best_pos
                    improved = True
            
            if not improved:
                break
        
        # Evaluate
        score = _min_pairwise_dist_sq(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback
        pts = np.zeros((N, 2))
        for i in range(N):
            u, v = rng.random(), rng.random()
            while u + v >= 1.0:
                u, v = rng.random(), rng.random()
            pts[i] = [u + 0.5 * v, H * v]
        best_pts = pts
    
    return best_pts
>>>>>>> REPLACE

