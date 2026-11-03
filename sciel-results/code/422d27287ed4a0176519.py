"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import random

def _delaunay_edges(tri):
    """Return set of unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edges.add((min(a, b), max(a, b)))
    return edges

def _compute_radii(points):
    """
    Compute radii for circles centered at points such that:
    - Circles are disjoint (interiors)
    - Circles stay inside [0,1]^2
    - Radii are determined by half the shortest incident edge length (for internal constraints)
      and boundary distance.
    
    We use a conservative approach: for each circle, its radius is limited by:
    1. Distance to boundary (min(x, 1-x, y, 1-y))
    2. Half the distance to each neighboring circle (from Delaunay edges)
    
    To ensure disjoint interiors, we set radius_i = 0.5 * min over neighbors j of dist(i,j)
    but also capped by boundary distance. However, this might be too conservative.
    
    A better approach: use a relaxation where we start with small radii and grow them
    until they touch. But for simplicity and robustness, let's use the Delaunay-based
    constraint: for each point, radius = 0.5 * min(shortest edge incident to it, boundary distance * 2)
    
    Actually, let's think more carefully. If two circles i and j are neighbors in the
    Delaunay triangulation, they should be tangent or close. The constraint is:
    r_i + r_j <= dist(i, j)
    
    And for boundary:
    r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    We want to maximize sum of r_i. This is a linear program! Let's set it up.
    """
    n = len(points)
    if n == 0:
        return np.array([])
    
    # Build Delaunay triangulation
    tri = Delaunay(points)
    edges = _delaunay_edges(tri)
    
    # Build adjacency: for each point, list of neighbor indices
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    
    # For each point, compute boundary constraint
    boundary_dist = np.array([min(x, 1 - x, y, 1 - y) for x, y in points])
    
    # We'll use a simple iterative method: start with small radii, then grow them
    # subject to constraints. Use a water-filling approach.
    
    # Initialize radii to small value
    eps = 1e-6
    r = np.full(n, eps)
    
    # Iterate: for each point, try to increase its radius as much as possible
    # given current radii of neighbors. Repeat until convergence.
    
    max_iter = 200
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            # Max radius for point i given boundary
            max_r = boundary_dist[i]
            
            # Constraint from neighbors
            for j in adj[i]:
                d = np.linalg.norm(points[i] - points[j])
                # r_i + r_j <= d => r_i <= d - r_j
                max_r = min(max_r, d - r[j])
            
            # Also ensure non-negative
            max_r = max(max_r, eps)
            
            # If we can increase radius i
            if max_r > r[i] + 1e-9:
                r[i] = max_r
                changed = True
        
        if not changed:
            break
    
    # Final check: ensure all constraints are satisfied
    for i in range(n):
        # Boundary
        r[i] = min(r[i], boundary_dist[i])
        # Neighbors
        for j in adj[i]:
            d = np.linalg.norm(points[i] - points[j])
            if r[i] + r[j] > d:
                # Adjust the larger one
                excess = r[i] + r[j] - d
                if r[i] >= r[j]:
                    r[i] = max(0, r[i] - excess)
                else:
                    r[j] = max(0, r[j] - excess)
    
    return r

def _force_model_optimize(points, n_iter=500, lr=0.01):
    """
    Optimize center positions using a repulsion-based force model.
    Points are pushed apart if circles would overlap, and pulled toward
    a balanced configuration.
    """
    n = len(points)
    points = points.copy().astype(float)
    
    for it in range(n_iter):
        forces = np.zeros_like(points)
        
        # Compute current radii
        r = _compute_radii(points)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = points[i] - points[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-6:
                    d = 1e-6
                    d_vec = np.array([1e-6, 0.0])
                else:
                    d_vec = d_vec / d
                
                # If circles overlap or are too close, repel
                min_dist = r[i] + r[j]
                if d < min_dist * 1.1:
                    # Force proportional to overlap
                    force_mag = (min_dist - d) * 0.5
                    forces[i] += force_mag * d_vec
                    forces[j] -= force_mag * d_vec
        
        # Boundary forces: push away from boundaries if circle would exit
        for i in range(n):
            x, y = points[i]
            ri = r[i]
            if x < ri:
                forces[i, 0] += (ri - x) * 0.5
            if x > 1 - ri:
                forces[i, 0] -= (x - (1 - ri)) * 0.5
            if y < ri:
                forces[i, 1] += (ri - y) * 0.5
            if y > 1 - ri:
                forces[i, 1] -= (y - (1 - ri)) * 0.5
        
        # Update positions
        points += lr * forces
        
        # Clamp to [0,1]^2 with margin
        points = np.clip(points, 0.0, 1.0)
    
    return points

def construct():
    # Start with a random or grid-based initial configuration
    # Use a grid as starting point
    side = int(np.ceil(np.sqrt(N)))
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) < N:
                points.append([(i + 0.5) / side, (j + 0.5) / side])
    points = np.array(points[:N])
    
    # Add some randomness to break symmetry
    random.seed(42)
    for _ in range(3):
        # Try different random perturbations
        pts = points.copy()
        pts += np.random.uniform(-0.02, 0.02, size=pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
        # Optimize
        pts_opt = _force_model_optimize(pts, n_iter=300, lr=0.005)
        
        # Compute radii
        radii = _compute_radii(pts_opt)
        
        # Validate: check all constraints
        valid = True
        for i in range(N):
            x, y = pts_opt[i]
            ri = radii[i]
            if x < -1e-6 or x > 1 + 1e-6 or y < -1e-6 or y > 1 + 1e-6:
                valid = False
                break
            if ri < -1e-6:
                valid = False
                break
            # Boundary
            if x - ri < -1e-6 or x + ri > 1 + 1e-6 or y - ri < -1e-6 or y + ri > 1 + 1e-6:
                valid = False
                break
            # Pairwise
            for j in range(i + 1, N):
                d = np.linalg.norm(pts_opt[i] - pts_opt[j])
                if d < radii[i] + radii[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return np.column_stack([pts_opt, radii])
    
    # Fallback: use the grid with conservative radii
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cx, cy = (i + 0.5) / side, (j + 0.5) / side
                r = min(cx, 1 - cx, cy, 1 - cy) * 0.99
                cells.append([cx, cy, r])
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
