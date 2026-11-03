"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # wall-clock limit
    
    # Regular hexagon with side length 1, flat-top orientation (angle=0)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # For a regular hexagon of side s, vertices are at distance s from center
    hex_vertices_0 = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # The 6 outward normals for a flat-top hexagon (perpendicular to edges)
    # Edge normals point in directions 30, 90, 150, 210, 270, 330 degrees
    # Actually, for containment in a regular hexagon of side S centered at origin,
    # the condition is that for each of the 6 directions (normals to the 6 edges),
    # the support function of the contained set must be <= S * cos(30 deg) = S * sqrt(3)/2
    # Wait, let me reconsider.
    
    # A regular hexagon of side S centered at origin with flat top:
    # The distance from center to each edge (apothem) is S * sqrt(3)/2
    # The 6 edge normals are at angles: 0, 60, 120, 180, 240, 300? No.
    # Let's think carefully.
    
    # For a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 (pointy-top),
    # the edges are between consecutive vertices. The normal to the edge between
    # vertex at angle 0 and vertex at angle 60 points in direction 30 degrees.
    
    # For flat-top hexagon (vertices at 30, 90, 150, 210, 270, 330),
    # the normals are at 0, 60, 120, 180, 240, 300.
    
    # Let's use the standard: a regular hexagon of side S centered at origin.
    # If we orient it so that one pair of edges is horizontal (flat-top),
    # the apothem (inradius) is S * sqrt(3)/2.
    # The 6 outward unit normals are at angles: 0, 60, 120, 180, 240, 300 degrees.
    # Containment condition: for each normal direction n_i, 
    #   max_{p in contained_set} n_i · p <= S * sqrt(3)/2
    # and min_{p in contained_set} n_i · p >= -S * sqrt(3)/2
    
    # Actually, for a convex set to be inside a regular hexagon of side S centered at origin
    # with the above orientation, we need:
    # For each of the 6 directions (the 6 edge normals), the projection of the set
    # onto that direction must lie within [-S*sqrt(3)/2, S*sqrt(3)/2].
    
    # So the minimal S is determined by:
    # S = (2 / sqrt(3)) * max over all 12 half-spaces of |extreme projection|
    # where the 12 half-spaces correspond to the 6 normals and their opposites.
    
    # The 6 normal directions (unit vectors) for flat-top hexagon:
    normal_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For a unit hexagon with center at c and rotation angle theta,
    # its vertices are c + R(theta) @ hex_vertices_0
    # where R(theta) is the rotation matrix.
    
    def hex_vertices(center, theta):
        """Return vertices of unit hexagon with given center and rotation angle (radians)."""
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return center + R @ hex_vertices_0.T
    
    def compute_min_side(centers, angles):
        """
        Given centers (N, 2) and angles (N,) in radians, compute the minimal outer side length.
        The outer hexagon is centered at origin, flat-top orientation.
        """
        # For each inner hexagon, compute its 6 vertices
        # Then find the max projection in each of the 6 normal directions
        max_projs = np.zeros(6)
        
        for i in range(N):
            verts = hex_vertices(centers[i], angles[i])  # shape (2, 6)
            # Project onto each normal
            projs = normals.T @ verts  # shape (6, 6)
            # For each normal direction, take the max and min
            max_projs = np.maximum(max_projs, np.max(projs, axis=1))
            min_projs = np.min(projs, axis=1)
            # We need |min_projs| too, since the constraint is both upper and lower bound
            max_projs = np.maximum(max_projs, -min_projs)
        
        # The minimal S satisfies S * sqrt(3)/2 >= max_projs[i] for all i
        # So S = (2/sqrt(3)) * max(max_projs)
        return (2.0 / np.sqrt(3)) * np.max(max_projs)
    
    # We want to minimize the outer side length.
    # Variables: positions (2N) and angles (N) of the 11 hexagons.
    # Total: 3N = 33 variables.
    
    # Initial guess: place hexagons in a hexagonal packing pattern
    # For unit hexagons in hexagonal close packing, the distance between centers
    # is sqrt(3) (for flat-top orientation with adjacent hexagons sharing edges)
    # Actually, for two unit hexagons sharing an edge, center-to-center distance is sqrt(3).
    
    # Let's start with a simple linear arrangement or small cluster
    # Hexagonal lattice: centers at (sqrt(3)*i + 0.5*sqrt(3)*j, 1.5*j) for appropriate i,j
    # Actually, for flat-top hexagons, the lattice vectors are:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 1.5)
    
    # Let's place 11 hexagons in a small cluster
    # Center + 6 around it + 4 more = 11
    # Or just use a linear chain for simplicity first
    
    def initial_guess():
        """Generate an initial configuration."""
        centers = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Hexagonal close packing positions
        # Center at origin, 6 neighbors at distance sqrt(3), etc.
        sqrt3 = np.sqrt(3)
        positions = [
            (0, 0),
            (sqrt3, 0),
            (sqrt3/2, 1.5),
            (-sqrt3/2, 1.5),
            (-sqrt3, 0),
            (-sqrt3/2, -1.5),
            (sqrt3/2, -1.5),
            # Next ring
            (3*sqrt3/2, 1.5),
            (0, 3),
            (-3*sqrt3/2, 1.5),
            (-3*sqrt3/2, -1.5)
        ]
        
        for i in range(N):
            centers[i] = positions[i]
        
        return centers, angles
    
    def objective(params):
        """Objective function: minimize outer side length."""
        centers = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        return compute_min_side(centers, angles)
    
    # Start with initial guess
    centers0, angles0 = initial_guess()
    params0 = np.concatenate([centers0.flatten(), angles0])
    
    best_params = params0.copy()
    best_side = objective(params0)
    
    # Try a few different initial configurations
    # and use local optimization
    
    # Since the objective is non-smooth (due to max), we'll use Nelder-Mead
    # which doesn't require gradients
    
    # Try multiple starting points
    start_points = []
    
    # Starting point 1: hexagonal cluster
    start_points.append(params0.copy())
    
    # Starting point 2: linear chain
    centers_lin = np.zeros((N, 2))
    for i in range(N):
        centers_lin[i] = [i * np.sqrt(3), 0]
    angles_lin = np.zeros(N)
    start_points.append(np.concatenate([centers_lin.flatten(), angles_lin]))
    
    # Starting point 3: 2-row arrangement
    centers_2row = np.zeros((N, 2))
    sqrt3 = np.sqrt(3)
    for i in range(N):
        if i < 6:
            centers_2row[i] = [i * sqrt3, 0]
        else:
            centers_2row[i] = [(i - 6) * sqrt3 + sqrt3/2, 1.5]
    angles_2row = np.zeros(N)
    start_points.append(np.concatenate([centers_2row.flatten(), angles_2row]))
    
    # Starting point 4: random
    np.random.seed(42)
    centers_rand = np.random.randn(N, 2) * 2
    angles_rand = np.random.randn(N) * np.pi / 6
    start_points.append(np.concatenate([centers_rand.flatten(), angles_rand]))
    
    # Optimize from each starting point
    for sp in start_points:
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            result = minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerances
    if time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final configuration
    centers_final = best_params[:2*N].reshape(N, 2)
    angles_final = best_params[2*N:3*N]
    
    # Convert angles to degrees
    angles_deg = np.degrees(angles_final)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(centers_final[i, 0]), float(centers_final[i, 1]), float(angles_deg[i])])
    
    # Outer hexagon: centered at origin, side length = best_side, rotation = 0
    return inner, (0.0, 0.0), float(best_side), 0.0
# EVOLVE-BLOCK-END
