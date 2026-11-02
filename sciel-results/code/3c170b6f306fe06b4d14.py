"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Returns:
    inner: list of [x, y, angle_degrees] for N unit hexagons
    center: (x, y) of outer hexagon
    side: side length of outer hexagon
    angle: rotation of outer hexagon (degrees)
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Unit hexagon vertices (side 1, apothem sqrt(3)/2, circumradius 1)
    # Flat-top orientation: vertices at 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))] 
        for a in range(0, 360, 60)
    ])
    
    # Generate candidate center positions on a triangular lattice
    # Triangular lattice: spacing = sqrt(3) for unit hexagons (edge-to-edge distance)
    # Actually, for unit side hexagons, the distance between centers of adjacent 
    # non-overlapping hexagons in a honeycomb is sqrt(3)
    
    # Generate a dense grid of potential centers
    max_radius = 8.0
    spacing = 0.5  # Fine spacing for candidate centers
    
    # Generate lattice points
    candidates = []
    for i in range(int(-max_radius/spacing), int(max_radius/spacing) + 1):
        for j in range(int(-max_radius/spacing), int(max_radius/spacing) + 1):
            x = i * spacing
            y = j * spacing
            if math.sqrt(x*x + y*y) <= max_radius:
                candidates.append([x, y])
    
    # We need to place N=11 hexagons. Start with the 11 points closest to origin
    # But we should optimize the configuration
    
    # Strategy: Use a greedy approach or try known good configurations
    
    # Known good configuration for N=11: hexagonal arrangement
    # A hexagonal number: 1 + 6 + 12 = 19 for 3 rings
    # 1 + 6 = 7 for 2 rings
    # For 11, we can have 1 center + 6 around it + 4 more
    
    # Let's try several configurations and pick the best
    
    def hex_vertices(cx, cy, angle_deg=0.0):
        """Get vertices of unit hexagon centered at (cx, cy) rotated by angle_deg"""
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = base_vertices @ rot.T + np.array([cx, cy])
        return verts
    
    def outer_hexagon_side(centers, angles):
        """
        Compute minimum side length of outer hexagon that contains all inner hexagons.
        centers: list of (x, y)
        angles: list of rotation angles in degrees
        Returns: minimum side length and optimal outer hexagon rotation
        """
        # Get all vertices of all inner hexagons
        all_verts = []
        for i, (cx, cy) in enumerate(centers):
            verts = hex_vertices(cx, cy, angles[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # The outer hexagon has 6 symmetry axes at angles 0, 60, 120, 180, 240, 300
        # plus the optimal rotation theta
        
        # For a given outer hexagon rotation theta, the 6 outward normals are at
        # theta, theta+60, theta+120, theta+180, theta+240, theta+300
        # The apothem must be >= max projection of any vertex onto each normal
        
        def side_for_rotation(theta):
            """Compute required side for outer hexagon rotated by theta (radians)"""
            max_proj = 0
            for k in range(6):
                angle = theta + k * math.pi / 3
                nx, ny = math.cos(angle), math.sin(angle)
                # Project all vertices onto this normal
                projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                max_proj = max(max_proj, np.max(projs))
            # Apothem = max_proj, side = 2 * apothem / sqrt(3)
            return 2 * max_proj / math.sqrt(3)
        
        # Optimize over theta
        # The optimal theta is likely near one of the hexagon symmetry directions
        # Try a range of thetas
        best_side = float('inf')
        best_theta = 0
        for theta_deg in range(0, 30, 1):  # By symmetry, only need 0-30 degrees
            theta = math.radians(theta_deg)
            s = side_for_rotation(theta)
            if s < best_side:
                best_side = s
                best_theta = theta
        
        # Refine around best_theta
        for theta_deg in np.linspace(best_theta - math.pi/60, best_theta + math.pi/60, 20):
            s = side_for_rotation(theta_deg)
            if s < best_side:
                best_side = s
                best_theta = theta_deg
        
        return best_side, math.degrees(best_theta)
    
    # Configuration 1: Hexagonal packing
    # Center at origin, 6 around it, then 4 more
    
    # Ring 0: 1 hex at (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) from center
    # Ring 2: 12 hexagons at distance 2*sqrt(3) from center (but we only need 4 more)
    
    d1 = math.sqrt(3)  # Distance to first ring
    d2 = 2 * math.sqrt(3)  # Distance to second ring
    
    centers_hex = [(0.0, 0.0)]
    # First ring: 6 hexagons
    for k in range(6):
        angle = math.radians(60 * k)
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        centers_hex.append((x, y))
    
    # Second ring: 12 positions, pick 4 that minimize the bounding hexagon
    # The 12 positions are at:
    # 6 at distance 2*sqrt(3) at angles 0, 60, 120, 180, 240, 300
    # 6 at distance sqrt(3)*sqrt(3) = 3 at angles 30, 90, 150, 210, 270, 330
    # Wait, let me recalculate. The second ring of a hexagonal lattice:
    # Positions are at (2a, 0), (a, a*sqrt(3)), (-a, a*sqrt(3)), (-2a, 0), (-a, -a*sqrt(3)), (a, -a*sqrt(3))
    # where a = sqrt(3)
    # Plus (1.5a, 0.5a*sqrt(3)), etc.
    
    # Actually, the neighbors of the 6 first-ring hexagons that are not the center or other first-ring:
    # Each first-ring hex has 6 neighbors. One is the center, two are adjacent first-ring hexes.
    # So 3 new neighbors per first-ring hex, but some are shared.
    
    # Let me just enumerate all lattice points within distance 3*sqrt(3) and pick 11
    
    lattice_points = []
    a = math.sqrt(3)
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = a * (i + j/2.0)
            y = a * (j * math.sqrt(3) / 2.0)
            dist = math.sqrt(x*x + y*y)
            lattice_points.append((x, y, dist))
    
    lattice_points.sort(key=lambda p: p[2])
    
    # Take the 11 closest
    centers_lattice = [(p[0], p[1]) for p in lattice_points[:11]]
    
    # Try both configurations
    best_result = None
    
    # Configuration A: Lattice
    angles_zero = [0.0] * N
    side_a, theta_a = outer_hexagon_side(centers_lattice, angles_zero)
    if best_result is None or side_a < best_result[0]:
        best_result = (side_a, centers_lattice, [0.0] * N, theta_a)
    
    # Check deadline
    if time.time() > deadline - 20:
        side, centers, angles, theta = best_result
        inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
        return inner, (0.0, 0.0), side, theta
    
    # Configuration B: Try rotating individual hexagons to improve packing
    # For each hexagon, try a few rotation angles
    improved_centers = list(centers_lattice)
    improved_angles = [0.0] * N
    
    # Try rotating some hexagons
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        # Randomly perturb
        new_angles = improved_angles[:]
        for i in range(N):
            if np.random.random() < 0.3:
                new_angles[i] = np.random.uniform(0, 60)
        
        # Also try small perturbations of centers
        new_centers = []
        for i in range(N):
            dx = np.random.normal(0, 0.1)
            dy = np.random.normal(0, 0.1)
            new_centers.append((improved_centers[i][0] + dx, improved_centers[i][1] + dy))
        
        side_c, theta_c = outer_hexagon_side(new_centers, new_angles)
        
        if side_c < best_result[0] - 1e-6:
            best_result = (side_c, new_centers, new_angles, theta_c)
            improved_centers = new_centers
            improved_angles = new_angles
    
    side, centers, angles, theta = best_result
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    return inner, (0.0, 0.0), side, theta

# EVOLVE-BLOCK-END
