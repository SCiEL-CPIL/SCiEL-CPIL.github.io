# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, minimizing the side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        """
        Returns vertices of a regular hexagon.
        angle_deg is the angle of the first vertex (i=0) relative to x-axis.
        """
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def get_normals(vertices):
        """
        Returns unit outward normals for the edges of a convex polygon given in CCW order.
        """
        n = len(vertices)
        normals = np.zeros((n, 2))
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            edge = p2 - p1
            # Outward normal for CCW polygon: rotate edge by -90 deg
            # (x, y) -> (y, -x)
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normals[i] = normal / norm
        return normals

    def project_vertices(vertices, normal):
        """
        Projects vertices onto a normal vector, returns (min, max) dot products.
        """
        dots = vertices @ normal
        return np.min(dots), np.max(dots)

    def check_separation(hex1_verts, hex1_normals, hex2_verts, hex2_normals):
        """
        Returns True if hexagons are separated (disjoint), False if they overlap.
        Uses Separating Axis Theorem.
        """
        # Check normals of hex1
        for n in hex1_normals:
            min1, max1 = project_vertices(hex1_verts, n)
            min2, max2 = project_vertices(hex2_verts, n)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True # Separated
        
        # Check normals of hex2
        for n in hex2_normals:
            min1, max1 = project_vertices(hex1_verts, n)
            min2, max2 = project_vertices(hex2_verts, n)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True # Separated
        
        return False # Overlapping

    def objective(params):
        """
        params: array of 36 values (12 * (x, y, angle_deg))
        Returns: Minimal side length S of outer hexagon (centered at 0,0, angle 0) 
                 required to contain all inner hexagons, plus penalty if they overlap.
        """
        n_hex = 12
        hexes = []
        for i in range(n_hex):
            x = params[3*i]
            y = params[3*i+1]
            angle = params[3*i+2]
            verts = get_hex_vertices(x, y, 1.0, angle)
            normals = get_normals(verts)
            hexes.append((verts, normals))
        
        # Check separation
        overlap_penalty = 0.0
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                if not check_separation(hexes[i][0], hexes[i][1], hexes[j][0], hexes[j][1]):
                    # Calculate overlap depth roughly for penalty? 
                    # Just a large constant penalty is often enough for Nelder-Mead to push them apart
                    overlap_penalty += 10.0
        
        if overlap_penalty > 0:
            return overlap_penalty + 100.0 # Ensure it's worse than any valid S

        # Calculate required S for outer hexagon (centered at 0,0, angle 0)
        # Outer hexagon normals (for angle 0, vertex at (S,0)):
        # Vertices at 0, 60, 120, 180, 240, 300 degrees.
        # Edges are between these.
        # Normal to edge between 0 and 60: 
        # Edge vector: (cos60-1, sin60-0) = (-0.5, 0.866). 
        # Outward normal: (0.866, 0.5) -> angle 30 deg.
        # Normals are at 30, 90, 150, 210, 270, 330 degrees.
        outer_normals = []
        for k in range(6):
            theta = np.deg2rad(30 + 60*k)
            outer_normals.append(np.array([np.cos(theta), np.sin(theta)]))
        
        max_dist = 0.0
        for i in range(n_hex):
            verts, _ = hexes[i]
            for n in outer_normals:
                # Distance from origin to the plane defined by normal n and passing through vertex v
                # We need S such that all vertices v satisfy v . n <= S * sqrt(3) / 2
                # So S >= 2/sqrt(3) * v . n
                # Note: v . n can be negative, but we take max over all vertices and normals.
                # Since normals point out, and hex is centered at 0, we need max(v . n) for all v, n.
                dots = verts @ n
                max_dot = np.max(dots)
                if max_dot > 0:
                    required_s = (2.0 / np.sqrt(3.0)) * max_dot
                    if required_s > max_dist:
                        max_dist = required_s
        
        return max_dist

    # Initial guesses
    # 1. Seed configuration scaled
    seed_inner = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ]
    )
    # Scale seed to be tighter? The seed S=8.
    # Let's try to optimize from the seed positions directly.
    initial_params = seed_inner.flatten()
    
    # 2. Random hexagonal grid start
    # Centers in hex grid:
    # (0,0)
    # (sqrt(3), 0), (-sqrt(3), 0)
    # (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5) ...
    # Let's generate a compact cluster of 12
    hex_centers = []
    # Center
    hex_centers.append([0, 0, 0])
    # Ring 1 (6)
    for k in range(6):
        angle = np.deg2rad(60*k)
        hex_centers.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle), 0])
    # Ring 2 (first 5)
    # Positions in ring 2:
    # (2*sqrt(3), 0)
    # (sqrt(3), 3) -> dist sqrt(3+9)=sqrt(12)? No.
    # Let's just use the seed as the primary start, it's a valid packing.
    
    best_params = initial_params
    best_score = objective(initial_params)
    
    # Run Nelder-Mead optimization from multiple starts
    starts = [initial_params]
    
    # Add a few random perturbations
    for _ in range(5):
        noise = np.random.normal(0, 0.5, 36)
        starts.append(initial_params + noise)
        
    # Add a "tight" random start
    for _ in range(5):
        rand_params = np.random.uniform(-3, 3, 36)
        starts.append(rand_params)

    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract final result
    inner_hex_data = best_params.reshape(12, 3)
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_score

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
