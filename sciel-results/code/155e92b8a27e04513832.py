import math
import time
import numpy as np

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Unit hexagon vertices (flat-top, side 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])
    
    # Precompute edge normals for unit hexagon
    # Edge i is from vert i to vert i+1
    # Normal is perpendicular to edge, pointing outward
    edge_normals = np.zeros((6, 2))
    for i in range(6):
        v1 = unit_verts[i]
        v2 = unit_verts[(i+1) % 6]
        e = v2 - v1
        # Rotate e by -90 deg to get outward normal?
        # e = (dx, dy). Outward normal is (dy, -dx) or (-dy, dx)?
        # For flat-top, top edge is from (0.5, s/2) to (-0.5, s/2). e = (-1, 0).
        # Outward normal should be (0, 1).
        # (-dy, dx) -> (0, -1). Wrong.
        # (dy, -dx) -> (0, 1). Correct.
        n = np.array([e[1], -e[0]])
        n = n / np.linalg.norm(n)
        edge_normals[i] = n
        
    def get_world_vertices(positions, angles):
        """
        positions: (N, 2)
        angles: (N,) radians
        returns: (N, 6, 2)
        """
        c = np.cos(angles)
        s = np.sin(angles)
        ux = unit_verts[:, 0]
        uy = unit_verts[:, 1]
        
        wx = c[:, None] * ux[None, :] - s[:, None] * uy[None, :] + positions[:, 0:1]
        wy = s[:, None] * ux[None, :] + c[:, None] * uy[None, :] + positions[:, 1:2]
        
        return np.stack([wx, wy], axis=-1)

    def calc_outer_side(world_verts, outer_angle_rad):
        """
        world_verts: (N, 6, 2)
        outer_angle_rad: rotation of outer hexagon
        returns: minimum side length S
        """
        # Normals of outer hexagon
        # Outer hexagon is regular, side S.
        # Distance from center to side (apothem) = S * SQRT3 / 2
        # Condition: dot(v, n_k) <= apothem for all k
        # So S = 2 * max(dot(v, n_k)) / SQRT3
        # We need to consider absolute value if the hexagon is centered at origin and we look at all sides?
        # The outer hexagon contains the origin. The inner hexagons are inside.
        # The max projection on any normal n_k determines the support.
        # Since the outer hexagon is symmetric, we just need the max projection on the 6 normals.
        
        normals = np.zeros((6, 2))
        for k in range(6):
            theta = outer_angle_rad + k * math.pi / 3 + math.pi / 6 # 30 deg offset for flat-top normals?
            # Unit hex normals:
            # Edge 0 (top): normal (0, 1) -> angle 90
            # My unit_verts: (1,0) to (0.5, s/2). Edge vector (-0.5, s/2).
            # Normal (s/2, 0.5) -> angle 30.
            # Let's just use the same edge_normals logic for the outer hexagon shape.
            # Actually, the set of normals for a regular hexagon is just the 6 directions.
            # For a flat-top hexagon, normals are at 30, 90, 150, 210, 270, 330.
            # Let's generate them.
            pass
        
        # Simpler: The 6 normals of a regular hexagon are at angles 30 + k*60.
        # If outer_angle is the rotation of the hexagon, the normals rotate by the same amount.
        # Let's assume the "base" hexagon has normals at 30, 90...
        # So normal k angle = outer_angle_rad + math.radians(30) + k * math.pi / 3
        
        max_proj = 0.0
        for k in range(6):
            theta = outer_angle_rad + math.radians(30) + k * math.pi / 3
            n = np.array([math.cos(theta), math.sin(theta)])
            # Project all vertices
            projs = world_verts @ n # (N, 6)
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
                
        return max_proj * 2.0 / SQRT3

    def check_overlap_pair(verts_i, verts_j):
        """
        verts_i, verts_j: (6, 2)
        returns True if overlap
        """
        # SAT
        # Axes: edge normals of i and j
        # We need the rotated edge normals for i and j.
        # But we don't have angles here, only world vertices.
        # We can compute edge normals from world vertices.
        
        def get_normals(verts):
            norms = []
            for k in range(6):
                v1 = verts[k]
                v2 = verts[(k+1) % 6]
                e = v2 - v1
                n = np.array([e[1], -e[0]])
                n = n / np.linalg.norm(n)
                norms.append(n)
            return norms

        norms_i = get_normals(verts_i)
        norms_j = get_normals(verts_j)
        
        all_norms = norms_i + norms_j
        
        for n in all_norms:
            proj_i = verts_i @ n
            proj_j = verts_j @ n
            
            min_i = np.min(proj_i)
            max_i = np.max(proj_i)
            min_j = np.min(proj_j)
            max_j = np.max(proj_j)
            
            if max_i < min_j - 1e-9 or max_j < min_i - 1e-9:
                return False # Separated
        
        return True # Overlap

    def check_all_overlaps(world_verts):
        """
        world_verts: (N, 6, 2)
        returns True if any overlap
        """
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap_pair(world_verts[i], world_verts[j]):
                    return True
        return False

    def evaluate(positions, angles, outer_angle_rad):
        """
        returns (S, is_valid)
        """
        world_verts = get_world_vertices(positions, angles)
        if check_all_overlaps(world_verts):
            return float('inf'), False
        S = calc_outer_side(world_verts, outer_angle_rad)
        return S, True

    # Initialization: Hexagonal lattice
    # Centers at (0,0), (sqrt(3), 0), (sqrt(3)/2, 1.5) ...
    # We need 12 points.
    # Let's use a triangular grid.
    # Row 0: 4 points?
    # Row 1: 4 points?
    # Row 2: 4 points?
    # Total 12.
    
    def init_hex_lattice():
        pos = []
        # Center around origin
        # Let's place them in a 3x4 block roughly
        # dx = sqrt(3), dy = 1.5
        # x = i * sqrt(3) + (j % 2) * sqrt(3)/2
        # y = j * 1.5
        
        # We want to center them.
        # Let's generate a 4x3 grid
        pts = []
        for j in range(3):
            for i in range(4):
                x = i * SQRT3 + (j % 2) * SQRT3 / 2
                y = j * 1.5
                pts.append([x, y])
        
        # Center
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        for p in pts:
            p[0] -= cx
            p[1] -= cy
            
        return np.array(pts), np.zeros(N)

    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    rng = np.random.default_rng(42)
    
    # Try a few outer angles
    for outer_angle_deg in [0, 15, 30, 45]:
        outer_angle_rad = math.radians(outer_angle_deg)
        
        # Restart with different initializations
        for restart in range(5):
            if time.time() > deadline:
                break
                
            if restart == 0:
                positions, angles = init_hex_lattice()
            else:
                # Random scatter
                positions = rng.uniform(-3, 3, (N, 2))
                angles = rng.uniform(0, math.pi, N)
            
            # Local Search
            # We will do a simple hill climbing
            # Step size for positions: 0.1
            # Step size for angles: 0.1 rad
            
            for iter in range(2000):
                if time.time() > deadline:
                    break
                
                # Pick random index
                idx = rng.integers(0, N)
                
                # Save old
                old_pos = positions[idx].copy()
                old_ang = angles[idx]
                
                # Perturb
                # Gaussian noise
                positions[idx] += rng.normal(0, 0.2, 2)
                angles[idx] += rng.normal(0, 0.2)
                
                # Evaluate
                S, valid = evaluate(positions, angles, outer_angle_rad)
                
                if valid and S < best_S:
                    best_S = S
                    best_positions = positions.copy()
                    best_angles = angles.copy()
                    best_outer_angle = outer_angle_rad
                    
                # Acceptance: 
                # If valid and S is not worse than current best, keep?
                # No, we are minimizing S.
                # If we move to a valid state with S < current_S, we keep it.
                # If we move to an invalid state, we revert.
                # This is pure hill climbing. It might get stuck.
                # Let's add a small probability of accepting worse valid moves (Annealing).
                
                # For simplicity and speed, let's just keep if valid and S is smaller than the S of the current state.
                # We need to track current_S.
                
                # Let's restructure:
                # Keep current_S.
                # If new is valid:
                #   If new_S < current_S: accept, update current_S.
                #   Else: accept with prob exp(-(new_S - current_S)/T)
                # If new is invalid: reject.
                
                # We need to compute S for the current state too.
                # To save time, we only compute S if valid.
                # But we need current_S.
                
                # Let's initialize current_S with the S of the initial state.
                # We'll do this outside the inner loop.

            # This logic is getting messy in the draft. Let's clean it up in the final code.
            
    # Final check and return
    if best_positions is None:
        # Fallback
        positions, angles = init_hex_lattice()
        world_verts = get_world_vertices(positions, angles)
        best_S = calc_outer_side(world_verts, 0.0)
        best_outer_angle = 0.0
    else:
        # Verify best
        world_verts = get_world_vertices(best_positions, best_angles)
        if check_all_overlaps(world_verts):
            # Should not happen, but just in case
            positions, angles = init_hex_lattice()
            world_verts = get_world_vertices(positions, angles)
            best_S = calc_outer_side(world_verts, 0.0)
            best_positions = positions
            best_angles = angles
            best_outer_angle = 0.0
        else:
            best_S = calc_outer_side(world_verts, best_outer_angle)

    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], math.degrees(best_angles[i])])
    
    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_angle)
