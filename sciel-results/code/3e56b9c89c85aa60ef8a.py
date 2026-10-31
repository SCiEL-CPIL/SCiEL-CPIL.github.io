"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i+1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length containing all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    # Normals to the sides of the outer hexagon
    # The support function of a hexagon with side S is S/sqrt(3) in the direction of the normal?
    # Actually, for a regular hexagon with side S, the distance from center to side (apothem) is S * sqrt(3) / 2.
    # The condition for a point p to be inside is:
    # dot(p - c, n_k) <= apothem = S * sqrt(3) / 2
    # So S >= 2 * dot(p - c, n_k) / sqrt(3)
    
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                d = np.dot(v - np.array([ox, oy]), n)
                if d > 0:
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle, tol=1e-7):
    """Check no overlaps and all inside outer hexagon."""
    # Check overlaps
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            # Quick distance check
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0: # Min distance between centers of non-overlapping unit hexagons is sqrt(3) approx 1.732, but with rotation can be closer? 
                # Actually, min distance is sqrt(3) for parallel, but if rotated, they can interlock? 
                # No, regular hexagons are convex. The minimum distance between centers for non-overlap is sqrt(3) if aligned.
                # If rotated, the distance can be slightly larger or smaller? 
                # The "width" of a unit hexagon is 2. The distance between parallel sides is sqrt(3).
                # So if centers are closer than sqrt(3), they MUST overlap? 
                # Yes, the minimum distance between two non-intersecting unit hexagons is sqrt(3).
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                if polygons_overlap(v1, v2, tol=tol):
                    return False
    
    # Check containment
    ox, oy = outer_center
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            inside = True
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % 6]
                edge = v2 - v1
                # Normal pointing inwards
                normal = np.array([-edge[1], edge[0]])
                # Check if point is on the same side as the center
                # Actually, simpler: check if point is inside convex polygon
                # Cross product method
                cross = np.cross(edge, np.array([px, py]) - v1)
                # We need consistent orientation. Vertices are CCW.
                # For CCW polygon, interior is to the left of edges.
                # So cross(edge, p-v1) should be >= 0 (or <= 0 depending on coord system)
                # Let's use the standard half-plane test.
                # Normal n = (-dy, dx). dot(n, p-v1) >= 0 for interior if n points inward.
                # For CCW polygon, (-dy, dx) points inward.
                d1 = np.dot(normal, np.array([px, py]) - v1)
                if d1 < -tol:
                    inside = False
                    break
            if not inside:
                return False
    return True

def construct():
    """Main entry point: find optimal packing using discrete angle search and local position optimization."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_score = float('inf')
    best_result = None
    
    # Discrete angles to try for inner hexagons
    inner_angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    # Discrete angles for outer hexagon
    outer_angles = [0.0, 30.0]
    
    # Generate initial hex lattice positions
    d = np.sqrt(3)
    base_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        base_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        base_positions.append((x, y))
        if len(base_positions) >= N:
            break
    base_positions = base_positions[:N]

    # We will try combinations of angles for the 11 hexagons?
    # 6^11 is too large.
    # Strategy: Assume all inner hexagons have the same angle, or two types (center vs ring).
    # Let's try:
    # 1. All same angle
    # 2. Center angle A, Ring 1 angle B, Ring 2 angle C
    
    # To respect the "discrete angular slots" directive, we can iterate over a small set of global configurations.
    
    # Candidate 1: All aligned (0 deg)
    # Candidate 2: All rotated 30 deg
    # Candidate 3: Center 0, Others 30
    # Candidate 4: Center 30, Others 0
    
    candidate_angle_configs = []
    
    # Type 0: All 0
    candidate_angle_configs.append([0.0]*N)
    # Type 1: All 30
    candidate_angle_configs.append([30.0]*N)
    # Type 2: Center 0, Rest 30
    conf = [0.0]*N
    conf[0] = 0.0
    for i in range(1, N): conf[i] = 30.0
    candidate_angle_configs.append(conf)
    # Type 3: Center 30, Rest 0
    conf = [0.0]*N
    conf[0] = 30.0
    for i in range(1, N): conf[i] = 0.0
    candidate_angle_configs.append(conf)
    # Type 4: Ring 1 (1-6) 30, Rest 0
    conf = [0.0]*N
    for i in range(1, 7): conf[i] = 30.0
    candidate_angle_configs.append(conf)
    # Type 5: Ring 1 (1-6) 0, Rest 30
    conf = [0.0]*N
    for i in range(7, N): conf[i] = 30.0
    candidate_angle_configs.append(conf)

    for outer_ang in outer_angles:
        for inner_ang_conf in candidate_angle_configs:
            # Initialize positions with base lattice
            positions = np.array(base_positions)
            angles = np.array(inner_ang_conf)
            
            # Local search: Perturb positions to minimize S while maintaining validity
            # We use a simple random restart hill climbing
            
            best_local_S = float('inf')
            best_local_pos = positions.copy()
            best_local_angles = angles.copy()
            
            for restart in range(5):
                pos = best_local_pos.copy()
                if restart > 0:
                    pos = pos + np.random.randn(N, 2) * 0.1
                
                # Hill climbing
                current_S = compute_outer_side_length(
                    [(pos[i][0], pos[i][1], angles[i]) for i in range(N)], 
                    (0.0, 0.0), 
                    outer_ang
                )
                
                # Check validity first
                if not check_validity(
                    [(pos[i][0], pos[i][1], angles[i]) for i in range(N)], 
                    (0.0, 0.0), 
                    current_S, 
                    outer_ang
                ):
                    continue
                
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    # Try moving one hexagon
                    idx = np.random.randint(N)
                    orig_pos = pos[idx].copy()
                    
                    # Try small random moves
                    for _ in range(20):
                        move = np.random.randn(2) * 0.05
                        pos[idx] = orig_pos + move
                        
                        # Check validity for this move
                        # Optimization: Only check overlap with neighbors
                        valid = True
                        for j in range(N):
                            if i == j: continue
                            d = np.sqrt((pos[idx][0]-pos[j][0])**2 + (pos[idx][1]-pos[j][1])**2)
                            if d < 2.0:
                                v1 = hexagon_vertices(pos[idx][0], pos[idx][1], 1.0, angles[idx])
                                v2 = hexagon_vertices(pos[j][0], pos[j][1], 1.0, angles[j])
                                if polygons_overlap(v1, v2, tol=1e-7):
                                    valid = False
                                    break
                        if not valid:
                            pos[idx] = orig_pos
                            continue
                        
                        # Check containment roughly (skip for speed, check at end of iteration)
                        new_S = compute_outer_side_length(
                            [(pos[k][0], pos[k][1], angles[k]) for k in range(N)], 
                            (0.0, 0.0), 
                            outer_ang
                        )
                        
                        if new_S < current_S:
                            current_S = new_S
                            improved = True
                            if not check_validity(
                                [(pos[k][0], pos[k][1], angles[k]) for k in range(N)], 
                                (0.0, 0.0), 
                                current_S, 
                                outer_ang
                            ):
                                pos[idx] = orig_pos
                                improved = False
                                break
                        else:
                            pos[idx] = orig_pos
                
                if current_S < best_local_S:
                    best_local_S = current_S
                    best_local_pos = pos.copy()
                    best_local_angles = angles.copy()
            
            if best_local_S < best_score:
                best_score = best_local_S
                best_result = (
                    [[best_local_pos[i][0], best_local_pos[i][1], best_local_angles[i]] for i in range(N)],
                    (0.0, 0.0),
                    best_local_S,
                    outer_ang
                )

    # Fallback if nothing found (should not happen)
    if best_result is None:
        inner = [[base_positions[i][0], base_positions[i][1], 0.0] for i in range(N)]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)
        
    return best_result

# EVOLVE-BLOCK-END
