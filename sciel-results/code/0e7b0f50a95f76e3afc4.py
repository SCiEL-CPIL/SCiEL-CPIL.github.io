import numpy as np
import time
import random
import math
from scipy.spatial import ConvexHull

N = 12
TIME_LIMIT = 140.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment of all inner vertices in outer hexagon
    # For each inner vertex, check if it's inside the outer hexagon
    # A convex polygon contains point P if P is to the left (or on) of every edge
    for i in range(len(inner_hexes)):
        for p in inner_verts[i]:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                # Cross product of (b-a) and (p-a)
                cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
    
    # Check disjointness of inner hexagons
    # Two convex polygons are disjoint if there exists a separating axis
    # We can check all edges of both polygons
    def _separating_axis_exists(v1, v2, eps=1e-7):
        n1, n2 = len(v1), len(v2)
        # Try all edges of v1 and v2 as potential separating axes
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            # Normal to edge p1-p2
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            # Project all vertices of both polygons onto this normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # If intervals don't overlap, we have separation
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        for i in range(n2):
            p1 = v2[i]
            p2 = v2[(i+1)%n2]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        return False
    
    for i in range(len(inner_hexes)):
        for j in range(i+1, len(inner_hexes)):
            if _separating_axis_exists(inner_verts[i], inner_verts[j], eps=1e-7):
                continue  # They are disjoint
            else:
                return False, "overlap"
    
    return True, "valid"

def _compute_outer_side(centers, inner_angle_deg, outer_center=(0.0, 0.0), outer_angle_deg=0.0):
    """
    Compute the minimum outer side length such that all inner hexagons 
    (at given centers with given angle) fit inside the outer hexagon.
    """
    inner_verts_list = []
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i], 1.0, inner_angle_deg)
        inner_verts_list.append(verts)
    
    # Collect all vertices of all inner hexagons
    all_verts = np.vstack(inner_verts_list)
    
    # The outer hexagon has center outer_center, angle outer_angle_deg
    # For a regular hexagon with side s, center C, angle A, the distance from center 
    # to each edge (apothem) is s * sqrt(3) / 2
    # The 6 outward normals are at angles A + k * 60 degrees for k = 0..5
    # A point P is inside if dot(P - C, n_k) <= s * sqrt(3) / 2 for all k
    # So s >= max_k dot(P - C, n_k) * 2 / sqrt(3)
    
    theta_outer = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = theta_outer + k * (np.pi / 3)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    
    cx, cy = outer_center
    all_verts = all_verts - np.array([cx, cy])
    
    max_proj = -np.inf
    for n in normals:
        projs = np.dot(all_verts, n)
        max_proj = max(max_proj, np.max(projs))
    
    if max_proj < 0:
        # All vertices are on the negative side of all normals? That means they're all at center?
        # Actually, if max_proj < 0, it means the hexagons are entirely in the negative half-plane
        # for all 6 directions, which is impossible unless they're at the center and small enough.
        # But side 1 hexagons have radius 1, so they'll always have positive projections somewhere.
        # Actually, the 6 normals point outward, so if a point is inside, its projection onto 
        # outward normal should be <= apothem. If max_proj is negative, it means the points are 
        # on the opposite side, which means the hexagon center might be far away.
        # For safety, use max(0, max_proj)
        max_proj = 0.0
    
    side = max_proj * 2.0 / np.sqrt(3.0)
    return side

def _simulate_annealing(centers0, angle0, T_start=2.0, T_end=0.01, steps_per_temp=50, 
                        time_deadline=None):
    """
    Simulated Annealing to optimize positions and angle.
    Returns (best_centers, best_angle, best_side).
    """
    best_centers = centers0.copy()
    best_angle = angle0
    best_side = _compute_outer_side(best_centers, best_angle)
    
    current_centers = centers0.copy()
    current_angle = angle0
    current_side = best_side
    
    T = T_start
    total_steps = 2000
    num_temps = total_steps // steps_per_temp
    
    for t in range(num_temps):
        for s in range(steps_per_temp):
            if time_deadline is not None and time.time() > time_deadline:
                return best_centers, best_angle, best_side
            
            # Random move: pick one hexagon and move it
            idx = random.randint(0, N-1)
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            
            new_centers = current_centers.copy()
            new_centers[idx] = [current_centers[idx][0] + dx, current_centers[idx][1] + dy]
            
            # Maybe also change angle
            new_angle = current_angle
            if random.random() < 0.1:
                new_angle = current_angle + random.gauss(0, 5.0)
            
            # Compute new side
            new_side = _compute_outer_side(new_centers, new_angle)
            
            # Check if valid
            inner_hexes = [[new_centers[i][0], new_centers[i][1], new_angle] for i in range(N)]
            is_valid, _ = _check_validity(inner_hexes, new_side, (0.0, 0.0), new_angle)
            
            if not is_valid:
                # Penalty for invalid
                new_side_penalty = new_side + 5.0
            else:
                new_side_penalty = new_side
            
            delta = new_side_penalty - current_side
            
            # Acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_centers = new_centers
                current_angle = new_angle
                current_side = new_side_penalty
                
                if is_valid and new_side < best_side:
                    best_centers = new_centers.copy()
                    best_angle = new_angle
                    best_side = new_side
        
        # Cooling
        T *= 0.95
        if T < T_end:
            T = T_end
    
    return best_centers, best_angle, best_side

def _local_refine(centers, angle, steps=100, step_size=0.1):
    """
    Simple local refinement: try small moves in each direction.
    """
    best_centers = centers.copy()
    best_angle = angle
    best_side = _compute_outer_side(best_centers, best_angle)
    
    for step in range(steps):
        improved = False
        for i in range(N):
            for dx, dy in [(-step_size,0),(step_size,0),(0,-step_size),(0,step_size),
                          (-step_size,-step_size),(step_size,-step_size),
                          (-step_size,step_size),(step_size,step_size)]:
                new_centers = best_centers.copy()
                new_centers[i] = [best_centers[i][0]+dx, best_centers[i][1]+dy]
                new_side = _compute_outer_side(new_centers, best_angle)
                
                inner_hexes = [[new_centers[j][0], new_centers[j][1], best_angle] for j in range(N)]
                is_valid, _ = _check_validity(inner_hexes, new_side, (0.0, 0.0), best_angle)
                
                if is_valid and new_side < best_side - 1e-10:
                    best_centers = new_centers
                    best_side = new_side
                    improved = True
        
        # Try angle changes
        for dangle in [-step_size, step_size]:
            new_angle = best_angle + dangle
            new_side = _compute_outer_side(best_centers, new_angle)
            inner_hexes = [[best_centers[j][0], best_centers[j][1], new_angle] for j in range(N)]
            is_valid, _ = _check_validity(inner_hexes, new_side, (0.0, 0.0), new_angle)
            
            if is_valid and new_side < best_side - 1e-10:
                best_angle = new_angle
                best_side = new_side
                improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 0.001:
                break
    
    return best_centers, best_angle, best_side

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Initial configuration: triangular lattice
    s3 = np.sqrt(3)
    centers_init = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * s3 + 0.5 * j * s3
            y = j * 1.5
            centers_init.append((x, y))
    
    # Sort by distance from origin and take 12
    centers_init.sort(key=lambda p: p[0]**2 + p[1]**2)
    centers_init = np.array(centers_init[:12])
    angle_init = 0.0
    
    # Run simulated annealing with multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Perturb initial configuration
        if restart == 0:
            centers = centers_init.copy()
            angle = angle_init
        else:
            centers = centers_init.copy()
            for i in range(N):
                centers[i] = [centers[i][0] + random.gauss(0, 2.0), 
                             centers[i][1] + random.gauss(0, 2.0)]
            angle = random.uniform(0, 60)
        
        # Run SA
        result_centers, result_angle, result_side = _simulate_annealing(
            centers, angle, 
            T_start=3.0, T_end=0.01, 
            steps_per_temp=30,
            time_deadline=deadline - 15
        )
        
        if result_side < best_side:
            best_side = result_side
            best_center = (0.0, 0.0)
            best_angle = result_angle
            best_inner = [[result_centers[i][0], result_centers[i][1], result_angle] 
                         for i in range(N)]
        
        # Local refinement
        if time.time() < deadline - 10:
            ref_centers, ref_angle, ref_side = _local_refine(
                result_centers, result_angle, steps=50, step_size=0.2
            )
            if ref_side < best_side:
                best_side = ref_side
                best_center = (0.0, 0.0)
                best_angle = ref_angle
                best_inner = [[ref_centers[i][0], ref_centers[i][1], ref_angle] 
                             for i in range(N)]
    
    # Final verification
    if best_inner is not None:
        is_valid, _ = _check_validity(best_inner, best_side, best_center, best_angle)
        if is_valid:
            return (best_inner, best_center, best_side, best_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
