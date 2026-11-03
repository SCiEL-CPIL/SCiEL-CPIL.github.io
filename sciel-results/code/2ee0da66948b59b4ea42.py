"""
Strategy: Simulated Annealing with Rigid Body Clustering for Hexagon Packing.

This approach abandons the fixed topology constraint of the seed program.
Instead, it treats the 11 unit hexagons as a deformable cluster.
Key innovations:
1. Hard constraint enforcement: Moves causing overlap are rejected, keeping the search
   on the manifold of valid non-overlapping configurations.
2. Rigid Body Moves: Occasionally, the entire cluster is rotated and translated.
   This allows the global shape of the packing to adapt to the outer hexagon's
   boundaries, which is often more effective than tweaking individual hexagons
   when the cluster is tightly packed.
3. Simulated Annealing: Uses a cooling schedule to escape local minima.
   The "temperature" controls the acceptance of worse solutions (larger outer side).
"""

import time
import numpy as np
import math

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin for 120s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(inner_positions, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin
    # For a regular hexagon with side S centered at origin, aligned with axes:
    # The constraint for a point (lx, ly) to be inside is:
    # |lx| <= S
    # |lx + ly/sqrt3| <= S
    # |lx - ly/sqrt3| <= S
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _check_overlap(pos, ang, tol=1e-7):
    """
    Check if any two inner hexagons overlap.
    Uses Separating Axis Theorem (SAT) for convex polygons.
    Returns True if overlap exists, False otherwise.
    """
    verts_list = []
    for i in range(N):
        verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        verts_list.append(verts)
    
    for i in range(N):
        for j in range(i+1, N):
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            # Quick check: distance between centers
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            # Minimum distance between centers of two touching unit hexagons is 2*sqrt(3)
            # If dist < 2*sqrt(3) - epsilon, they might overlap.
            # But orientation matters. Use SAT for accuracy.
            
            axes = []
            # Normals of edges of v1
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
            
            # Normals of edges of v2
            for k in range(6):
                p1 = v2[k]
                p2 = v2[(k+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
            
            overlap = True
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                    overlap = False
                    break
            
            if overlap:
                return True
                
    return False

def _initial_configuration():
    """
    Generate a valid initial configuration.
    We use the 'flower' pattern as it is known to be valid and reasonably compact.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2 (4 hexagons)
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6] # 30, 150, 210, 330
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

def construct():
    global START_TIME, DEADLINE
    
    # Initialize
    pos, ang = _initial_configuration()
    best_pos = pos.copy()
    best_ang = ang.copy()
    
    # Calculate initial outer side
    # We fix outer center at (0,0) and angle at 0 for simplicity, 
    # as the relative orientation can be handled by rotating the inner cluster.
    # However, allowing the outer hexagon to rotate might help? 
    # The problem says "outer hexagon center (x, y), side length and rotation".
    # But since the inner cluster can rotate, fixing the outer to 0 is WLOG 
    # if we allow the inner cluster to rotate freely.
    # Wait, the score is the side length. The outer rotation doesn't change the side length 
    # required for a given set of points if we can also rotate the points?
    # Actually, the "minimum outer side length" calculation `_min_outer_side` 
    # assumes the outer hexagon is aligned with the axes (angle 0).
    # If we allow the outer hexagon to rotate, we are effectively allowing the 
    # "frame" of measurement to rotate. This is equivalent to rotating the 
    # inner cluster in the fixed frame.
    # So, we can fix outer_angle = 0 and outer_center = (0,0) WLOG, 
    # provided we allow the inner cluster to rotate.
    
    current_side = _min_outer_side(pos, ang, (0.0, 0.0), 0.0)
    best_side = current_side
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    steps = 0
    max_steps = 50000
    
    # Precompute some constants
    sqrt3 = np.sqrt(3)
    
    while time.time() < DEADLINE and steps < max_steps:
        steps += 1
        
        # Generate a new candidate configuration
        new_pos = pos.copy()
        new_ang = ang.copy()
        
        move_type = np.random.rand()
        
        if move_type < 0.1:
            # Rigid Body Move: Rotate and Translate the entire cluster
            # This is crucial for fitting the cluster into the hexagon
            d_rot = np.random.normal(0, 0.1)
            d_tx = np.random.normal(0, 0.5)
            d_ty = np.random.normal(0, 0.5)
            
            # Rotate all positions around the centroid
            cx = np.mean(pos[:, 0])
            cy = np.mean(pos[:, 1])
            
            # Translate to origin
            pts = new_pos - np.array([cx, cy])
            
            # Rotate
            c, s = np.cos(d_rot), np.sin(d_rot)
            rot_matrix = np.array([[c, -s], [s, c]])
            pts = pts @ rot_matrix.T
            
            # Translate back and add new translation
            new_pos = pts + np.array([cx, cy]) + np.array([d_tx, d_ty])
            
            # Rotate all angles
            new_ang = (ang + np.degrees(d_rot)) % 360.0
            
        elif move_type < 0.7:
            # Local Move: Perturb a single hexagon
            i = np.random.randint(N)
            d_x = np.random.normal(0, 0.2)
            d_y = np.random.normal(0, 0.2)
            d_a = np.random.normal(0, 1.0)
            
            new_pos[i, 0] += d_x
            new_pos[i, 1] += d_y
            new_ang[i] += d_a
            
        else:
            # Pair Move: Perturb two adjacent hexagons slightly to maintain contact
            i = np.random.randint(N)
            j = (i + 1) % N # Simple neighbor heuristic, not topologically accurate
            # For better results, we could use a distance-based neighbor check,
            # but for speed, we just pick two random ones and nudge them towards each other
            d_x = np.random.normal(0, 0.1)
            d_y = np.random.normal(0, 0.1)
            
            new_pos[i, 0] += d_x
            new_pos[i, 1] += d_y
            new_pos[j, 0] -= d_x * 0.5
            new_pos[j, 1] -= d_y * 0.5
            
        # Check validity (no overlap)
        if _check_overlap(new_pos, new_ang):
            # Reject if overlap
            continue
            
        # Calculate new side length
        new_side = _min_outer_side(new_pos, new_ang, (0.0, 0.0), 0.0)
        
        # Acceptance criterion
        delta = new_side - current_side
        if delta < 0:
            # Better solution
            pos = new_pos
            ang = new_ang
            current_side = new_side
            
            if current_side < best_side:
                best_pos = pos.copy()
                best_ang = ang.copy()
                best_side = current_side
        else:
            # Worse solution, accept with probability exp(-delta / T)
            if T > T_min:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    pos = new_pos
                    ang = new_ang
                    current_side = new_side
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If T is very low and we haven't improved in a while, reheat slightly?
        # For simplicity, we just let it cool.
        
    # Finalize
    # Ensure the best solution is valid
    if not _check_overlap(best_pos, best_ang):
        # Fallback to initial
        pos_f, ang_f = _initial_configuration()
        best_pos = pos_f
        best_ang = ang_f
        best_side = _min_outer_side(best_pos, best_ang, (0.0, 0.0), 0.0)
    else:
        best_side = _min_outer_side(best_pos, best_ang, (0.0, 0.0), 0.0)
        
    inner_list = [[float(best_pos[i,0]), float(best_pos[i,1]), float(best_ang[i])] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
