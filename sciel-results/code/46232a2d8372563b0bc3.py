import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def _sat_overlap_depth(v1, v2):
    """
    Compute the minimum translation distance to separate two convex polygons v1, v2.
    Returns 0 if no overlap, positive value if overlapping.
    v1, v2 are (6, 2) arrays of vertices.
    """
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            # Ensure normal points outward
            mid = (p1 + p2) / 2
            center = np.mean(verts, axis=0)
            if np.dot(normal, center - mid) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    if min_overlap == float('inf'):
        return 0.0
    
    return min_overlap

def _containment_violation(verts, outer_center, outer_side, outer_angle):
    """
    Compute the maximum violation of the vertices being inside the outer hexagon.
    Returns 0 if all vertices are inside, otherwise the max distance outside.
    """
    cx, cy = outer_center
    a = np.radians(outer_angle)
    apothem = outer_side * np.sqrt(3) / 2.0
    normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    
    max_viol = 0.0
    for v in verts:
        dx = v[0] - cx
        dy = v[1] - cy
        for na in normal_angles:
            proj = dx * np.cos(na) + dy * np.sin(na)
            if proj > apothem:
                viol = proj - apothem
                if viol > max_viol:
                    max_viol = viol
                break
    return max_viol

def _force_directed_step(positions, angles, outer_side, outer_angle, dt=0.01, damping=0.9):
    """
    One step of force-directed simulation.
    positions: (N, 2) array of centers
    angles: (N,) array of rotation angles in radians
    Returns updated positions, angles, and total force magnitude.
    """
    forces = np.zeros_like(positions)
    torques = np.zeros(N)
    
    # Pairwise repulsion forces
    for i in range(N):
        v1 = _hex_verts(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
        for j in range(i+1, N):
            v2 = _hex_verts(positions[j, 0], positions[j, 1], 1.0, np.degrees(angles[j]))
            depth = _sat_overlap_depth(v1, v2)
            if depth > 0:
                # Repulsion force proportional to overlap depth
                force_mag = depth * 5.0
                # Direction: from j to i (push i away from j)
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random direction if exactly overlapping
                    angle = np.random.uniform(0, 2*np.pi)
                    diff = np.array([np.cos(angle), np.sin(angle)])
                    dist = 1.0
                force_dir = diff / dist
                forces[i] += force_dir * force_mag
                forces[j] -= force_dir * force_mag
    
    # Containment force: push vertices back inside outer hexagon
    cx, cy = 0.0, 0.0
    apothem = outer_side * np.sqrt(3) / 2.0
    normal_angles = np.radians(outer_angle) + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    
    for i in range(N):
        v = _hex_verts(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
        for vertex in v:
            dx = vertex[0] - cx
            dy = vertex[1] - cy
            for na in normal_angles:
                proj = dx * np.cos(na) + dy * np.sin(na)
                if proj > apothem:
                    viol = proj - apothem
                    # Force pushing vertex back inside
                    force_x = -viol * 3.0 * np.cos(na)
                    force_y = -viol * 3.0 * np.sin(na)
                    forces[i] += np.array([force_x, force_y])
    
    # Update positions with damping
    positions = positions + forces * dt
    positions *= damping  # Simple damping to prevent oscillation
    
    # Clamp positions to stay within reasonable bounds
    max_pos = outer_side * 1.5
    positions = np.clip(positions, -max_pos, max_pos)
    
    total_force = np.linalg.norm(forces)
    
    return positions, angles, total_force

def _try_pack_force(outer_side, time_budget=5.0, max_restarts=5):
    """
    Try to find a valid packing using force-directed simulation.
    Returns (success, positions, angles, outer_angle).
    """
    start_time = time.time()
    best_positions = None
    best_angles = None
    best_penalty = float('inf')
    
    # Initial configurations
    def get_hex_grid():
        """Hexagonal grid initialization."""
        pos = []
        # 3 rows of 4
        for r in range(3):
            for c in range(4):
                x = c * 2.1 + (r % 2) * 1.05 - 3.15
                y = r * 1.8 - 1.8
                pos.append([x, y])
        return np.array(pos[:N])
    
    def get_random():
        """Random initialization."""
        return np.random.uniform(-3, 3, size=(N, 2))
    
    inits = [get_hex_grid(), get_random(), get_random()]
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        if restart < len(inits):
            positions = inits[restart].copy()
        else:
            positions = get_random()
        
        angles = np.zeros(N)
        outer_angle = 0.0
        
        # Run force-directed simulation
        for step in range(500):
            if time.time() - start_time > time_budget:
                break
                
            positions, angles, total_force = _force_directed_step(
                positions, angles, outer_side, outer_angle, dt=0.02, damping=0.95
            )
            
            # Check convergence
            if total_force < 0.01:
                break
        
        # Evaluate final configuration
        penalty = 0.0
        for i in range(N):
            v1 = _hex_verts(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
            for j in range(i+1, N):
                v2 = _hex_verts(positions[j, 0], positions[j, 1], 1.0, np.degrees(angles[j]))
                depth = _sat_overlap_depth(v1, v2)
                if depth > 0:
                    penalty += depth * depth
            
            viol = _containment_violation(v1, (0.0, 0.0), outer_side, outer_angle)
            if viol > 0:
                penalty += viol * viol
        
        if penalty < best_penalty:
            best_penalty = penalty
            best_positions = positions.copy()
            best_angles = angles.copy()
            
            if penalty < 1e-5:
                return True, best_positions, best_angles, outer_angle
    
    return False, best_positions, best_angles, outer_angle

def construct():
    start_time = time.time()
    
    # Binary search for the minimum side length
    low = 3.5
    high = 6.0
    
    best_positions = None
    best_angles = None
    best_side = high
    best_outer_angle = 0.0
    
    # First, ensure we have a valid packing at 'high'
    success, positions, angles, outer_angle = _try_pack_force(high, time_budget=15.0, max_restarts=5)
    if not success:
        # Try larger
        high = 7.0
        success, positions, angles, outer_angle = _try_pack_force(high, time_budget=15.0, max_restarts=5)
        if not success:
            # Fallback: return a very large hexagon with a valid grid
            inner = []
            for r in range(3):
                for c in range(4):
                    x = c * 2.1 + (r % 2) * 1.05
                    y = r * 1.8
                    x -= 3 * 2.1 / 2
                    y -= 2 * 1.8 / 2
                    inner.append([x, y, 0.0])
            return inner, (0.0, 0.0), 8.0, 0.0
    
    best_positions = positions
    best_angles = angles
    best_side = high
    best_outer_angle = outer_angle
    
    # Binary Search
    for i in range(10):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        mid = (low + high) / 2.0
        success, positions, angles, outer_angle = _try_pack_force(mid, time_budget=10.0, max_restarts=5)
        
        if success:
            # Feasible
            high = mid
            best_side = mid
            best_positions = positions
            best_angles = angles
            best_outer_angle = outer_angle
        else:
            # Infeasible
            low = mid
    
    # Extract final result
    inner = []
    for i in range(N):
        x = best_positions[i, 0]
        y = best_positions[i, 1]
        a = np.degrees(best_angles[i])
        inner.append([x, y, a])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle

if __name__ == "__main__":
    # Simple test
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}")
