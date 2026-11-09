import time
import numpy as np

N = 12
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def _point_in_hex(vx, vy, cx, cy, side, angle_deg):
    """Check if point (vx,vy) is inside hexagon centered at (cx,cy) with given side and angle.
    Returns max violation (0 if inside, positive if outside)."""
    a = np.radians(angle_deg)
    apothem = side * np.sqrt(3) / 2.0
    # The 6 inward-facing normals of the hexagon
    # For a hexagon with flat top (angle=0), normals point at 0, 60, 120, 180, 240, 300 degrees
    # But we need outward normals. The edges are perpendicular to directions at 30, 90, 150, 210, 270, 330
    normal_angles = a + np.pi/6.0 + np.arange(6) * np.pi / 3.0
    
    dx = vx - cx
    dy = vy - cy
    max_viol = 0.0
    for na in normal_angles:
        proj = dx * np.cos(na) + dy * np.sin(na)
        if proj > apothem:
            viol = proj - apothem
            if viol > max_viol:
                max_viol = viol
    return max_viol

def _hex_overlap_depth(v1, v2):
    """Compute overlap depth between two hexagons using SAT. Returns 0 if no overlap."""
    def get_edge_normals(verts):
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
            # Ensure outward: check against center
            center = np.mean(verts, axis=0)
            mid = (p1 + p2) / 2
            if np.dot(normal, center - mid) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_edge_normals(v1)
    n2 = get_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        
        if proj1.max() <= proj2.min() or proj2.max() <= proj1.min():
            return 0.0
        
        overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
        min_overlap = min(min_overlap, overlap)
    
    return 0.0 if min_overlap == float('inf') else min_overlap

def _check_validity(inner, outer_side, outer_angle, outer_center=(0.0, 0.0)):
    """Check if configuration is valid. Returns (is_valid, max_violation)."""
    cx, cy = outer_center
    max_viol = 0.0
    
    # Check containment
    for i in range(N):
        x, y, a = inner[i]
        verts = _hex_verts(x, y, 1.0, a)
        for v in verts:
            viol = _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle)
            if viol > max_viol:
                max_viol = viol
    
    # Check disjointness
    for i in range(N):
        v1 = _hex_verts(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, N):
            v2 = _hex_verts(inner[j][0], inner[j][1], 1.0, inner[j][2])
            depth = _hex_overlap_depth(v1, v2)
            if depth > max_viol:
                max_viol = depth
    
    return max_viol < 1e-6, max_viol

def _relax(inner, outer_side, outer_angle, outer_center=(0.0, 0.0), iterations=200, dt=0.01):
    """Force-directed relaxation to fix overlaps and containment violations."""
    cx, cy = outer_center
    inner = [list(p) for p in inner]
    
    for it in range(iterations):
        forces = [[0.0, 0.0, 0.0] for _ in range(N)]
        
        # Containment forces
        for i in range(N):
            x, y, a = inner[i]
            verts = _hex_verts(x, y, 1.0, a)
            for v in verts:
                viol = _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle)
                if viol > 0:
                    # Push center toward hex center
                    dx = cx - x
                    dy = cy - y
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                    forces[i][0] += dx / dist * viol * 10.0
                    forces[i][1] += dy / dist * viol * 10.0
        
        # Overlap forces
        for i in range(N):
            v1 = _hex_verts(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for j in range(i+1, N):
                v2 = _hex_verts(inner[j][0], inner[j][1], 1.0, inner[j][2])
                depth = _hex_overlap_depth(v1, v2)
                if depth > 0:
                    # Push apart along line connecting centers
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-10
                    fx = dx / dist * depth * 5.0
                    fy = dy / dist * depth * 5.0
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
        
        # Apply forces with damping
        for i in range(N):
            inner[i][0] += forces[i][0] * dt
            inner[i][1] += forces[i][1] * dt
            # Don't change angles in relaxation
    
    return [tuple(p) for p in inner]

def _make_hex_grid(n=12):
    """Create a hexagonal grid of n unit hexagons, centered at origin."""
    # For n=12, use a 3x4 grid or similar
    # Hex grid: row offset pattern
    positions = []
    
    # Try different grid layouts
    # Layout 1: 3 rows of 4
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (r % 2) * 1.0
            y = r * np.sqrt(3)
            positions.append([x, y, 0.0])
    
    # Center the grid
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return positions

def _try_pack(outer_side, time_budget=5.0):
    """Try to find a valid packing. Returns (success, inner, max_violation)."""
    start_time = time.time()
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Start with hex grid
    inner = _make_hex_grid(N)
    
    # Try several relaxations with different initial perturbations
    best_inner = None
    best_viol = float('inf')
    
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Perturb initial configuration
        perturbed = []
        for i in range(N):
            x, y, a = inner[i]
            px = x + np.random.uniform(-0.5, 0.5)
            py = y + np.random.uniform(-0.5, 0.5)
            pa = np.random.uniform(0, 60)
            perturbed.append([px, py, pa])
        
        # Relax
        relaxed = _relax(perturbed, outer_side, outer_angle, outer_center, 
                        iterations=100, dt=0.05)
        
        valid, viol = _check_validity(relaxed, outer_side, outer_angle, outer_center)
        if viol < best_viol:
            best_viol = viol
            best_inner = relaxed
        
        if valid:
            return True, relaxed, viol
    
    return best_viol < 1e-6, best_inner, best_viol

def construct():
    start_time = time.time()
    
    # Binary search for minimum side length
    low = 3.0
    high = 6.0
    
    # First ensure we have a valid packing at high
    success, inner, viol = _try_pack(high, time_budget=10.0)
    if not success or viol > 1e-4:
        # Try larger
        high = 7.0
        success, inner, viol = _try_pack(high, time_budget=10.0)
        if not success or viol > 1e-4:
            # Fallback: return safe large value
            inner = _make_hex_grid(N)
            return inner, (0.0, 0.0), 8.0, 0.0
    
    best_inner = inner
    best_side = high
    
    # Binary search
    for i in range(12):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        mid = (low + high) / 2.0
        success, inner, viol = _try_pack(mid, time_budget=8.0)
        
        if success and viol < 1e-5:
            high = mid
            best_side = mid
            best_inner = inner
        else:
            low = mid
    
    return best_inner, (0.0, 0.0), best_side, 0.0

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}")
