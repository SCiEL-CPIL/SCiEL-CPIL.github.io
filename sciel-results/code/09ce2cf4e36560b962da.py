import numpy as np
import time

N = 12
TIME_LIMIT = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

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
    
    # Check disjointness via segment intersection
    def _seg_seg_dist(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(r)
        elif a < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            return np.linalg.norm(r + t * d2)
        elif e < 1e-12:
            s = -np.dot(d1, r) / a
            s = max(0.0, min(1.0, s))
            return np.linalg.norm(s * d1 - r)
        
        c = np.dot(d1, r)
        b = np.dot(d1, d2)
        denom = a * e - b * b
        
        if denom < 1e-12:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
            s = max(0.0, min(1.0, (b * t - c) / a))
        else:
            s = (b * f - c * e) / denom
            if s < 0.0 or s > 1.0:
                s = 0.0
                t = f / e
                t = max(0.0, min(1.0, t))
            else:
                t = (b * s + f) / e
                t = max(0.0, min(1.0, t))
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))
        
        return np.linalg.norm(s * d1 - r - t * d2)

    def _polygons_overlap(v1, v2, eps=1e-9):
        n1, n2 = len(v1), len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i + 1) % n1]
            for j in range(n2):
                q1 = v2[j]
                q2 = v2[(j + 1) % n2]
                if _seg_seg_dist(p1, p2, q1, q2) < eps:
                    return True
        return False

    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], eps=1e-9):
                return False, "overlap"
            
            # Check containment
            is_inside = True
            for p in inner_verts[i]:
                inside = True
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k+1)%6]
                    cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                    if cross_ab < -eps:
                        inside = False
                        break
                if not inside:
                    is_inside = False
                    break
            
            if not is_inside:
                return False, "outside"
                
    return True, "valid"

def _compute_required_side(centers, angles, outer_angle=0.0):
    """Compute the minimum outer side length required to contain all hexagons."""
    max_proj = 0.0
    # Outer hexagon normals
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def _simulate(centers, angles, target_side, outer_angle=0.0, steps=2000, dt=0.01):
    """
    Run a rigid-body simulation to pack hexagons.
    Returns updated centers and angles.
    """
    # State: positions, velocities, angles, angular velocities
    pos = centers.copy()
    vel = np.zeros_like(pos)
    ang = angles.copy()
    ang_vel = np.zeros_like(angles)
    
    # Parameters
    k_repel = 50.0
    k_container = 10.0
    damping = 0.95
    ang_damping = 0.95
    
    # Precompute outer hexagon vertices for current target_side
    # We simulate in a fixed container size, but the "target_side" defines the container size.
    # Actually, for the simulation to converge, we should simulate in a container slightly larger than target,
    # or simulate in a fixed container and then compute the required side.
    # Let's simulate in a container defined by target_side.
    
    outer_verts = _hex_vertices((0,0), target_side, outer_angle)
    
    for step in range(steps):
        forces = np.zeros_like(pos)
        torques = np.zeros_like(angles)
        
        # 1. Repulsive forces between hexagons
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hex_vertices(pos[i], 1.0, angles[i])
                v2 = _hex_vertices(pos[j], 1.0, angles[j])
                
                # Simple overlap check: distance between centers
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                
                # Approximate collision distance for hexagons
                # If dist < 2 * apothem (sqrt(3)), they might overlap.
                # For a more accurate force, we'd need Minkowski sum, but let's use a potential.
                # Potential: (R - dist)^2 if dist < R, else 0
                R = 2.0  # Rough contact distance
                if dist < R:
                    # Force direction: away from each other
                    if dist > 1e-6:
                        f_dir = d / dist
                    else:
                        f_dir = np.array([1.0, 0.0])
                    
                    # Force magnitude
                    f_mag = k_repel * (R - dist)
                    
                    forces[i] += f_mag * f_dir
                    forces[j] -= f_mag * f_dir
                    
                    # Torque: try to rotate to reduce overlap
                    # Simple heuristic: rotate towards alignment or away
                    # For now, just add small random torque to escape local minima
                    torques[i] += np.random.uniform(-0.1, 0.1)
                    torques[j] += np.random.uniform(-0.1, 0.1)

        # 2. Container forces
        # For each hexagon, check if any vertex is outside the outer hexagon
        for i in range(N):
            verts = _hex_vertices(pos[i], 1.0, angles[i])
            for v in verts:
                # Check if v is inside outer hexagon
                inside = True
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k+1)%6]
                    cross_ab = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                    if cross_ab < 0:
                        inside = False
                        # Compute force pushing back in
                        # Normal of the edge
                        edge = b - a
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-9:
                            normal = normal / norm
                            # Ensure normal points inward
                            mid = (a + b) / 2
                            if np.dot(normal, mid) > 0:
                                normal = -normal
                            # Penetration depth approx
                            pen = -cross_ab / norm
                            f_mag = k_container * pen
                            forces[i] += f_mag * normal
                        break
                if not inside:
                    break
                    
        # 3. Integration (Symplectic Euler)
        vel = (vel + forces * dt) * damping
        ang_vel = (ang_vel + torques * dt) * ang_damping
        
        pos = pos + vel * dt
        ang = ang + ang_vel * dt
        
    return pos, ang

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Strategy:
    # 1. Start with a large container.
    # 2. Simulate to let hexagons settle.
    # 3. Compute required side.
    # 4. Shrink container and repeat.
    # 5. Try different outer angles.
    
    # Initial configuration: Triangular lattice
    np.random.seed(42)
    
    # Generate a set of initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal packing
    pts = [[0,0]]
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    for k in range(6):
        angle = np.pi/3 * k + np.pi/6
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
        if len(pts) == 12:
            break
    initial_configs.append((np.array(pts), np.zeros(N)))
    
    # Config 2: Random
    for _ in range(3):
        pts = np.random.uniform(-4, 4, (N, 2))
        initial_configs.append((pts, np.zeros(N)))
        
    # Config 3: Linear
    pts = np.array([[i*2.0, 0.0] for i in range(N)])
    initial_configs.append((pts, np.zeros(N)))
    
    # Optimization loop
    for outer_angle in [0.0, 30.0]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for init_pos, init_ang in initial_configs:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                # Start with a large container
                current_side = 10.0
                
                # Simulate and shrink
                pos = init_pos.copy()
                ang = init_ang.copy()
                
                # Multiple cooling steps
                for step in range(20):
                    if time.time() - start_time > TIME_LIMIT - 5:
                        break
                        
                    # Simulate
                    pos, ang = _simulate(pos, ang, current_side, outer_angle, steps=500, dt=0.02)
                    
                    # Compute required side
                    req_side = _compute_required_side(pos, ang, outer_angle)
                    
                    # If it fits, try to shrink
                    if req_side < current_side * 0.98:
                        current_side = req_side * 1.01 # Slight buffer
                    else:
                        # If it doesn't fit well, maybe we are stuck.
                        # Try to increase temperature (randomness) or break.
                        # For now, just stop shrinking this config.
                        pass
                        
                    # Update best
                    if req_side < best_side:
                        # Verify validity before updating best
                        inner_hexes = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
                        is_valid, reason = _check_validity(inner_hexes, req_side, (0.0, 0.0), outer_angle)
                        if is_valid:
                            best_side = req_side
                            best_inner = inner_hexes
                            best_outer_center = (0.0, 0.0)
                            best_outer_angle = outer_angle

            except Exception as e:
                pass
                
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
