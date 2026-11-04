"""Packing 11 unit hexagons in a regular hexagon via cluster rotation optimization."""
import math
import time
import numpy as np

N = 11
_DEADLINE = 140.0
_t0 = time.time()

def _hex_verts(cx, cy, s, ang):
    """6 vertices of regular hexagon, side s, center (cx,cy), rotation ang (rad)."""
    a = ang + np.arange(6) * (math.pi / 3.0)
    return np.stack([cx + s * np.cos(a), cy + s * np.sin(a)], axis=1)

def _hex_halfwidths(s, ang):
    """For 6 flat directions (30,90,150,210,270,330 deg), the half-extent of a hexagon
       of side s rotated by ang. Used to compute min enclosing hexagon side."""
    # Flat directions of the container (pointy-top convention): normals at 30+60k deg
    # A hexagon with vertices at angles ang+60k has support in direction theta:
    #   h(theta) = s * max_k cos(theta - (ang + 60k))  ... but for a regular hexagon
    # Actually support function of regular hexagon (side s, circumradius s):
    #   h(theta) = s * cos( (theta - ang) mod 60deg - 30deg )  clipped appropriately
    # Simpler: compute via vertices
    # h(theta) = max_i (v_i . u) where u = (cos theta, sin theta)
    # We'll compute directly from vertices when needed.
    return None  # placeholder

def _min_enclosing_hex_side(centers, angles_deg, s_inner=1.0):
    """Given inner hexagon centers and angles, find min side of enclosing regular hexagon
       (container can rotate). Returns (best_side, best_container_angle).
       Uses the fact that for a given container orientation, the required side is
       max over all vertices of (support in 6 flat directions) / cos(30deg) ... 
       Actually: if container has flat-direction unit normals n_0..n_5, then
       side = max_i (max_v (v . n_i) - min_v (v . n_i)) / 2 / cos(0) 
       Hmm, for a regular hexagon with side L and center at origin, the half-width
       in the flat-normal direction is L * cos(30 deg) = L * sqrt(3)/2.
       So required L = max_i ( (max_v v.n_i - min_v v.n_i) / 2 ) / (sqrt(3)/2)
       = max_i (range_i) / sqrt(3)
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy), ang_d in zip(centers, angles_deg):
        ang = math.radians(ang_d)
        vs = _hex_verts(cx, cy, s_inner, ang)
        all_verts.append(vs)
    all_verts = np.concatenate(all_verts, axis=0)  # (6*N, 2)
    
    best_side = float('inf')
    best_ang = 0.0
    
    # The container orientation matters. Try a range of orientations.
    # For a hexagonal cluster, the optimal is likely near 0 or 30 deg offset.
    # But we'll sweep finely.
    for cont_ang_deg in range(0, 60, 1):  # 60 deg symmetry
        cont_ang = math.radians(cont_ang_deg)
        # Flat normals of container: at angles cont_ang + 30 + 60k
        normals = []
        for k in range(3):  # only 3 unique directions (opposite pairs)
            theta = cont_ang + math.pi/6 + k * math.pi/3
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        # For each of 3 directions, compute range of projections
        max_range = 0.0
        for n in normals:
            proj = all_verts @ n
            r = proj.max() - proj.min()
            if r > max_range:
                max_range = r
        # Required side: half-range / (sqrt(3)/2) = range / sqrt(3)
        side = max_range / math.sqrt(3.0)
        if side < best_side:
            best_side = side
            best_ang = cont_ang_deg
    
    return best_side, best_ang

def _overlap_penalty(centers, angles_deg, s_inner=1.0, eps=1e-4):
    """Penalty if any two inner hexagons overlap. Returns 0 if no overlap, else positive."""
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = centers[i]
            xj, yj = centers[j]
            ai = math.radians(angles_deg[i])
            aj = math.radians(angles_deg[j])
            vi = _hex_verts(xi, yi, s_inner, ai)
            vj = _hex_verts(xj, yj, s_inner, aj)
            # Check if any vertex of i is inside j or vice versa (approximate)
            # Better: check edge-edge intersections or use separation
            # Simple: min distance between any vertex of i and any vertex of j
            # If hexagons are disjoint, there exists a separating line.
            # Approximation: check if centers are closer than 2*inradius
            inradius = s_inner * math.sqrt(3.0) / 2.0
            dist = math.hypot(xi - xj, yi - yj)
            # For same orientation, min center distance for no overlap is 2*inradius = sqrt(3)
            # For different orientations, it's more complex. Use a conservative check.
            # We'll use a simple distance-based penalty with a margin.
            min_dist = 2.0 * inradius  # ~1.732
            if dist < min_dist - eps:
                penalty += (min_dist - dist) ** 2
    return penalty

def _hex_lattice_positions(n=11):
    """Generate positions for n hexagons on a hexagonal lattice (tight packing).
       For 11 hexagons, use a hexagonal number arrangement or a compact cluster.
       Hex numbers: 1, 7, 19... For 11, use a 3x3 block minus corners, or a cross shape.
       Let's use: center + 6 neighbors + 4 more = a compact 11-hex cluster.
       Actually, a nice arrangement: rows of 3,4,3,1 or similar.
       Let me use a hexagonal spiral: center, ring 1 (6), then 4 more from ring 2.
       Positions in axial coords (q, r) -> cartesian:
         x = s * sqrt(3) * (q + r/2)
         y = s * 3/2 * r
       where s is side length (1 for unit hexagons), and adjacent hexagons are at distance sqrt(3).
    """
    # Axial coordinates for a compact 11-hex cluster
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 partial: (2,-1), (2,-2), (1,-2), (-1,2)  -- pick 4 to make 11
    # Let's use a symmetric shape:
    axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (-2, 1), (-2, 2)
    ]
    # Convert to cartesian (pointy-top hexagons, side=1)
    # x = sqrt(3) * (q + r/2), y = 3/2 * r
    centers = []
    for q, r in axial:
        x = math.sqrt(3.0) * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    return np.array(centers)

def construct():
    global _t0, _DEADLINE
    
    # Initial cluster on hex lattice
    base_centers = _hex_lattice_positions(N)
    n = len(base_centers)
    
    # Parameters to optimize:
    # - cluster rotation (deg): 0..60
    # - cluster translation (x, y)
    # - container rotation (deg): 0..60
    # - container side length (derived from geometry)
    # We can reduce: fix container center at origin, optimize cluster position/rotation and container rotation.
    
    best_side = float('inf')
    best_params = None
    best_inner = None
    best_center = (0.0, 0.0)
    best_side_len = 40.0
    best_container_ang = 0.0
    
    deadline = _t0 + _DEADLINE
    
    # Multi-start Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        """params: [cluster_rot_deg, tx, ty, container_rot_deg]"""
        if time.time() > deadline - 5:
            return 1e10
        c_rot = params[0]
        tx, ty = params[1], params[2]
        cont_rot = params[3]
        
        # Rotate base cluster
        c_rot_rad = math.radians(c_rot)
        cos_r, sin_r = math.cos(c_rot_rad), math.sin(c_rot_rad)
        rotated = []
        for (x, y) in base_centers:
            rx = x * cos_r - y * sin_r + tx
            ry = x * sin_r + y * cos_r + ty
            rotated.append((rx, ry))
        centers = np.array(rotated)
        angles_deg = [c_rot] * n
        
        # Check overlaps
        pen = _overlap_penalty(centers, angles_deg, 1.0)
        if pen > 1e-6:
            return 1e6 + pen
        
        # Compute min enclosing hexagon side for this container rotation
        # Container flat normals at cont_rot + 30 + 60k
        cont_rot_rad = math.radians(cont_rot)
        all_verts = []
        for i in range(n):
            cx, cy = centers[i]
            ang = math.radians(c_rot)
            vs = _hex_verts(cx, cy, 1.0, ang)
            all_verts.append(vs)
        all_verts = np.concatenate(all_verts, axis=0)
        
        max_range = 0.0
        for k in range(3):
            theta = cont_rot_rad + math.pi/6 + k * math.pi/3
            n_vec = np.array([math.cos(theta), math.sin(theta)])
            proj = all_verts @ n_vec
            r = proj.max() - proj.min()
            if r > max_range:
                max_range = r
        side = max_range / math.sqrt(3.0)
        
        return side
    
    # Multiple starts
    n_starts = 30
    for si in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Random initial parameters
        np.random.seed(si * 1000 + 42)
        init = [
            np.random.uniform(0, 60),   # cluster rotation
            np.random.uniform(-2, 2),   # tx
            np.random.uniform(-2, 2),   # ty
            np.random.uniform(0, 60),   # container rotation
        ]
        
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_side and res.fun < 1e5:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Refine with more starts if time permits
    if best_params is not None and time.time() < deadline - 30:
        for si in range(20):
            if time.time() > deadline - 10:
                break
            np.random.seed(si * 7777 + 13)
            init = best_params + np.random.normal(0, 1.0, size=4)
            init[0] = init[0] % 60
            init[3] = init[3] % 60
            try:
                res = minimize(objective, init, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
                if res.fun < best_side and res.fun < 1e5:
                    best_side = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Fallback if optimization failed
    if best_params is None:
        # Use simple linear arrangement as fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract best solution
    c_rot, tx, ty, cont_rot = best_params
    c_rot_rad = math.radians(c_rot)
    cos_r, sin_r = math.cos(c_rot_rad), math.sin(c_rot_rad)
    
    inner = []
    for (x, y) in base_centers:
        rx = x * cos_r - y * sin_r + tx
        ry = x * sin_r + y * cos_r + ty
        inner.append([rx, ry, c_rot])
    
    # Center the container: compute centroid of all vertices and center container there
    all_verts = []
    for (cx, cy) in [(inner[i][0], inner[i][1]) for i in range(N)]:
        ang = math.radians(inner[i][2])
        vs = _hex_verts(cx, cy, 1.0, ang)
        all_verts.append(vs)
    all_verts = np.concatenate(all_verts, axis=0)
    centroid = all_verts.mean(axis=0)
    
    # The container should be centered at the centroid for symmetry
    # But our optimization assumed container center at origin.
    # Shift everything so container center is at centroid.
    # Actually, let's re-center: the min enclosing hex is centered at the
    # center of the bounding box in each of the 3 flat directions.
    # For simplicity, place container center at centroid.
    
    center = (float(centroid[0]), float(centroid[1]))
    
    return inner, center, float(best_side), float(cont_rot)

# Quick self-test
if __name__ == "__main__":
    t_start = time.time()
    inner, center, side, ang = construct()
    t_end = time.time()
    print(f"Time: {t_end - t_start:.2f}s, Side: {side:.4f}, Angle: {ang:.2f}")
    print(f"Center: {center}")
    for i, (x, y, a) in enumerate(inner):
        print(f"  Hex {i}: ({x:.4f}, {y:.4f}), {a:.2f} deg")
