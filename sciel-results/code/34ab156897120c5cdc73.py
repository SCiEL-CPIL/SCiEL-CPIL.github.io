"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_verts_fast(cx, cy, side, angle_rad):
    """Faster vertex computation using precomputed cos/sin."""
    base = np.array([
        [side * np.cos(angle_rad + np.radians(60 * i)),
         side * np.sin(angle_rad + np.radians(60 * i))]
        for i in range(6)
    ])
    return base + np.array([cx, cy])

def _point_in_polygon(pt, verts):
    """Check if point is inside polygon using winding number / cross products."""
    n = len(verts)
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i + 1) % n]
        edge = v2 - v1
        rel = pt - v1
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < -1e-10:
            return False
    return True

def _containment_penalty(inner_verts, outer_verts, smooth_beta=100.0):
    """Smooth penalty for inner hexagon not being fully inside outer hexagon.
    Uses soft-max to make the penalty differentiable."""
    n_out = len(outer_verts)
    total_penalty = 0.0
    for iv in inner_verts:
        # Compute signed distances to each edge (negative = outside)
        dists = np.zeros(n_out)
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            # Cross product gives signed distance * edge_len
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            dists[i] = cross / edge_len  # positive = inside, negative = outside
        
        # Soft-max of negative distances (penalizes most-violated edge)
        # We want to penalize when any dist < 0
        neg_dists = -dists  # positive when outside
        # Soft-max for smoothness
        max_nd = np.max(neg_dists)
        if max_nd < 0:
            total_penalty += 0.0
        else:
            # Use log-sum-exp style soft-max
            shifted = neg_dists - max_nd
            weights = np.exp(smooth_beta * shifted)
            soft_max = max_nd + np.log(np.sum(weights)) / smooth_beta
            total_penalty += soft_max
    return total_penalty

def _sat_overlap_depth(h1, h2, smooth_beta=50.0):
    """Smooth minimum penetration depth between two convex polygons using SAT.
    Returns 0 if disjoint (with smooth transition)."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    overlaps = []
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap = max(min1, min2) - min(max1, max2)  # positive if separated
        if gap >= 0:
            return 0.0  # Disjoint on this axis
        
        overlap = -gap
        overlaps.append(overlap)
        if overlap < min_depth:
            min_depth = overlap
    
    if min_depth == float('inf'):
        return 0.0
    
    return min_depth

def _total_penalty(params, outer_side, outer_center, outer_angle, smooth_beta=100.0):
    """Total penalty: containment + pairwise disjointness with smooth surrogates."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts, smooth_beta)
    
    # Disjointness
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j], smooth_beta)
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def build_hexagonal_lattice(nrows, ncols):
        """Build a hexagonal lattice of positions (offset rows)."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions
    
    def build_ring_config():
        """6 around center + 5 more in second ring."""
        positions = [(0.0, 0.0)]
        # First ring: 6 hexagons at distance 2 from center
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        # Second ring: pick 5 from the 6 positions at distance ~3.46
        for i in range(6):
            ang = np.radians(60 * i + 30)
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        # We need exactly 12, take first 12
        return positions[:12]
    
    def build_compact_config():
        """Compact 3-4-3-2 arrangement optimized for hexagonal symmetry."""
        positions = []
        # Row 0: 3 hexagons
        for c in range(3):
            positions.append((c * 2.0 - 2.0, 0.0))
        # Row 1: 4 hexagons (offset)
        for c in range(4):
            positions.append((c * 2.0 - 2.0 + 1.0, np.sqrt(3)))
        # Row 2: 3 hexagons
        for c in range(3):
            positions.append((c * 2.0 - 2.0, 2.0 * np.sqrt(3)))
        # Row 3: 2 hexagons (offset)
        for c in range(2):
            positions.append((c * 2.0 - 1.0 + 1.0, 3.0 * np.sqrt(3)))
        return positions[:12]
    
    def build_symmetric_config():
        """Highly symmetric configuration: 1 center + 6 ring + 5 outer."""
        positions = [(0.0, 0.0)]
        # Inner ring at distance 2
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        # Outer ring: 5 of 6 positions
        for i in range(5):
            ang = np.radians(60 * i + 30)
            d = 2.0 * np.sqrt(3)
            positions.append((d * np.cos(ang), d * np.sin(ang)))
        return positions[:12]
    
    # Generate candidate initial configurations
    configs = []
    
    # 3 rows x 4 cols hex grid
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 4 rows x 3 cols hex grid
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 2 rows x 6 cols
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # Ring configuration
    ring_pos = build_ring_config()
    if len(ring_pos) == N:
        arr = np.array(ring_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 6 rows x 2 cols
    pos_6x2 = build_hexagonal_lattice(6, 2)
    if len(pos_6x2) == N:
        arr = np.array(pos_6x2)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # Compact 3-4-3-2
    compact_pos = build_compact_config()
    if len(compact_pos) == N:
        arr = np.array(compact_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # Symmetric config
    sym_pos = build_symmetric_config()
    if len(sym_pos) == N:
        arr = np.array(sym_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
    
    # 12 in a line (degenerate but valid start)
    line_pos = [(float(i - 5.5), 0.0) for i in range(12)]
    configs.append(line_pos)
    
    # Try different outer angles and sides
    outer_angles = [0.0, 30.0]
    best_score = float('inf')
    best_result = None
    
    for outer_angle in outer_angles:
        # Try sides from large to small
        for outer_side in [5.0, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2]:
            if time.time() > deadline - 20.0:
                break
            
            outer_center = (0.0, 0.0)
            
            for config in configs:
                if time.time() > deadline - 15.0:
                    break
                
                init_params = []
                for (x, y) in config:
                    init_params.extend([float(x), float(y), 0.0])
                init_params = np.array(init_params)
                
                # Quick penalty check with smooth penalty
                pen = _total_penalty(init_params, outer_side, outer_center, outer_angle, smooth_beta=100.0)
                
                if pen < 1e-4:
                    # Already fits (smooth penalty close to 0)
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (init_params.copy(), outer_center, outer_side, outer_angle)
                    continue
                
                # Try to optimize
                if time.time() > deadline - 15.0:
                    continue
                
                def objective(p):
                    return _total_penalty(p, outer_side, outer_center, outer_angle, smooth_beta=100.0)
                
                try:
                    result = minimize(
                        objective,
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue
                
                if opt_val < 1e-4:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (opt_params, outer_center, outer_side, outer_angle)
                
                # If close, try smaller
                elif opt_val < 0.05 and outer_side > 3.0:
                    if time.time() > deadline - 15.0:
                        continue
                    smaller = outer_side - 0.05
                    try:
                        result2 = minimize(
                            lambda p: _total_penalty(p, smaller, outer_center, outer_angle, smooth_beta=100.0),
                            opt_params,
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
                        )
                        if result2.fun < 1e-4:
                            if smaller < best_score:
                                best_score = smaller
                                best_result = (result2.x, outer_center, smaller, outer_angle)
                    except Exception:
                        pass
    
    # Final high-precision polish pass
    if best_result is not None and time.time() < deadline - 5.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Binary search for minimal valid side
        lo = outer_side - 0.5
        hi = outer_side
        for _ in range(20):
            if time.time() > deadline - 3.0:
                break
            mid = (lo + hi) / 2.0
            try:
                result = minimize(
                    lambda p: _total_penalty(p, mid, outer_center, outer_angle, smooth_beta=200.0),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 8000, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if result.fun < 1e-6:
                    hi = mid
                    opt_params = result.x
                else:
                    lo = mid
            except Exception:
                lo = mid
        
        outer_side = hi
        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Fallback
    if best_result is None:
        # Use a safe large hexagon
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation with strict (non-smooth) checks
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Strict containment check
    cont_pen = 0.0
    for iv in inner_verts_list:
        for pt in iv:
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % 6]
                edge = v2 - v1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-12:
                    continue
                rel = pt - v1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    cont_pen += (-cross) / edge_len
    
    # Strict disjointness check
    disj_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d
    
    if cont_pen > 1e-4 or disj_pen > 1e-4:
        # Invalid, use fallback
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
