"""Packing 11 unit hexagons in a regular hexagon - hex grid cluster search."""
import numpy as np
import time

N = 11

def _hex_vertices_flat_top(cx, cy, s, angle_deg):
    """Vertices of flat-top regular hexagon centered at (cx,cy) with side s, rotated by angle_deg.
    Flat-top means vertices at 30, 90, 150, 210, 270, 330 degrees (before rotation)."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    verts = []
    for i in range(6):
        theta = np.radians(30 + 60 * i)
        x = s * np.cos(theta)
        y = s * np.sin(theta)
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        verts.append((cx + xr, cy + yr))
    return np.array(verts)

def _min_enclosing_hexagon(inner_hexagons, n_angles=180):
    """
    Find minimal side length of a regular hexagon enclosing all inner hexagons.
    Outer hexagon orientation is variable. For orientation phi, the 6 face normals
    are at phi + 60*k. The apothem is s*sqrt(3)/2.
    For each face normal direction, the projection range of all vertices must be <= 2*apothem = s*sqrt(3).
    So s = max_k (range_k) / sqrt(3).
    """
    all_verts = []
    for (cx, cy, ang) in inner_hexagons:
        v = _hex_vertices_flat_top(cx, cy, 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    best_s = float('inf')
    best_center = (0.0, 0.0)
    best_phi = 0.0
    
    for phi_deg in range(0, 30, 1):
        phi = np.radians(phi_deg)
        max_range = 0.0
        for k in range(6):
            theta = phi + np.radians(60 * k)
            d = np.array([np.cos(theta), np.sin(theta)])
            projs = all_verts @ d
            rng = np.max(projs) - np.min(projs)
            if rng > max_range:
                max_range = rng
        
        s = max_range / np.sqrt(3)
        if s < best_s:
            best_s = s
            # Compute center: for each direction, center_proj = (max+min)/2
            cx_sum = 0.0
            cy_sum = 0.0
            for k in range(6):
                theta = phi + np.radians(60 * k)
                d = np.array([np.cos(theta), np.sin(theta)])
                projs = all_verts @ d
                mid = (np.max(projs) + np.min(projs)) / 2.0
                cx_sum += mid * np.cos(theta)
                cy_sum += mid * np.sin(theta)
            best_center = (cx_sum / 6.0, cy_sum / 6.0)
            best_phi = phi_deg
    
    return best_s, best_center, best_phi

def _generate_clusters(n):
    """Generate compact hex grid clusters of n hexagons."""
    sqrt3 = np.sqrt(3)
    clusters = []
    seen = set()
    
    max_r = 5
    positions = []
    for r in range(-max_r, max_r + 1):
        for q in range(-max_r, max_r + 1):
            x = sqrt3 * (q + r / 2.0)
            y = 1.5 * r
            dist_sq = x * x + y * y
            positions.append((dist_sq, q, r, x, y))
    
    positions.sort(key=lambda p: p[0])
    
    # Multiple selection strategies
    strategies = []
    
    # Strategy 1: Closest to origin
    strategies.append(sorted(positions, key=lambda p: p[0]))
    
    # Strategy 2: Prefer horizontal (low |y|)
    strategies.append(sorted(positions, key=lambda p: (abs(p[4]), abs(p[3]), p[0])))
    
    # Strategy 3: Prefer vertical (low |x|)  
    strategies.append(sorted(positions, key=lambda p: (abs(p[3]), abs(p[4]), p[0])))
    
    # Strategy 4: Closest to (0, 0.75) - slight vertical shift
    strategies.append(sorted(positions, key=lambda p: (p[3]-0)**2 + (p[4]-0.75)**2))
    
    # Strategy 5: Closest to (sqrt3/2, 0)
    strategies.append(sorted(positions, key=lambda p: (p[3]-sqrt3/2)**2 + p[4]**2))
    
    for strat in strategies:
        selected = []
        used = set()
        for dist_sq, q, r, x, y in strat:
            if len(selected) >= n:
                break
            key = (q, r)
            if key not in used:
                used.add(key)
                selected.append((x, y, 0.0))
        if len(selected) == n:
            # Canonical form for dedup
            sig = tuple(sorted((round(x,6), round(y,6)) for x,y,_ in selected))
            if sig not in seen:
                seen.add(sig)
                clusters.append(selected)
    
    return clusters

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_score = float('inf')
    
    clusters = _generate_clusters(N)
    
    for cluster in clusters:
        if time.time() > deadline:
            break
        
        s, center, phi = _min_enclosing_hexagon(cluster)
        
        if s < best_score:
            best_score = s
            best_result = (cluster, center, s, phi)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, s, phi = best_result
    return inner, center, s, phi

# EVOLVE-BLOCK-END
