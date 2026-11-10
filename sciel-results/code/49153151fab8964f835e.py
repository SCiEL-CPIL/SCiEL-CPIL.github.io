"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from collections import Counter

_DEADLINE = 85.0  # safety margin before 90s kill

# Precompute triple indices for n=30 once
_N = 30
_i_idx, _j_idx, _k_idx = [], [], []
for _i in range(_N - 2):
    for _j in range(_i + 1, _N - 1):
        for _k in range(_j + 1, _N):
            _i_idx.append(_i)
            _j_idx.append(_j)
            _k_idx.append(_k)
_I_IDX = np.array(_i_idx, dtype=np.int32)
_J_IDX = np.array(_j_idx, dtype=np.int32)
_K_IDX = np.array(_k_idx, dtype=np.int32)
_NUM_TRIPLES = len(_I_IDX)  # C(30,3) = 4060

def _score_fast(points):
    """Fast scoring using precomputed indices. Returns (max_count, area_counts_dict)."""
    pts = np.array(points, dtype=np.int64)
    xi = pts[_I_IDX, 0]; yi = pts[_I_IDX, 1]
    xj = pts[_J_IDX, 0]; yj = pts[_J_IDX, 1]
    xk = pts[_K_IDX, 0]; yk = pts[_K_IDX, 1]
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0, {}
    vals, counts = np.unique(det, return_counts=True)
    area_dict = dict(zip(vals.tolist(), counts.tolist()))
    return int(counts.max()), area_dict

def _score_incremental(points, changed_idx, old_pt, new_pt, area_counts, current_max):
    """Incrementally update area counts after moving one point.
    Returns (new_max, new_area_counts_dict, new_target_count)."""
    pts = np.array(points, dtype=np.int64)
    # Find all triples involving changed_idx
    mask_i = _I_IDX == changed_idx
    mask_j = _J_IDX == changed_idx
    mask_k = _K_IDX == changed_idx
    mask = mask_i | mask_j | mask_k
    n_affected = int(np.sum(mask))
    
    # Update area counts: subtract old contributions, add new
    # For simplicity, recompute affected determinants
    ii = _I_IDX[mask]; jj = _J_IDX[mask]; kk = _K_IDX[mask]
    xi = pts[ii, 0]; yi = pts[ii, 1]
    xj = pts[jj, 0]; yj = pts[jj, 1]
    xk = pts[kk, 0]; yk = pts[kk, 1]
    det_new = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    
    # Recompute old determinants (need old points)
    pts_old = pts.copy()
    pts_old[changed_idx] = old_pt
    xi_o = pts_old[ii, 0]; yi_o = pts_old[ii, 1]
    xj_o = pts_old[jj, 0]; yj_o = pts_old[jj, 1]
    xk_o = pts_old[kk, 0]; yk_o = pts_old[kk, 1]
    det_old = np.abs((xj_o - xi_o) * (yk_o - yi_o) - (yj_o - yi_o) * (xk_o - xi_o))
    
    # Update counts
    for d_old, d_new in zip(det_old, det_new):
        if d_old > 0:
            area_counts[d_old] = area_counts.get(d_old, 0) - 1
            if area_counts[d_old] <= 0:
                del area_counts[d_old]
        if d_new > 0:
            area_counts[d_new] = area_counts.get(d_new, 0) + 1
    
    new_max = max(area_counts.values()) if area_counts else 0
    return new_max, dict(area_counts)

def _score_max_only(points):
    """Return just the max multiplicity."""
    pts = np.array(points, dtype=np.int64)
    xi = pts[_I_IDX, 0]; yi = pts[_I_IDX, 1]
    xj = pts[_J_IDX, 0]; yj = pts[_J_IDX, 1]
    xk = pts[_K_IDX, 0]; yk = pts[_K_IDX, 1]
    det = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    det = det[det > 0]
    if len(det) == 0:
        return 0
    vals, counts = np.unique(det, return_counts=True)
    return int(counts.max())

def _gen_rect_grid(rows, cols, dx, dy):
    """Generate a centered rectangular grid, return exactly 30 points or None."""
    pts = set()
    total_w = (cols - 1) * dx
    total_h = (rows - 1) * dy
    if total_w > 60 or total_h > 60:
        return None
    ox = -total_w // 2
    oy = -total_h // 2
    for i in range(rows):
        for j in range(cols):
            x = ox + i * dx
            y = oy + j * dy
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
    if len(pts) < 30:
        return None
    # Take first 30 in sorted order for determinism
    return sorted(pts)[:30]

def _gen_hex_config(radius):
    """Generate a hexagonal lattice configuration."""
    pts = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r
            if abs(s) > radius:
                continue
            # Convert axial to cartesian
            x = q + s // 2
            y = r
            # Scale and center
            cx = x * 2
            cy = y * 2 + (q % 2)
            if -30 <= cx <= 30 and -30 <= cy <= 30:
                pts.add((cx, cy))
    if len(pts) < 30:
        return None
    return sorted(pts)[:30]

def _gen_lattice(a, b, c, d, count=30):
    """Generate points on a 2D lattice spanned by vectors (a,b) and (c,d)."""
    pts = set()
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * a + j * c
            y = i * b + j * d
            if -30 <= x <= 30 and -30 <= y <= 30:
                pts.add((x, y))
        if len(pts) >= count:
            break
    if len(pts) < count:
        return None
    return sorted(pts)[:count]

def _gen_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # Rectangular grids
    for rows in range(3, 9):
        for cols in range(3, 11):
            if rows * cols < 30:
                continue
            for dx in [1, 2, 3]:
                for dy in [1, 2, 3]:
                    cfg = _gen_rect_grid(rows, cols, dx, dy)
                    if cfg and len(cfg) == 30:
                        configs.append(cfg)
    
    # Hexagonal
    for r in range(3, 6):
        cfg = _gen_hex_config(r)
        if cfg:
            configs.append(cfg)
    
    # Various lattices
    for a, b in [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (3, 1)]:
        for c, d in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (1, 3)]:
            if a * d - b * c == 0:
                continue
            cfg = _gen_lattice(a, b, c, d)
            if cfg:
                configs.append(cfg)
    
    # Random configurations
    rng = np.random.RandomState(42)
    for _ in range(15):
        pts = set()
        while len(pts) < 30:
            x = rng.randint(-30, 31)
            y = rng.randint(-30, 31)
            pts.add((x, y))
        configs.append(sorted(pts))
    
    # Remove duplicates
    unique_configs = []
    seen = set()
    for cfg in configs:
        key = tuple(sorted(cfg))
        if key not in seen:
            seen.add(key)
            unique_configs.append(cfg)
    
    return unique_configs

def _local_search(points, deadline, target_area=None):
    """Simulated annealing local search with target area focus and incremental scoring."""
    start_time = time.time()
    current_points = list(points)
    current_score, current_areas_dict = _score_fast(current_points)
    current_areas = dict(current_areas_dict)  # mutable dict for incremental updates
    best_points = list(current_points)
    best_score = current_score
    best_areas = dict(current_areas)
    
    n = len(current_points)
    point_set = set(current_points)
    
    if target_area is None and current_areas:
        target_area = max(current_areas, key=current_areas.get)
    
    temp = 8.0
    cooling = 0.997
    min_temp = 0.05
    no_improve = 0
    steps = 0
    target_count = current_areas.get(target_area, 0) if target_area else 0
    best_target_count = target_count
    
    while time.time() < deadline and temp > min_temp:
        # Pick a point to move
        idx = int(np.random.randint(n))
        old_pt = current_points[idx]
        
        # Generate new position - biased toward small moves for refinement
        r = np.random.random()
        if r < 0.5:
            # Small perturbation
            radius = int(np.random.choice([1, 2, 3], p=[0.4, 0.35, 0.25]))
            new_x = old_pt[0] + int(np.random.randint(-radius, radius + 1))
            new_y = old_pt[1] + int(np.random.randint(-radius, radius + 1))
        elif r < 0.8:
            # Medium perturbation
            new_x = old_pt[0] + int(np.random.randint(-5, 6))
            new_y = old_pt[1] + int(np.random.randint(-5, 6))
        else:
            # Large perturbation
            new_x = int(np.random.randint(-30, 31))
            new_y = int(np.random.randint(-30, 31))
        
        new_pt = (new_x, new_y)
        
        if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
            temp *= cooling
            continue
        if new_pt == old_pt or new_pt in point_set:
            temp *= cooling
            continue
        
        # Apply move
        point_set.discard(old_pt)
        point_set.add(new_pt)
        current_points[idx] = new_pt
        
        # Incremental scoring
        new_score, new_areas = _score_incremental(current_points, idx, old_pt, new_pt, current_areas, current_score)
        steps += 1
        
        new_target_count = new_areas.get(target_area, 0) if target_area else 0
        
        # Acceptance: primary = max score, secondary = target area count
        accept = False
        if new_score > current_score:
            accept = True
        elif new_score == current_score and target_area and new_target_count > target_count:
            accept = True
        elif new_score == current_score and target_area and new_target_count == target_count:
            # Slight acceptance for neutral moves to explore
            if np.random.random() < 0.3:
                accept = True
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(temp, 0.01))
                if np.random.random() < prob:
                    accept = True
            else:
                accept = True
        
        if accept:
            current_score = new_score
            current_areas = new_areas
            target_count = new_target_count
            no_improve = 0
            if new_score > best_score or (new_score == best_score and new_target_count > best_target_count):
                best_score = new_score
                best_points = list(current_points)
                best_areas = dict(new_areas)
                best_target_count = new_target_count
        else:
            # Revert
            current_points[idx] = old_pt
            point_set.discard(new_pt)
            point_set.add(old_pt)
            no_improve += 1
        
        temp *= cooling
        
        # Reheat if stuck
        if no_improve > 150:
            temp = max(temp, 5.0)
            no_improve = 0
    
    return best_points, best_score

def _polish(points, deadline):
    """Final high-precision polish with multiple passes and wider search."""
    current_points = list(points)
    current_score, current_areas_dict = _score_fast(current_points)
    current_areas = dict(current_areas_dict)
    best_points = list(current_points)
    best_score = current_score
    point_set = set(current_points)
    n = len(current_points)
    
    if current_areas:
        target_area = max(current_areas, key=current_areas.get)
    else:
        target_area = None
    
    # Multiple passes with decreasing neighborhood
    for pass_num, radius in enumerate([4, 3, 2, 1]):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(n):
                if time.time() > deadline:
                    return best_points, best_score
                old_pt = current_points[idx]
                best_for_idx = None
                best_for_idx_score = current_score
                best_for_idx_target = current_areas.get(target_area, 0) if target_area else 0
                
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                        if not (-30 <= new_pt[0] <= 30 and -30 <= new_pt[1] <= 30):
                            continue
                        if new_pt == old_pt or new_pt in point_set:
                            continue
                        
                        # Try this move
                        point_set.discard(old_pt)
                        point_set.add(new_pt)
                        current_points[idx] = new_pt
                        
                        sc, ar = _score_fast(current_points)
                        tc = ar.get(target_area, 0) if target_area else 0
                        
                        if sc > best_for_idx_score or (sc == best_for_idx_score and tc > best_for_idx_target):
                            best_for_idx_score = sc
                            best_for_idx_target = tc
                            best_for_idx = new_pt
                        else:
                            # Revert
                            current_points[idx] = old_pt
                            point_set.discard(new_pt)
                            point_set.add(old_pt)
                
                if best_for_idx is not None:
                    # Apply the best move for this index
                    point_set.discard(old_pt)
                    point_set.add(best_for_idx)
                    current_points[idx] = best_for_idx
                    current_score, current_areas_dict = _score_fast(current_points)
                    current_areas = dict(current_areas_dict)
                    if current_score > best_score:
                        best_score = current_score
                        best_points = list(current_points)
                    improved = True
    
    return best_points, best_score

def _try_all_configs():
    """Main search: evaluate initial configs, then local search on top ones."""
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Generate and score initial configs
    configs = _gen_initial_configs()
    
    scored_configs = []
    for config in configs:
        if time.time() > deadline - 10.0:
            break
        if len(set(config)) != 30:
            continue
        score, areas = _score_fast(config)
        scored_configs.append((score, areas, config))
    
    if not scored_configs:
        # Fallback
        fallback = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
        return fallback
    
    scored_configs.sort(key=lambda x: -x[0])
    
    best_points = scored_configs[0][2]
    best_score = scored_configs[0][0]
    best_areas = scored_configs[0][1]
    
    # Run local search on top configurations
    top_k = min(10, len(scored_configs))
    for i in range(top_k):
        if time.time() > deadline - 8.0:
            break
        score, areas, config = scored_configs[i]
        target_area = max(areas, key=areas.get) if areas else None
        remaining = deadline - time.time() - 5.0
        if remaining < 5.0:
            break
        try:
            lp, ls = _local_search(config, time.time() + remaining, target_area=target_area)
            if ls > best_score:
                best_score = ls
                best_points = lp
                _, best_areas = _score_fast(lp)
        except Exception:
            pass
    
    # Polish the best result
    if time.time() < deadline - 1.0:
        try:
            pp, ps = _polish(best_points, deadline - 0.5)
            if ps > best_score:
                best_score = ps
                best_points = pp
        except Exception:
            pass
    
    return best_points

def construct():
    """Main entry point: find 30 points maximizing max area multiplicity."""
    start_time = time.time()
    
    try:
        result = _try_all_configs()
    except Exception:
        result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
    
    # Verify validity
    pts_set = set(result)
    if len(pts_set) != 30:
        result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
    for (x, y) in result:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            result = [(x, y) for x in range(-3, 3) for y in range(-3, 3)]
            break
    
    return result
# EVOLVE-BLOCK-END
