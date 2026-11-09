"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - t0

    def doubled_areas_np(points):
        """Return dict: doubled_area -> count (positive only) using NumPy."""
        pts = np.array(points, dtype=np.int32)
        n = len(pts)
        i_idx, j_idx, k_idx = np.indices((n, n, n))
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_f = i_idx[mask]
        j_f = j_idx[mask]
        k_f = k_idx[mask]
        
        xi = pts[i_f, 0].astype(np.int64)
        yi = pts[i_f, 1].astype(np.int64)
        xj = pts[j_f, 0].astype(np.int64)
        yj = pts[j_f, 1].astype(np.int64)
        xk = pts[k_f, 0].astype(np.int64)
        yk = pts[k_f, 1].astype(np.int64)
        
        d = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        d = d[d > 0]
        if len(d) == 0:
            return {}
        vals, counts = np.unique(d, return_counts=True)
        return dict(zip(vals.tolist(), counts.tolist()))

    def score_of(points):
        if len(points) != 30 or len(set(points)) != 30:
            return -1
        for (x, y) in points:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return -1
        cnt = doubled_areas_np(points)
        if not cnt:
            return 0
        return max(cnt.values())

    def delta_score(pts, idx, new_pos):
        """Compute change in max multiplicity when moving point idx to new_pos.
        Returns (new_score, area_counter) for efficiency."""
        n = 30
        old_x, old_y = pts[idx]
        new_x, new_y = new_pos
        
        # Compute all areas involving point idx (with other pairs)
        # For each pair (j, k) where j != idx, k != idx, j < k:
        # old area: det(pts[idx], pts[j], pts[k])
        # new area: det(new_pos, pts[j], pts[k])
        
        # Build arrays of other points
        others = [pts[i] for i in range(n) if i != idx]
        other_idx = [i for i in range(n) if i != idx]
        
        n_others = len(others)
        # Compute old and new determinants for all pairs of others
        # det(p, q, r) = px*(qy-ry) + qx*(ry-py) + rx*(py-qy)
        
        # For old point:
        old_dets = []
        new_dets = []
        for a in range(n_others):
            for b in range(a + 1, n_others):
                j = other_idx[a]
                k = other_idx[b]
                xj, yj = pts[j]
                xk, yk = pts[k]
                # Old determinant with pts[idx]
                old_d = abs(old_x * (yj - yk) + xj * (yk - old_y) + xk * (old_y - yj))
                old_dets.append(old_d)
                # New determinant with new_pos
                new_d = abs(new_x * (yj - yk) + xj * (yk - new_y) + xk * (new_y - yj))
                new_dets.append(new_d)
        
        old_dets = np.array(old_dets, dtype=np.int64)
        new_dets = np.array(new_dets, dtype=np.int64)
        
        # Areas NOT involving idx remain unchanged
        # Compute full area multiset for old config
        # We need to track the count of each area value
        
        # For efficiency, compute the area histogram
        # Areas involving idx: old_dets
        # Areas not involving idx: computed from all triples not containing idx
        
        # Let's just compute both full histograms
        pts_old = list(pts)
        pts_new = list(pts)
        pts_new[idx] = new_pos
        
        # Full area computation for both
        def full_area_hist(p):
            n = len(p)
            areas = []
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        xi, yi = p[i]
                        xj, yj = p[j]
                        xk, yk = p[k]
                        d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                        if d > 0:
                            areas.append(d)
            if not areas:
                return {}
            arr = np.array(areas, dtype=np.int64)
            vals, counts = np.unique(arr, return_counts=True)
            return dict(zip(vals.tolist(), counts.tolist()))
        
        # This is too slow for local search. Let me use a smarter approach.
        # Actually, for 30 points, C(30,3) = 4060 triples.
        # The delta involves only triples containing idx: C(29,2) = 406 triples.
        # So we can precompute the non-idx areas and just update the idx areas.
        
        return None  # placeholder, we'll use a different approach

    def fast_local_search(points, max_iters=200):
        """Local search using incremental area updates."""
        pts = list(points)
        n = 30
        best_s = score_of(pts)
        if best_s < 0:
            return pts, best_s
        
        # Precompute: for each point idx, store the areas of all triples containing idx
        # and the areas of all triples NOT containing idx
        
        def compute_area_tables(p):
            """Return (idx_areas, non_idx_areas) where:
            idx_areas[i] = dict of area->count for triples containing point i
            non_idx_areas[i] = dict of area->count for triples NOT containing point i
            """
            idx_areas = [defaultdict(int) for _ in range(n)]
            non_idx_areas = [defaultdict(int) for _ in range(n)]
            
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        xi, yi = p[i]
                        xj, yj = p[j]
                        xk, yk = p[k]
                        d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                        if d == 0:
                            continue
                        # This triple involves points i, j, k
                        # It contributes to idx_areas[i], idx_areas[j], idx_areas[k]
                        # And to non_idx_areas[l] for all l not in {i,j,k}
                        idx_areas[i][d] += 1
                        idx_areas[j][d] += 1
                        idx_areas[k][d] += 1
                        for l in range(n):
                            if l not in (i, j, k):
                                non_idx_areas[l][d] += 1
            return idx_areas, non_idx_areas
        
        idx_areas, non_idx_areas = compute_area_tables(pts)
        
        def get_max_mult(idx_areas, non_idx_areas, idx):
            """Get max multiplicity for point idx's move."""
            # Combine: areas from triples containing idx + areas not containing idx
            # The total area multiset = non_idx_areas[idx] + idx_areas[idx]
            # But wait, idx_areas[idx] contains areas of triples containing idx
            # non_idx_areas[idx] contains areas of triples NOT containing idx
            # Together they form the complete area multiset
            combined = defaultdict(int)
            for a, c in non_idx_areas[idx].items():
                combined[a] += c
            for a, c in idx_areas[idx].items():
                combined[a] += c
            if not combined:
                return 0
            return max(combined.values())
        
        for it in range(max_iters):
            if elapsed() > DEADLINE - 3:
                break
            idx = it % n
            ox, oy = pts[idx]
            improved = False
            
            for radius in [1, 2, 3, 4, 5, 6, 8]:
                if improved:
                    break
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        if abs(dx) + abs(dy) > radius:
                            continue
                        nx, ny = ox + dx, oy + dy
                        if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                            continue
                        new_pos = (nx, ny)
                        # Check uniqueness
                        if new_pos in set(pts[i] for i in range(n) if i != idx):
                            continue
                        
                        # Compute new idx_areas[idx]
                        new_idx_areas = defaultdict(int)
                        for j in range(n):
                            if j == idx:
                                continue
                            for k in range(j+1, n):
                                if k == idx:
                                    continue
                                xj, yj = pts[j]
                                xk, yk = pts[k]
                                d = abs(nx * (yj - yk) + xj * (yk - ny) + xk * (ny - yj))
                                if d > 0:
                                    new_idx_areas[d] += 1
                        
                        # New max multiplicity
                        combined = defaultdict(int)
                        for a, c in non_idx_areas[idx].items():
                            combined[a] += c
                        for a, c in new_idx_areas.items():
                            combined[a] += c
                        new_s = max(combined.values()) if combined else 0
                        
                        if new_s > best_s:
                            # Accept move, update tables
                            pts[idx] = new_pos
                            best_s = new_s
                            improved = True
                            # Update idx_areas and non_idx_areas
                            idx_areas, non_idx_areas = compute_area_tables(pts)
                            break
                    if improved:
                        break
                    if elapsed() > DEADLINE - 3:
                        break
                if improved:
                    break
            if elapsed() > DEADLINE - 3:
                break
        
        return pts, best_s

    best_points = None
    best_score = -1

    def try_points(pts):
        nonlocal best_points, best_score
        if len(pts) != 30 or len(set(pts)) != 30:
            return
        for (x, y) in pts:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return
        s = score_of(pts)
        if s > best_score:
            best_score = s
            best_points = pts[:]

    # Family A: two horizontal lines y=0 and y=h
    for h in [1, 2, 3, 5, 7, 10]:
        for m in range(5, 26):
            n = 30 - m
            if n < 1:
                continue
            if elapsed() > DEADLINE:
                break
            line = [(i, 0) for i in range(m)]
            off = [(i, h) for i in range(n)]
            pts = line + off
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family B: three parallel lines
    splits = [(10, 10, 10), (12, 9, 9), (11, 10, 9), (14, 8, 8), (16, 7, 7),
              (15, 8, 7), (9, 10, 11), (8, 11, 11), (13, 9, 8), (15, 10, 5),
              (10, 12, 8), (11, 11, 8), (12, 12, 6), (13, 10, 7), (14, 9, 7)]
    y_configs = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 1, 4), (0, 2, 3), (1, 2, 3)]
    for split in splits:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in y_configs:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30:
                continue
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family C: four parallel lines
    splits4 = [(8, 8, 7, 7), (10, 8, 7, 5), (9, 9, 7, 5), (10, 10, 5, 5),
               (12, 8, 6, 4), (10, 8, 8, 4), (9, 8, 7, 6), (10, 9, 6, 5),
               (8, 8, 8, 6), (9, 9, 8, 4), (11, 8, 6, 5), (10, 10, 6, 4)]
    y_configs4 = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 1, 2, 5),
                  (0, 2, 3, 4), (0, 1, 4, 5), (1, 2, 3, 4)]
    for split in splits4:
        if sum(split) != 30:
            continue
        if elapsed() > DEADLINE:
            break
        for ys in y_configs4:
            pts = []
            for count, y in zip(split, ys):
                pts += [(i, y) for i in range(count)]
            if len(pts) != 30:
                continue
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family D: two lines with staggered x-offsets
    for h in [1, 2, 3]:
        for m in range(8, 23):
            n = 30 - m
            if n < 1:
                continue
            if elapsed() > DEADLINE:
                break
            for x_off in range(-5, 6):
                line = [(i, 0) for i in range(m)]
                off = [(i + x_off, h) for i in range(n)]
                if any(x < -30 or x > 30 for x, y in off):
                    continue
                pts = line + off
                try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family E: Grid-like configurations
    for rows in range(2, 9):
        if elapsed() > DEADLINE:
            break
        for cols in range(2, 15):
            total = rows * cols
            if total > 30:
                continue
            base_pts = [(x, y) for y in range(rows) for x in range(cols)]
            remaining = 30 - total
            if remaining < 0:
                continue
            for x_start in range(-3, 6):
                for y_extra in [rows, rows + 1]:
                    extra = [(x_start + i, y_extra) for i in range(remaining)]
                    if any(x < -30 or x > 30 for x, y in extra):
                        continue
                    pts = base_pts + extra
                    try_points(pts)
            if elapsed() > DEADLINE:
                break

    # Family F: Optimized 2-line configs
    for m in range(10, 21):
        n = 30 - m
        if elapsed() > DEADLINE:
            break
        for off in range(-4, 5):
            pts2 = [(i, 0) for i in range(m)] + [(i + off, 1) for i in range(n)]
            if all(-30 <= x <= 30 for x, y in pts2):
                try_points(pts2)
        if elapsed() > DEADLINE:
            break

    # Family G: 3 lines with optimized distribution
    for m0 in range(6, 18):
        for m1 in range(6, 18):
            m2 = 30 - m0 - m1
            if m2 < 1:
                continue
            if elapsed() > DEADLINE:
                break
            pts = [(i, 0) for i in range(m0)] + [(i, 1) for i in range(m1)] + [(i, 2) for i in range(m2)]
            try_points(pts)
        if elapsed() > DEADLINE:
            break

    # Family H: Concentric circle-like / regular polygon approximations
    import math
    for r in range(5, 31, 3):
        if elapsed() > DEADLINE:
            break
        for start_angle in range(0, 360, 30):
            pts = []
            for i in range(30):
                angle = math.radians(start_angle + i * 360.0 / 30)
                x = round(r * math.cos(angle))
                y = round(r * math.sin(angle))
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
                else:
                    pts.append((0, 0))
            seen = set()
            unique_pts = []
            for p in pts:
                if p not in seen:
                    seen.add(p)
                    unique_pts.append(p)
            if len(unique_pts) == 30:
                try_points(unique_pts)
        if elapsed() > DEADLINE:
            break

    # Family I: Hexagonal / triangular lattice patches
    for r in range(3, 12):
        if elapsed() > DEADLINE:
            break
        pts = set()
        for q in range(-r, r + 1):
            for rr in range(-r, r + 1):
                s = -q - rr
                if abs(s) <= r:
                    x = q + rr / 2
                    y = rr * math.sqrt(3) / 2
                    pts.add((round(x * 2), round(y * 2)))
        pts_list = list(pts)
        if len(pts_list) >= 30:
            import random
            random.seed(42)
            random.shuffle(pts_list)
            for i in range(0, min(10, len(pts_list) - 29)):
                subset = pts_list[i:i+30]
                if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in subset):
                    try_points(subset)
        if elapsed() > DEADLINE:
            break

    # Apply fast local search to the best candidate
    if best_points is not None and elapsed() < DEADLINE - 5:
        refined_pts, refined_s = fast_local_search(best_points, max_iters=300)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    # Also try fast local search on a few other good candidates
    if elapsed() < DEADLINE - 15:
        candidates = []
        candidates.append([(i, 0) for i in range(15)] + [(i, 1) for i in range(15)])
        candidates.append([(i, 0) for i in range(10)] + [(i, 1) for i in range(10)] + [(i, 2) for i in range(10)])
        candidates.append([(i, 0) for i in range(12)] + [(i, 1) for i in range(9)] + [(i, 2) for i in range(9)])
        candidates.append([(i, 0) for i in range(8)] + [(i, 1) for i in range(8)] + [(i, 2) for i in range(7)] + [(i, 3) for i in range(7)])
        
        for cand in candidates:
            if elapsed() > DEADLINE - 10:
                break
            s_cand = score_of(cand)
            if s_cand >= best_score - 100:
                refined_pts, refined_s = fast_local_search(cand, max_iters=200)
                if refined_s > best_score:
                    best_score = refined_s
                    best_points = refined_pts

    # Final polish with more iterations
    if best_points is not None and elapsed() < DEADLINE - 5:
        refined_pts, refined_s = fast_local_search(best_points, max_iters=200)
        if refined_s > best_score:
            best_score = refined_s
            best_points = refined_pts

    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]

    return best_points
# EVOLVE-BLOCK-END
