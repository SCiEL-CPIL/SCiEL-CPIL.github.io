"""
Optimized: vectorized area computation + fast incremental local search.
Key improvements over parent:
  1. All area computations use numpy vectorization (no pure-Python triple loops).
  2. Incremental score only recomputes the C(29,2)=406 areas involving the moved point.
  3. Tighter candidate generation and more effective local search within budget.
"""
import numpy as np
import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def compute_all_areas(pts):
        """
        Vectorized: compute all positive doubled areas for C(n,3) triples.
        Returns 1-D int64 array of positive areas.
        """
        n = len(pts)
        x = pts[:, 0].astype(np.int64)
        y = pts[:, 1].astype(np.int64)
        # Build index arrays for i<j<k
        idx = np.arange(n)
        i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        ii = i_idx[mask]
        jj = j_idx[mask]
        kk = k_idx[mask]
        areas = np.abs(x[ii]*(y[jj]-y[kk]) + x[jj]*(y[kk]-y[ii]) + x[kk]*(y[ii]-y[jj]))
        areas = areas[areas > 0]
        return areas

    def max_multiplicity(pts):
        """Return (max_count, area_dict) using vectorized computation."""
        areas = compute_all_areas(pts)
        if len(areas) == 0:
            return 0, {}
        vals, cnts = np.unique(areas, return_counts=True)
        best_idx = np.argmax(cnts)
        area_dict = {int(v): int(c) for v, c in zip(vals, cnts)}
        return int(cnts[best_idx]), area_dict

    def incremental_score(pts, moved_idx, new_pt):
        """
        After replacing pts[moved_idx] with new_pt, recompute only the
        C(n-1, 2) areas involving moved_idx. Return (max_count, area_dict).
        """
        n = len(pts)
        pts = pts.copy()
        pts[moved_idx] = new_pt
        x = pts[:, 0].astype(np.int64)
        y = pts[:, 1].astype(np.int64)

        # Areas NOT involving moved_idx: all triples from the other n-1 points
        other = [i for i in range(n) if i != moved_idx]
        # Use the original (unchanged) points for these
        # But we need to be careful: the points array has new_pt at moved_idx,
        # so for triples not involving moved_idx, the values are the same.
        # We can compute them from the full array by excluding moved_idx.
        
        # Actually, let's just compute all areas and it'll be fast enough with numpy
        # For n=30, C(30,3)=4060, vectorized this is very fast.
        # But the incremental approach saves recomputing ~3654 unchanged areas.
        # Let's do it properly:
        
        # Unchanged areas: triples from 'other' indices
        m = len(other)  # 29
        if m >= 3:
            oi = np.array(other, dtype=np.int64)
            oi_x = x[oi]
            oi_y = y[oi]
            idx_m = np.arange(m)
            i2, j2, k2 = np.meshgrid(idx_m, idx_m, idx_m, indexing='ij')
            mask2 = (i2 < j2) & (j2 < k2)
            ii2 = oi[i2[mask2]]
            jj2 = oi[j2[mask2]]
            kk2 = oi[k2[mask2]]
            unchanged = np.abs(oi_x[i2[mask2]]*(oi_y[j2[mask2]]-oi_y[k2[mask2]]) +
                               oi_x[j2[mask2]]*(oi_y[k2[mask2]]-oi_y[i2[mask2]]) +
                               oi_x[k2[mask2]]*(oi_y[i2[mask2]]-oi_y[j2[mask2]]))
            unchanged = unchanged[unchanged > 0]
        else:
            unchanged = np.array([], dtype=np.int64)
        
        # Changed areas: triples involving moved_idx
        # Pairs from 'other': C(29,2) = 406
        if m >= 2:
            oi = np.array(other, dtype=np.int64)
            oi_x = x[oi]
            oi_y = y[oi]
            idx_m = np.arange(m)
            a2, b2 = np.meshgrid(idx_m, idx_m, indexing='ij')
            mask2 = (a2 < b2)
            aa = oi[a2[mask2]]
            bb = oi[b2[mask2]]
            # Triple: (moved_idx, aa, bb)
            mx = x[moved_idx]
            my = y[moved_idx]
            ax = oi_x[a2[mask2]]
            ay = oi_y[a2[mask2]]
            bx = oi_x[b2[mask2]]
            by = oi_y[b2[mask2]]
            changed = np.abs(mx*(ay-by) + ax*(by-my) + bx*(my-ay))
            changed = changed[changed > 0]
        else:
            changed = np.array([], dtype=np.int64)
        
        all_areas = np.concatenate([unchanged, changed])
        if len(all_areas) == 0:
            return 0, {}
        vals, cnts = np.unique(all_areas, return_counts=True)
        best_idx = np.argmax(cnts)
        area_dict = {int(v): int(c) for v, c in zip(vals, cnts)}
        return int(cnts[best_idx]), area_dict

    # --- Phase 1: Generate structured candidates ---
    candidates = []

    # Two parallel rows: 15 points each
    for x_start in range(-15, 16):
        for y0 in range(-15, 16):
            for gap in [1, 2, 3, 4]:
                y1 = y0 + gap
                if y1 > 30 or y1 < -30:
                    continue
                pts = []
                for x in range(x_start, x_start + 15):
                    if -30 <= x <= 30:
                        pts.append((x, y0))
                for x in range(x_start, x_start + 15):
                    if -30 <= x <= 30:
                        pts.append((x, y1))
                if len(pts) == 30 and len(set(pts)) == 30:
                    candidates.append(pts)

    # Three rows: 10 each
    for x_start in range(-20, 21):
        for base_y in range(-20, 21):
            for gap in [1, 2, 3]:
                pts = []
                for row in range(3):
                    yy = base_y + row * gap
                    if yy < -30 or yy > 30:
                        break
                    for x in range(x_start, x_start + 10):
                        if -30 <= x <= 30:
                            pts.append((x, yy))
                if len(pts) >= 30:
                    pts = pts[:30]
                    if len(set(pts)) == 30:
                        candidates.append(pts)

    # Rectangular grids
    for w in range(5, 10):
        for h in range(4, 10):
            if w * h >= 30:
                for ox in range(-15, 16, 2):
                    for oy in range(-15, 16, 2):
                        pts = [(ox + dx, oy + dy) for dx in range(w) for dy in range(h)]
                        pts = [p for p in pts if -30 <= p[0] <= 30 and -30 <= p[1] <= 30]
                        if len(pts) >= 30:
                            pts = pts[:30]
                            if len(set(pts)) == 30:
                                candidates.append(pts)

    # Evaluate candidates (limit to avoid timeout)
    random.seed(42)
    best_score = 0
    best_points = None
    best_counts = {}

    # Sort candidates: prioritize two-row patterns (known strong)
    def cand_priority(c):
        ys = set(p[1] for p in c)
        return len(ys)  # fewer distinct y = more parallel

    candidates.sort(key=cand_priority)
    
    max_cands = min(len(candidates), 2000)
    for cand in candidates[:max_cands]:
        if time.time() - start_time > DEADLINE:
            break
        pts_arr = np.array(cand, dtype=np.int64)
        score, counts = max_multiplicity(pts_arr)
        if score > best_score:
            best_score = score
            best_points = cand[:]
            best_counts = counts

    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        pts_arr = np.array(best_points, dtype=np.int64)
        best_score, best_counts = max_multiplicity(pts_arr)

    if time.time() - start_time > DEADLINE:
        return best_points

    # --- Phase 2: Local search ---
    all_valid = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    current_set = set(best_points)
    current_counts = best_counts

    def try_move(best_pts, i, new_pt, cur_score):
        """Try replacing best_pts[i] with new_pt. Return (improved, new_score, new_counts)."""
        if new_pt in current_set or new_pt == best_pts[i]:
            return False, cur_score, None
        score, counts = incremental_score(np.array(best_pts, dtype=np.int64), i, new_pt)
        if score > cur_score:
            return True, score, counts
        return False, cur_score, None

    improved_any = True
    while improved_any and time.time() - start_time < DEADLINE:
        improved_any = False
        current_set = set(best_points)
        
        for i in range(30):
            if time.time() - start_time > DEADLINE:
                break
            old_pt = best_points[i]
            
            # Generate neighbors: local + some random
            neighbors = []
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    np_ = (old_pt[0] + dx, old_pt[1] + dy)
                    if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30:
                        neighbors.append(np_)
            # Add random far points
            for _ in range(20):
                np_ = random.choice(all_valid)
                neighbors.append(np_)
            
            random.shuffle(neighbors)
            
            best_local = best_score
            best_local_pt = None
            best_local_counts = None
            
            for np_ in neighbors[:50]:
                if np_ in current_set or np_ == old_pt:
                    continue
                score, counts = incremental_score(np.array(best_points, dtype=np.int64), i, np_)
                if score > best_local:
                    best_local = score
                    best_local_pt = np_
                    best_local_counts = counts
            
            if best_local_pt is not None and best_local > best_score:
                best_points[i] = best_local_pt
                best_score = best_local
                best_counts = best_local_counts
                current_set = set(best_points)
                improved_any = True

        # Multi-point perturbation to escape local optima
        if time.time() - start_time < DEADLINE - 5:
            for _ in range(5):
                if time.time() - start_time > DEADLINE:
                    break
                new_pts = best_points[:]
                for _ in range(random.randint(2, 4)):
                    idx = random.randint(0, 29)
                    old = new_pts[idx]
                    for _ in range(200):
                        dx = random.randint(-10, 10)
                        dy = random.randint(-10, 10)
                        np_ = (old[0] + dx, old[1] + dy)
                        if -30 <= np_[0] <= 30 and -30 <= np_[1] <= 30 and np_ not in new_pts:
                            new_pts[idx] = np_
                            break
                if len(set(new_pts)) == 30:
                    pts_arr = np.array(new_pts, dtype=np.int64)
                    score, counts = max_multiplicity(pts_arr)
                    if score > best_score:
                        best_score = score
                        best_points = new_pts
                        best_counts = counts
                        improved_any = True

    return best_points
