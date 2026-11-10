import time
import random
import numpy as np
from collections import Counter

# Precompute all C(30,3) triple indices
_TRIPLE_INDICES = np.array([(i, j, k) for i in range(30) for j in range(i+1, 30) for k in range(j+1, 30)], dtype=np.int32)

def _score_fast(points):
    """
    Fast scoring using precomputed triple indices.
    points: list of tuples (x, y)
    Returns: max multiplicity of a non-zero doubled area.
    """
    P = np.array(points, dtype=np.int32)
    n = len(P)
    
    if n < 3:
        return 0
    
    # Use precomputed triple indices
    idx = _TRIPLE_INDICES[:len(_TRIPLE_INDICES)]  # All C(30,3) = 4060 triples
    if n < 30:
        # Filter to valid indices
        valid = idx[:, 2] < n
        idx = idx[valid]
    
    p0 = P[idx[:, 0]]
    p1 = P[idx[:, 1]]
    p2 = P[idx[:, 2]]
    
    # det = x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1)
    dets = np.abs(p0[:, 0] * (p1[:, 1] - p2[:, 1]) + 
                  p1[:, 0] * (p2[:, 1] - p0[:, 1]) + 
                  p2[:, 0] * (p0[:, 1] - p1[:, 1]))
    
    # Filter non-zero
    dets = dets[dets > 0]
    
    if len(dets) == 0:
        return 0
    
    # Count multiplicities
    _, counts = np.unique(dets, return_counts=True)
    return int(np.max(counts))

def _delta_score(points, idx, new_pos):
    """
    Compute the change in score when moving point at index idx to new_pos.
    Returns (new_max_count, old_max_count, changed_area_counts)
    """
    P = np.array(points, dtype=np.int32)
    n = len(P)
    
    # Find all triples involving point idx
    mask = (_TRIPLE_INDICES[:, 0] == idx) | (_TRIPLE_INDICES[:, 1] == idx) | (_TRIPLE_INDICES[:, 2] == idx)
    triples = _TRIPLE_INDICES[mask]
    
    if len(triples) == 0:
        return 0, 0, {}, {}
    
    # Compute old areas for these triples
    old_areas = []
    new_areas = []
    
    for t in triples:
        i, j, k = t
        # Old area
        pi = P[i]
        pj = P[j]
        pk = P[k]
        old_det = abs(pi[0] * (pj[1] - pk[1]) + pj[0] * (pk[1] - pi[1]) + pk[0] * (pi[1] - pj[1]))
        old_areas.append(old_det)
        
        # New area (replace point idx with new_pos)
        if i == idx:
            pi_new = np.array(new_pos)
            new_det = abs(pi_new[0] * (pj[1] - pk[1]) + pj[0] * (pk[1] - pi_new[1]) + pk[0] * (pi_new[1] - pj[1]))
        elif j == idx:
            pj_new = np.array(new_pos)
            new_det = abs(pi[0] * (pj_new[1] - pk[1]) + pj_new[0] * (pk[1] - pi[1]) + pk[0] * (pi[1] - pj_new[1]))
        else:
            pk_new = np.array(new_pos)
            new_det = abs(pi[0] * (pj[1] - pk_new[1]) + pj[0] * (pk_new[1] - pi[1]) + pk_new[0] * (pi[1] - pj[1]))
        new_areas.append(new_det)
    
    old_areas = np.array(old_areas)
    new_areas = np.array(new_areas)
    
    # Build area count changes
    old_counts = Counter(old_areas[old_areas > 0])
    new_counts = Counter(new_areas[new_areas > 0])
    
    return old_counts, new_counts

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin before 90s limit
    
    best_score = 0
    best_points = None
    
    def _try_config(points):
        nonlocal best_score, best_points
        score = _score_fast(points)
        if score > best_score:
            best_score = score
            best_points = points[:]
        return score
    
    # 1. Systematic Grid Search
    factors = [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), (30, 1)]
    
    for W, H in factors:
        max_sx = 60 // (W - 1) if W > 1 else 60
        max_sy = 60 // (H - 1) if H > 1 else 60
        
        for sx in range(1, max_sx + 1):
            for sy in range(1, max_sy + 1):
                grid_width = (W - 1) * sx
                grid_height = (H - 1) * sy
                
                if grid_width <= 60 and grid_height <= 60:
                    points = [(x * sx, y * sy) for x in range(W) for y in range(H)]
                    _try_config(points)
                    
                    # Also try centered grids
                    if W > 1 and H > 1:
                        cx = -grid_width // 2
                        cy = -grid_height // 2
                        points_c = [(x * sx + cx, y * sy + cy) for x in range(W) for y in range(H)]
                        if all(-30 <= px <= 30 and -30 <= py <= 30 for px, py in points_c):
                            _try_config(points_c)
    
    if best_points is None:
        best_points = [(i, 0) for i in range(30)]
        best_score = _score_fast(best_points)
    
    # 2. Targeted search: greedy for specific area values
    def _greedy_for_area(A, n_points=30):
        """
        Greedily select points to maximize count of triangles with doubled area A.
        """
        if time.time() > deadline - 5:
            return None, 0
        
        random.seed(42)
        selected = [(random.randint(-30, 30), random.randint(-30, 30)) for _ in range(5)]
        selected = list(set(selected))[:5]
        
        def _count_new_area(p, selected_pts, target_A):
            count = 0
            for i in range(len(selected_pts)):
                for j in range(i+1, len(selected_pts)):
                    x1, y1 = p
                    x2, y2 = selected_pts[i]
                    x3, y3 = selected_pts[j]
                    det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if det == target_A:
                        count += 1
            return count
        
        for step in range(n_points - len(selected)):
            if time.time() > deadline - 5:
                break
            
            candidates = set()
            
            # Points near existing selected points
            for (sx, sy) in selected:
                for dx in range(-8, 9):
                    for dy in range(-8, 9):
                        nx, ny = sx + dx, sy + dy
                        if -30 <= nx <= 30 and -30 <= ny <= 30:
                            candidates.add((nx, ny))
            
            # Random candidates
            for _ in range(150):
                candidates.add((random.randint(-30, 30), random.randint(-30, 30)))
            
            selected_set = set(selected)
            candidates = [c for c in candidates if c not in selected_set]
            
            if not candidates:
                break
            
            best_cand = None
            best_count = -1
            
            for cand in candidates:
                new_count = _count_new_area(cand, selected, A)
                if new_count > best_count:
                    best_count = new_count
                    best_cand = cand
            
            if best_cand is None or best_count <= 0:
                best_cand = random.choice(candidates)
            
            selected.append(best_cand)
        
        score = _score_fast(selected)
        return selected, score
    
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30]
    
    for A in target_areas:
        if time.time() > deadline - 10:
            break
        points, score = _greedy_for_area(A)
        if points is not None:
            _try_config(points)
    
    # 3. Local search with delta computation
    if time.time() > deadline - 10:
        return best_points
    
    current_points = best_points[:]
    current_score = best_score
    n = 30
    
    # Precompute which triples involve each point
    triples_per_point = [[] for _ in range(n)]
    for t in _TRIPLE_INDICES:
        for idx in t:
            triples_per_point[idx].append(tuple(t))
    
    # Hill climbing: try moving one point at a time with delta evaluation
    for round_num in range(15):
        if time.time() > deadline - 3:
            break
        
        improved = False
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 3:
                break
            
            ox, oy = current_points[idx]
            
            # Try neighbors in a spiral pattern
            for r in range(1, 5):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        if abs(dx) != r and abs(dy) != r:
                            continue
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx = ox + dx
                        ny = oy + dy
                        
                        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                            continue
                        
                        if (nx, ny) in current_points:
                            continue
                        
                        # Compute delta score
                        old_counts, new_counts = _delta_score(current_points, idx, (nx, ny))
                        
                        # Build combined counts to find new max
                        # Start with full score, subtract old contributions, add new
                        # This is expensive, so just do full rescore for simplicity
                        # but only for promising moves
                        
                        # Quick check: if the delta doesn't change the max area count, skip
                        # Actually, let's just do full rescore - it's fast enough with precomputed indices
                        current_points[idx] = (nx, ny)
                        new_score = _score_fast(current_points)
                        
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = current_points[:]
                        else:
                            current_points[idx] = (ox, oy)
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_points
