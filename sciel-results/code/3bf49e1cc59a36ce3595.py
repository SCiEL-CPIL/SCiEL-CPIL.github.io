import time
import random
import numpy as np
from collections import Counter

def _score_fast(points):
    """
    Fast scoring using NumPy with preallocated arrays.
    points: list of tuples (x, y)
    Returns: max multiplicity of a non-zero doubled area.
    """
    P = np.array(points, dtype=np.int32)
    n = len(P)
    
    if n < 3:
        return 0
    
    # Maximum possible doubled area: points in [-30,30]^2, max det ~ 60*60*2 = 7200
    hist_size = 8001
    counts = np.zeros(hist_size, dtype=np.int32)
    
    # Preallocate arrays for maximum pair count (29*28/2 = 406)
    max_pairs = 406
    v1 = np.empty((max_pairs, 2), dtype=np.int32)
    v2 = np.empty((max_pairs, 2), dtype=np.int32)
    dets = np.empty(max_pairs, dtype=np.int32)
    temp_counts = np.zeros(hist_size, dtype=np.int32)
    
    for i in range(n - 2):
        pi = P[i]
        Pj = P[i+1:]
        m = len(Pj)
        if m < 2:
            continue
        
        j_idx, k_idx = np.triu_indices(m, k=1)
        num_pairs = len(j_idx)
        
        v1[:num_pairs] = Pj[j_idx] - pi
        v2[:num_pairs] = Pj[k_idx] - pi
        
        dets[:num_pairs] = np.abs(v1[:num_pairs, 0] * v2[:num_pairs, 1] - v1[:num_pairs, 1] * v2[:num_pairs, 0])
        
        mask = (dets[:num_pairs] > 0) & (dets[:num_pairs] < hist_size)
        if np.any(mask):
            valid_dets = dets[:num_pairs][mask]
            temp_counts[:] = 0
            np.add.at(temp_counts, valid_dets, 1)
            counts += temp_counts
    
    return int(np.max(counts)) if n >= 3 else 0

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
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
                    
                    if W > 1 and H > 1:
                        cx = -grid_width // 2
                        cy = -grid_height // 2
                        points_c = [(x * sx + cx, y * sy + cy) for x in range(W) for y in range(H)]
                        if all(-30 <= px <= 30 and -30 <= py <= 30 for px, py in points_c):
                            _try_config(points_c)
    
    if best_points is None:
        best_points = [(i, 0) for i in range(30)]
        best_score = _score_fast(best_points)
    
    # 2. Graph-based greedy selection for target areas
    # Precompute all candidate points and their pairwise area relationships
    
    all_candidates = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    cand_idx = {pt: i for i, pt in enumerate(all_candidates)}
    n_cands = len(all_candidates)
    
    # For each target area A, build adjacency: for each pair (i,j), 
    # which other points k form a triangle of area A with i,j
    # This is expensive, so we'll compute on-the-fly per target area
    
    # Precompute arrays for all candidates
    cand_x = np.array([c[0] for c in all_candidates], dtype=np.int32)
    cand_y = np.array([c[1] for c in all_candidates], dtype=np.int32)
    
    def _greedy_graph_select(target_A, n_points=30, max_time_budget=10.0):
        """
        Greedy selection using graph structure.
        For target area A, at each step select the point that creates
        the most new triangles of area A with already-selected pairs.
        """
        if time.time() > deadline - 5:
            return None
        
        # Start with a few well-separated seed points
        random.seed(42)
        seeds = []
        attempts = 0
        while len(seeds) < 4 and attempts < 1000:
            p = random.choice(all_candidates)
            if all(abs(p[0]-s[0]) + abs(p[1]-s[1]) > 10 for s in seeds):
                seeds.append(p)
            attempts += 1
        
        selected = list(seeds)
        selected_set = set(selected)
        
        # Initialize contribution counts
        contrib = np.zeros(n_cands, dtype=np.int32)
        
        # Compute initial contributions
        for idx, s in enumerate(selected):
            for jdx in range(idx + 1, len(selected)):
                sj = selected[jdx]
                det_vals = np.abs(cand_x * (s[1] - sj[1]) + s[0] * (sj[1] - cand_y) + sj[0] * (cand_y - s[1]))
                mask = (det_vals == target_A) & (det_vals > 0)
                contrib[mask] += 1
        
        # Greedily select remaining points
        for step in range(n_points - len(selected)):
            if time.time() > deadline - 5:
                break
            
            # Exclude already selected points
            for s in selected:
                contrib[cand_idx[s]] = -1
            
            # Find best candidate
            best_idx = int(np.argmax(contrib))
            best_contrib = contrib[best_idx]
            
            if best_contrib <= 0:
                available = [i for i in range(n_cands) if contrib[i] >= 0]
                if not available:
                    break
                best_idx = random.choice(available)
            
            best_pt = all_candidates[best_idx]
            selected.append(best_pt)
            selected_set.add(best_pt)
            
            # Update contributions for new pairs involving best_pt
            for idx in range(len(selected) - 2):
                s = selected[idx]
                det_vals = np.abs(cand_x * (s[1] - best_pt[1]) + s[0] * (best_pt[1] - cand_y) + best_pt[0] * (cand_y - s[1]))
                mask = (det_vals == target_A) & (det_vals > 0)
                contrib[mask] += 1
            
            contrib[cand_idx[best_pt]] = -1
        
        if len(selected) == n_points:
            return selected
        return None
    
    # Try multiple target areas
    target_areas = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 60, 80]
    
    for A in target_areas:
        if time.time() > deadline - 8:
            break
        
        points = _greedy_graph_select(A)
        if points is not None:
            _try_config(points)
    
    # 3. Local search / hill climbing with incremental scoring
    if time.time() > deadline - 5:
        return best_points
    
    current_points = best_points[:]
    current_score = best_score
    n = 30
    current_set = set(current_points)
    
    # Precompute all candidate positions
    all_pos = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    pos_set = set(all_pos)
    
    def _delta_score(old_pt, new_pt, pts):
        """Compute change in score when replacing old_pt with new_pt."""
        # Remove contributions involving old_pt
        # Add contributions involving new_pt
        # This is complex, so we just recompute for now but with early exit
        
        # Quick check: if new_pt creates collinear triples, skip
        # Full recompute
        temp_pts = pts[:]
        temp_pts[pts.index(old_pt)] = new_pt
        return _score_fast(temp_pts)
    
    for round_num in range(30):
        if time.time() > deadline - 2:
            break
        
        improved = False
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 2:
                break
            
            ox, oy = current_points[idx]
            
            # Try moves in increasing radius
            for r in range(1, 6):
                if time.time() > deadline - 2:
                    break
                
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
                        
                        if (nx, ny) in current_set:
                            continue
                        
                        current_points[idx] = (nx, ny)
                        new_score = _score_fast(current_points)
                        
                        if new_score > current_score:
                            current_set.discard((ox, oy))
                            current_set.add((nx, ny))
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_points = current_points[:]
                            # Reset radius search after improvement
                            break
                        else:
                            current_points[idx] = (ox, oy)
                    else:
                        continue
                    break  # Break out of dx loop
                else:
                    continue
                break  # Break out of r loop
            
            if improved:
                break
        
        if not improved:
            # Try larger jumps
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                idx = random.randint(0, n - 1)
                ox, oy = current_points[idx]
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                if (nx, ny) in current_set:
                    continue
                current_points[idx] = (nx, ny)
                new_score = _score_fast(current_points)
                if new_score > current_score:
                    current_set.discard((ox, oy))
                    current_set.add((nx, ny))
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points[:]
                    break
                else:
                    current_points[idx] = (ox, oy)
            
            if current_score == best_score:
                break
    
    return best_points
