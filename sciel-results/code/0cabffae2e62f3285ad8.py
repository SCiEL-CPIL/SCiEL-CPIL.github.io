import time
import random
import math

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return da * da + da * db + db * db

def _edge_count(pts_list):
    """Count edges among a list of points. Uses set for O(1) lookup."""
    S = set(pts_list)
    cnt = 0
    for p in S:
        for q in _neighbors(p):
            if q in S and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    if len(Sset) != 32:
        return False
    if len(set(Sset)) != 32:
        return False
    for a, b in Sset:
        if abs(a) > 64 or abs(b) > 64:
            return False
    pts = list(Sset)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _dist_sq(pts[i], pts[j]) < 1:
                return False
    return True

def _hexagonal_config(n=32):
    """
    Generate a compact hexagonal shape by taking points in order of hexagonal distance.
    Hexagonal distance: max(|a|, |b|, |a+b|) in axial coords.
    For n=32, we take the 19-point hex of radius 2 plus 13 more from radius 3.
    """
    candidates = []
    for a in range(-7, 8):
        for b in range(-7, 8):
            hex_dist = max(abs(a), abs(b), abs(a + b))
            # Tie-break by angle for symmetry
            x = a + 0.5 * b
            y = math.sqrt(3) / 2 * b
            angle = math.atan2(y, x)
            candidates.append((hex_dist, angle, (a, b)))
    
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    points = []
    seen = set()
    for hd, angle, p in candidates:
        if p in seen:
            continue
        valid = True
        for q in points:
            if _dist_sq(p, q) < 1:
                valid = False
                break
        if valid:
            points.append(p)
            seen.add(p)
            if len(points) == n:
                break
    return points

def _simulated_annealing(initial, max_iters=8000, deadline=None):
    """
    Simulated Annealing to maximize edge count.
    """
    rng = random.Random(999)
    current = list(initial)
    S = set(current)
    current_edges = _edge_count(current)
    best = list(current)
    best_edges = current_edges
    
    T = 1.0
    cooling = 0.998
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        idx = rng.randrange(len(current))
        old_pos = current[idx]
        
        # Generate candidate moves: neighbors + random nearby
        candidates = list(_neighbors(old_pos))
        # Add a few random positions near the cluster
        min_a = min(p[0] for p in current) - 3
        max_a = max(p[0] for p in current) + 3
        min_b = min(p[1] for p in current) - 3
        max_b = max(p[1] for p in current) + 3
        for _ in range(4):
            candidates.append((rng.randint(min_a, max_a), rng.randint(min_b, max_b)))
        
        rng.shuffle(candidates)
        
        moved = False
        for new_pos in candidates:
            if new_pos == old_pos:
                continue
            if new_pos in S:
                continue
            
            valid = True
            for other in current:
                if other == old_pos:
                    continue
                if _dist_sq(new_pos, other) < 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new edge count efficiently
            # Remove old_pos edges, add new_pos edges
            old_edges_removed = 0
            for nb in _neighbors(old_pos):
                if nb in S and nb != old_pos:
                    old_edges_removed += 1
            new_edges_added = 0
            for nb in _neighbors(new_pos):
                if nb in S and nb != old_pos:
                    new_edges_added += 1
            
            # Adjust: edges between old_pos and new_pos counted twice if both in S
            # But new_pos is not in S (we checked), so no double count
            # However, old_pos is in S, so edges from old_pos to other points are removed
            # Edges from new_pos to other points are added
            # The net change is: new_edges_added - old_edges_removed
            # But we also need to account for the fact that old_pos was connected to some
            # points that new_pos might also be connected to (shared neighbors)
            # Actually, the edge count changes by:
            #   - edges from old_pos to S\{old_pos}  (removed)
            #   + edges from new_pos to S\{old_pos}  (added)
            # Since new_pos not in S, S\{old_pos} is the same set for both
            delta = new_edges_added - old_edges_removed
            
            if delta > 0 or rng.random() < math.exp(delta / T):
                S.discard(old_pos)
                S.add(new_pos)
                current[idx] = new_pos
                current_edges += delta
                if current_edges > best_edges:
                    best_edges = current_edges
                    best = list(current)
                moved = True
                break
            # else: don't move
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best

def _greedy_construct(seed):
    """
    Greedily construct a set of 32 points.
    """
    rng = random.Random(seed)
    points = set()
    
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
            
    rng.shuffle(candidates)
    
    for _ in range(32):
        best_edge_gain = -1
        best_pos = None
        
        start_idx = rng.randrange(len(candidates))
        
        for i in range(len(candidates)):
            idx = (start_idx + i) % len(candidates)
            p = candidates[idx]
            if p in points:
                continue
            
            valid = True
            for q in points:
                if _dist_sq(p, q) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            gain = 0
            for nb in _neighbors(p):
                if nb in points:
                    gain += 1
            
            if gain > best_edge_gain:
                best_edge_gain = gain
                best_pos = p
        
        if best_pos is not None:
            points.add(best_pos)
        else:
            break
            
    return list(points)

def _local_search_with_swaps(Sset, max_iters=1000, deadline=None):
    """
    Local search: try moving points and swapping pairs to improve edge count.
    """
    pts = list(Sset)
    S = set(pts)
    current_edges = _edge_count(pts)
    best_edges = current_edges
    best_pts = list(pts)
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Choose between single move and swap
        if rng.random() < 0.6:
            # Single move
            idx = rng.randrange(len(pts))
            old_pos = pts[idx]
            
            move_candidates = list(_neighbors(old_pos))
            rng.shuffle(move_candidates)
            for _ in range(3):
                move_candidates.append((rng.randint(-8, 8), rng.randint(-8, 8)))
            
            improved = False
            for new_pos in move_candidates:
                if new_pos == old_pos or new_pos in S:
                    continue
                
                valid = True
                for other in pts:
                    if other == old_pos:
                        continue
                    if _dist_sq(new_pos, other) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute delta
                old_edges_removed = sum(1 for nb in _neighbors(old_pos) if nb in S and nb != old_pos)
                new_edges_added = sum(1 for nb in _neighbors(new_pos) if nb in S and nb != old_pos)
                delta = new_edges_added - old_edges_removed
                
                if delta > 0:
                    S.discard(old_pos)
                    S.add(new_pos)
                    pts[idx] = new_pos
                    current_edges += delta
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_pts = list(pts)
                    improved = True
                    break
        else:
            # Swap two points
            if len(pts) < 2:
                continue
            i, j = rng.sample(range(len(pts)), 2)
            pi, pj = pts[i], pts[j]
            
            # Try swapping: pi goes to pj's location, pj goes to pi's location
            # This is only useful if it creates more edges
            # Check validity
            valid = True
            for k in range(len(pts)):
                if k == i or k == j:
                    continue
                if _dist_sq(pj, pts[k]) < 1 or _dist_sq(pi, pts[k]) < 1:
                    valid = False
                    break
            if not valid:
                continue
            if _dist_sq(pi, pj) < 1:
                continue
            
            # Compute delta for swap
            # Remove edges involving pi and pj, add edges with swapped positions
            old_edges = 0
            for nb in _neighbors(pi):
                if nb in S and nb != pi and nb != pj:
                    old_edges += 1
            for nb in _neighbors(pj):
                if nb in S and nb != pi and nb != pj:
                    old_edges += 1
            if _dist_sq(pi, pj) == 1:
                old_edges += 1  # edge between pi and pj
            
            new_edges = 0
            for nb in _neighbors(pj):  # pi is now at pj's old position
                if nb in S and nb != pi and nb != pj:
                    new_edges += 1
            for nb in _neighbors(pi):  # pj is now at pi's old position
                if nb in S and nb != pi and nb != pj:
                    new_edges += 1
            if _dist_sq(pi, pj) == 1:
                new_edges += 1
            
            delta = new_edges - old_edges
            if delta > 0:
                pts[i], pts[j] = pj, pi
                current_edges += delta
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_pts = list(pts)
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_result = None
    best_edges = -1
    
    # Strategy 1: Hexagonal + Simulated Annealing
    hexagon = _hexagonal_config(32)
    if len(hexagon) == 32:
        edges_hex = _edge_count(hexagon)
        if edges_hex > best_edges:
            best_edges = edges_hex
            best_result = hexagon
        
        if edges_hex < 76:
            result = _simulated_annealing(hexagon, max_iters=8000, deadline=deadline - 15)
            edges = _edge_count(result)
            if edges > best_edges:
                best_edges = edges
                best_result = result
    
    # Strategy 2: Hexagonal + Local Search with Swaps
    if time.time() < deadline - 10:
        result = _local_search_with_swaps(hexagon, max_iters=1000, deadline=deadline - 5)
        edges = _edge_count(result)
        if edges > best_edges:
            best_edges = edges
            best_result = result
    
    # Strategy 3: Greedy + Local Search (multiple seeds)
    for i in range(8):
        if time.time() > deadline - 5:
            break
        seed = i * 1000 + 42
        initial = _greedy_construct(seed)
        if len(initial) != 32:
            continue
        result = _local_search_with_swaps(initial, max_iters=500, deadline=deadline - 3)
        edges = _edge_count(result)
        if edges > best_edges:
            best_edges = edges
            best_result = result
    
    if best_result is None or len(best_result) != 32:
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    if not _valid(best_result):
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    return [[a, b] for a, b in best_result]
