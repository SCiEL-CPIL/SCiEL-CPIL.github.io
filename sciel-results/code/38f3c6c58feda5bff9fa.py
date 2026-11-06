import time
import random
import math

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    """Squared distance between two lattice points a+b*omega."""
    da = p[0] - q[0]
    db = p[1] - q[1]
    # |a + b*omega|^2 = a^2 + ab + b^2 for omega = exp(i*pi/3)
    return da * da + da * db + db * db

def _edge_count(Sset):
    """Count pairs at distance exactly 1."""
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
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

def _edge_delta_move(Sset, old_pos, new_pos):
    """
    Compute the change in edge count when moving a point from old_pos to new_pos.
    Sset contains old_pos but not new_pos.
    Returns (delta, is_valid) where is_valid checks distance >= 1 constraint.
    """
    # Check validity: new_pos must be at distance >= 1 from all other points
    for q in Sset:
        if q == old_pos:
            continue
        if _dist_sq(new_pos, q) < 1:
            return 0, False
    if new_pos in Sset:
        return 0, False
    
    # Remove edges from old_pos
    lost = 0
    for nb in _neighbors(old_pos):
        if nb in Sset:
            lost += 1
    
    # Add edges to new_pos
    gained = 0
    for nb in _neighbors(new_pos):
        if nb in Sset and nb != old_pos:
            gained += 1
    
    return gained - lost, True

def _hexagon_config():
    """
    Generate a hexagonal cluster of 32 points using BFS rings.
    Hexagonal numbers: 1, 7, 19, 37. So 32 is between 19 and 37.
    We take a hexagon of radius 2 (19 points) and add 13 more points from ring 3.
    """
    from collections import deque
    points = set()
    visited = set()
    queue = deque([(0, 0)])
    visited.add((0, 0))
    dist = {(0, 0): 0}
    while queue:
        a, b = queue.popleft()
        d = dist[(a, b)]
        if d >= 3:
            continue
        for nb in _neighbors((a, b)):
            if nb not in visited:
                visited.add(nb)
                dist[nb] = d + 1
                queue.append(nb)
    
    # Add all points at hex distance <= 2 (19 points)
    for p in visited:
        if dist[p] <= 2:
            points.add(p)
    
    # We have 19 points. Need 13 more from hex distance 3.
    ring3 = [p for p in visited if dist[p] == 3]
    
    # Sort by number of neighbors in points (to keep it compact)
    ring3.sort(key=lambda p: -sum(1 for nb in _neighbors(p) if nb in points))
    
    for p in ring3[:13]:
        points.add(p)
    
    return list(points)

def _greedy_construct(seed):
    """
    Greedily construct a set of 32 points.
    At each step, pick the position that maximizes the number of new edges,
    with random tie-breaking.
    """
    rng = random.Random(seed)
    points = set()
    
    # Define a search region for candidates
    # A 15x15 region should be enough for 32 points
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
            
    # Shuffle candidates to break symmetry randomly
    rng.shuffle(candidates)
    
    for _ in range(32):
        best_score = -1
        best_pos = None
        # To break ties randomly, we can add a small random noise
        # Or just iterate and pick the first max with a random start offset
        start_idx = rng.randrange(len(candidates))
        
        best_edge_gain = -1
        best_pos = None
        best_tie = 0
        
        for i in range(len(candidates)):
            idx = (start_idx + i) % len(candidates)
            p = candidates[idx]
            if p in points:
                continue
            
            # Check validity: distance >= 1 from all existing points
            valid = True
            for q in points:
                if _dist_sq(p, q) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Count new edges
            gain = 0
            for nb in _neighbors(p):
                if nb in points:
                    gain += 1
            
            # Tie-breaking: random noise to explore different shapes
            # We want to maximize gain, but if gain is same, pick randomly
            # This specific implementation picks the first one with max gain in the shuffled order
            # To make it more robust, we can collect all max gain candidates and pick one
            # But for speed, let's just use the gain and a small random factor
            # Actually, just picking the first max in a shuffled list is a good heuristic
            
            if gain > best_edge_gain:
                best_edge_gain = gain
                best_pos = p
                best_tie = rng.random()
            elif gain == best_edge_gain:
                # If gain is equal, we might want to pick randomly among them
                # But for simplicity and speed, we'll stick to the first one found
                # in the shuffled order, which is effectively random.
                pass
        
        if best_pos is None:
            # Fallback: pick a random valid position
            # This shouldn't happen with a large enough candidate pool
            for p in candidates:
                if p in points:
                    continue
                valid = True
                for q in points:
                    if _dist_sq(p, q) < 1:
                        valid = False
                        break
                if valid:
                    best_pos = p
                    break
        
        if best_pos is not None:
            points.add(best_pos)
        else:
            # Error: no valid position found
            break
            
    return list(points)

def _local_search(Sset, max_iters=5000, deadline=None):
    """
    Local search: try moving points to improve edge count using incremental edge counting.
    """
    pts = list(Sset)
    current_edges = _edge_count(set(pts))
    best_edges = current_edges
    best_set = set(pts)
    
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.randrange(len(pts))
        old_pos = pts[idx]
        pts_set = set(pts)
        
        # Try all 6 neighbors plus a few random positions
        move_candidates = list(_neighbors(old_pos))
        rng.shuffle(move_candidates)
        
        best_delta = 0
        best_new_pos = None
        
        for new_pos in move_candidates:
            if new_pos == old_pos or new_pos in pts_set:
                continue
            
            delta, valid = _edge_delta_move(pts_set, old_pos, new_pos)
            if not valid:
                continue
            
            if delta > best_delta:
                best_delta = delta
                best_new_pos = new_pos
        
        if best_new_pos is not None:
            pts[idx] = best_new_pos
            current_edges += best_delta
            if current_edges > best_edges:
                best_edges = current_edges
                best_set = set(pts)
    
    return list(best_set)

def _simulated_annealing(Sset, deadline=None, seed=42):
    """
    Simulated annealing to escape local optima using incremental edge counting.
    """
    rng = random.Random(seed)
    current = list(Sset)
    current_edges = _edge_count(set(current))
    best = list(current)
    best_edges = current_edges
    
    T = 1.0
    T_min = 0.01
    cooling = 0.997
    
    while T > T_min:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point and a random new position
        idx = rng.randrange(len(current))
        old_pos = current[idx]
        
        # Try nearby positions
        new_pos = rng.choice(list(_neighbors(old_pos)))
        
        if new_pos == old_pos:
            continue
        
        pts_set = set(current)
        if new_pos in pts_set:
            continue
        
        delta, valid = _edge_delta_move(pts_set, old_pos, new_pos)
        if not valid:
            continue
        
        if delta > 0 or rng.random() < math.exp(delta / T):
            current[idx] = new_pos
            current_edges += delta
            if current_edges > best_edges:
                best_edges = current_edges
                best = list(current)
        
        T *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_result = None
    best_edges = -1
    
    # First try the hexagonal configuration
    hex_config = _hexagon_config()
    if len(hex_config) == 32:
        # Polish with local search (more iterations since it's faster now)
        polished = _local_search(hex_config, max_iters=10000, deadline=deadline - 30)
        edges = _edge_count(polished)
        if edges > best_edges:
            best_edges = edges
            best_result = polished
        
        # If we haven't hit optimum, try SA from the polished result
        if best_edges < 76 and time.time() < deadline - 20:
            sa_result = _simulated_annealing(polished, deadline=deadline - 10, seed=42)
            edges = _edge_count(sa_result)
            if edges > best_edges:
                best_edges = edges
                best_result = sa_result
    
    # Run multiple greedy + local search iterations
    for i in range(50):
        if time.time() > deadline - 5:
            break
            
        seed = i * 1000 + 42
        initial = _greedy_construct(seed)
        
        if len(initial) != 32:
            continue
            
        # Local search
        result = _local_search(initial, max_iters=2000, deadline=deadline - 2)
        
        edges = _edge_count(result)
        
        if edges > best_edges:
            best_edges = edges
            best_result = result
            
        # If we hit the theoretical maximum, we can stop early
        if best_edges >= 76:
            break
    
    # If still not at optimum, try SA from best result
    if best_edges < 76 and best_result is not None and time.time() < deadline - 5:
        sa_result = _simulated_annealing(best_result, deadline=deadline - 2, seed=999)
        edges = _edge_count(sa_result)
        if edges > best_edges:
            best_edges = edges
            best_result = sa_result
    
    if best_result is None or len(best_result) != 32:
        # Fallback: simple grid
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    # Final verification
    if not _valid(best_result):
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    return [[a, b] for a, b in best_result]
