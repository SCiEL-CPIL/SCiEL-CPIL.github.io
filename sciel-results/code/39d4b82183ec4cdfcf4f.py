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

def _local_search(Sset, max_iters=2000, deadline=None):
    """
    Local search: try moving points to improve edge count.
    """
    current_edges = _edge_count(Sset)
    best_edges = current_edges
    best_set = set(Sset)
    
    # Candidate positions to try
    candidates = []
    for a in range(-8, 9):
        for b in range(-8, 9):
            candidates.append((a, b))
            
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.randrange(len(Sset))
        old_pos = Sset[idx]
        
        # Try a few random candidate positions
        # We prioritize neighbors of the current position for locality
        move_candidates = list(_neighbors(old_pos))
        rng.shuffle(move_candidates)
        # Add some random distant candidates
        for _ in range(3):
            move_candidates.append(rng.choice(candidates))
            
        improved = False
        for new_pos in move_candidates:
            if new_pos == old_pos:
                continue
            if new_pos in Sset:
                continue
            
            # Check validity
            valid = True
            for other in Sset:
                if other == old_pos:
                    continue
                if _dist_sq(new_pos, other) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute new edge count
            Sset[idx] = new_pos
            new_edges = _edge_count(Sset)
            
            if new_edges > current_edges:
                current_edges = new_edges
                improved = True
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_set = set(Sset)
                break # Move accepted, next iteration
            else:
                Sset[idx] = old_pos
                # Do not break, try next candidate for this point
        
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_result = None
    best_edges = -1
    
    # Run multiple greedy + local search iterations
    # 50 iterations should be fast enough
    num_iterations = 50
    
    for i in range(num_iterations):
        if time.time() > deadline - 5:
            break
            
        seed = i * 1000 + 42
        initial = _greedy_construct(seed)
        
        if len(initial) != 32:
            continue
            
        # Local search
        result = _local_search(initial, max_iters=500, deadline=deadline - 2)
        
        edges = _edge_count(result)
        
        if edges > best_edges:
            best_edges = edges
            best_result = result
            
        # If we hit the theoretical maximum, we can stop early
        if best_edges >= 76:
            break
            
    if best_result is None or len(best_result) != 32:
        # Fallback: simple grid
        # This is a valid configuration but likely suboptimal
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    # Final verification
    if not _valid(best_result):
        # Try to repair or fallback
        # For safety, return a known valid config
        best_result = [(a, b) for a in range(8) for b in range(4)]
        
    return [[a, b] for a, b in best_result]
