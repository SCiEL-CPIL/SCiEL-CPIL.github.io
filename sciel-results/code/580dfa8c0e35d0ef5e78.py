"""Combined program: Robust global local search on triangular lattice for 32 points."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _dist2(a1, b1, a2, b2):
    """Squared distance on triangular lattice."""
    da = a1 - a2
    db = b1 - b2
    return da*da + da*db + db*db

def _count_edges(points):
    """Count unit-distance pairs among given (a,b) points."""
    point_set = set(tuple(p) for p in points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in point_set:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _generate_candidates(max_r):
    """Generate all lattice points within hex distance max_r."""
    candidates = []
    for a in range(-max_r, max_r + 1):
        for b in range(-max_r, max_r + 1):
            # Hex distance is max(|a|, |b|, |a+b|)
            if max(abs(a), abs(b), abs(a+b)) <= max_r:
                candidates.append((a, b))
    return candidates

def _greedy_init(candidates, n_points):
    """Greedy initialization: pick points that maximize edges incrementally."""
    selected = []
    selected_set = set()
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Start with a central point to ensure connectivity
    # Find point closest to (0,0)
    candidates_sorted = sorted(candidates, key=lambda p: p[0]*p[0] + p[0]*p[1] + p[1]*p[1])
    
    if candidates_sorted:
        first = candidates_sorted[0]
        selected.append(first)
        selected_set.add(first)
    
    while len(selected) < n_points:
        best_gain = -1
        best_pt = None
        for p in candidates:
            if p in selected_set:
                continue
            gain = 0
            for da, db in neighbors:
                np = (p[0]+da, p[1]+db)
                if np in selected_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_pt = p
            elif gain == best_gain and best_pt is None:
                best_pt = p
        if best_pt is None:
            break
        selected.append(best_pt)
        selected_set.add(best_pt)
        
    return selected

def _local_search(points, candidates, deadline, target_edges=76):
    """Local search using single and double swaps."""
    current_points = list(points)
    current_set = set(current_points)
    current_edges = _count_edges(current_points)
    
    if current_edges >= target_edges:
        return current_points, current_edges

    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    candidate_set = set(candidates)
    
    # Precompute indices for faster swapping? 
    # For N=32, full recount is O(32*6) = 192 ops. Very fast.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Single Swaps
        # Randomize order to avoid bias
        pts_copy = list(current_points)
        random.shuffle(pts_copy)
        
        for p_remove in pts_copy:
            if time.time() >= deadline:
                break
            # Find best p_add
            best_gain = 0
            best_add = None
            # We can just iterate all candidates not in set
            for p_add in candidates:
                if p_add in current_set:
                    continue
                # Calculate new edge count
                # Remove edges connected to p_remove
                # Add edges connected to p_add
                # This is O(1) if we track degrees, but O(N) to compute is fine.
                
                # Simulate swap
                new_set = current_set - {p_remove} | {p_add}
                # Count edges
                e = 0
                for a, b in new_set:
                    for da, db in neighbors:
                        if (a+da, b+db) in new_set:
                            e += 1
                e //= 2
                
                if e > current_edges:
                    current_edges = e
                    current_points = [p for p in current_points if p != p_remove] + [p_add]
                    current_set = new_set
                    improved = True
                    if current_edges >= target_edges:
                        return current_points, current_edges
                    break # Restart search with new state
            
        # Double Swaps (if time permits and still not optimal)
        if time.time() < deadline - 2.0 and current_edges < target_edges:
            # Pick 2 to remove, 2 to add
            # This is expensive to do exhaustively, so do random sampling
            for _ in range(100): # Limit iterations per pass
                if time.time() >= deadline:
                    break
                if len(current_points) < 2:
                    break
                p_r1, p_r2 = random.sample(current_points, 2)
                p_a1, p_a2 = random.sample([c for c in candidates if c not in current_set], 2)
                
                new_set = current_set - {p_r1, p_r2} | {p_a1, p_a2}
                e = 0
                for a, b in new_set:
                    for da, db in neighbors:
                        if (a+da, b+db) in new_set:
                            e += 1
                e //= 2
                
                if e > current_edges:
                    current_edges = e
                    current_points = [p for p in current_points if p not in (p_r1, p_r2)] + [p_a1, p_a2]
                    current_set = new_set
                    improved = True
                    if current_edges >= target_edges:
                        return current_points, current_edges

    return current_points, current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Generate a sufficient candidate pool
    # Hex radius 4 has 1 + 3*4*5 = 61 points.
    # Hex radius 3 has 37 points.
    # 32 points fit in radius 3, but radius 4 provides more flexibility for local search.
    candidates = _generate_candidates(4)
    
    # Initial guess: Greedy
    initial_points = _greedy_init(candidates, 32)
    
    # Refine with local search
    best_points, best_edges = _local_search(initial_points, candidates, deadline)
    
    # If we didn't reach the theoretical max, try a few random restarts
    if best_edges < 76 and time.time() < deadline - 10.0:
        for _ in range(10):
            if time.time() >= deadline - 10.0:
                break
            # Random initialization
            random_points = random.sample(candidates, 32)
            pts, edges = _local_search(random_points, candidates, deadline)
            if edges > best_edges:
                best_points = pts
                best_edges = edges
                if best_edges >= 76:
                    break
    
    # Final Validation
    assert len(best_points) == 32, f"Got {len(best_points)} points"
    assert len(set(tuple(p) for p in best_points)) == 32, "Points not distinct"
    for a, b in best_points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
        
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
