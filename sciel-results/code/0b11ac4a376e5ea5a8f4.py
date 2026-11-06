import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    # Directions in the triangular lattice for neighbors
    # A point (a, b) has neighbors:
    # (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    # Let's verify distances:
    # Distance squared formula: d^2 = a^2 + ab + b^2 (if omega = e^{i pi/3})?
    # Actually, standard basis: v1 = (1,0), v2 = (1/2, sqrt(3)/2).
    # Point P = a*v1 + b*v2.
    # |P|^2 = a^2 + ab + b^2.
    # Neighbors are at distance 1.
    # (1,0) -> dist^2 = 1.
    # (0,1) -> dist^2 = 1.
    # (1,-1) -> dist^2 = 1 - 1 + 1 = 1.
    # (-1,1) -> dist^2 = 1 - 1 + 1 = 1.
    # (1,1) -> dist^2 = 1 + 1 + 1 = 3 (not neighbor).
    # (0,-1) -> dist^2 = 1.
    # (-1,0) -> dist^2 = 1.
    # So neighbors of (a,b) are:
    # (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    
    neighbors = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1)
    ]
    
    # We want to add 32 points.
    num_points = 32
    
    # Start with the origin
    S = set()
    S.add((0, 0))
    
    # Candidates are neighbors of points in S
    # We store candidates as a set of tuples
    candidates = set()
    
    # Initialize candidates
    for dx, dy in neighbors:
        p = (0 + dx, 0 + dy)
        candidates.add(p)
        
    best_score = 0
    best_points = None
    
    # Greedy construction
    while len(S) < num_points:
        if time.time() > deadline:
            break
            
        # Find the best candidate to add
        # Score: number of neighbors in S
        # Tie-breaker: distance to centroid (minimize)
        
        # Calculate centroid of S for tie-breaking
        if S:
            cx = sum(p[0] for p in S) / len(S)
            cy = sum(p[1] for p in S) / len(S)
        else:
            cx, cy = 0, 0
            
        best_cand = None
        max_edges = -1
        min_dist_to_centroid = float('inf')
        
        # To ensure determinism and handle ties, we can sort or just iterate.
        # Since the number of candidates is small (boundary of the cluster),
        # we can just find the max.
        
        # Optimization: Pre-calculate neighbor counts?
        # For each candidate, count how many of its neighbors are in S.
        
        for p in candidates:
            # Check bounds (though unlikely to be hit for n=32)
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                continue
                
            edges = 0
            for dx, dy in neighbors:
                if (p[0] + dx, p[1] + dy) in S:
                    edges += 1
                    
            # Tie-breaking: closer to centroid is better
            dist_sq = (p[0] - cx)**2 + (p[1] - cy)**2
            
            if edges > max_edges or (edges == max_edges and dist_sq < min_dist_to_centroid):
                max_edges = edges
                min_dist_to_centroid = dist_sq
                best_cand = p
                
        if best_cand is None:
            # This shouldn't happen unless we run out of valid candidates,
            # which implies the graph is disconnected or we hit bounds.
            # But for a connected cluster growing on an infinite lattice,
            # there are always candidates.
            break
            
        # Add the best candidate
        S.add(best_cand)
        
        # Remove from candidates
        candidates.remove(best_cand)
        
        # Add new neighbors of best_cand to candidates
        for dx, dy in neighbors:
            new_p = (best_cand[0] + dx, best_cand[1] + dy)
            if new_p not in S:
                candidates.add(new_p)
                
        # Update best score/points for safety return
        # Calculate total edges for current S
        # This is O(N^2) which is fine for N=32.
        # Or we can track it incrementally.
        # Incremental: current_edges += max_edges (since best_cand adds max_edges new edges)
        # We started with 0 edges.
        
        if len(S) == num_points:
            break
            
    # If we broke early due to time, we might have fewer than 32 points.
    # The problem requires EXACTLY 32 distinct pairs.
    # If we don't have 32, we need to add arbitrary valid points?
    # But the greedy approach is very fast. 32 steps is trivial.
    # So we should have 32 points.
    
    if len(S) < num_points:
        # Fallback: add random valid points if we failed to grow (should not happen)
        # This is just a safety net.
        while len(S) < num_points:
            # Find a random point far away? No, just scan a grid.
            found = False
            for a in range(-64, 65):
                if found: break
                for b in range(-64, 65):
                    if (a, b) not in S:
                        S.add((a, b))
                        found = True
                        break
            if not found:
                break # No more points? Unlikely with 64 bounds.

    # Convert to list of lists
    result = [[a, b] for a, b in S]
    
    # Verify distinctness and bounds
    # The set S ensures distinctness.
    # The bounds were checked during addition, but let's be sure.
    # If any point is out of bounds, we'd have a problem, but n=32 is small.
    
    return result

if __name__ == "__main__":
    pts = construct()
    print(f"Number of points: {len(pts)}")
    # Count edges
    S = set(tuple(p) for p in pts)
    edges = 0
    for p in S:
        for q in S:
            if p < q:
                dx = p[0] - q[0]
                dy = p[1] - q[1]
                dist_sq = dx*dx + dx*dy + dy*dy
                if dist_sq == 1:
                    edges += 1
    print(f"Number of edges: {edges}")
