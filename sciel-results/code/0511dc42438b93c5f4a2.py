"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_dist2():
    """Precompute squared distances between all pairs of grid points."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    return points, dist2

# Precompute at module load time to save time during construct()
POINTS, DIST2 = _precompute_dist2()
V = len(POINTS)

def _check_isosceles_with_new_point(new_idx, selected_indices):
    """
    Check if adding new_idx to selected_indices creates any isosceles triangle.
    """
    if len(selected_indices) < 2:
        return True
    
    S = list(selected_indices)
    
    # Check condition 1: d(new, a) distinct for all a in S
    d_new_to_S = {}
    for a in S:
        d = DIST2[new_idx, a]
        if d in d_new_to_S:
            return False
        d_new_to_S[d] = a
        
    # Check condition 2: d(new, a) != d(a, b) for all a, b in S, a != b
    for a in S:
        d_na = DIST2[new_idx, a]
        for b in S:
            if a == b:
                continue
            d_ab = DIST2[a, b]
            if d_na == d_ab:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # 10s buffer for safety
    
    best_size = V
    best_set = []
    
    # Strategy: Weighted Greedy Construction
    # 1. Maintain a set of "valid" points that can be added to the current set.
    # 2. At each step, evaluate each valid point p:
    #    - Temporarily add p.
    #    - Count how many other currently valid points become INVALID.
    #    - This count is the "blocking power" of p.
    # 3. Select the point with the highest blocking power (with random tie-breaking).
    # 4. Add it to the set, update the valid set, and repeat until no valid points remain.
    # 5. The resulting set is maximal. We record its size.
    # 6. Repeat with different random seeds/orders to escape local optima.
    
    # We will run multiple restarts.
    # To make the greedy choice robust, we can randomize the tie-breaking or
    # slightly perturb the blocking power.
    
    for restart in range(2000):
        if time.time() > deadline:
            break
        
        # Initialize with empty set
        current_set = []
        current_set_set = set()
        
        # All points are initially valid candidates
        valid_candidates = set(range(V))
        
        while valid_candidates:
            # Find the best candidate
            best_candidate = -1
            best_block_count = -1
            best_random = random.random()
            
            # To speed up, we might not check all candidates if we have a strong lead?
            # But V=256 is small. Checking 256 candidates * O(|S|^2) is fast.
            # |S| will grow, but we stop when valid_candidates is empty.
            
            # Optimization: If the current set is already larger than best_size, 
            # we can abort this restart early? 
            # No, because we haven't finished the set. But if len(current_set) >= best_size,
            # the final set will be >= best_size. So we can abort.
            if len(current_set) >= best_size:
                break
                
            for p in valid_candidates:
                # Check if p is actually valid (it should be, by definition of valid_candidates,
                # but we must ensure it doesn't form isosceles with current_set)
                # Wait, valid_candidates is maintained as "points that CAN be added".
                # So they are valid by construction.
                
                # Calculate blocking power:
                # If we add p, which other valid candidates q will become invalid?
                # q becomes invalid if _check_isosceles_with_new_point(q, current_set ∪ {p}) is False.
                
                # This is expensive to check for all q.
                # Alternative heuristic:
                # A point p "blocks" q if p and q form an isosceles triangle with some r in current_set?
                # Or if p and q have the same distance to some r in current_set?
                
                # Let's use a faster proxy for blocking power:
                # Count how many valid candidates q have a "conflict" with p given current_set.
                # Conflict if:
                # 1. d(p, q) == d(p, r) for some r in current_set? No, that's not quite it.
                # 2. Adding p makes q invalid.
                
                # Let's just do the explicit check for a subset or all if fast enough.
                # With |S| ~ 10-20, and |valid| ~ 200, checking 200*200 = 40k pairs per step is too slow?
                # 40k * 20 steps = 800k checks per restart. 2000 restarts = 1.6B checks. Too slow in Python.
                
                # We need a faster heuristic.
                # Heuristic: 
                # The "conflict" of adding p is related to how many other points share distances with p 
                # that are already "used" or create isosceles with existing points.
                
                # Let's try a simpler heuristic:
                # Score(p) = Number of valid candidates q such that d(p, q) is "small" or "common"?
                # Actually, let's just pick randomly from the top 10% of candidates based on a fast metric.
                
                # Fast Metric: 
                # For each valid candidate p, calculate the number of distances d(p, r) for r in current_set 
                # that are duplicated among other valid candidates?
                
                # Let's stick to a simpler, robust heuristic that is fast:
                # Pick the point p that minimizes the distance to the centroid of current_set?
                # Or maximizes the number of "close" neighbors?
                
                # Let's go back to the explicit check but optimize it.
                # We can precompute for each pair (p, q) if they would form an isosceles with ANY r in current_set?
                # No, current_set changes.
                
                # Alternative Strategy:
                # Instead of calculating blocking power for ALL candidates, 
                # we can just pick a random valid candidate, but with a bias towards 
                # points that are "central" or have high degree in the "conflict graph".
                
                # Let's try this:
                # 1. For each valid candidate p, compute a "cost" = sum of 1/d(p, q)^2 for q in valid_candidates?
                #    Points close to many others might block more?
                
                # Actually, the most effective simple heuristic for set cover-like problems is:
                # Pick the element that covers the most uncovered items.
                # Here, "covering" means "making invalid".
                
                # Let's approximate the blocking power.
                # If we add p, a candidate q becomes invalid if:
                # - d(p, q) == d(p, r) for some r in S
                # - d(p, q) == d(q, r) for some r in S
                # - d(p, q) == d(r, s) for some r, s in S? No, that's for the triangle (p, r, s).
                
                # Let's just use a random choice but weighted by 1/(1 + distance_to_centroid).
                # Or, let's just do the full check but limit the restarts or optimize the check.
                
                # Optimization for _check_isosceles_with_new_point:
                # We can precompute for each point p, the set of distances to all other points.
                # But the condition depends on S.
                
                # Let's try a different angle:
                # Start with a random valid point.
                # Then, repeatedly pick a random valid point.
                # This is what the current code does.
                
                # To improve, let's try "Lookahead 1":
                # For each candidate p, simulate adding p.
                # Count how many valid candidates remain.
                # Pick p that minimizes remaining valid candidates.
                
                # Cost: |valid| * (|valid| * |S|) operations.
                # If |valid|=100, |S|=10, cost = 100 * 100 * 10 = 100,000 ops.
                # 20 steps per restart. 2,000,000 ops per restart.
                # 2000 restarts = 4e9 ops. Too slow.
                
                # We need to reduce restarts or make the check faster.
                # Let's reduce restarts to 200 and make the check faster.
                
                # Fast check for "remaining valid candidates":
                # We can't easily know this without checking each q.
                
                # Let's use a heuristic:
                # Pick the point p that has the most "conflicts" with other valid points.
                # Conflict(p, q) if d(p, q) is in the set of distances {d(p, r) for r in S} U {d(q, r) for r in S}?
                
                # Let's just use a simple random restart with a twist:
                # Sort valid candidates by their distance to the first point in S?
                
                # Actually, let's just run the current greedy but with a "greedy selection" based on 
                # number of valid neighbors.
                
                # Let's try: Pick the point p that has the FEWEST valid neighbors?
                # No, that would be "least blocking". We want MOST blocking.
                
                # Let's try: Pick the point p that has the MOST valid neighbors?
                # Intuition: If p is close to many q, it might block them?
                
                # Let's implement the "Lookahead 1" but with a time limit per step.
                # If it takes too long, fall back to random.
                
                # For now, let's just do Randomized Greedy with a bias towards 
                # points that are "isolated" from the current set?
                
                # Let's try a completely different simple heuristic:
                # Pick the point p that minimizes the maximum distance to any point in S?
                # (Minimize the "radius" of the set).
                
                # Or pick the point p that maximizes the minimum distance to S?
                
                # Let's go with "Maximize Blocking Power" via approximation:
                # For each p, count how many q in valid_candidates have d(p, q) equal to 
                # any d(p, r) for r in S.
                
                # This is O(|S| * |valid|) per p. Total O(|S| * |valid|^2).
                # Still too slow for 2000 restarts.
                
                # Let's reduce restarts to 100 and use the full lookahead.
                # 100 restarts * 20 steps * 10000 ops = 20,000,000 ops. This is fine.
                
                # Let's implement the full lookahead for the top 10 candidates?
                # No, let's just do it for all if V is small enough.
                
                # Let's try 100 restarts with full lookahead.
                
                # Re-initialize loop variables for this specific restart
                # (We are inside the restart loop)
                
                # We will pick the best p by minimizing the number of remaining valid candidates.
                
                best_p = -1
                min_remaining = V
                
                # To speed up, we can break early if we find a p that blocks all? Unlikely.
                
                for p in valid_candidates:
                    # Simulate adding p
                    # new_set = current_set + {p}
                    # Count valid q in valid_candidates \ {p} such that q is valid with new_set
                    
                    # Optimization: 
                    # q is invalid with new_set if:
                    # 1. q is invalid with current_set (but q is in valid_candidates, so this is false)
                    # 2. Adding p makes q invalid.
                    #    This happens if triangle (p, q, r) is isosceles for some r in current_set,
                    #    OR if d(p, q) == d(p, r) for some r in current_set,
                    #    OR if d(p, q) == d(q, r) for some r in current_set.
                    
                    # Let's precompute distances from p to S and from p to all valid q.
                    
                    # This is getting complex to optimize inline.
                    # Let's just use a simpler heuristic for now to ensure it runs.
                    
                    # Heuristic: Pick the point p that is farthest from the centroid of S?
                    # Or closest?
                    
                    # Let's try: Pick the point p that has the most "shared distances" with S?
                    
                    # For now, let's just pick a random point but with a slight bias.
                    # Actually, the current code's random greedy is already decent.
                    # The key is to get STUCK EARLY.
                    
                    # Let's try "Greedy with Backtracking" is too complex.
                    
                    # Let's try "Minimize Diameter" heuristic:
                    # Pick the point p that minimizes the max distance to existing points.
                    # This keeps the set compact, which might lead to more local conflicts and earlier blocking.
                    
                    if not current_set:
                        # First point is arbitrary, pick random
                        best_p = random.choice(list(valid_candidates))
                        break
                    
                    # Calculate max distance to S for each p
                    min_max_dist = float('inf')
                    for p in valid_candidates:
                        max_d = 0
                        for r in current_set:
                            d = DIST2[p, r]
                            if d > max_d:
                                max_d = d
                        if max_d < min_max_dist:
                            min_max_dist = max_d
                            best_p = p
                            
                    # Add some randomness to avoid deterministic local optima
                    if random.random() < 0.1:
                        best_p = random.choice(list(valid_candidates))
                    
                    # Add best_p to current_set
                    current_set.append(best_p)
                    current_set_set.add(best_p)
                    
                    # Update valid_candidates
                    # Remove best_p
                    valid_candidates.remove(best_p)
                    
                    # Remove any q that is no longer valid
                    to_remove = []
                    for q in valid_candidates:
                        if not _check_isosceles_with_new_point(q, current_set_set):
                            to_remove.append(q)
                    for q in to_remove:
                        valid_candidates.remove(q)
                    
                    # If no valid candidates, we are done with this restart
                    if not valid_candidates:
                        break
                    
                    # If current set size exceeds best, break
                    if len(current_set) > best_size:
                        break
        
        # After the while loop, if we broke out naturally (valid_candidates empty), 
        # we have a maximal set.
        if valid_candidates and len(current_set) > best_size:
            # We broke early because size exceeded best, not because maximal
            pass
        else:
            # Check if we actually reached a maximal state
            # The while loop exits when valid_candidates is empty OR we break due to size.
            # If we break due to size, it's not maximal.
            # We need to know if it's maximal.
            # If valid_candidates is empty, it is maximal.
            if not valid_candidates:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    if best_size <= 8:
                        break

    # Convert indices back to points
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
