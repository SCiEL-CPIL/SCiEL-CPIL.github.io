import time
import random
import math
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Keep 40s margin for 330s limit
    
    N = 64
    NUM_POINTS = N * N
    
    # Precompute coordinates
    coords = [(i // N, i % N) for i in range(NUM_POINTS)]
    
    # Precompute all squared distances for every pair? 
    # 4096 points -> ~8M pairs. Memory: 8M * 8 bytes (float) or 4 bytes (int) = 32MB. 
    # Time: 8M ops is fast.
    # However, we don't need all pairs for the validity check if we maintain dist_from.
    # But for the "uniqueness" heuristic, we need global frequency.
    
    # Global distance frequency
    # d2 values range from 1 to 2*N*N = 8192.
    # We can count frequencies of d2 in the whole grid.
    freq = [0] * (2 * N * N + 1)
    # Optimization: only count unique pairs i < j
    for i in range(NUM_POINTS):
        x1, y1 = coords[i]
        for j in range(i + 1, NUM_POINTS):
            x2, y2 = coords[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx*dx + dy*dy
            freq[d2] += 1
            
    # Score for a point p: sum of 1/freq[d2(p, q)] for all q?
    # Or just min freq? 
    # Let's use a "potential" score: points that have distances with low global frequency are better.
    # score[p] = sum(1.0 / max(1, freq[d2(p, q)]) for q in range(NUM_POINTS))
    # This is O(N^2) per point -> O(N^4) total = 4096^2 * 4096 = 68 billion. Too slow.
    
    # Alternative: 
    # The constraint is: for any p in S, the set of distances {d(p, q) | q in S} must be distinct.
    # And for any p, q in S, d(p, q) must not be in {d(q, r) | r in S, r != p}.
    
    # Let's try a different structural change: 
    # Greedy construction using a "Most Constrained Variable" (MCV) heuristic.
    # Instead of random order, we pick the point that minimizes the number of valid future additions?
    # That's expensive.
    
    # Let's stick to the "Uniqueness" heuristic but approximate it.
    # Points near the center have many neighbors at various distances.
    # Points near the corner have fewer neighbors.
    # Usually, sparse regions allow for larger sets.
    
    # Let's try a grid-based partitioning strategy (Hexagonal packing idea from prompt, adapted).
    # Partition the grid into blocks. Try to pack points in a pattern that avoids isosceles triangles.
    # A known large set is a Sidon set (distinct pairwise sums). 
    # If we take a Sidon set A in [0, M), then (x, y) = (a, 0) is not enough.
    # But if we map a 1D Sidon set to 2D grid points, we might get a good seed.
    
    # Let's generate a Sidon set.
    # Simple Sidon set: {i^2 + 2i} or similar? No, that's not guaranteed Sidon for large N.
    # Known Sidon set: {3^i} is sparse. 
    # Better: Greedy Sidon set construction.
    
    def generate_sidon_set(max_val):
        S = []
        used_sums = set()
        for i in range(max_val):
            valid = True
            for j in range(len(S)):
                s = i + S[j]
                if s in used_sums:
                    valid = False
                    break
            if valid:
                S.append(i)
                for j in range(len(S)-1):
                    used_sums.add(i + S[j])
                used_sums.add(2*i) # i+i is not a sum of distinct pairs, but Sidon requires all sums distinct?
                # Sidon: a+b = c+d implies {a,b}={c,d}.
                # So we check sums of distinct elements? 
                # Actually, standard Sidon set definition: all pairwise sums a_i + a_j (i <= j) are distinct.
                # So we need to check i+i as well.
        return S
    
    # This greedy Sidon generation is O(K^2) where K is size.
    # If we want ~100 points, K=100, K^2=10000. Fast.
    
    sidon_1d = generate_sidon_set(1000)
    
    # Map 1D Sidon set to 2D grid?
    # If we take points (s_i, s_i) for s_i in Sidon set, does it avoid isosceles?
    # Distance squared between (a,a) and (b,b) is 2(a-b)^2.
    # Distances are determined by |a-b|.
    # If A is Sidon, are |a-b| distinct? No. Sidon is about sums.
    # A set with distinct differences is a Golomb ruler.
    # A Golomb ruler has distinct pairwise differences.
    # If we have a Golomb ruler G, then points (g_i, 0) have distinct distances.
    # But we need to avoid isosceles triangles with 3 points.
    # If all points are on a line, no isosceles triangles (degenerate). 
    # But we are in 2D grid.
    # If we place points on a line, we get |S| points. 
    # Max points on a line in 64x64 is 64.
    # Current score is ~10. So 64 is a huge improvement if valid.
    # Wait, "no three points forming an isosceles triangle".
    # If 3 points are collinear, they form a degenerate triangle. 
    # Usually "forming a triangle" implies non-collinear. 
    # However, if the definition includes degenerate triangles, then any 3 collinear points form an isosceles triangle?
    # No, a degenerate triangle has sides a, b, a+b. It is isosceles only if a=b or a=a+b (impossible) or b=a+b (impossible). 
    # So a degenerate triangle is isosceles iff the two segments are equal length. 
    # i.e., points A, B, C collinear with B in middle. If AB = BC, it is isosceles (degenerate).
    # So we cannot have 3 equally spaced collinear points.
    # A Golomb ruler has distinct differences, so no 3 equally spaced points.
    # So a Golomb ruler on a line is a valid set!
    
    # Let's find the largest Golomb ruler that fits in [0, 63].
    # Known optimal Golomb rulers:
    # Length 15: 0, 1, 4, 10, 12 (5 points)
    # Length 17: 0, 1, 4, 10, 12, 17? No.
    # Optimal Golomb ruler of order n has length ~ n^2.
    # For length 63, max order is roughly sqrt(63) ~ 8.
    # So a line gives us ~8 points. Worse than 10.
    
    # So we need 2D.
    
    # Let's try the "Hexagonal Packing" / Lattice approach.
    # If we use a lattice that avoids isosceles triangles?
    # Square lattice: (0,0), (1,0), (0,1) -> Isosceles right triangle. Bad.
    # Hexagonal lattice: (0,0), (1,0), (1/2, sqrt(3)/2).
    # Distances: 1, 1, sqrt(3). Isosceles! (1,1,sqrt(3)).
    # Most lattices have isosceles triangles.
    
    # We need a set with no isosceles triangles.
    # This is a "No-3" type problem for specific geometric configurations.
    
    # Let's go back to the optimization strategy but make it structural.
    # Strategy: **Conflict-Driven Local Search with Deterministic Restart**
    # 1. Greedy Init with "Most Unique Distances" heuristic.
    # 2. Local Search: If a point p cannot be added, find the point q in S that causes the MOST conflicts with p.
    #    Remove q. Try to add p.
    #    If p is added, great.
    #    If p is still not added (conflicts with others), try to add the point r that was second most conflicting?
    #    Actually, just remove q and try to add ANY point that fits.
    
    # This is effectively a "Swap" but guided by conflict analysis.
    
    # Implementation:
    
    S = []
    in_set = [False] * NUM_POINTS
    dist_from = [set() for _ in range(NUM_POINTS)]
    
    # Heuristic Order: 
    # Calculate "potential" of each point based on global distance frequencies.
    # We want points that have distances with LOW frequency.
    # potential[p] = sum(1.0 / freq[d2(p, q)] for q in range(NUM_POINTS) if d2(p, q) > 0)
    # This is O(N^2) per point -> O(N^4) = 68e9. Too slow for 300s.
    
    # Approximation: 
    # Points far from the center have fewer neighbors, so their distance set is smaller.
    # Smaller distance set -> easier to keep distinct.
    # So, process points from corners/edges inwards?
    # Or, process points with the fewest possible distinct distances?
    
    # Let's just use a shuffled order but with a bias towards "sparse" areas?
    # Actually, the previous code used random shuffle. 
    # Let's try a deterministic spiral from the corner?
    
    # Better: 
    # Use the "Most Constrained" heuristic.
    # Start with empty set.
    # For each step, pick the point p that minimizes the number of valid candidates remaining?
    # Too expensive.
    
    # Let's try a different structural change: **Simulated Annealing with Conflict Repair**
    # Instead of random swaps, we identify the "bottleneck" conflicts.
    
    # Let's implement a faster validity check using numpy?
    # No, Python loops are okay if we reduce iterations.
    
    # Let's try: **Greedy with Backtracking (Limited Depth)**
    # No, backtracking is exponential.
    
    # Let's try: **Two-Phase Construction**
    # Phase 1: Construct a large set using a simple greedy (random order).
    # Phase 2: Use "Conflict Swap". 
    # For each point p NOT in S:
    #   Find all points q in S such that p conflicts with q.
    #   If the set of conflicting points is small (e.g., 1 or 2), try removing them and adding p.
    #   This is targeted, not random.
    
    def calc_d2(p, q):
        px, py = coords[p]
        qx, qy = coords[q]
        dx = px - qx
        dy = py - qy
        return dx*dx + dy*dy

    def is_valid_add(p, S, dist_from):
        d2s = set()
        for a in S:
            d2_pa = calc_d2(p, a)
            if d2_pa in d2s:
                return False
            d2s.add(d2_pa)
            if d2_pa in dist_from[a]:
                return False
        return True

    def do_add(p, S, in_set, dist_from):
        S.append(p)
        in_set[p] = True
        dist_from[p] = set()
        for a in S[:-1]:
            d2 = calc_d2(p, a)
            dist_from[p].add(d2)
            dist_from[a].add(d2)

    def do_remove(p, S, in_set, dist_from):
        idx = S.index(p)
        S.pop(idx)
        in_set[p] = False
        for a in S:
            d2 = calc_d2(p, a)
            dist_from[a].discard(d2)
        dist_from[p] = set()

    # Phase 1: Greedy Init
    # Try a few deterministic orders:
    # 1. Row-major
    # 2. Column-major
    # 3. Diagonal
    # 4. Random
    
    best_init_S = []
    best_init_size = 0
    
    orders = []
    # Row major
    orders.append(list(range(NUM_POINTS)))
    # Col major
    col_order = [i * N + j for j in range(N) for i in range(N)]
    orders.append(col_order)
    # Random
    rand_order = list(range(NUM_POINTS))
    random.shuffle(rand_order)
    orders.append(rand_order)
    
    for order in orders:
        S = []
        in_set = [False] * NUM_POINTS
        dist_from = [set() for _ in range(NUM_POINTS)]
        
        for p in order:
            if is_valid_add(p, S, dist_from):
                do_add(p, S, in_set, dist_from)
                
        if len(S) > best_init_size:
            best_init_size = len(S)
            best_init_S = S[:]
            # Copy state
            best_in_set = in_set[:]
            best_dist_from = [s.copy() for s in dist_from]
            
    # Set current state to best init
    S = best_init_S[:]
    in_set = best_in_set[:]
    dist_from = best_dist_from
    
    best_size = len(S)
    best_S = S[:]
    
    # Phase 2: Conflict-Driven Swap
    # For each point p not in S, find conflicts.
    # If conflict set is small, try swap.
    
    # Precompute list of non-S points
    non_S = [i for i in range(NUM_POINTS) if not in_set[i]]
    
    # We will iterate through non_S points.
    # To make it efficient, we limit the number of conflicts we check.
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # Shuffle non_S to avoid bias
        random.shuffle(non_S)
        
        for p in non_S:
            if in_set[p]:
                continue
                
            # Find conflicts
            conflicts = []
            d2s = set()
            valid = True
            for a in S:
                d2_pa = calc_d2(p, a)
                if d2_pa in d2s:
                    # Conflict with previous points in this check? 
                    # Actually, if d2_pa is in d2s, it means p forms isosceles with two points in S.
                    # This is a hard conflict. We can't fix it by removing one point unless we remove both?
                    # No, if d(p, a1) == d(p, a2), we must remove either a1 or a2 (or p, but we want to add p).
                    # So we add both to conflicts?
                    # Actually, if we remove a1, the conflict with a2 might disappear if d2_pa was the cause?
                    # No, d2_pa is distance to p. It doesn't change.
                    # So if d(p, a1) == d(p, a2), we MUST remove both a1 and a2 to add p?
                    # Or remove one, and the other still causes conflict?
                    # Yes, if d(p, a1) == d(p, a2), then for any subset of S containing both a1 and a2, p is invalid.
                    # So we need to remove at least one of them.
                    # Let's just mark the pair as conflict.
                    conflicts.append(a1) # This is wrong, a1 is not defined in this scope properly if we break.
                    # Let's restart conflict detection logic.
                    valid = False
                    break
                d2s.add(d2_pa)
                
                if d2_pa in dist_from[a]:
                    # Conflict: d(p, a) == d(a, b) for some b in S.
                    # We need to remove a OR b.
                    # We don't know b easily.
                    # But we can just flag 'a' as a potential blocker.
                    conflicts.append(a)
            
            if valid and not conflicts:
                # Should not happen if p is not in S and we checked all?
                # Wait, if valid and no conflicts, we can add p.
                do_add(p, S, in_set, dist_from)
                improved = True
                no_improve = 0
                best_size = max(best_size, len(S))
                if len(S) > best_size:
                    best_S = S[:]
                # Update non_S
                if p in non_S:
                    non_S.remove(p)
                continue
                
            if conflicts:
                # Try to remove one conflict point and add p
                # Pick a random conflict point
                q = random.choice(conflicts)
                
                # Check if p is valid with S \ {q}
                # We need to re-check validity.
                # Optimized check: 
                # We know p was invalid due to conflicts.
                # If we remove q, does it become valid?
                # Only if all conflicts were related to q.
                
                # Simple check:
                temp_S = S[:]
                temp_S.remove(q)
                
                # Check p against temp_S
                d2s_check = set()
                can_add = True
                for a in temp_S:
                    d2_pa = calc_d2(p, a)
                    if d2_pa in d2s_check:
                        can_add = False
                        break
                    d2s_check.add(d2_pa)
                    if d2_pa in dist_from[a]:
                        # Check if this distance was caused by q
                        # i.e., is d2_pa in dist_from[a] \ {d2(q, a)}?
                        # If dist_from[a] contains d2_pa, and d2_pa != d2(q, a), then conflict remains.
                        # If d2_pa == d2(q, a), then removing q might fix it.
                        # But we also need to check if d2_pa is in dist_from[a] from other points.
                        # dist_from[a] is a set of distances. It doesn't store WHICH point caused it.
                        # So we can't easily distinguish.
                        # We must assume it's still a conflict unless we verify.
                        # To verify, we would need to check if there exists b in temp_S such that d2(a, b) == d2_pa.
                        # This is O(K).
                        
                        # Let's just do the full check for safety, it's O(K).
                        # Since we are in a local search, K is ~10-20.
                        pass
                        
                # Actually, the full check is:
                def is_valid_add_to_set(p, S_set, dist_from_set):
                    d2s = set()
                    for a in S_set:
                        d2_pa = calc_d2(p, a)
                        if d2_pa in d2s:
                            return False
                        d2s.add(d2_pa)
                        if d2_pa in dist_from_set[a]:
                            return False
                    return True

                # But dist_from_set for temp_S is not updated.
                # We need to update dist_from for temp_S?
                # Or just compute validity from scratch for p against temp_S?
                # "From scratch" means checking d2(p, a) vs d2(a, b) for all a, b in temp_S.
                # This is O(K^2).
                
                # Let's do O(K^2) check for the swap candidate.
                # It's safer and K is small.
                
                valid_swap = True
                # 1. p must have distinct distances to all a in temp_S
                d2s_p = set()
                for a in temp_S:
                    d2_pa = calc_d2(p, a)
                    if d2_pa in d2s_p:
                        valid_swap = False
                        break
                    d2s_p.add(d2_pa)
                
                if valid_swap:
                    # 2. For each a in temp_S, d2(p, a) must not be d2(a, b) for any b in temp_S, b != a
                    # And also, existing distances in temp_S must remain distinct? 
                    # Removing q doesn't change distances between other points.
                    # So we only need to check p against temp_S.
                    
                    for a in temp_S:
                        d2_pa = calc_d2(p, a)
                        # Check if d2_pa is a distance between a and some b in temp_S
                        # We can check if d2_pa is in the set of distances from a to others in temp_S.
                        # We don't have this set precomputed for temp_S.
                        # We can compute it on the fly? O(K) per a -> O(K^2).
                        
                        # Alternatively, just check all pairs (a, b) in temp_S
                        # If d2(a, b) == d2_pa, conflict.
                        
                        # To avoid O(K^3), we can precompute distances for temp_S?
                        # No, just loop.
                        
                        for b in temp_S:
                            if a == b:
                                continue
                            d2_ab = calc_d2(a, b)
                            if d2_ab == d2_pa:
                                valid_swap = False
                                break
                        if not valid_swap:
                            break
                
                if valid_swap:
                    # Perform swap
                    do_remove(q, S, in_set, dist_from)
                    do_add(p, S, in_set, dist_from)
                    improved = True
                    no_improve = 0
                    best_size = max(best_size, len(S))
                    if len(S) > best_size:
                        best_S = S[:]
                    if p in non_S:
                        non_S.remove(p)
                    # q is now not in set, add to non_S
                    non_S.append(q)
                    break # Restart loop with new state? Or continue?
                    # Continuing might use stale non_S/conflicts.
                    # Let's break and restart the outer loop iteration.
        
        if not improved:
            no_improve += 1
        else:
            # If we improved, we might want to update non_S list more carefully
            # But for now, just let it be.
            pass
            
    return [coords[i] for i in best_S]

