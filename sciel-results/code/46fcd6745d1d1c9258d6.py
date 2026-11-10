"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    """
    Return 32 distinct integer pairs [a, b] on the triangular lattice
    (point = a + b*omega, omega = exp(i*pi/3)) with |a|,|b| <= 64,
    mutual distance >= 1, maximizing the number of pairs at distance exactly 1.
    
    Uses deterministic branch-and-bound over the 37-point hexagonal region
    (radius 3), removing 5 points. Upper bounds use degree-sum relaxation
    to prove the Harborth optimum of 76 edges.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Unit distance vectors on triangular lattice in (a,b) coordinates
    unit_vectors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        pts = set(tuple(p) for p in points)
        edges = 0
        for p in pts:
            for u in unit_vectors:
                q = (p[0] + u[0], p[1] + u[1])
                if q in pts:
                    edges += 1
        return edges // 2
    
    def degree(p, points_set):
        """Count neighbors at unit distance."""
        d = 0
        for u in unit_vectors:
            if (p[0] + u[0], p[1] + u[1]) in points_set:
                d += 1
        return d
    
    def hex_points(radius):
        """Generate hexagonal cluster of given radius centered at origin."""
        pts = set()
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                c = -(a + b)
                hex_dist = max(abs(a), abs(b), abs(c))
                if hex_dist <= radius:
                    pts.add((a, b))
        return pts
    
    # The 37-point hexagon of radius 3
    hex3 = hex_points(3)
    hex3_list = sorted(hex3, key=lambda p: (p[1], p[0]))
    n_total = len(hex3_list)  # 37
    n_remove = n_total - 32   # 5
    
    # Precompute adjacency in the full hexagon
    adj = {}
    for p in hex3_list:
        adj[p] = set()
    for i, p in enumerate(hex3_list):
        for u in unit_vectors:
            q = (p[0] + u[0], p[1] + u[1])
            if q in hex3:
                adj[p].add(q)
    
    # Full hexagon has E_total edges
    E_total = sum(len(adj[p]) for p in hex3_list) // 2
    
    # Best solution found so far
    best_edges = 0
    best_removed = None
    
    # Branch and bound: choose which 5 points to remove
    # Key insight: removing a point with degree d loses d edges, but may also
    # lose additional edges if two removed points were adjacent to each other.
    # Upper bound: if we've removed set R so far, and need to remove k more,
    # the max additional edges we can preserve is bounded by the sum of the
    # top-k degrees among remaining points (minus edges among them).
    
    # Order points by "removal cost" - points with lower degree are cheaper to remove
    # Sort by degree in full hexagon, then by hex distance from center
    def removal_cost(p):
        d = len(adj[p])
        hex_d = max(abs(p[0]), abs(p[1]), abs(p[0]+p[1]))
        return (d, hex_d)
    
    # For B&B, we need to decide which 5 to remove.
    # Strategy: use the fact that the optimal solution removes 5 boundary points.
    # We'll do a systematic search over combinations of 5 points to remove,
    # but prune aggressively using upper bounds.
    
    # Upper bound function: given a partial removal set R (size r < 5),
    # what's the max possible edge count?
    # Current edges in hex3 \ R = E_total - (edges lost by removing R)
    # Edges lost by removing R = sum of degrees of points in R within (hex3 \ R_before)
    # But for an upper bound on the FINAL result (after removing 5 total),
    # we can say: final_edges <= E_total - min_edges_lost
    # where min_edges_lost is the minimum number of edges lost when removing 5 points.
    
    # A simpler upper bound: for any set S of 32 points, 
    # |E(S)| <= sum of top-32 degrees in the induced subgraph... 
    # Actually, let's use: |E(S)| <= (1/2) * sum_{v in S} deg(v)
    # And deg(v) in S <= deg(v) in hex3
    # So |E(S)| <= (1/2) * sum of top-32 degrees in hex3
    
    # Compute all degrees in hex3
    degs = sorted([len(adj[p]) for p in hex3_list], reverse=True)
    # Upper bound: (1/2) * sum of top 32 degrees
    UB_global = sum(degs[:32]) // 2
    
    # For a tighter bound during search: if we've decided to remove some points,
    # the remaining 32 points have degrees at most their degrees in hex3.
    # But if we remove point p, all its neighbors lose one degree.
    
    # Let's do a direct combinatorial search with pruning.
    # We'll enumerate which 5 points to remove, but use symmetry and bounds.
    
    # First, let's try the greedy approach as a baseline
    def greedy_remove():
        pts = set(hex3_list)
        for _ in range(n_remove):
            # Find point with minimum degree in current set
            min_deg = min(degree(p, pts) for p in pts)
            candidates = [p for p in pts if degree(p, pts) == min_deg]
            # Remove the one farthest from center
            to_remove = max(candidates, key=lambda p: max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])))
            pts.remove(to_remove)
        return pts
    
    greedy_result = greedy_remove()
    greedy_edges = count_edges(greedy_result)
    best_edges = max(best_edges, greedy_edges)
    best_points = set(greedy_result)
    
    # Now try systematic search over removal sets
    # The 37 points: let's index them
    # For efficiency, only consider removing points from the outer ring (degree <= 4)
    # Inner points have higher degree and are more costly to remove
    
    # Classify points by their degree in the full hexagon
    outer_points = [p for p in hex3_list if len(adj[p]) <= 4]
    mid_points = [p for p in hex3_list if len(adj[p]) == 5]
    inner_points = [p for p in hex3_list if len(adj[p]) == 6]
    
    # The 37-point hexagon: center has degree 6, ring 1 has 6 points with degree 5,
    # ring 2 has 12 points with degree 5 or 4, ring 3 has 18 points with degree 3 or 4
    # Let's verify
    print(f"Outer (deg<=4): {len(outer_points)}, Mid (deg=5): {len(mid_points)}, Inner (deg=6): {len(inner_points)}")
    
    # For the optimal solution, we should primarily remove outer points.
    # Let's search over combinations of removals, prioritizing outer points.
    
    # Use itertools.combinations but with pruning
    # To keep it tractable, limit the search space:
    # - At most 2 from mid_points
    # - At most 1 from inner_points
    # - Rest from outer_points
    
    # Actually, let's be more systematic. We'll do a depth-first search
    # choosing which points to remove, with pruning.
    
    def bnb_search():
        nonlocal best_edges, best_points
        
        # Sort candidate points to remove: prefer lower degree, then farther from center
        all_pts = sorted(hex3_list, key=lambda p: (len(adj[p]), max(abs(p[0]), abs(p[1]), abs(p[0]+p[1]))))
        
        # We need to choose 5 points to remove from the 37.
        # Use recursive search with pruning.
        
        removed = []
        remaining_set = set(hex3_list)
        
        def current_edges():
            return count_edges(remaining_set)
        
        def upper_bound(removed_so_far, remaining_to_remove):
            """Upper bound on final edge count."""
            # Current edges in remaining_set
            cur_e = current_edges()
            # When we remove more points, we lose at least 0 edges each
            # (a point with degree 0 in remaining_set loses 0 edges)
            # But we need a TIGHTER bound.
            # Each additional removal of point p loses degree(p, remaining_set) edges.
            # The minimum additional loss is the sum of the smallest remaining_to_remove degrees.
            # Compute degrees of all remaining points
            if remaining_to_remove == 0:
                return cur_e
            degs_remaining = sorted([degree(p, remaining_set) for p in remaining_set])
            # We'll remove remaining_to_remove more points, losing at least
            # the sum of the smallest remaining_to_remove degrees
            min_loss = sum(degs_remaining[:remaining_to_remove])
            return cur_e - min_loss  # Wait, this is a LOWER bound on loss, so UPPER bound on edges
        
        # Actually, the upper bound should be: current_edges - (minimum possible additional loss)
        # Minimum additional loss when removing k more points = sum of k smallest degrees among remaining
        # So upper bound = current_edges - sum_of_k_smallest_degrees
        
        # Let me redo: we want an upper bound on the FINAL edge count.
        # final_edges = current_edges - (edges lost by removing remaining_to_remove more points)
        # edges lost >= sum of the remaining_to_remove smallest degrees in remaining_set
        # So final_edges <= current_edges - sum_of_k_smallest_degrees
        # This is correct as an upper bound.
        
        def upper_bound_correct(remaining_set, remaining_to_remove):
            if remaining_to_remove == 0:
                return count_edges(remaining_set)
            degs = sorted([degree(p, remaining_set) for p in remaining_set])
            min_loss = sum(degs[:remaining_to_remove])
            return count_edges(remaining_set) - min_loss
        
        # DFS
        def dfs(removed_count, start_idx):
            nonlocal best_edges, best_points
            
            if time.time() > deadline:
                return
            
            if removed_count == n_remove:
                e = count_edges(remaining_set)
                if e > best_edges:
                    best_edges = e
                    best_points = set(remaining_set)
                return
            
            # Upper bound check
            ub = upper_bound_correct(remaining_set, n_remove - removed_count)
            if ub <= best_edges:
                return
            
            # Try removing each candidate point from start_idx onward
            # Only consider points that are "reasonable" to remove
            for i in range(start_idx, len(all_pts)):
                p = all_pts[i]
                if p not in remaining_set:
                    continue
                
                # Remove p
                remaining_set.remove(p)
                dfs(removed_count + 1, i + 1)
                remaining_set.add(p)
                
                # If we found the theoretical optimum, we can stop
                if best_edges >= 76:
                    return
        
        dfs(0, 0)
    
    bnb_search()
    
    # If we haven't reached 76, try a few more heuristics
    if best_edges < 76:
        # Try different removal orderings
        for perm_seed in range(10):
            if time.time() > deadline:
                break
            
            # Different tie-breaking
            pts = set(hex3_list)
            for _ in range(n_remove):
                min_deg = min(degree(p, pts) for p in pts)
                candidates = [p for p in pts if degree(p, pts) == min_deg]
                # Different tie-breaking based on perm_seed
                to_remove = candidates[perm_seed % len(candidates)]
                pts.remove(to_remove)
            
            e = count_edges(pts)
            if e > best_edges:
                best_edges = e
                best_points = pts
    
    # Final local search to polish
    if best_edges < 76 and time.time() < deadline - 10:
        pts = set(best_points)
        cur_e = best_edges
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for p in list(pts):
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        if da == 0 and db == 0:
                            continue
                        new_pos = (p[0] + da, p[1] + db)
                        if new_pos in pts:
                            continue
                        if abs(new_pos[0]) > 64 or abs(new_pos[1]) > 64:
                            continue
                        pts.remove(p)
                        pts.add(new_pos)
                        new_e = count_edges(pts)
                        if new_e > cur_e:
                            cur_e = new_e
                            improved = True
                            break
                        else:
                            pts.remove(new_pos)
                            pts.add(p)
                    if improved:
                        break
                if improved:
                    break
        if cur_e > best_edges:
            best_edges = cur_e
            best_points = pts
    
    result = sorted(best_points, key=lambda p: (p[1], p[0]))
    
    # Verify constraints
    assert len(result) == 32, f"Expected 32 points, got {len(result)}"
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64, f"Point {p} out of bounds"
    
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d2 = dist_sq(result[i], result[j])
            assert d2 >= 1, f"Points {result[i]} and {result[j]} too close: d^2={d2}"
    
    return [[p[0], p[1]] for p in result]
# EVOLVE-BLOCK-END
