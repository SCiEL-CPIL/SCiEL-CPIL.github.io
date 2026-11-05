"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # safety margin before 330s kill
    
    total_points = N * N
    # Map (x,y) to index
    def idx(x, y):
        return x * N + y
    
    # Precompute all point coordinates
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute squared distances between all pairs (total_points x total_points)
    # This is 4096 x 4096 = 16.7M entries, int32 = ~67MB
    # Compute in chunks to save memory
    dx = coords[:, 0:1] - coords[1:, 0]  # (4096, 4095)
    dy = coords[:, 1:2] - coords[1:, 1]
    # Actually let's do full matrix
    dx_full = coords[:, 0:1] - coords[0:1, 0]  # (4096, 1) - (1, 4096) = (4096, 4096)
    dy_full = coords[:, 1:2] - coords[0:1, 1]
    dist_sq = dx_full * dx_full + dy_full * dy_full  # (4096, 4096) int32
    
    # Build conflict matrix: conflict[i,j] = True if points i and j cannot both be selected
    # Conflict if there exists k such that dist(i,k)==dist(j,k) or dist(i,k)==dist(i,j) or dist(j,k)==dist(i,j)
    # 
    # For efficiency: for each k, group points by distance to k.
    # Points in same group from k are pairwise conflicting.
    # Also, for pair (i,j): if dist(i,j) == dist(i,k) for some k, then conflict.
    # 
    # Let's use the first criterion primarily: for each k, all pairs at same distance from k conflict.
    # This covers the case where k is the apex.
    # For the other two cases (i or j is apex), we need: dist(i,k) == dist(i,j) means k is at same distance from i as j is from i.
    # This is equivalent to: for pair (i,j), if there exists k with dist(i,k) = dist(i,j), conflict.
    # Which means: for each i, all points j and k with dist(i,j) = dist(i,k) conflict.
    # That's the same as: for each i, group by distance, all pairs in same group conflict.
    # 
    # So actually, the conflict condition is simply: for some vertex v, two other points are at the same distance from v.
    # i.e., conflict(i,j) = exists v such that dist(v,i) == dist(v,j)
    # 
    # This is exactly: for each v, all pairs of points at the same distance from v are conflicting.
    
    conflict = np.zeros((total_points, total_points), dtype=np.uint8)
    
    # For each vertex v, group other points by distance to v
    for v in range(total_points):
        # Get distances from v to all other points
        d_to_v = dist_sq[:, v]  # (total_points,)
        # For each unique distance, all points at that distance from v conflict with each other
        # Use unique to group
        unique_dists, inverse = np.unique(d_to_v, return_inverse=True)
        # For each group, set conflict among all members
        for dist_val, group_idx in enumerate(inverse):
            if group_idx < 0:  # v itself
                continue
            members = np.where(group_idx == dist_val)[0]
            if len(members) < 2:
                continue
            # Set conflict for all pairs in members
            conflict[np.ix_(members, members)] = 1
    
    # Note: this only captures the case where the apex is a grid point that IS in our candidate set.
    # But actually, the isosceles triangle condition is: three points A, B, C form isosceles if
    # any two distances among {|AB|, |AC|, |BC|} are equal.
    # So conflict(A,B) exists if there's a C such that |AC|=|BC| or |AC|=|AB| or |BC|=|AB|.
    # The first case: |AC|=|BC| means C is equidistant from A and B. C doesn't need to be in our set,
    # but C must be a grid point (since we're only selecting grid points).
    # Wait, no - C must be one of the selected points. So C is in our set.
    # 
    # So conflict(A,B) = exists C in grid such that C != A, C != B, and (|AC|=|BC| or |AC|=|AB| or |BC|=|AB|)
    # But C only matters if C is also selected. In the independent set formulation, we want:
    # A and B are adjacent in conflict graph if NO valid set can contain both A and B.
    # A and B can't both be in a valid set if there EXISTS any grid point C such that {A,B,C} forms isosceles.
    # Because if we select A and B, and later try to select C, it would fail. But we could also never select C.
    # 
    # Actually, the constraint is: for any three selected points, no isosceles triangle.
    # So A and B conflict only if for EVERY possible C that could be selected, {A,B,C} is isosceles? No.
    # 
    # Let me reconsider. The constraint is: no three points in the selected set form an isosceles triangle.
    # So when building the set greedily, we add points one by one. When considering adding point P,
    # we check that for every pair of already-selected points (A, B), the triple {A, B, P} is not isosceles.
    # 
    # For the conflict graph / independent set approach: two points A and B are in conflict if there
    # exists a third point C (any grid point) such that {A, B, C} forms an isosceles triangle.
    # This is a sufficient condition for conflict: if A and B are in conflict, we can't select both
    # because... wait, that's not right either. If A and B conflict with some C, we can still select
    # A and B as long as we don't select C.
    # 
    # Hmm, the independent set formulation needs: A and B are adjacent if they CANNOT both be in any
    # valid solution. A and B can both be in a valid solution iff for ALL grid points C, {A,B,C} is
    # NOT isosceles (or C is not selected). But we control which C's are selected.
    # 
    # Actually, A and B can both be selected if there's no C that we MUST select that would form
    # isosceles with them. Since we're choosing the set, A and B conflict only if EVERY valid set
    # containing A must exclude B. This is complex.
    # 
    # Simpler approach: just use the greedy with fast validity checking. The conflict graph idea
    # from the directive is: "edges connect point pairs that share a common isosceles vertex".
    # I think the intent is: for each potential apex point v, all pairs of points at the same
    # distance from v are in conflict. Then find a maximum independent set.
    # 
    # But as I noted, this over-constrains. However, it might still give good results as a heuristic.
    # Let me just improve the greedy approach with better data structures instead.
    
    # Let me use a smarter greedy: maintain for each point the set of distances to already-selected points.
    # A new point P can be added if:
    # 1. For all selected A, dist(P,A) is not in dist_set[A] (no isosceles with A as apex)
    # 2. All dist(P,A) for selected A are distinct (no isosceles with P as apex)
    # 
    # The current code does this but in pure Python which is slow. Let me use numpy for the
    # distance checks.
    
    def greedy_with_numpy(seed_val):
        rng = np.random.default_rng(seed_val)
        order = rng.permutation(total_points)
        
        selected = []
        selected_set = set()
        # dist_to_selected[i] = set of squared distances from point i to selected points
        # For points not yet selected, this grows as we add points
        dist_sets = [set() for _ in range(total_points)]
        
        for pi in order:
            if time.time() > deadline:
                break
            
            px, py = coords[pi]
            
            # Check condition 2: all distances from P to selected points are distinct
            # And condition 1: for each selected A, dist(P,A) not in dist_sets[A]
            
            # Get selected points
            if not selected:
                selected.append(pi)
                selected_set.add(pi)
                continue
            
            sel_arr = np.array(selected, dtype=np.int32)
            # Compute distances from pi to all selected
            dx = coords[sel_arr, 0] - px
            dy = coords[sel_arr, 1] - py
            d_sqs = dx * dx + dy * dy
            
            # Check distinctness
            if len(set(d_sqs.tolist())) != len(d_sqs):
                continue
            
            # Check against each selected point's distance set
            valid = True
            for si, d in zip(selected, d_sqs.tolist()):
                if d in dist_sets[si]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add pi
            selected.append(pi)
            selected_set.add(pi)
            # Update dist_sets for all existing selected points
            for si, d in zip(selected[:-1], d_sqs.tolist()):
                dist_sets[si].add(d)
            # Initialize dist_set for pi
            dist_sets[pi] = set(d_sqs.tolist())
        
        return selected
    
    best = []
    
    # Try multiple seeds
    for s in range(50):
        if time.time() > deadline:
            break
        pts = greedy_with_numpy(s)
        if len(pts) > len(best):
            best = pts
    
    # Convert indices to (x,y)
    result = [tuple(coords[i]) for i in best]
    
    if len(result) < 5:
        # Fallback
        xs, used = [], set()
        x = 0
        while x < N:
            diffs = {abs(x - y) for y in xs}
            if not (diffs & used):
                used |= diffs
                xs.append(x)
            x += 1
        result = [(x, 0) for x in xs]
    
    return result
# EVOLVE-BLOCK-END
