"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _edge_count(points):
    """Count unit-distance edges among the given set of lattice points.
    Distance 1 on the triangular lattice (axial coords a,b) corresponds to
    the six neighbours: (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1).
    """
    s = set(tuple(p) for p in points)
    count = 0
    for (a, b) in s:
        for da, db in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            if (a + da, b + db) in s:
                count += 1
    return count // 2

def construct():
    deadline = time.time() + 110.0  # safety margin under 120s limit
    N = 32
    OPT = 76  # Harborth's bound for n=32
    import math

    # Hexagonal seed: take hexagon of radius 3 (37 points), remove 5 outer points
    # to get 32 points with maximal edges.
    # Hexagon radius 3: all (a,b) with max(|a|,|b|,|a+b|) <= 3 in axial coords... 
    # Actually in axial coords, hexagon radius r is |a|+|b|+|a+b| <= 2r
    hex_pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) + abs(b) + abs(a + b) <= 6:
                hex_pts.append([a, b])
    # hex_pts should have 37 points
    # Remove 5 points from the outer ring to minimize edge loss
    # Outer ring points have |a|+|b|+|a+b| == 6
    inner = [p for p in hex_pts if abs(p[0]) + abs(p[1]) + abs(p[0]+p[1]) < 6]
    outer = [p for p in hex_pts if abs(p[0]) + abs(p[1]) + abs(p[0]+p[1]) == 6]
    # inner has 19 points, outer has 18. Need to keep 32-19=13 from outer.
    # To maximize edges, keep outer points that have the most neighbors in the kept set.
    # Greedy: repeatedly pick the outer point with most neighbors in (inner + already picked)
    kept_outer = []
    remaining_outer = list(outer)
    while len(kept_outer) < 13 and remaining_outer:
        best_pt = None
        best_count = -1
        for p in remaining_outer:
            a, b = p
            cnt = 0
            for da, db in [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]:
                nb = (a+da, b+db)
                if any(q[0]==nb[0] and q[1]==nb[1] for q in inner) or any(q[0]==nb[0] and q[1]==nb[1] for q in kept_outer):
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_pt = p
        kept_outer.append(best_pt)
        remaining_outer.remove(best_pt)
    
    hex_seed = inner + kept_outer
    hex_edges = _edge_count(hex_seed)
    
    best_pts = hex_seed
    best_edges = hex_edges

    if best_edges >= OPT:
        return best_pts

    # Spiral construction as fallback
    R = 12
    candidates = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            d = (abs(a) + abs(b) + abs(a + b)) // 2
            x = a + 0.5 * b
            y = (math.sqrt(3) / 2) * b
            angle = math.atan2(y, x)
            candidates.append((d, angle, a, b))
    candidates.sort(key=lambda x: (x[0], x[1]))
    spiral_pts = [[a, b] for d, angle, a, b in candidates[:N]]
    spiral_edges = _edge_count(spiral_pts)
    if spiral_edges > best_edges:
        best_pts = spiral_pts
        best_edges = spiral_edges
    if best_edges >= OPT:
        return best_pts

    # Local search: move one point to an adjacent empty cell if it improves
    # the edge count. Also try 2-point swaps.
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    random.seed(42)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        pts = [list(p) for p in best_pts]
        pt_set = set(tuple(p) for p in pts)
        for i in range(N):
            if time.time() > deadline:
                break
            a, b = pts[i]
            for da, db in directions:
                na, nb = a + da, b + db
                if (na, nb) in pt_set:
                    continue
                if abs(na) > 64 or abs(nb) > 64:
                    continue
                # Tentative move
                pts[i] = [na, nb]
                pt_set.discard((a, b))
                pt_set.add((na, nb))
                new_edges = _edge_count(pts)
                if new_edges > best_edges:
                    best_pts = [list(p) for p in pts]
                    best_edges = new_edges
                    improved = True
                    if best_edges >= OPT:
                        break
                else:
                    # Revert
                    pts[i] = [a, b]
                    pt_set.discard((na, nb))
                    pt_set.add((a, b))
            if best_edges >= OPT:
                break

        # 2-point swap search (only if not at optimum)
        if best_edges < OPT and time.time() < deadline:
            pts = [list(p) for p in best_pts]
            pt_set = set(tuple(p) for p in pts)
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(i+1, N):
                    if time.time() > deadline:
                        break
                    a1, b1 = pts[i]
                    a2, b2 = pts[j]
                    # Try swapping: move i to where j is, j to where i is? No, that's just a swap of positions.
                    # Better: try moving i to a neighbor of j, and j to a neighbor of i, if both are empty.
                    # Simpler: try all pairs of empty cells adjacent to i and j.
                    for da1, db1 in directions:
                        na1, nb1 = a1 + da1, b1 + db1
                        if (na1, nb1) in pt_set or abs(na1) > 64 or abs(nb1) > 64:
                            continue
                        for da2, db2 in directions:
                            na2, nb2 = a2 + da2, b2 + db2
                            if (na2, nb2) in pt_set or abs(na2) > 64 or abs(nb2) > 64:
                                continue
                            if (na1, nb1) == (na2, nb2):
                                continue
                            # Tentative 2-move
                            pts[i] = [na1, nb1]
                            pts[j] = [na2, nb2]
                            pt_set.discard((a1, b1))
                            pt_set.discard((a2, b2))
                            pt_set.add((na1, nb1))
                            pt_set.add((na2, nb2))
                            new_edges = _edge_count(pts)
                            if new_edges > best_edges:
                                best_pts = [list(p) for p in pts]
                                best_edges = new_edges
                                improved = True
                                if best_edges >= OPT:
                                    break
                            else:
                                # Revert
                                pts[i] = [a1, b1]
                                pts[j] = [a2, b2]
                                pt_set.discard((na1, nb1))
                                pt_set.discard((na2, nb2))
                                pt_set.add((a1, b1))
                                pt_set.add((a2, b2))
                        if best_edges >= OPT:
                            break
                    if best_edges >= OPT:
                        break

    # Post-processing: Center the optimal configuration to minimize
    # sum of squared distances from origin, in case of tie-breakers.
    if best_edges >= OPT:
        sa = sum(p[0] for p in best_pts)
        sb = sum(p[1] for p in best_pts)
        # Shift to make centroid as close to (0,0) as possible
        # Since coords are integers, we can shift by integers.
        # Ideal shift is -centroid.
        # But we must keep integer coords.
        # Let's try shifts around the ideal centroid.
        ideal_a = -sa / N
        ideal_b = -sb / N
        
        best_shift = (0, 0)
        best_metric = float('inf')
        
        # Check integer shifts near the ideal
        for da in range(int(ideal_a) - 2, int(ideal_a) + 3):
            for db in range(int(ideal_b) - 2, int(ideal_b) + 3):
                # Apply shift
                shifted = [[p[0] + da, p[1] + db] for p in best_pts]
                # Check bounds
                if any(abs(x) > 64 or abs(y) > 64 for x, y in shifted):
                    continue
                # Check distinctness (shift preserves distinctness)
                # Metric: sum of squared distances from (0,0)
                metric = sum(x*x + y*y for x, y in shifted)
                if metric < best_metric:
                    best_metric = metric
                    best_shift = (da, db)
        
        # Apply best shift
        best_pts = [[p[0] + best_shift[0], p[1] + best_shift[1]] for p in best_pts]

    return best_pts
# EVOLVE-BLOCK-END
