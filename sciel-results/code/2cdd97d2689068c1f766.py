# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 620.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a structural re-formulation:
    instead of greedily adding points to a fixed base complement, we treat
    the problem as a maximum independent set (MIS) on a conflict graph.
    Vertices = points that can be removed from N (i.e., added to S).
    Edges = pairs of points that cannot both be in S (because they lie on
    a common line for some direction, and that direction is the only valid
    witness for some third point, or simply because adding both would break
    validity).
    
    We use a greedy MIS heuristic with randomization and multiple restarts,
    which is a fundamentally different search paradigm than incremental
    single-point addition.
    """
    p = 7
    k = 3  # alpha^2 = 3

    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)
    
    def field_sub(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)
    
    # Enumerate all points in F_49^2
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    n_total = len(all_points)  # 2401
    
    # Directions: (1, m) for all m in F_49, plus (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 1), (0, 0)))  # (0, 1)
    
    n_dirs = len(directions)  # 50
    
    # Precompute line neighbors: line_neighbors[d][i] = list of other points on line through i with direction d
    line_neighbors = []
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        line_map = {}
        
        for pt_idx, pt in enumerate(all_points):
            (x, y) = pt
            if vx == (1, 0):  # direction (1, m)
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:  # direction (0, 1)
                c = x
            
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        
        neigh = [[] for _ in range(n_total)]
        for c, pts_on_line in line_map.items():
            for i in pts_on_line:
                neigh[i] = [j for j in pts_on_line if j != i]
        
        line_neighbors.append(neigh)
    
    # Base construction: unital-based
    squares = {0, 1, 2, 4}
    S_base = set()
    for pt_idx, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S_base.add(pt_idx)
    
    # Initialize validity counters
    # valid_dir[x][d] = True if direction d is valid for point x (no other point on line is in S)
    valid_dir = [[False] * n_dirs for _ in range(n_total)]
    valid_count = [0] * n_total
    
    for x in range(n_total):
        for d in range(n_dirs):
            is_valid = True
            for other in line_neighbors[d][x]:
                if other in S_base:
                    is_valid = False
                    break
            valid_dir[x][d] = is_valid
            if is_valid:
                valid_count[x] += 1
    
    # Identify candidate points that can potentially be added to S
    # A point can be added if valid_count > 0
    candidates = [i for i in range(n_total) if i not in S_base and valid_count[i] > 0]
    
    # Build conflict graph: two candidates conflict if adding both to S
    # would break validity for some point.
    # For efficiency, we use a greedy approach with random restarts.
    
    best_S = set(S_base)
    best_size = len(S_base)
    
    np.random.seed(42)
    
    # Multiple restarts with different random orderings
    n_restarts = 20
    for restart in range(n_restarts):
        if _time_left() < 60:
            break
        
        S = set(S_base)
        # Copy validity state
        valid_dir_cur = [row[:] for row in valid_dir]
        valid_count_cur = valid_count[:]
        
        # Shuffle candidates
        cand_list = candidates[:]
        np.random.shuffle(cand_list)
        
        # Greedy MIS: add point if it doesn't conflict with already-added points
        # A point p conflicts with already-added point q if:
        # For some direction d, p and q are on the same line, and there exists
        # a point x such that d is x's only valid direction, and both p and q
        # are on the line through x with direction d.
        
        # More precisely: adding p breaks x if valid_dir_cur[x][d] was True and
        # p is on line(x, d). If we've already added q which also breaks x,
        # then x is already broken, so adding p doesn't cause additional harm.
        # But if x still has valid_count >= 1 and adding p would reduce it to 0,
        # that's bad.
        
        # Simplified greedy: try adding each candidate in random order.
        # A candidate c can be added if:
        # 1. valid_count_cur[c] > 0 (c itself has a valid witness)
        # 2. For all neighbors x of c (in any direction), if valid_dir_cur[x][d] is True,
        #    then valid_count_cur[x] > 1 (so removing this direction doesn't break x)
        
        added = 0
        for cand in cand_list:
            if _time_left() < 30:
                break
            
            if valid_count_cur[cand] == 0:
                continue
            
            # Check if adding cand breaks any point
            breaks = False
            for d in range(n_dirs):
                for x in line_neighbors[d][cand]:
                    if valid_dir_cur[x][d]:
                        if valid_count_cur[x] == 1:
                            breaks = True
                            break
                if breaks:
                    break
            
            if breaks:
                continue
            
            # Accept cand
            S.add(cand)
            added += 1
            
            # Update validity
            for d in range(n_dirs):
                for x in line_neighbors[d][cand]:
                    if valid_dir_cur[x][d]:
                        valid_dir_cur[x][d] = False
                        valid_count_cur[x] -= 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Build final result
    points = []
    for pt_idx in range(n_total):
        if pt_idx not in best_S:
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])
    
    arr = np.array(points, dtype=np.int64)
    return arr


# EVOLVE-BLOCK-END
