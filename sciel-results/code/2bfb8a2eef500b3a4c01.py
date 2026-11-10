# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49 = 7^2).
    
    Strategy: Model as maximum independent set on a conflict graph where two points
    conflict if they lie on the same line in the same direction (i.e., they are on the
    same parallel line for some direction). Actually, the constraint is:
    For each point x, there must exist a direction d such that the line through x in 
    direction d has no other points in S.
    
    Equivalently: S is a set where for every point x (in S or not), there exists a direction d
    such that no other point of S lies on the line through x in direction d.
    
    This means: for each x, at least one of the 50 lines through x (one per direction) 
    contains no other point of S.
    
    We use greedy independent set construction with random restarts and local search.
    """
    p = 7
    q = p * p  # 49
    deadline = time.time() + 600.0

    # Field arithmetic
    k_const = 3  # alpha^2 = 3

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k_const) % p,
                (a[0]*b[1] + a[1]*b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    # Precompute all field elements as tuples
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Directions: 50 total
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    
    num_dirs = len(directions)  # 50

    # Precompute line membership: for each direction d, group points into lines
    # line_of[d][x] = set of points on the same line as x in direction d
    # More efficiently: for each direction d, for each point x, compute the line_id
    # and group points by line_id.
    
    # For direction d = (vx, vy), the line through x is determined by a "line identifier".
    # For v = (1, m): line through (px, py) is {(px + t, py + t*m) : t in F_49}
    #   The line is determined by (py - px*m) ... but in F_49.
    #   Actually, two points x1, x2 are on the same line in direction (1,m) iff
    #   x2 - x1 = t*(1, m) for some t, i.e., x2x - x1x = t and x2y - x1y = t*m.
    #   So (x2y - x1y) = m * (x2x - x1x).
    #   The line through x in direction (1,m) can be identified by the value c = py - px*m.
    #   For vertical direction (0,1): lines are identified by px.
    
    # Let's precompute for each direction, the line_id for each point.
    # line_ids[d][x] = integer 0..48 identifying which line in direction d contains x.
    
    line_ids = np.zeros((num_dirs, 2401), dtype=np.int32)
    
    # For direction (1, m) where m = (ma, mb):
    # c = py - px * m
    # For direction (0, 1): c = px
    
    # We need to map field elements to integers 0..48 for line ids.
    # Map (a, b) -> a * 7 + b
    
    def field_to_int(f):
        return f[0] * 7 + f[1]
    
    for d_idx, (vx, vy) in enumerate(directions):
        if vx == (1, 0):
            # Direction (1, m) where m = vy
            for x_idx in range(2401):
                pt = idx_to_pt(x_idx)
                px, py = pt[0], pt[1]
                # c = py - px * vy
                pxv = px
                c = sub_result = ((py[0] - mul(px, vy)[0]) % p, (py[1] - mul(px, vy)[1]) % p)
                line_ids[d_idx, x_idx] = field_to_int(c)
        else:
            # Vertical direction (0, 1)
            for x_idx in range(2401):
                pt = idx_to_pt(x_idx)
                px = pt[0]
                line_ids[d_idx, x_idx] = field_to_int(px)

    # Now, for the Nikodym constraint:
    # S is valid if for every point x, there exists a direction d such that
    # no other point in S is on the same line as x in direction d.
    # i.e., for every x, exists d such that |{s in S : line_ids[d][s] == line_ids[d][x]}| <= 1 (only x itself if x in S)
    
    # Equivalently: for every x and for every d, if x is the only point of S on its line in direction d, that's fine.
    # The constraint fails for x if for ALL d, there is some other s in S on the same line as x in direction d.
    
    # Let's define: for each point x and direction d, count_d[x][d] = number of points in S on the same line as x in direction d.
    # The constraint for x is: exists d such that count_d[x][d] == 1 if x in S, or count_d[x][d] == 0 if x not in S.
    # Wait, let me re-read the problem.
    
    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49},
    #  with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    
    # So for each x, there exists a direction v such that all points on the line through x in direction v,
    # EXCEPT x itself, are in N (i.e., not in S).
    
    # This means: for each x, exists d such that the line through x in direction d contains no points of S other than possibly x.
    
    # So: for each x, exists d such that |S ∩ {points on line through x in direction d} \ {x}| = 0.
    
    # Let line_count[x][d] = |S ∩ {points on line through x in direction d} \ {x}|
    # Constraint: for each x, exists d such that line_count[x][d] == 0.
    
    # To maximize |S|, we want to add points to S while maintaining this property.
    
    # Approach: Greedy with random restarts.
    # Start with S = empty. Add points one by one. A point can be added if adding it doesn't violate the constraint.
    # When we add point p to S, for each direction d, all points x on the line through p in direction d (x != p)
    # will have their line_count[x][d] incremented by 1.
    # The constraint for x is violated if before adding p, line_count[x][d] == 0 and d was the only direction
    # with line_count[x][d] == 0 for x.
    
    # Let's track zero_count[x] = number of directions d with line_count[x][d] == 0.
    # Initially, all zero_count[x] = 50 (all directions have count 0).
    # When we add p, for each direction d, for each x on line through p in direction d (x != p):
    #   if line_count[x][d] == 0, then zero_count[x] decreases by 1.
    #   if zero_count[x] becomes 0, constraint is violated.
    
    # We need to precompute, for each point p and direction d, the list of other points on the line.
    # This is expensive to store, but we can compute on the fly.
    
    # Actually, let's precompute for each (p, d) the set of other points.
    # line_others[p][d] = list of point indices on the line through p in direction d, excluding p.
    # There are 2401 * 50 * 48 = 5,762,400 entries. As a list of lists, this might be too much memory.
    # Let's compute on the fly but cache.
    
    # Better approach: for each direction d, precompute the line groups.
    # For direction d, points are partitioned into 49 lines of 49 points each.
    # line_groups[d][line_id] = list of point indices on that line.
    
    line_groups = []
    for d_idx in range(num_dirs):
        groups = {}
        for x_idx in range(2401):
            lid = line_ids[d_idx, x_idx]
            if lid not in groups:
                groups[lid] = []
            groups[lid].append(x_idx)
        line_groups.append(groups)
    
    # Now, for a point p and direction d, the other points on the line are:
    # line_groups[d][line_ids[d][p]] \ {p}
    
    # Let's implement the greedy algorithm with random restarts.
    
    best_S = set()
    best_size = 0
    
    np.random.seed(12345)
    
    num_restarts = 50  # will adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        S = set()
        line_count = np.zeros((2401, num_dirs), dtype=np.int32)
        zero_count = np.full(2401, num_dirs, dtype=np.int32)
        
        # Create a random ordering of all points
        order = np.random.permutation(2401)
        
        for p_idx in order:
            if time.time() > deadline - 30:
                break
            
            # Check if we can add p_idx to S
            # For each direction d, find all x on line through p in direction d (x != p)
            # For each such x, if line_count[x][d] == 0, zero_count[x] would decrease.
            # If any x would have zero_count[x] become 0, we can't add p.
            
            can_add = True
            affected = []  # (x_idx, d_idx) where line_count[x][d] == 0
            
            for d_idx in range(num_dirs):
                lid = line_ids[d_idx, p_idx]
                group = line_groups[d_idx][lid]
                for x_idx in group:
                    if x_idx == p_idx:
                        continue
                    if line_count[x_idx, d_idx] == 0:
                        affected.append((x_idx, d_idx))
                        if zero_count[x_idx] == 1:
                            can_add = False
                            break
                if not can_add:
                    break
            
            if can_add:
                S.add(p_idx)
                for x_idx, d_idx in affected:
                    line_count[x_idx, d_idx] += 1
                    zero_count[x_idx] -= 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Local search on the best S: try to add more points
    if time.time() < deadline - 30:
        S = set(best_S)
        line_count = np.zeros((2401, num_dirs), dtype=np.int32)
        zero_count = np.full(2401, num_dirs, dtype=np.int32)
        
        # Recompute line_count and zero_count for current S
        for s_idx in S:
            for d_idx in range(num_dirs):
                lid = line_ids[d_idx, s_idx]
                group = line_groups[d_idx][lid]
                for x_idx in group:
                    if x_idx == s_idx:
                        continue
                    line_count[x_idx, d_idx] += 1
        
        for x_idx in range(2401):
            zero_count[x_idx] = np.sum(line_count[x_idx] == 0)
        
        # Try to add points not in S
        candidates = [i for i in range(2401) if i not in S]
        np.random.shuffle(candidates)
        
        for p_idx in candidates:
            if time.time() > deadline - 15:
                break
            
            can_add = True
            affected = []
            
            for d_idx in range(num_dirs):
                lid = line_ids[d_idx, p_idx]
                group = line_groups[d_idx][lid]
                for x_idx in group:
                    if x_idx == p_idx:
                        continue
                    if line_count[x_idx, d_idx] == 0:
                        affected.append((x_idx, d_idx))
                        if zero_count[x_idx] == 1:
                            can_add = False
                            break
                if not can_add:
                    break
            
            if can_add:
                S.add(p_idx)
                for x_idx, d_idx in affected:
                    line_count[x_idx, d_idx] += 1
                    zero_count[x_idx] -= 1
    
    # Build the output: N = F_49^2 \ S
    points = []
    for i in range(2401):
        if i not in S:
            points.append(idx_to_pt(i))
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
