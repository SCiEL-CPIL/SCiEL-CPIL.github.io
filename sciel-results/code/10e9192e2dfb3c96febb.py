import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    deadline = 600
    
    # Field elements: (a, b) -> a + b*alpha, alpha^2 = 3
    # Map to int 0..48
    def to_int(a, b):
        return a * p + b
        
    def to_pair(i):
        return (i // p, i % p)
        
    # Precompute addition and multiplication tables
    add = np.zeros((q, q), dtype=np.int8)
    mul = np.zeros((q, q), dtype=np.int8)
    
    for i in range(q):
        ia, ib = to_pair(i)
        for j in range(q):
            ja, jb = to_pair(j)
            sa = (ia + ja) % p
            sb = (ib + jb) % p
            add[i, j] = to_int(sa, sb)
            ma = (ia * ja + ib * jb * 3) % p
            mb = (ia * jb + ib * ja) % p
            mul[i, j] = to_int(ma, mb)
            
    # Directions: v = (1, m) for m in F_q, and v = (0, 1)
    # Each direction is (vx, vy) where vx, vy are elements of F_q (0..48)
    v_coords = []
    for ma in range(p):
        for mb in range(p):
            vx = to_int(1, 0)
            vy = to_int(ma, mb)
            v_coords.append((vx, vy))
    # Vertical direction
    vx = to_int(0, 0)
    vy = to_int(1, 0)
    v_coords.append((vx, vy))
    num_dirs = len(v_coords)  # 50
    
    # Precompute for each point y and each direction v, the set of points on the line
    # line_points[y_idx][v_idx] = list of point indices (excluding y_idx)
    line_points = [[None] * num_dirs for _ in range(total_points)]
    
    for y_idx in range(total_points):
        # y_idx = yx * q + yy, where yx, yy are elements of F_q
        yx = y_idx // q
        yy = y_idx % q
        
        for v_idx in range(num_dirs):
            vx, vy = v_coords[v_idx]
            pts = []
            for t in range(q):
                if t == 0:
                    continue
                tx = mul[t, vx]
                ty = mul[t, vy]
                px = add[yx, tx]
                py = add[yy, ty]
                pts.append(px * q + py)
            line_points[y_idx][v_idx] = pts
    
    # Initial construction: Start with all points in N
    # Then remove points that form a unital-like structure
    in_N = [True] * total_points
    
    # Remove a set of points to create the initial Nikodym set
    # We want every point x to be on a line where all other points are in N
    # Strategy: Remove points such that for each removed point x, 
    # there exists a direction v where all points on the line through x with direction v 
    # (except x itself) are still in N
    
    # Start by removing all points, then add back? No.
    # Let's try: Remove a sparse set of points.
    
    # Actually, the simplest approach:
    # N = all points except a set S.
    # For every point x (in or not in N), there must be a line through x
    # such that all other points on that line are in N.
    
    # If x is not in N (x in S), then for x, we need a line through x where all other points are in N.
    # This means the line through x must not contain any other point from S.
    # So for each x in S, there must be a direction v such that the line through x with direction v
    # contains no other points from S.
    
    # If x is in N, we also need a line through x where all other points are in N.
    # This is automatically satisfied if the line contains no points from S.
    
    # So the condition is: For every point x (whether in S or not), there exists a direction v
    # such that the line through x with direction v contains at most one point from S (namely x if x in S).
    
    # This is equivalent to: S is a set of points such that no two points of S lie on a "forbidden" line
    # where the forbidden lines are defined by... wait.
    
    # Actually, for x in S, we need a line through x with no other points of S.
    # For x not in S, we need a line through x with no points of S.
    
    # So: For every point x in F_q^2, there exists a direction v such that the line L(x, v)
    # contains at most one point of S (and if it contains one, it's x).
    
    # This is the definition of a "perfect 1-error-correcting code" in some sense?
    # Or rather, S is a set such that every point is "covered" by a line with at most one S-point.
    
    # Let's try to find a small S.
    # If S is empty, N = all points, which is valid but score = 0.
    # We want S to be as large as possible (so N is as small as possible, complement large).
    # Wait, score = 2401 - |N| = |S|. So we want to maximize |S|.
    
    # But S must satisfy: for every x, there is a line through x with at most 1 point of S.
    
    # Let's try S = all points on a unital? No, that would be too dense.
    
    # Actually, let's think differently.
    # The condition is: S is a set of points such that for every x in F_q^2, 
    # there exists v such that |S ∩ L(x, v)| ≤ 1.
    
    # This is equivalent to: S is a "1-saturating set" in the dual? No.
    
    # Let's try: S is a set of points no two of which are "too close" in some sense.
    
    # Actually, if we take S to be a "line-free" set, meaning no line contains 2 or more points of S,
    # then for any x, if x in S, the line in any direction has at most 1 point of S (namely x).
    # If x not in S, we need a line through x with 0 points of S.
    # So we need: S is a set of points such that no line contains 2 or more points of S,
    # AND for every x not in S, there is a line through x containing no points of S.
    
    # The first condition means S is a "cap set" in the affine plane.
    # The second condition means S does not block all lines through any x.
    
    # Let's try to construct S as a cap set.
    # In AG(2, q), a cap is a set of points no three of which are collinear.
    # But we need no two on a line? That would be an "independent set" in the line graph.
    # Actually, if no two points of S are on the same line, then S is a set of points
    # such that no two are collinear. But in AG(2, q), any two distinct points determine a line.
    # So "no two on the same line" is impossible for |S| > 1.
    
    # Wait, I misread. The line L(x, v) is a specific line. 
    # The condition is: for each x, there exists v such that L(x, v) contains at most 1 point of S.
    # This does NOT mean no two points of S are on the same line. It means that for each x,
    # at least one line through x has at most 1 S-point.
    
    # So S can have many collinear points, as long as for each x, there's "some" direction
    # that avoids most of S.
    
    # Let's try: S = all points. Then for x in S, we need a line through x with at most 1 S-point.
    # But every line through x contains many S-points (all 49). So S = all points is invalid.
    
    # Let's try: S = empty. Valid, |S| = 0.
    
    # Let's try: S = one point. Valid, |S| = 1.
    
    # We want to maximize |S|.
    
    # This is a hard combinatorial problem. Let's try a heuristic approach.
    
    # Start with S = empty. Try to add points one by one.
    # When adding x, check if the condition still holds.
    # The condition holds if for every y in F_q^2, there exists v such that |S ∪ {x} ∩ L(y, v)| ≤ 1.
    # This is equivalent to: for every y, either:
    #   - There was already a v with |S ∩ L(y, v)| ≤ 1, and x is not on L(y, v) with |S ∩ L(y, v)| = 1
    #   - Or x is on some L(y, v) with |S ∩ L(y, v)| = 0, making it 1.
    
    # This is getting complex. Let's just do a greedy search with backtracking.
    
    # Actually, let's try a different approach: start with N = all points, and remove points one by one.
    # When removing x from N (adding x to S), check if the Nikodym condition still holds.
    # The condition fails if there exists some y such that for all v, L(y, v) contains a point not in N
    # (i.e., in S) other than possibly y.
    # But y might be in N or in S.
    
    # This is still complex. Let's just do a simple greedy: 
    # Start with N = all points. 
    # For each x in random order, try to remove x from N.
    # Check if the Nikodym condition still holds.
    # If yes, keep x removed. If no, put x back.
    
    # But checking the full Nikodym condition is O(2401 * 50 * 49) which is too slow for 2401 removals.
    
    # Alternative: Precompute for each x, the directions v such that L(x, v) contains no points outside N.
    # When we remove a point y from N, this affects the status of all x that share a line with y.
    
    # Let's try this:
    # Maintain for each x, a count c[x][v] = number of points on L(x, v) that are NOT in N.
    # x is "good" for direction v if c[x][v] == 0.
    # x is "satisfied" if there exists v with c[x][v] == 0.
    # N is valid if all x are satisfied.
    
    # When we remove y from N, for each x and each v such that y is on L(x, v), c[x][v] increases by 1.
    # We need to check if any x becomes unsatisfied.
    
    # How many (x, v) pairs are affected by removing y?
    # y is on L(x, v) iff x is on L(y, v).
    # So for each v, there are q points x on L(y, v) (including y).
    # So 50 * 49 = 2450 (x, v) pairs are affected.
    
    # This is manageable.
    
    # Let's implement this.
    
    # Precompute: for each y and each v, the list of x such that y is on L(x, v).
    # This is the same as: x is on L(y, v).
    # So affected_pairs[y][v] = [x1, x2, ..., x49] where x_i are on L(y, v).
    
    # Actually, L(y, v) contains q points. So affected_pairs[y][v] has q entries.
    
    # Let's precompute this.
    
    affected_pairs = [[None] * num_dirs for _ in range(total_points)]
    for y_idx in range(total_points):
        for v_idx in range(num_dirs):
            # L(y, v) contains y and q-1 other points
            # x is on L(y, v) iff y is on L(x, v)
            # The points on L(y, v) are y + t*v for t in F_q
            # So x = y + t*v
            # We already computed line_points[y_idx][v_idx] = [x1, ..., x48] (excluding y)
            # So affected_pairs[y_idx][v_idx] = line_points[y_idx][v_idx] + [y_idx]
            affected_pairs[y_idx][v_idx] = line_points[y_idx][v_idx] + [y_idx]
    
    # Now, greedy removal:
    import random
    
    # Shuffle the points
    order = list(range(total_points))
    random.shuffle(order)
    
    # c[x][v] = number of points on L(x, v) not in N
    # Initially N = all points, so c[x][v] = 0 for all x, v.
    c = np.zeros((total_points, num_dirs), dtype=np.int8)
    
    # satisfied[x] = True if there exists v with c[x][v] == 0
    satisfied = [True] * total_points
    
    # Count how many directions satisfy x: sat_count[x] = number of v with c[x][v] == 0
    sat_count = [num_dirs] * total_points
    
    for x_idx in order:
        # Try to remove x_idx from N
        # Check if after removal, all points are still satisfied
        # Removing x_idx affects c[y][v] for all (y, v) such that x_idx is on L(y, v)
        # i.e., y is on L(x_idx, v)
        # So affected (y, v) pairs are: for each v, y in affected_pairs[x_idx][v]
        
        # Save old values
        affected = []
        for v_idx in range(num_dirs):
            for y_idx in affected_pairs[x_idx][v_idx]:
                affected.append((y_idx, v_idx, c[y_idx, v_idx]))
                c[y_idx, v_idx] += 1
                if c[y_idx, v_idx] == 1:
                    sat_count[y_idx] -= 1
                    if sat_count[y_idx] == 0:
                        satisfied[y_idx] = False
        
        # Check if any y became unsatisfied
        any_unsatisfied = False
        for y_idx, v_idx, old_c in affected:
            if not satisfied[y_idx]:
                any_unsatisfied = True
                break
        
        if any_unsatisfied:
            # Rollback
            for y_idx, v_idx, old_c in affected:
                c[y_idx, v_idx] = old_c
                if old_c == 0:
                    sat_count[y_idx] += 1
                    satisfied[y_idx] = True
        else:
            # Keep x_idx removed
            in_N[x_idx] = False
    
    # Convert to array
    pts = []
    for i in range(total_points):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            pts.append([[xa, xb], [ya, yb]])
            
    return np.array(pts, dtype=np.int64)
