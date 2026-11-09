# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using a greedy heuristic to maximize
    the excluded set S (complement of N).
    
    A set N is a Nikodym set if for every point x in F_49^2, there exists a line L 
    passing through x such that L \ {x} is a subset of N.
    Equivalently, if S = F_49^2 \ N, then for every x in F_49^2, there exists a line L
    through x such that (L \ {x}) is disjoint from S.
    Note: If x is in S, the condition is on L \ {x}. If x is in N, the condition is on L \ {x}.
    Actually, the definition says: "every point x ... must lie on a line ... whose points 
    other than x all belong to N".
    This means: For all x, exists direction v such that for all t != 0, x + t*v in N.
    This implies that S (the complement of N) cannot contain any two points x, y such that 
    the line segment between them (excluding x) is fully in S? No.
    
    Let's re-read carefully:
    "every point x ... must lie on a line ... whose points other than x all belong to N"
    So, if x is in S, we need a line through x where all OTHER points are in N.
    This means S cannot contain a line with 2 or more points? 
    If S contained two points x and y, and the line through them had only x and y from S, 
    then for point x, the line contains y (which is in S), so y is NOT in N. 
    So that line would fail for x.
    However, x might have another line.
    
    The condition is: For each x, there is at least one line L_x through x such that 
    L_x \ {x} is subset of N.
    This is equivalent to: For each x, there is at least one line L_x through x such that 
    (L_x \ {x}) ∩ S = empty set.
    
    This is a "maximal S such that every point has a 'clean' line".
    This is a complex constraint satisfaction problem.
    
    Heuristic:
    1. Start with S = empty.
    2. We want to add points to S.
    3. When we add a point p to S, we must ensure that for every point x (including p), 
       there still exists a line through x disjoint from S \ {x} (wait, disjoint from S except maybe x?).
       The condition for x is: exists line L through x such that (L \ {x}) ∩ S = ∅.
       
    Let's define `valid(x, S)`: True if there exists a line L through x such that (L \ {x}) ∩ S is empty.
    We want to maximize |S| such that valid(x, S) is True for ALL x in F_49^2.
    
    Greedy approach:
    - Iterate through all points.
    - Try to add point p to S.
    - Check if adding p breaks the condition for any x.
      Adding p to S only affects the validity of x if p is on a line L through x and p != x.
      Specifically, for a fixed x, let `good_lines(x)` be the set of lines through x that are currently disjoint from S (excluding x).
      If we add p to S, then for any x such that p is on a line L through x (and p != x), the line L is no longer "good" for x if p is the only point in S on L? No, if p is in S, then L contains a point in S (p), so L is NOT disjoint from S. Thus L is removed from `good_lines(x)`.
      If `good_lines(x)` becomes empty, then x is no longer valid.
      
    So, to check if we can add p:
    For each x != p:
      Let L be the unique line through x and p.
      If L is currently "good" for x (i.e., L \ {x} is disjoint from current S), then adding p will make L "bad" for x (since p is in L \ {x} and p is in new S).
      If L was the ONLY good line for x, then adding p invalidates x.
      
    We need to track `good_lines_count[x]` for each x.
    Initially, S is empty. Every line is good for every x. So `good_lines_count[x]` = q+1 (number of lines through a point in F_q^2 is q+1).
    
    When we add p to S:
    For each x != p:
      Identify the line L(x,p).
      Check if L(x,p) is currently "good" for x.
      How to check if L is good? We need to know if L contains any point in S (other than x, but x is not in L unless x is on L, which it is, but we exclude x from the check).
      Wait, the condition is (L \ {x}) ∩ S = ∅.
      So L is good for x if no point in L \ {x} is in S.
      
      This requires tracking for each line L, which points are in S.
      If we add p to S, it affects all lines passing through p.
      Let Lines(p) be the set of lines passing through p.
      For each line L in Lines(p):
        For each point x on L (x != p):
          Before adding p, was L good for x?
          L was good for x if (L \ {x}) ∩ S_old = ∅.
          Since p is in L and p != x, p is in L \ {x}.
          If p was NOT in S_old, then p was not a blocker.
          If there were no other points in S_old on L \ {x}, then L was good.
          Now we add p to S. So L \ {x} now contains p (which is in S_new).
          So L is NO LONGER good for x.
          
          Therefore, for each x on L (x != p):
            If L was good for x before adding p, then `good_lines_count[x]` decreases by 1.
            If `good_lines_count[x]` becomes 0, then x is invalid. We must not add p.
            
    So the algorithm:
    1. Precompute lines.
    2. Initialize `good_lines_count[x] = q+1` for all x.
    3. Initialize `S = set()`.
    4. Maintain for each line L, the set of points in S that are on L? Or just a count?
       We need to know if L \ {x} is disjoint from S.
       This is equivalent to: count of points in S on L, excluding x if x in S? No, x is not in L \ {x}.
       So we need `count_S_on_L[L]` = number of points in S that lie on L.
       L is good for x if `count_S_on_L[L] == 0` (if x not in S) or `count_S_on_L[L] == 1` and the point is x? No.
       The condition is (L \ {x}) ∩ S = ∅.
       This means all points in S on L must be x itself? But x is not in L \ {x}.
       So actually, it means NO point in S lies on L \ {x}.
       This is equivalent to:
       If x is in S: The other points on L must not be in S. So `count_S_on_L[L]` must be 1 (only x).
       If x is not in S: The points on L must not be in S. So `count_S_on_L[L]` must be 0.
       
       Let's refine the "good" check.
       For a given x and line L through x:
       L is good for x iff (L \ {x}) ∩ S is empty.
       
       When we add p to S:
       We update `count_S_on_L` for all lines L through p.
       For each such line L:
         For each x on L (x != p):
           We need to check if L was good for x BEFORE adding p, and if it becomes bad AFTER.
           It becomes bad iff p is in L \ {x} (which is true since x != p) AND p is in S_new (true).
           So L becomes bad for x IF L was good for x before.
           
           How to check if L was good for x before?
           L was good for x iff (L \ {x}) ∩ S_old is empty.
           This is true iff:
           - If x in S_old: count_S_on_L_old[L] == 1 (only x).
           - If x not in S_old: count_S_on_L_old[L] == 0.
           
           We can track this state.
           
    5. Iterate through points in a randomized order (or multiple random orders).
       For each point p:
         Simulate adding p to S.
         Check if any x becomes invalid.
         If no x becomes invalid, commit the addition.
         Else, skip p.
         
    6. To maximize S, we might want to try multiple random permutations and keep the best S.
    
    Complexity:
    Number of points N_pts = 2401.
    Number of lines per point = 50.
    For each p, we check 50 lines.
    For each line, we check 49 other points x.
    So per p, ~2450 checks.
    Total ~ 2401 * 2450 ~ 6 million operations.
    This is fast enough for one pass.
    We can do multiple passes or random restarts.
    
    Let's implement this.
    """
    p = 7
    q = p * p
    
    # Field arithmetic
    def mul(a, b, c, d):
        # (a + b*alpha) * (c + d*alpha) = (ac + bd*3) + (ad + bc)*alpha
        return ((a * c + b * d * 3) % p, (a * d + b * c) % p)

    def add(a, b, c, d):
        return ((a + c) % p, (b + d) % p)

    def sub(a, b, c, d):
        return ((a - c) % p, (b - d) % p)

    # Map point (xa, xb, ya, yb) to integer 0..2400
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * q + (ya * p + yb))

    # Map integer to point
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)

    # Precompute all lines
    # A line is defined by a point and a slope m in F_49 or vertical.
    # Slopes m = (ma, mb) in F_49. Vertical is special.
    # Total slopes: 49 + 1 = 50.
    
    # We will store lines as lists of point indices.
    # Also, for each point, we need the list of lines passing through it.
    point_to_lines = [[] for _ in range(q * q)]
    lines = [] # list of sets of point indices? or lists?
    
    # Generate lines
    # For each point, for each slope, generate the line.
    # But this creates duplicates. Better to generate unique lines.
    # A line is uniquely defined by its set of points.
    # We can use a dictionary mapping frozenset(points) to line_id.
    
    line_map = {}
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = point_to_idx(xa, xb, ya, yb)
                    # Vertical line: x = const
                    # Points: (xa, xb, y_a, y_b) for all y
                    v_points = []
                    for ya2 in range(p):
                        for yb2 in range(p):
                            v_points.append(point_to_idx(xa, xb, ya2, yb2))
                    key = frozenset(v_points)
                    if key not in line_map:
                        line_id = len(lines)
                        lines.append(v_points)
                        line_map[key] = line_id
                        for pt in v_points:
                            point_to_lines[pt].append(line_id)
                    
                    # Non-vertical lines: y = m * x + c
                    # We iterate over slopes m.
                    # To avoid duplicates, we can just iterate all m and all c?
                    # No, for a fixed x, the line is determined by m and the point.
                    # But different (m, c) pairs might give same line?
                    # In F_q^2, a line with slope m is y = mx + c.
                    # For a fixed point (x,y) and slope m, c is determined.
                    # So for each point and each slope, we get a unique line (unless it's the same as an existing one).
                    
                    for ma in range(p):
                        for mb in range(p):
                            # Slope m = ma + mb*alpha
                            # Line through (xa, xb) with slope m:
                            # y - y0 = m * (x - x0)
                            # y = m*x + (y0 - m*x0)
                            # c = y0 - m*x0
                            c = sub(ya, yb, *mul(ma, mb, xa, xb))
                            ca, cb = c
                            
                            # Generate points on this line
                            l_points = []
                            for xa2 in range(p):
                                for xb2 in range(p):
                                    # y = m*x + c
                                    y = add(*mul(ma, mb, xa2, xb2), ca, cb)
                                    ya2, yb2 = y
                                    l_points.append(point_to_idx(xa2, xb2, ya2, yb2))
                            
                            key = frozenset(l_points)
                            if key not in line_map:
                                line_id = len(lines)
                                lines.append(l_points)
                                line_map[key] = line_id
                                for pt in l_points:
                                    point_to_lines[pt].append(line_id)
    
    # Now we have `lines` and `point_to_lines`.
    # `lines` is a list of lists of point indices.
    # `point_to_lines` is a list of lists of line ids.
    
    # State variables
    # count_S_on_line[line_id]: number of points in S on this line.
    count_S_on_line = [0] * len(lines)
    # good_lines_count[x]: number of lines through x that are "good" for x.
    good_lines_count = [q + 1] * (q * q)
    
    S = set()
    
    def is_line_good_for_point(line_id, x_idx):
        # Check if (L \ {x}) is disjoint from S.
        # L is good for x iff count of points in S on L \ {x} is 0.
        # count_S_on_line[line_id] includes x if x is in S.
        # If x in S: count on L \ {x} is count_S_on_line[line_id] - 1.
        # If x not in S: count on L \ {x} is count_S_on_line[line_id].
        if x_idx in S:
            return count_S_on_line[line_id] == 1
        else:
            return count_S_on_line[line_id] == 0

    def try_add_point(p_idx):
        # Returns True if adding p_idx to S keeps all points valid.
        # If True, we actually add it.
        # We need to simulate the change.
        
        # Identify lines affected: those passing through p_idx.
        affected_lines = point_to_lines[p_idx]
        
        # For each affected line, identify points x on that line (x != p_idx).
        # For each such x, check if the line was good for x before, and will be bad after.
        # If it was the ONLY good line for x, then x becomes invalid.
        
        # We need to know, for each x, how many good lines it has BEFORE adding p.
        # And whether the specific line L was one of them.
        
        # Let's collect the "damage".
        # damage[x] = 1 if L becomes bad for x and L was good for x before.
        # If good_lines_count[x] - damage[x] < 1, then invalid.
        
        # Note: Multiple lines through p might affect the same x?
        # No, for a fixed x and p, there is only ONE line through x and p.
        # So each x is affected by at most one line from the set of lines through p.
        
        # Wait, is that true?
        # Yes, in a projective plane (or affine plane), two distinct points determine a unique line.
        # So for a fixed p and x, there is a unique line L.
        # So we can just iterate over x != p, find the line L through x and p, and check.
        
        # However, iterating over all x != p is 2400 iterations.
        # Iterating over lines through p (50 lines) and points on each line (49 points) is also 2450.
        # Let's use the line iteration.
        
        # We need a temporary count of "lost good lines" for each x.
        lost_good = {}
        
        for line_id in affected_lines:
            L = lines[line_id]
            # For each x in L, x != p_idx
            for x_idx in L:
                if x_idx == p_idx:
                    continue
                # Check if L was good for x before adding p
                if is_line_good_for_point(line_id, x_idx):
                    # L was good. After adding p, p is in S and p is in L \ {x}.
                    # So L is no longer good.
                    # So x loses one good line.
                    lost_good[x_idx] = lost_good.get(x_idx, 0) + 1
                    # Check if x becomes invalid
                    if good_lines_count[x_idx] - lost_good[x_idx] < 1:
                        return False
        # Also check p_idx itself?
        # For x = p_idx, the condition is on lines through p_idx.
        # Adding p_idx to S affects the "goodness" of lines through p_idx for p_idx?
        # For x = p_idx, L is good if (L \ {p_idx}) is disjoint from S.
        # Before adding p_idx, p_idx not in S.
        # After adding p_idx, p_idx in S.
        # Does p_idx being in S affect (L \ {p_idx})? No, p_idx is excluded.
        # So the goodness of lines for p_idx does NOT change when we add p_idx to S.
        # Wait, is that right?
        # Condition: (L \ {x}) ∩ S = ∅.
        # If x = p_idx, then L \ {p_idx} does not contain p_idx.
        # So whether p_idx is in S or not doesn't matter for the intersection with L \ {p_idx}.
        # So p_idx's validity is unchanged.
        
        # If we passed the checks, commit the change.
        S.add(p_idx)
        for line_id in affected_lines:
            count_S_on_line[line_id] += 1
            L = lines[line_id]
            for x_idx in L:
                if x_idx == p_idx:
                    continue
                if is_line_good_for_point(line_id, x_idx):
                    # This line is no longer good for x
                    good_lines_count[x_idx] -= 1
        # Note: We don't update good_lines_count for p_idx because it didn't change.
        
        return True

    # Greedy loop
    # We can try multiple random orders.
    best_S_size = 0
    best_S = set()
    
    # Time limit guard
    start_time = time.time()
    time_limit = 600.0
    
    # Try a few random permutations
    for attempt in range(10):
        if time.time() - start_time > time_limit - 60:
            break
            
        # Reset state
        count_S_on_line = [0] * len(lines)
        good_lines_count = [q + 1] * (q * q)
        S = set()
        
        # Random order of points
        points_order = list(range(q * q))
        random.shuffle(points_order)
        
        for p_idx in points_order:
            if time.time() - start_time > time_limit - 60:
                break
            try_add_point(p_idx)
        
        if len(S) > best_S_size:
            best_S_size = len(S)
            best_S = set(S)
            
        # Optional: Local search? Swap points?
        # For now, just multiple restarts.
        
    # Construct N = F_49^2 \ best_S
    # N is the Nikodym set.
    # We need to return points in N.
    
    points = []
    for idx in range(q * q):
        if idx not in best_S:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
