import time
import numpy as np
from collections import defaultdict

p = 7
k_alpha_sq = 3
START_TIME = time.time()
DEADLINE = START_TIME + 600.0  # 30s safety margin

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _generate_lines_for_point(xa, xb, ya, yb):
    """
    Generate all 50 lines through point (xa, xb, ya, yb).
    Returns list of sets of point indices (excluding the point itself).
    """
    lines = []
    
    # Vertical line: v = (0, 1)
    vert_line = set()
    for ta in range(p):
        for tb in range(p):
            if ta == 0 and tb == 0:
                continue
            nya = (ya + ta) % p
            nyb = (yb + tb) % p
            vert_line.add(_point_to_idx(xa, xb, nya, nyb))
    lines.append(vert_line)
    
    # Slanted lines: v = (1, m) for m in F_49
    for ma in range(p):
        for mb in range(p):
            line_pts = set()
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    nx_a = (xa + ta) % p
                    nx_b = (xb + tb) % p
                    # y + t*m
                    ny_a = (ya + ta * ma + 3 * tb * mb) % p
                    ny_b = (yb + ta * mb + tb * ma) % p
                    line_pts.add(_point_to_idx(nx_a, nx_b, ny_a, ny_b))
            lines.append(line_pts)
    
    return lines

def _build_constraint_graph():
    """
    For each point x, compute the set of lines through x.
    A point x is "covered" (blocked) by S if for EVERY line L through x,
    L intersects S.
    
    Equivalently, x is "uncovered" if there exists a line L through x
    such that L is disjoint from S.
    
    We want to find the smallest S such that all points are covered.
    This is a hitting set problem.
    """
    total_points = p ** 4
    # constraints[x] = list of lines (each line is a set of point indices)
    # We need S to hit every line in constraints[x] for all x.
    
    # To make this tractable, we'll use a greedy approach:
    # 1. Start with S = empty.
    # 2. Find the point x with the most "uncovered" lines (lines disjoint from current S).
    # 3. For that x, find the point s that lies on the most uncovered lines of x.
    # 4. Add s to S.
    # 5. Repeat until all points are covered.
    
    # Precompute all lines for all points
    print("Precomputing lines...")
    all_lines = [None] * total_points
    for idx in range(total_points):
        xa, xb, ya, yb = _idx_to_point(idx)
        all_lines[idx] = _generate_lines_for_point(xa, xb, ya, yb)
        if time.time() > DEADLINE - 30:
            break
    
    # For each point, we need to hit all 50 lines.
    # Let's track which lines are still "open" (disjoint from S) for each point.
    # open_lines[x] = set of line_indices that are still disjoint from S
    
    # This is still very complex. Let's try a different angle.
    # A simpler sufficient condition for a Nikodym set:
    # If S is a "blocking set" such that every line in the affine plane
    # intersects S, then N = complement is a Nikodym set?
    # No, that's not quite right. The definition requires that for each x,
    # there exists a line through x such that all OTHER points on that line are in N.
    # This is equivalent to: for each x, there exists a line L through x
    # such that L \ {x} is disjoint from S.
    
    # So S must NOT be a blocking set for the family of punctured lines.
    # Actually, the condition is: for each x, there is at least one line L through x
    # such that L \ {x} ∩ S = ∅.
    
    # This means: for each x, NOT all 50 lines through x have S-points on them.
    # Equivalently: for each x, at least one line through x is disjoint from S.
    
    # So we need to find the smallest S such that for every x,
    # at least one of the 50 lines through x is disjoint from S.
    
    # This is the complement of the "every line hits S" condition.
    # It's a hitting set avoidance problem.
    
    # Greedy approach:
    # Start with S = ∅. All points are covered (every line is disjoint from S).
    # We want to MINIMIZE |S| while maintaining the property.
    # But we want to MAXIMIZE |N| = 2401 - |S|, so we want to MINIMIZE |S|.
    # Wait, the metric is 2401 - |N| = |S|. Higher is better.
    # So we want to MAXIMIZE |S|? No, the metric is "complement size 2401 - |N|".
    # N is the Nikodym set. Complement of N is S.
    # Metric = 2401 - |N| = |S|.
    # So we want to MAXIMIZE |S| while keeping N a valid Nikodym set.
    
    # So the problem is: find the largest S such that for every x,
    # at least one line through x is disjoint from S.
    
    # This is a maximal S problem. We can start with S = all points,
    # and remove points until the condition is satisfied.
    # Or start with S = ∅ and add points as long as the condition holds.
    
    # Let's use a greedy addition approach:
    # Start with S = ∅.
    # For each point s (in some order), check if adding s to S
    # still satisfies the Nikodym condition for N = complement.
    # If yes, add s to S.
    
    # The condition: for every x, there exists a line L through x
    # such that L \ {x} is disjoint from S.
    
    # When we add s to S, we might break the condition for some x
    # if all lines through x that were previously disjoint from S
    # now contain s.
    
    # To check this efficiently, for each x, we need to know
    # how many lines through x are disjoint from S.
    # If any count is 0, the condition is violated.
    
    # Let's maintain for each x the number of lines through x
    # that are disjoint from S.
    
    # This is still complex. Let's try a simpler heuristic:
    # Use the seed construction as a starting point, then try to
    # add more points to S while maintaining validity.
    
    # Actually, let's just return the seed for now, as the greedy
    # optimization might be too slow for 600s.
    
    # But the directive says to use a DIFFERENT strategy.
    # Let's implement a simple local search:
    # 1. Start with the seed S.
    # 2. Try to add random points to S, checking validity.
    # 3. If valid, keep the addition.
    # 4. Repeat until time runs out.
    
    return all_lines

def _validate_nikodym_fast(excluded):
    """
    Fast validation: for each point x, check if at least one line
    through x is disjoint from S.
    """
    total_points = p ** 4
    for idx in range(total_points):
        xa, xb, ya, yb = _idx_to_point(idx)
        
        # Check vertical
        vert_ok = True
        for ta in range(p):
            for tb in range(p):
                if ta == 0 and tb == 0:
                    continue
                nya = (ya + ta) % p
                nyb = (yb + tb) % p
                if excluded[_point_to_idx(xa, xb, nya, nyb)]:
                    vert_ok = False
                    break
            if not vert_ok:
                break
        if vert_ok:
            continue
        
        # Check slanted
        found = False
        for ma in range(p):
            for mb in range(p):
                line_ok = True
                for ta in range(p):
                    for tb in range(p):
                        if ta == 0 and tb == 0:
                            continue
                        nx_a = (xa + ta) % p
                        nx_b = (xb + tb) % p
                        ny_a = (ya + ta * ma + 3 * tb * mb) % p
                        ny_b = (yb + ta * mb + tb * ma) % p
                        if excluded[_point_to_idx(nx_a, nx_b, ny_a, ny_b)]:
                            line_ok = False
                            break
                    if line_ok:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return False
    
    return True

def _generate_seed_s():
    """Generate the seed excluded set S."""
    squares = set((u * u) % p for u in range(p))
    excluded = np.zeros(p ** 4, dtype=np.int8)
    count = 0
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k_alpha_sq) % p == 0:
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
                            count += 1
    return excluded, count

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Generate seed S
    excluded, s_size = _generate_seed_s()
    print(f"Seed S size: {s_size}")
    
    # Validate seed
    if _validate_nikodym_fast(excluded):
        print("Seed is valid")
    else:
        print("Seed is INVALID!")
    
    # Try to grow S by adding points
    # Use a random order
    total_points = p ** 4
    indices = list(range(total_points))
    np.random.shuffle(indices)
    
    added = 0
    for idx in indices:
        if time.time() > DEADLINE - 60:
            break
        
        if excluded[idx]:
            continue
        
        # Try adding this point
        excluded[idx] = 1
        # Check validity
        if _validate_nikodym_fast(excluded):
            added += 1
        else:
            excluded[idx] = 0
    
    print(f"Added {added} points, total S size: {s_size + added}")
    
    # Build N
    points = []
    for idx in range(total_points):
        if excluded[idx] == 0:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
