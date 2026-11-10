# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using a greedy optimization approach.
    """
    p = 7
    q = p * p
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds wall-clock limit

    # Field F_49 = F_7[alpha] with alpha^2 = 3
    k_const = 3  # alpha^2 = 3, which is a quadratic non-residue mod 7

    def field_add(a, b):
        """Add two field elements represented as (a1, a2) and (b1, b2)."""
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def field_mul(a, b):
        """Multiply two field elements."""
        a1, a2 = a
        b1, b2 = b
        # (a1 + a2*alpha) * (b1 + b2*alpha) = a1*b1 + a1*b2*alpha + a2*b1*alpha + a2*b2*alpha^2
        # = a1*b1 + a2*b2*3 + (a1*b2 + a2*b1)*alpha
        return ((a1 * b1 + a2 * b2 * k_const) % p, (a1 * b2 + a2 * b1) % p)

    def field_neg(a):
        """Negate a field element."""
        return ((-a[0]) % p, (-a[1]) % p)

    # Precompute all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    field_index = {elem: i for i, elem in enumerate(field_elements)}

    # Precompute multiplication table
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i, a in enumerate(field_elements):
        for j, b in enumerate(field_elements):
            mul_table[i, j] = field_index[field_mul(a, b)]

    # All points in F_49^2
    all_points = [(x, y) for x in field_elements for y in field_elements]
    point_index = {pt: i for i, pt in enumerate(all_points)}

    # Lines: for each direction v = (1, m) or v = (0, 1), and each point x,
    # the line through x in direction v is {x + t*v : t in F_49}
    # We need: for every point x, there exists a direction v such that
    # all other points on the line through x in direction v are in N.

    # Precompute lines. A line is identified by (point, direction).
    # For direction (1, m): line through (x, y) contains points (x + t, y + t*m) for t in F_49
    # For direction (0, 1): line through (x, y) contains points (x, y + t) for t in F_49

    # For each point, store which lines pass through it
    # line_id -> set of points on that line
    # point -> list of line_ids

    # Directions: m in F_49 gives direction (1, m), plus (0,1)
    # Total directions: q + 1 = 50
    directions = []
    for m in field_elements:
        directions.append((1, m))  # direction (1, m)
    directions.append((0, 1))  # vertical direction

    # For each direction and each point, compute the line
    # A line in direction (1, m) through point (x, y) is:
    #   {(x + t, y + t*m) : t in F_49}
    # A line in direction (0, 1) through point (x, y) is:
    #   {(x, y + t) : t in F_49}

    # We can identify a line by (base_point, direction) but many points give the same line.
    # Better: for direction (1, m), a line is determined by the "intercept" y - m*x.
    # For direction (0, 1), a line is determined by x.

    # Let's enumerate all distinct lines.
    # For direction (1, m): lines are parameterized by c = y - m*x (in F_49)
    #   Line: {(x, y) : y = m*x + c}
    # For direction (0, 1): lines are parameterized by x (in F_49)
    #   Line: {(x, y) : x = const}

    # Total lines: q * q (for q slopes) + q (vertical) = q^2 + q = 49*50 = 2450

    # For each line, store the set of points on it
    # line_id -> list of point_ids

    lines = []  # list of sets of point indices
    point_to_lines = defaultdict(list)  # point_index -> list of line indices

    # Lines with direction (1, m)
    for m in field_elements:
        m_idx = field_index[m]
        for c in field_elements:
            c_idx = field_index[c]
            line_points = []
            for x in field_elements:
                # y = m * x + c
                mx = field_mul(m, x)
                y = field_add(mx, c)
                pt = (x, y)
                pt_idx = point_index[pt]
                line_points.append(pt_idx)
            line_id = len(lines)
            lines.append(line_points)
            for pt_idx in line_points:
                point_to_lines[pt_idx].append(line_id)

    # Vertical lines with direction (0, 1)
    for x in field_elements:
        x_idx = field_index[x]
        line_points = []
        for y in field_elements:
            pt = (x, y)
            pt_idx = point_index[pt]
            line_points.append(pt_idx)
        line_id = len(lines)
        lines.append(line_points)
        for pt_idx in line_points:
            point_to_lines[pt_idx].append(line_id)

    num_lines = len(lines)
    num_points = len(all_points)

    # For each point, we need at least one line through it such that all OTHER points on that line are in N.
    # Equivalently: for each point x, there exists a line L through x such that L \ {x} ⊆ N.
    # This means: for each point x, at least one line L through x has the property that
    # all points in L except x are in N.

    # Let's think of it as: we want to choose a set S of points to EXCLUDE (complement of N)
    # such that for every point x, there is a line L through x where L \ {x} ∩ S = ∅.
    # I.e., on every line through x, at most one point can be excluded, and if x itself is excluded,
    # then on at least one line through x, no other point is excluded.

    # Wait, let me re-read: "every point x of F_49^2 must lie on a line {x + t*v : t in F_49},
    # with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."

    # So for each x, there exists a direction v such that for all t != 0, x + t*v ∈ N.
    # This means: for each x, there is a line through x (in some direction) where all other points are in N.

    # Let S = F_49^2 \ N (the excluded set). We want to maximize |S|.
    # Constraint: for each x, there exists a line L through x such that (L \ {x}) ∩ S = ∅.

    # Greedy approach: start with S = ∅, try to add points to S.
    # When we add a point to S, we must ensure that for every point x on lines through that point,
    # x still has at least one line through it with no other excluded points.

    # Let's define for each point x:
    #   available_lines[x] = set of line_ids through x such that no other point on that line is in S.

    # Initially, all lines are available for all points.
    # When we add point p to S, for each line L through p, for each other point x on L,
    # line L is no longer available for x.

    # We can add point p to S if and only if available_lines[p] is non-empty (p still has some line).

    # Greedy: repeatedly pick a point p that can be added (available_lines[p] non-empty),
    # add it to S, and update available_lines for all other points.

    # To be efficient, we track:
    #   for each point x: count of available lines through x
    #   for each line L: set of points on L

    # Let's use a different representation.
    # For each point x, let avail[x] = number of lines through x that have no excluded points other than possibly x.
    # Initially avail[x] = number of lines through x = len(point_to_lines[x]) for all x.

    # When we exclude point p:
    #   For each line L through p:
    #       For each other point x on L:
    #           avail[x] -= 1
    #   avail[p] -= 1  (because now L has p excluded, so L is no longer "clean" for p either? No...)

    # Wait, let me reconsider. A line L is "available" for point x if (L \ {x}) ∩ S = ∅.
    # If we add p to S, then for any line L through p, L is no longer available for any other point x on L,
    # because now L contains p which is in S, and p != x.
    # For p itself: L is available for p if (L \ {p}) ∩ S = ∅. Before adding p, if L was available for p,
    # after adding p, L is still available for p as long as no other point on L is in S.
    # But we're adding p, so we need to check if (L \ {p}) ∩ S was empty before.

    # Let me redefine:
    #   For each point x and line L through x: L is "clean for x" if (L \ {x}) ∩ S = ∅.
    #   Point x is "satisfied" if there exists at least one line L through x that is clean for x.

    # We want to maximize |S| such that all points are satisfied.

    # Greedy: maintain for each point x the set of lines L through x that are clean for x.
    # A point can be added to S if it has at least one clean line.
    # When we add point p to S:
    #   For each line L through p:
    #       L is no longer clean for any x != p on L.
    #       L remains clean for p if it was clean for p before (i.e., no other point on L was in S).

    # Implementation:
    #   clean_lines[x] = set of line_ids L through x such that (L \ {x}) ∩ S = ∅
    #   Initially: clean_lines[x] = all lines through x.

    #   To add p: require clean_lines[p] non-empty.
    #   After adding p:
    #       For each L in all_lines_through[p]:
    #           For each x in lines[L] where x != p:
    #               L is no longer clean for x, so remove L from clean_lines[x].
    #       For p itself: clean_lines[p] stays the same? No, we need to check.
    #       Actually, clean_lines[p] contains lines L through p where (L\{p})∩S = ∅.
    #       Adding p to S doesn't change whether (L\{p})∩S = ∅ for other points.
    #       So clean_lines[p] remains unchanged.

    # But this is expensive: when we add p, we iterate over all lines through p and all points on those lines.

    # Let's precompute:
    #   lines_through[p] = list of line_ids through p
    #   lines[L] = list of point_ids on line L

    # We already have point_to_lines and lines.

    # Let's use a counter-based approach for efficiency.
    #   For each point x: clean_count[x] = number of lines L through x that are clean for x.
    #   Initially clean_count[x] = len(point_to_lines[x]).

    #   When we add p to S:
    #       For each L in point_to_lines[p]:
    #           For each x in lines[L] where x != p:
    #               clean_count[x] -= 1
    #       # clean_count[p] unchanged

    #   p can be added if clean_count[p] > 0.

    # This is still O(q * q) per addition in the worst case (each line has q points).
    # Total: O(|S| * q * q) which could be too slow.

    # Let's try a smarter approach. Use a priority queue or just iterate.

    # Actually, let's just do the greedy with the counter approach and see if it's fast enough.
    # q = 49, num_points = 2401, num_lines = 2450.
    # Each line has 49 points.
    # When we add p, we update clean_count for up to 50 * 48 = 2400 other points.
    # If we add ~1000 points, that's ~2.4 million operations. Should be fine.

    # Initialize
    clean_count = np.zeros(num_points, dtype=np.int32)
    for x in range(num_points):
        clean_count[x] = len(point_to_lines[x])

    # Points that can be added
    # We'll use a set of candidate points (those with clean_count > 0)
    candidates = set(range(num_points))

    S = set()  # excluded points

    # Greedy: while there are candidates, pick one and add it
    # To be deterministic, pick the one with smallest index or use some heuristic
    # Let's pick the one that causes the least damage (fewest clean_count reductions)
    # But that's expensive to compute. Let's just pick in order.

    # Actually, let's try a different heuristic: pick the point that has the fewest
    # "critical" lines, i.e., the point whose addition removes the fewest clean lines
    # from other points. But this is hard to compute quickly.

    # Simple approach: iterate through points in order, add if possible.
    # Repeat until no more can be added.

    # Let's just do a simple greedy: repeatedly pick any candidate and add it.
    # Use a list and filter.

    # For speed, let's convert point_to_lines to a list of lists
    pt_to_lines = [point_to_lines[i] for i in range(num_points)]
    line_pts = [lines[i] for i in range(num_lines)]

    # Convert to numpy for speed? Or just use lists.

    added_any = True
    while added_any:
        added_any = False
        for p in list(candidates):
            if p not in candidates:
                continue
            if clean_count[p] <= 0:
                candidates.discard(p)
                continue
            # Add p to S
            S.add(p)
            candidates.discard(p)
            added_any = True

            # Update clean_count for other points on lines through p
            for L in pt_to_lines[p]:
                for x in line_pts[L]:
                    if x != p:
                        clean_count[x] -= 1
                        if clean_count[x] <= 0:
                            candidates.discard(x)

            # Check deadline
            if time.time() > deadline - 30:
                break

        if time.time() > deadline - 30:
            break

    # Build the Nikodym set N = all_points \ S
    N = []
    for i, pt in enumerate(all_points):
        if i not in S:
            N.append([[pt[0][0], pt[0][1]], [pt[1][0], pt[1][1]]])

    result = np.array(N, dtype=np.int64) % 7
    return result

# EVOLVE-BLOCK-END
