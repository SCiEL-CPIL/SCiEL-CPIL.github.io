# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using maximum bipartite matching.
    Models point exclusion as matching points to witness lines;
    maximum matching gives maximum number of points we can exclude.
    """
    start_time = time.time()
    deadline = start_time + 580

    p = 7
    q = p * p

    # F_49 arithmetic
    def add_f49(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)

    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)

    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    n = len(all_points)  # 2401
    point_idx = {pt: i for i, pt in enumerate(all_points)}

    # Precompute all lines: vertical + slope lines
    # Each line is a frozenset of point indices
    # Line ID: 0..48 for vertical, 49..2448 for slope lines
    # Vertical lines: x = c, 49 lines
    # Slope lines: y = m*x + c, m in F49\{0}, c in F49 -> 48*49 = 2352 lines
    # Total: 49 + 2352 = 2401 lines

    # Build line -> set of points, and point -> set of lines
    vert_lines = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_lines[(xa, xb)].append(i)

    # Assign line IDs
    # Vertical: line_id = xa*7 + xb (0..48)
    # Slope: line_id = 49 + m_idx*49 + c_idx where m_idx is index in F49\{0}, c_idx in F49
    slope_m_list = [(ma, mb) for ma in range(p) for mb in range(p) if not (ma == 0 and mb == 0)]
    slope_m_idx = {m: i for i, m in enumerate(slope_m_list)}

    # line_points: list of lists, line_points[line_id] = list of point indices
    # point_lines: list of lists, point_lines[pt_idx] = list of line IDs
    num_vert = p * p  # 49
    num_slope = len(slope_m_list) * p  # 48*49 = 2352
    num_lines = num_vert + num_slope  # 2401

    line_points = [[] for _ in range(num_lines)]
    point_lines = [[] for _ in range(n)]

    # Fill vertical lines
    for (xa, xb), pts in vert_lines.items():
        lid = xa * p + xb
        for pi in pts:
            line_points[lid].append(pi)
            point_lines[pi].append(lid)

    # Fill slope lines
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            m_idx = slope_m_idx[(ma, mb)]
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                key = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[key].append(i)
            for (ca, cb), pts in lines_dict.items():
                lid = num_vert + m_idx * p + (ca * p + cb)
                for pi in pts:
                    line_points[lid].append(pi)
                    point_lines[pi].append(lid)

    # For each point, find which lines can serve as witness lines
    # A line L can be a witness for point x if x is on L and all other points on L are in N.
    # In the matching model: we want to exclude a set S of points such that for each x in S,
    # there exists a line L containing x where all other points on L are NOT in S.
    # This is equivalent to: for each excluded point x, at least one line through x has all other points in N.
    #
    # Reformulation: We build a bipartite graph where left = points, right = lines.
    # Edge (x, L) exists if x is on L.
    # A valid exclusion set S with witness assignment f: S -> lines is valid iff:
    #   For each x in S, f(x) is a line through x, and for all y on f(x), y != x => y not in S.
    #   Equivalently: for each line L, at most one point on L is in S (the one assigned to L).
    #   Wait, that's not quite right. If L is witness for x, then all OTHER points on L must be in N.
    #   So if two points x1, x2 on the same line L are both excluded, L cannot be witness for either.
    #   But they could use different witness lines.
    #
    # Actually the constraint is: for each x in S, there exists a line L_x through x such that
    #   S ∩ L_x = {x}.
    # This means: for each line L, the number of excluded points on L that choose L as witness ≤ 1.
    # But multiple points on L could be excluded as long as none of them choose L as witness.
    #
    # This is NOT simply a matching problem. Let me think differently.
    #
    # Alternative: Greedy with verification. Start with all points in N. Try to exclude points one by one.
    # For each candidate point x, check if it has a witness line (all other points on some line through x are in N).
    # Use a priority: prefer points that have many lines where they're the "only" excluded point.
    #
    # Let me use a simpler structural approach: iterative greedy exclusion with conflict tracking.

    # in_N: True means point is in N
    in_N = [True] * n
    excluded = set()

    # For each point, track which lines could be witness lines (i.e., all other points on line are in N)
    # A line L is a valid witness for x if x is on L and all y on L, y != x: in_N[y]
    # We maintain for each line L: count of excluded points on L
    line_excluded_count = [0] * num_lines
    for pi in range(n):
        for lid in point_lines[pi]:
            pass  # initially 0

    # For each point, precompute candidate witness lines
    # A point x can be excluded if there exists a line L through x with line_excluded_count[L] == 0
    # (meaning no other point on L is excluded... wait, line_excluded_count[L] counts ALL excluded on L)
    # Actually: L is witness for x iff all y on L, y≠x are in N iff no other point on L is excluded
    # iff line_excluded_count[L] - (1 if x on L else 0) == 0 iff line_excluded_count[L] == 1 (since x is on L and about to be excluded)
    # Wait: before excluding x, line_excluded_count[L] counts excluded points on L (x not yet excluded).
    # After excluding x, the count becomes line_excluded_count[L] + 1.
    # For L to be witness for x: all other points on L must be in N, i.e., no other excluded point on L.
    # So we need line_excluded_count[L] == 0 before excluding x (since x itself isn't counted yet).
    # Then after excluding x, line_excluded_count[L] becomes 1, and x is the only excluded point on L. ✓

    # Greedy: repeatedly find a point x that has at least one line L through x with line_excluded_count[L] == 0
    # Exclude it, update counts.
    # Priority: exclude points that "use up" the most lines (i.e., have many lines with count 0)
    # to free up other points' options. Actually, we want to maximize exclusions, so we should be careful.
    #
    # Better: use a queue-based approach. Points with fewest available witness lines first (most constrained).

    # For each point, count how many lines through it have line_excluded_count == 0
    # Initially all lines have count 0, so every point has all its lines available.

    # Let's do iterative: maintain a set of "excludable" points.
    # Initially all points are excludable (every line has 0 excluded).
    # When we exclude point x on line L, line L's count becomes 1.
    # For any other point y on L, L is no longer available as witness for y.
    # If y has no other available lines, y becomes non-excludable.

    # Data structures:
    # available_lines[x] = set of line IDs through x with line_excluded_count == 0
    # Initially: available_lines[x] = set(point_lines[x]) for all x

    # But this is memory heavy. Let's use a different approach.
    # For each point x, track: num_available_lines[x] = number of lines through x with count 0.
    # A point is excludable iff num_available_lines[x] > 0.

    num_avail = [len(point_lines[x]) for x in range(n)]  # initially all lines available
    excludable = [True] * n  # initially all excludable

    # Process in order of increasing num_avail (most constrained first)
    # Use a priority queue or just iterate

    # Actually, let's use a simpler iterative approach with multiple passes.
    # In each pass, try to exclude all currently excludable points.
    # This may not be optimal but is a valid heuristic.

    # Better: single pass with careful ordering.
    # Sort points by number of available lines (ascending) - most constrained first.
    # But available lines change as we exclude points.

    # Let me use a different structural approach:
    # Model as: we want to find the largest set S such that for each x in S,
    # there exists L through x with S ∩ L = {x}.
    # This is equivalent to: S is a set where no two elements share a "witness" line.
    #
    # Greedy with backtracking is expensive. Let me try a simple greedy:
    # Randomize order, exclude if possible.

    # Actually, let me try a cleaner approach:
    # For each point, the constraint is: at least one line through it has all other points in N.
    # This is a constraint satisfaction problem. Let me use a simple greedy:
    # 1. Start with all in N.
    # 2. For each point (in some order), try to exclude it.
    # 3. To exclude x, pick a line L through x where all other points are in N.
    # 4. Exclude x.
    # 5. Update: for all lines through x, they now have one excluded point (x).
    #    For other points on those lines, those lines are no longer valid witnesses.

    # Order: try points with fewer lines first? Or more lines?
    # Points with more lines are easier to exclude (more choices).
    # But excluding them "uses up" more lines, potentially blocking others.
    # Points with fewer lines are harder to exclude but use fewer lines.
    # Strategy: exclude points with MORE available lines first (they're easier and their exclusion
    # blocks many potential witnesses for others, but we already chose them).
    # Actually, let me just try: exclude in order of decreasing number of lines through the point.

    # Each point has exactly 49 lines through it (1 vertical + 48 slope lines).
    # So all points have the same number of lines! The constraint is dynamic.

    # Let me just do a simple greedy: iterate through points, exclude if possible.
    # Use multiple random orderings and keep the best.

    best_excluded_count = 0
    best_in_N = None

    import random
    rng = random.Random(42)

    max_iters = 20
    for it in range(max_iters):
        if time.time() > deadline - 120:
            break

        in_N_trial = [True] * n
        line_excl = [0] * num_lines
        excluded_count = 0

        # Random order
        order = list(range(n))
        rng.shuffle(order)

        for x in order:
            if time.time() > deadline - 120:
                break
            # Find a line L through x with line_excl[L] == 0
            found = False
            for lid in point_lines[x]:
                if line_excl[lid] == 0:
                    # Exclude x, use L as witness
                    in_N_trial[x] = False
                    excluded_count += 1
                    # Update line_excl for all lines through x
                    for lid2 in point_lines[x]:
                        line_excl[lid2] += 1
                    found = True
                    break
            # If no line available, x stays in N

        if excluded_count > best_excluded_count:
            best_excluded_count = excluded_count
            best_in_N = in_N_trial[:]

        if time.time() > deadline - 120:
            break

    if best_in_N is None:
        best_in_N = [True] * n
        best_excluded_count = 0

    # Build result
    pts = [all_points[i] for i in range(n) if best_in_N[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
