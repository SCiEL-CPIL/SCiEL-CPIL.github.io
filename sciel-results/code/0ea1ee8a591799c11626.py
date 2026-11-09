# EVOLVE-BLOCK-START
import numpy as np
import itertools


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q=49, p=7) using a maximum-flow /
    minimum-cut formulation to maximize the complement size.

    Field: F_49 = F_7[alpha] with alpha^2 = 3 (3 is the smallest quadratic
    non-residue mod 7).  A point of F_49^2 is written [[x_a, x_b], [y_a, y_b]].

    Strategy:
      For each point x, the valid directions are v = (1, m) for m in F_49 and
      v = (0, 1).  A direction is "waste" for x if the punctured line
      {x + t*v : t != 0} is fully contained in the complement (i.e., no point
      of that punctured line is in N).  We want to maximize the number of
      waste-directions across all points, subject to the constraint that for
      every point x at least one valid direction has its punctured line
      intersecting N.

      This is modeled as a maximum-flow problem:
        - Source S connects to each point x with capacity 1.
        - Each point x connects to each of its 50 valid directions with capacity 1.
        - Each direction-node connects to every point on its punctured line with
          capacity 1 (or a large capacity; we use 1 per edge to keep integral).
        - Actually, the standard reduction: a point y in N "covers" direction d
          for point x if y is on the punctured line of (x, d).  We want to pick
          a set N of points such that for every (x, d) where d is the chosen
          direction for x, N intersects the punctured line.  To maximize
          complement, we minimize |N|.

      A cleaner flow model:
        We have a bipartite graph between "point-direction" pairs (x, d) and
        points y.  Edge (x,d) -- y exists iff y is on the punctured line of (x,d).
        We want to select a minimum set N of points such that every (x,d) is
        covered by at least one y in N on its line.  But we only need ONE d per x
        to be covered; the other 49 can be waste.  So we choose for each x a
        direction d_x, and then N must hit the punctured line of (x, d_x) for
        all x.

      This is still complex.  Instead, we use a greedy local improvement
      starting from a known valid Nikodym set, but guided by the flow-based
      insight: remove a point y from N if doing so does not invalidate any
      point's coverage, i.e., for every point x that used y as part of its
      punctured line coverage, x has an alternative direction whose punctured
      line is still fully in N.

      Given the 600s time limit and the large search space, we implement an
      efficient greedy removal with caching of line memberships.
    """
    p = 7
    q = p * p  # 49
    k_const = 3  # alpha^2 = 3, smallest quadratic non-residue mod 7

    # Precompute field operations: addition, multiplication in F_49.
    # Represent field element as tuple (a, b) for a + b*alpha.
    def f_add(x, y):
        return ((x[0] + y[0]) % p, (x[1] + y[1]) % p)

    def f_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + b1*b2*alpha^2 + (a1*b2 + b1*a2)*alpha
        a = (a1 * a2 + b1 * b2 * k_const) % p
        b = (a1 * b2 + b1 * a2) % p
        return (a, b)

    def f_neg(x):
        return ((-x[0]) % p, (-x[1]) % p)

    def f_sub(x, y):
        return f_add(x, f_neg(y))

    def f_div(x, y):
        # x / y = x * y^{-1}; y^{-1} = (a - b*alpha)/(a^2 - b^2*k)
        a, b = y
        denom = (a * a - b * b * k_const) % p
        if denom == 0:
            raise ValueError("division by zero")
        denom_inv = pow(denom, p - 2, p)
        return f_mul(x, ((a * denom_inv) % p, (-b * denom_inv) % p))

    # Enumerate all field elements.
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    # Map field element to index for fast lookup.
    elem_to_idx = {e: i for i, e in enumerate(field_elements)}

    # Enumerate all points in F_49^2.
    all_points = [(x, y) for x in field_elements for y in field_elements]
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    num_points = len(all_points)  # 2401

    # Directions: v = (1, m) for m in F_49, and v = (0, 1).
    directions = [(1, m) for m in field_elements] + [(0, 1)]
    # Note: (1, m) where m is a field element; (1, m) means v_x = 1, v_y = m.
    # For v = (1, m): line through x is {x + t*(1, m) : t in F_49}.
    # For v = (0, 1): line through x is {x + t*(0, 1) : t in F_49} = vertical line.

    # Precompute for each point x and each direction d, the set of points on the
    # punctured line {x + t*v : t != 0}.
    # We store this as a list of point indices.
    # This is 2401 * 50 * 48 ~ 5.7M entries, which is large but manageable.

    # Actually, for efficiency, let's precompute lines differently.
    # For each direction d, the lines partition the space.  For v = (1, m),
    # the line through x = (x0, x1) is {(x0 + t, x1 + t*m) : t in F_49}.
    # The punctured line excludes t=0.

    # Let's build a dictionary: for each (point_idx, dir_idx), a frozenset of
    # point indices on the punctured line.  But storing 2401*50 frozensets of
    # size 48 is ~ 5.7M integers, ~ 45MB, which is fine.

    # However, for the greedy algorithm, we need to know for each point y in N,
    # which (x, d) pairs have y on their punctured line.  Let's build the reverse
    # mapping: for each point y, a list of (x_idx, d_idx) such that y is on the
    # punctured line of (x, d).

    # First, build the forward mapping.
    # punctured_lines[x_idx][d_idx] = list of point indices (excluding x_idx)
    punctured_lines = [[None] * len(directions) for _ in range(num_points)]

    for x_idx, (x0, x1) in enumerate(all_points):
        for d_idx, (vx, vy) in enumerate(directions):
            pts = []
            for t in field_elements:
                if t == (0, 0):
                    continue
                tx = f_add(x0, f_mul(t, vx))
                ty = f_add(x1, f_mul(t, vy))
                pt = (tx, ty)
                pts.append(point_to_idx[pt])
            punctured_lines[x_idx][d_idx] = pts

    # Build reverse mapping: for each point y, which (x, d) have y on punctured line.
    # reverse_map[y_idx] = list of (x_idx, d_idx)
    reverse_map = [[] for _ in range(num_points)]
    for x_idx in range(num_points):
        for d_idx in range(len(directions)):
            for y_idx in punctured_lines[x_idx][d_idx]:
                reverse_map[y_idx].append((x_idx, d_idx))

    # Start with a valid Nikodym set: all points.
    # Then greedily remove points.
    # A point y can be removed if for every (x, d) in reverse_map[y],
    # there exists another direction d' for x such that the punctured line
    # of (x, d') is fully contained in N (after removal of y, but since we're
    # removing y, we check if the punctured line of (x, d') has all points in N
    # except possibly y, and y is not on that line, or if y is on that line but
    # we still have coverage via another point... this is getting complex).

    # Simpler greedy: try to remove a point y.  After removal, N' = N \ {y}.
    # Check validity: for every point x, is there a direction d such that
    # punctured line of (x, d) is subset of N'?
    # We can speed this up by tracking for each x the set of directions d for
    # which the punctured line is fully in N.  Initially all 50 directions work
    # for all x (since N = all points).  When we remove y, for each (x, d) in
    # reverse_map[y], the direction d may no longer work for x if y was the only
    # point on that punctured line that was in N... but since we're removing y
    # from N, the punctured line of (x, d) is still fully in N' iff it was fully
    # in N and y is not on it, OR if y is on it but there are other points... 
    # Wait, the punctured line has 48 points.  If y is on the punctured line of
    # (x, d), then after removing y, the punctured line is NOT fully in N' unless
    # all other 47 points are still in N.  But since we started with all points
    # in N, and we're only removing points, the punctured line of (x, d) is fully
    # in N' iff all 48 points are in N'.  So if y is on the punctured line, and
    # we remove y, then the punctured line is not fully in N' (since y is missing).
    # So for each (x, d) in reverse_map[y], direction d becomes invalid for x.
    # x still has a valid direction iff at least one other direction d' is still
    # valid for x.

    # So we maintain for each x a count of valid directions.  Initially 50.
    # When we remove y, for each (x, d) in reverse_map[y], decrement count[x].
    # If count[x] becomes 0, removal is invalid.

    # But wait: a direction d is valid for x iff the punctured line of (x, d)
    # is fully in N.  When we remove y, if y is on the punctured line of (x, d),
    # then d is no longer valid for x.  But if y is NOT on the punctured line of
    # (x, d), then d remains valid for x (since all other points are still in N).
    # So the above logic is correct: only (x, d) pairs where y is on the punctured
    # line are affected.

    # Let's implement this greedy removal.

    # Start with N = all points.
    in_N = [True] * num_points
    valid_count = [len(directions)] * num_points  # each x has 50 valid directions

    # We'll try to remove points in some order.  To maximize complement, we want
    # to remove as many points as possible.  A good heuristic: remove points that
    # affect the fewest (x, d) pairs, i.e., points with small |reverse_map[y]|.
    # But removing a point y invalidates |reverse_map[y]| (x, d) pairs.  We want
    # to avoid making any x have 0 valid directions.

    # Let's compute the degree of each point (number of (x, d) pairs it covers).
    degree = [len(reverse_map[y]) for y in range(num_points)]

    # Sort points by degree (ascending) and try to remove them.
    # But we need to be careful: removing a point changes the validity of
    # subsequent removals.

    # Use a priority queue or just iterate.  Given 2401 points, we can try
    # all of them in order of increasing degree.

    # To make it faster, let's use a simple approach: repeatedly find a point
    # that can be removed (i.e., after removal, no x has 0 valid directions)
    # and remove it.  Use a heuristic to pick the "safest" point.

    # For simplicity and speed, let's try removing points in order of their
    # index, but check validity.  This may not be optimal but is fast.

    # Actually, let's try a smarter approach: use the fact that the problem is
    # symmetric.  Let's try to remove points in a structured way.

    # Given time constraints, let's just do a greedy pass: for each point y in
    # order of increasing degree, try to remove it.  If valid, remove it.

    # Precompute degree-sorted order.
    order = sorted(range(num_points), key=lambda y: degree[y])

    for y in order:
        if not in_N[y]:
            continue
        # Check if removing y is valid.
        # For each (x, d) in reverse_map[y], check if valid_count[x] > 1
        # (i.e., x has at least one other valid direction).
        can_remove = True
        for (x, d) in reverse_map[y]:
            if valid_count[x] <= 1:
                can_remove = False
                break
        if can_remove:
            in_N[y] = False
            for (x, d) in reverse_map[y]:
                valid_count[x] -= 1

    # Build the result.
    points = []
    for i, pt in enumerate(all_points):
        if in_N[i]:
            x0, x1 = pt[0]
            y0, y1 = pt[1]
            points.append([[x0, x1], [y0, y1]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
