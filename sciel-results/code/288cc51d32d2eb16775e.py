# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7. A set N is a Nikodym
    set if, for every point x in F_q^2, there exists a direction v != 0 such that
    the punctured line {x + t*v : t in F_q, t != 0} is fully contained in N. The
    goal is to make N as SMALL as possible, equivalently to make its complement
    (the excluded set) as LARGE as possible.

    Field model: F_q is built as F_p[alpha] with alpha^2 = k, where k = 3 is the
    smallest quadratic non-residue mod 7. An element of F_q is the pair (a, b)
    standing for a + b*alpha, with a, b in {0, ..., p-1}.

    Construction: Use the Hermitian unital U = {(x, y) : Trace(x) + Norm(y) = 0}
    as the base. The complement S is a subset of U. We need S to be a blocking set
    in the dual sense: for every point x and direction v, if x is not in N (i.e., x in S),
    then there must exist some other direction v' such that the punctured line
    {x + t*v' : t != 0} is contained in N.

    The baseline uses S = {points in U whose y-coordinate is NOT in Y}, where
    Y = {(u^2 mod p, v) : u, v in F_p}. This gives a valid Nikodym set.

    We attempt to enlarge S (shrink N) by greedily adding points from U \\ S
    while maintaining the Nikodym property.
    """
    p = 7
    k = 3  # alpha^2 = 3 mod 7

    # Precompute all points as flat indices 0..2400
    # Point index = xa * p^3 + xb * p^2 + ya * p + yb
    total_points = p * p * p * p  # 2401

    def point_to_idx(xa, xb, ya, yb):
        return xa * p**3 + xb * p**2 + ya * p + yb

    def idx_to_point(idx):
        xa = idx // (p**3)
        xb = (idx // (p**2)) % p
        ya = (idx // p) % p
        yb = idx % p
        return xa, xb, ya, yb

    # Compute which points are in the Hermitian unital U
    in_unital = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        in_unital[point_to_idx(xa, xb, ya, yb)] = True

    # Special y-coordinate set Y = {(u^2 mod p, v)}
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add((u * u % p, v))

    # Baseline excluded set S: points in U whose y is NOT in Y
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if in_unital[point_to_idx(xa, xb, ya, yb)] and (ya, yb) not in y_special:
                        S.add(point_to_idx(xa, xb, ya, yb))

    # Directions: (1, m) for m in F_q, and (0, 1)
    # Represent m as (ma, mb)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, mb))  # (vx_a, vx_b, vy_a, vy_b) where v = (1, m)
    directions.append((0, 0, 0, 1))  # v = (0, 1)

    # Precompute for each point and direction, the set of other points on that line
    # line_points[idx][dir_idx] = set of point indices on the line through idx in direction dir
    # This is expensive, so we compute on demand with caching

    def get_line_points(idx, dir_idx):
        """Get all points on the line through idx in direction dir_idx."""
        xa, xb, ya, yb = idx_to_point(idx)
        vx_a, vx_b, vy_a, vy_b = directions[dir_idx]

        # Line: (xa + t*vx_a, xb + t*vx_b, ya + t*vy_a, yb + t*vy_b) for t in F_q
        # t = (ta, tb), t in F_q
        pts = set()
        for ta in range(p):
            for tb in range(p):
                # t = ta + tb*alpha
                # t * vx_a = (ta + tb*alpha) * vx_a = ta*vx_a + tb*vx_a*alpha
                # So (t*vx_a)_a = ta*vx_a, (t*vx_a)_b = tb*vx_a
                # Similarly for vy
                # Point = (xa + ta*vx_a, xb + tb*vx_a, ya + ta*vy_a, yb + tb*vy_a)
                # Wait, that's not right. Let me redo.
                # In F_q, (ta + tb*alpha) * (va_a + va_b*alpha) = 
                #   ta*va_a + ta*va_b*alpha + tb*va_a*alpha + tb*va_b*alpha^2
                # = (ta*va_a + tb*va_b*k) + (ta*va_b + tb*va_a)*alpha
                # So if v = (vx_a, vx_b) and t = (ta, tb):
                # (t*v)_a = (ta*vx_a + tb*vx_b*k) % p
                # (t*v)_b = (ta*vx_b + tb*vx_a) % p

                tx_a = (ta * vx_a + tb * vx_b * k) % p
                tx_b = (ta * vx_b + tb * vx_a) % p
                ty_a = (ta * vy_a + tb * vy_b * k) % p
                ty_b = (ta * vy_b + tb * vy_a) % p

                nxa = (xa + tx_a) % p
                nxb = (xb + tx_b) % p
                nya = (ya + ty_a) % p
                nyb = (yb + ty_b) % p
                pts.add(point_to_idx(nxa, nxb, nya, nyb))
        return pts

    # Precompute line points for all points and directions
    # This is 2401 * 50 = 120050 lines, each with 49 points.
    # Total: ~5.8M point memberships. Let's precompute.
    print("Precomputing line points...")
    start_time = time.time()
    line_points_cache = {}
    for idx in range(total_points):
        for dir_idx in range(len(directions)):
            line_points_cache[(idx, dir_idx)] = get_line_points(idx, dir_idx)
    print(f"Precomputation done in {time.time() - start_time:.2f}s")

    # Nikodym validity check for a given S
    # For every point x: there exists a direction d such that
    # all points on the line through x in direction d, except x itself, are NOT in S.
    # Equivalently: for every x, the number of directions d for which
    # (line through x in direction d) \ {x} is fully contained in complement of S
    # must be at least 1.

    def is_valid_nikodym(S_set):
        S_arr = np.zeros(total_points, dtype=bool)
        for s in S_set:
            S_arr[s] = True

        # For each point, check if there's at least one direction where
        # all other points on that line are not in S
        for idx in range(total_points):
            found = False
            for dir_idx in range(len(directions)):
                line_pts = line_points_cache[(idx, dir_idx)]
                # Check if all points in line_pts except idx are not in S
                ok = True
                for pt in line_pts:
                    if pt != idx and S_arr[pt]:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if not found:
                return False
        return True

    # Greedy: try to add points to S from U \ S
    # Start with baseline S
    S_current = set(S)
    # Candidate points to add: points in U not already in S
    candidates = []
    for idx in range(total_points):
        if in_unital[idx] and idx not in S_current:
            candidates.append(idx)

    print(f"Starting S size: {len(S_current)}, candidates: {len(candidates)}")

    # Greedy: repeatedly add a candidate that keeps S valid
    # To speed up, we can check incrementally
    # But for simplicity, let's do a simple greedy with full validation
    # Actually, full validation is O(2401 * 50 * 49) ~ 5.8M per check, too slow for many iterations.

    # Better: for each candidate, check if adding it violates the property.
    # Adding point p to S means: for every point x on a line through p in some direction,
    # if that line was the ONLY valid direction for x, then x now has no valid direction.

    # Let's precompute for each point x, the set of valid directions
    def compute_valid_directions(S_set):
        S_arr = np.zeros(total_points, dtype=bool)
        for s in S_set:
            S_arr[s] = True

        valid_dirs = {}
        for idx in range(total_points):
            valid_dirs[idx] = set()
            for dir_idx in range(len(directions)):
                line_pts = line_points_cache[(idx, dir_idx)]
                ok = True
                for pt in line_pts:
                    if pt != idx and S_arr[pt]:
                        ok = False
                        break
                if ok:
                    valid_dirs[idx].add(dir_idx)
        return valid_dirs

    print("Computing initial valid directions...")
    valid_dirs = compute_valid_directions(S_current)
    print(f"Initial valid directions computed. Min valid dirs: {min(len(v) for v in valid_dirs.values())}")

    # Greedy: try to add candidates one by one
    # For each candidate, check if adding it would leave any point with 0 valid directions
    added_count = 0
    for candidate in candidates:
        # Check: for every point x on a line through candidate in some direction,
        # if candidate is on that line and candidate != x, then x's valid directions
        # that include that direction are now invalid (if all other points on that line
        # except x were not in S, but now candidate is in S).

        # Actually, adding candidate to S affects: for each direction d, for each point x
        # on the line through candidate in direction d (x != candidate), if d was valid for x
        # and all other points on that line (except x and candidate) were not in S, then d
        # is no longer valid for x.

        # Simpler: just check if after adding candidate, every point still has >= 1 valid direction.
        # We can do this incrementally: for each point x that shares a line with candidate,
        # check if x loses its last valid direction.

        # Find all points that share a line with candidate
        affected_points = set()
        for dir_idx in range(len(directions)):
            line_pts = line_points_cache[(candidate, dir_idx)]
            for pt in line_pts:
                if pt != candidate:
                    affected_points.add(pt)

        # For each affected point, check if it still has a valid direction
        # after candidate is added to S
        can_add = True
        for x in affected_points:
            # Check each valid direction of x
            for d in valid_dirs[x]:
                line_pts = line_points_cache[(x, d)]
                # Check if all points on this line except x are not in S (including candidate)
                ok = True
                for pt in line_pts:
                    if pt != x:
                        if pt == candidate:
                            # candidate would be in S, so this direction is invalid
                            ok = False
                            break
                        elif pt in S_current:
                            ok = False
                            break
                if ok:
                    break
            else:
                # No valid direction remains for x
                can_add = False
                break

        if can_add:
            S_current.add(candidate)
            # Update valid_dirs for affected points
            for x in affected_points:
                new_valid = set()
                for d in valid_dirs[x]:
                    line_pts = line_points_cache[(x, d)]
                    ok = True
                    for pt in line_pts:
                        if pt != x:
                            if pt == candidate or pt in S_current:
                                ok = False
                                break
                    if ok:
                        new_valid.add(d)
                valid_dirs[x] = new_valid
            added_count += 1

    print(f"Added {added_count} points. Final S size: {len(S_current)}")

    # Final validation
    if not is_valid_nikodym(S_current):
        print("WARNING: Final S is not a valid Nikodym complement! Falling back to baseline.")
        S_current = set(S)

    # Build the Nikodym set N = F_q^2 \ S
    points = []
    for idx in range(total_points):
        if idx not in S_current:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
