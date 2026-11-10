# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 620.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3

    # Field operations on F_49 = F_7[alpha], alpha^2 = 3
    # Elements represented as (a, b) meaning a + b*alpha

    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)

    def field_add(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 + a2) % p, (b1 + b2) % p)

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

    # Point indexing
    def point_to_int(pt):
        (xa, xb), (ya, yb) = pt
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    def int_to_point(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return ((xa, xb), (ya, yb))

    # Directions: (1, m) for all m in F_49 (49 directions), plus (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))
    n_dirs = len(directions)  # 50

    # Precompute line membership: for each direction, group points by their line
    # line_groups[d_idx] = dict mapping line_id -> list of point indices
    line_groups = []
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        line_map = {}
        for pt_idx, pt in enumerate(all_points):
            (x, y) = pt
            if vx == (1, 0):
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:
                c = x
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        line_groups.append(line_map)

    # Precompute for each direction and point, the set of other points on the line
    # Store as frozensets for fast membership testing
    line_others = []
    for d_idx in range(n_dirs):
        others = [None] * n_total
        for c, pts in line_groups[d_idx].items():
            for i in pts:
                others[i] = frozenset(j for j in pts if j != i)
        line_others.append(others)

    # S = complement of N. N = F_49^2 \ S.
    # Base: empty S (N = all points, trivially valid but small complement)
    # We want to maximize |S| while keeping N valid.

    # A point x can be in S if there exists a direction v such that
    # all other points on the line through x with direction v are in N (i.e., not in S).
    # For a set S to be valid, every point x (whether in S or not) must have at least
    # one direction v such that line(x, v) \ {x} is disjoint from S.

    # Strategy: Greedily add points to S.
    # For efficiency, maintain for each point x the count of "witness" directions
    # (directions where all other points on the line are not in S).
    # witness_count[x] = number of directions d where line_others[d][x] ∩ S = ∅

    # Initialize
    S = set()
    witness_count = [n_dirs] * n_total  # Initially all directions are witnesses (S is empty)

    # For each direction d and each point x, track which points on the line are in S
    # Actually, let's track for each direction d and line L, how many points on L are in S.
    # line_S_count[d][line_id] = number of points in S on that line
    # Then x has witness direction d iff line_S_count[d][line(x,d)] == 1 (only x itself, if x in S)
    # Wait, if x is not in S, then x has witness direction d iff line_S_count[d][line(x,d)] == 0
    # If x is in S, then x has witness direction d iff line_S_count[d][line(x,d)] == 1 (only x)

    # Let's use this approach.
    # line_id for point x in direction d:
    def get_line_id(pt_idx, d_idx):
        pt = all_points[pt_idx]
        vx, vy = directions[d_idx]
        x, y = pt
        if vx == (1, 0):
            mx = field_mul(vy, x)
            c = field_sub(y, mx)
        else:
            c = x
        return c

    # Precompute line_id for each point and direction
    line_id = [[get_line_id(i, d) for d in range(n_dirs)] for i in range(n_total)]

    # line_S_count[d][line_id] = number of points in S on that line
    # We need to initialize this.
    # Map from (d, line_id) to count
    # Use a dict for sparse storage, or just a list of dicts.
    line_S_count = [{} for _ in range(n_dirs)]

    # Also need to know for each point x, which direction-line pairs it belongs to
    # We already have line_id[x][d]

    # Initialize line_S_count (all zeros since S is empty)
    # We don't need to initialize explicitly; we'll use .get(key, 0)

    # Helper: get count of S-points on line (d, line_id)
    def get_line_S_count(d, lid):
        return line_S_count[d].get(lid, 0)

    def add_to_line_S_count(d, lid):
        line_S_count[d][lid] = line_S_count[d].get(lid, 0) + 1

    def remove_from_line_S_count(d, lid):
        cnt = line_S_count[d].get(lid, 0)
        if cnt <= 1:
            del line_S_count[d][lid]
        else:
            line_S_count[d][lid] = cnt - 1

    # witness_count[x]: number of directions d such that x has a witness.
    # x has witness in direction d iff:
    #   if x not in S: get_line_S_count(d, line_id[x][d]) == 0
    #   if x in S: get_line_S_count(d, line_id[x][d]) == 1

    def is_witness(x, d):
        cnt = get_line_S_count(d, line_id[x][d])
        if x in S:
            return cnt == 1
        else:
            return cnt == 0

    # Initialize witness_count
    for x in range(n_total):
        cnt = 0
        for d in range(n_dirs):
            if is_witness(x, d):
                cnt += 1
        witness_count[x] = cnt

    # When we add a point y to S:
    # - For each direction d, y's line L = (d, line_id[y][d]) gets one more S-point.
    # - For every point x on line L (including y), the witness status in direction d may change.
    #   Specifically, for x != y on line L:
    #     Before: x not in S, cnt was c. x had witness in d iff c == 0.
    #     After: cnt is c+1. x has witness in d iff c+1 == 0, i.e., never.
    #     So x loses witness in d iff c == 0 (i.e., before, x had witness; after, x doesn't).
    #   For y itself:
    #     Before: y not in S, cnt was c. y had witness in d iff c == 0.
    #     After: y in S, cnt is c+1. y has witness in d iff c+1 == 1, i.e., c == 0.
    #     So y loses witness in d iff c != 0. Wait:
    #     If c == 0: before y had witness (0==0), after y has witness (1==1). So y keeps witness.
    #     If c > 0: before y didn't have witness (c != 0), after y doesn't have witness (c+1 != 1). 
    #     So y loses witness iff c == 0? No, if c==0, y keeps. If c>0, y didn't have before and doesn't after.
    #     Actually, y's witness status in direction d:
    #     Before (y not in S): witness iff c == 0
    #     After (y in S): witness iff c+1 == 1 iff c == 0
    #     So y's witness status doesn't change! Interesting.
    #     
    #     Wait, let me re-examine. The witness condition for a point x in direction d is:
    #     All other points on the line through x with direction d are in N (not in S).
    #     "Other points" excludes x itself.
    #     So for x, the relevant count is the number of S-points on the line EXCLUDING x.
    #     If x not in S: count of S on line = c. All others in N iff c == 0.
    #     If x in S: count of S on line = c (includes x). Others in N iff c - 1 == 0 iff c == 1.
    #     
    #     So my is_witness function above is correct.
    #     
    #     Now when we add y to S:
    #     For x != y on the same line (d, L):
    #       Before: x not in S, count = c. Witness iff c == 0.
    #       After: x not in S, count = c+1. Witness iff c+1 == 0. Never true.
    #       So x loses witness iff c == 0.
    #     For y:
    #       Before: y not in S, count = c. Witness iff c == 0.
    #       After: y in S, count = c+1. Witness iff c+1 == 1 iff c == 0.
    #       So y's witness status is unchanged.

    # So when adding y to S:
    # For each direction d:
    #   L = line_id[y][d], c = current count of S on L
    #   For each x in L \ {y}:
    #     if c == 0: x loses witness in direction d
    #   Add y to S
    #   Increment count for L

    # Similarly, when removing y from S:
    # For each direction d:
    #   L = line_id[y][d], c = current count of S on L (includes y)
    #   c' = c - 1 (count after removal)
    #   For each x in L \ {y}:
    #     Before: x not in S, count = c. Witness iff c == 0. (But c >= 1 since y is in S, so c > 0, so x didn't have witness)
    #     After: x not in S, count = c'. Witness iff c' == 0.
    #     So x gains witness iff c' == 0.
    #   Remove y from S
    #   Decrement count for L
    #   For y:
    #     Before: y in S, count = c. Witness iff c == 1.
    #     After: y not in S, count = c'. Witness iff c' == 0.
    #     So y's witness status may change.

    # Let's implement this carefully.

    # Get all points on a line
    def get_line_points(d, lid):
        # Find the line from line_groups
        for c, pts in line_groups[d].items():
            if c == lid:
                return pts
        return []

    # For efficiency, precompute line_points as a dict
    line_points = []
    for d in range(n_dirs):
        lp = {}
        for c, pts in line_groups[d].items():
            lp[c] = pts
        line_points.append(lp)

    def get_line_points_fast(d, lid):
        return line_points[d].get(lid, [])

    # Greedy: try to add points to S
    # Candidate order: we want to add points that don't break too many witnesses
    # Score a candidate y: for each direction d, if c == 0, it breaks witnesses for all other points on L
    # Number of witnesses broken = sum over d of (len(L) - 1) if c == 0 else 0

    # Let's try multiple random orderings and keep the best S found.

    best_S_size = 0
    best_S = set()

    n_trials = 20
    for trial in range(n_trials):
        if _time_left() < 300:
            break

        S = set()
        witness_count = [n_dirs] * n_total
        line_S_count = [{} for _ in range(n_dirs)]

        # Initialize witness_count (should all be n_dirs since S is empty)
        for x in range(n_total):
            witness_count[x] = n_dirs

        np.random.seed(trial)
        candidates = list(range(n_total))
        np.random.shuffle(candidates)

        added = 0
        for y in candidates:
            if _time_left() < 100:
                break
            if y in S:
                continue

            # Check if adding y is feasible: y must have at least one witness direction after addition
            # y's witness status doesn't change (as shown above), so y must currently have witness_count[y] > 0
            if witness_count[y] == 0:
                continue

            # Count how many witnesses would be broken
            # And check if all affected points still have witnesses after the change
            feasible = True
            # We need to check: after adding y, every point still has witness_count >= 1
            # Only points on lines where c == 0 are affected (they lose a witness)
            # Plus y itself (no change)
            
            affected = []  # list of (x, d) where x loses witness in direction d
            for d in range(n_dirs):
                lid = line_id[y][d]
                c = line_S_count[d].get(lid, 0)
                if c == 0:
                    # All other points on this line lose witness in direction d
                    pts = get_line_points_fast(d, lid)
                    for x in pts:
                        if x != y:
                            affected.append((x, d))

            # Check if all affected points still have other witnesses
            for x, d in affected:
                if witness_count[x] <= 1:
                    # x would have 0 witnesses after losing this one
                    feasible = False
                    break

            if not feasible:
                continue

            # Also need to check that y still has at least one witness
            # y's witness count doesn't change, so if witness_count[y] > 0, we're fine

            # Commit the addition
            S.add(y)
            added += 1
            for d in range(n_dirs):
                lid = line_id[y][d]
                c = line_S_count[d].get(lid, 0)
                if c == 0:
                    # Update witness counts for all other points on this line
                    pts = get_line_points_fast(d, lid)
                    for x in pts:
                        if x != y:
                            witness_count[x] -= 1
                line_S_count[d][lid] = c + 1

        # Record best
        if len(S) > best_S_size:
            best_S_size = len(S)
            best_S = set(S)

    # Build N = F_49^2 \ S
    points = []
    for pt_idx in range(n_total):
        if pt_idx not in best_S:
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])

    arr = np.array(points, dtype=np.int64)
    return arr


# EVOLVE-BLOCK-END
