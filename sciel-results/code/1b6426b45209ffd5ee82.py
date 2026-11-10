# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (q = 49 = 7^2).
    Field: F_49 = F_7[alpha] with alpha^2 = 3.
    A point is [[xa, xb], [ya, yb]] representing (xa + xb*alpha, ya + yb*alpha).

    Strategy: Start from the baseline construction (complement size 147) and use
    incremental local search to add more points to the complement S while maintaining
    the Nikodym property. The key insight is that we can add a point to S if, for
    every line through that point, the line already has at least one other point in S
    OR we can verify the constraint is still satisfied.

    Actually, the correct constraint is: for every point x in F_q^2 and every direction v,
    the punctured line {x + t*v : t != 0} must be fully in N. Equivalently, for every
    point x and direction v, at most one point on the full line through x in direction v
    can be in S (the complement). Wait, that's not right either.

    Let me re-read: "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49},
    with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."

    So for each point x, there exists AT LEAST ONE direction v such that all other points on the line
    through x in direction v are in N. This means for each x, we need at least one line through x
    where the other q-1 = 48 points are all in N.

    Equivalently, for the complement S = F_q^2 \ N: for each point x, there exists a direction v
    such that the line through x in direction v contains NO points of S other than possibly x itself.
    Wait, "points other than x all belong to N" means the other 48 points are in N, i.e., not in S.
    So the line through x in direction v can have x in S or not, but the other 48 points must not be in S.

    So the constraint is: for each point x, there exists a direction v (among the 50 directions: 49 slopes + vertical)
    such that the line through x in direction v has at most 1 point in S (namely x itself if x in S).

    To maximize |S|, we want to add points to S greedily while maintaining this property.

    Let's use a different approach: start with S = empty (N = everything), and add points to S one by one,
    checking that the constraint is still satisfied. But that's expensive.

    Better: start with the baseline S (size 147) and try to add more points. For each candidate point p not in S,
    check if adding p to S keeps the Nikodym property valid. The property is violated if there exists some point x
    such that for ALL 50 directions through x, the line contains a point of S (other than x). Adding p to S can only
    hurt points x that lie on a line with p.

    For efficiency, track for each point x and direction v, how many points of S are on the line through x in direction v
    (excluding x). The constraint for x is satisfied if there exists a direction v where this count is 0.

    When we add a new point p to S, for each direction v, all points x on the line through p in direction v (x != p)
    will have their count for direction v incremented by 1. If any of those points had count 0 for direction v before,
    and now all their other directions have count > 0, then the constraint is violated.

    Let's implement this with efficient tracking.
    """
    p = 7
    q = p * p  # 49
    deadline = time.time() + 600.0  # safety margin

    # Field arithmetic helpers
    k_const = 3  # alpha^2 = 3 in F_7

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)

    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k_const) % p,
                (a[0]*b[1] + a[1]*b[0]) % p)

    def to_idx(pt):
        xa, xb, ya, yb = pt[0][0], pt[0][1], pt[1][0], pt[1][1]
        return xa * q * q + xb * q + ya * p + yb  # this is wrong, let me fix

    # Let's use a simpler indexing: point index = xa * 49 + xb * 7 + ya * 1 + yb... no
    # Total points = 49 * 49 = 2401
    # index = xa * 49 + xb * 7 + ya * 1 + yb doesn't work because xb, ya, yb each range 0..6
    # index = xa * (7*7*7) + xb * (7*7) + ya * 7 + yb = xa*343 + xb*49 + ya*7 + yb
    # That gives 0..2400 for xa,xb, ya, yb each in 0..6. Yes.

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

    # Directions: (1, m) for m in F_49, and (0, 1)
    # m is represented as (ma, mb)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (dx, dy) where dx=(1,0), dy=(ma,mb)
    directions.append(((0, 1), (0, 0)))  # vertical: (0, 1) means dx=(0,0), dy=(1,0)? No.
    
    # Let me reconsider. A line with direction v = (vx, vy) where vx, vy in F_49, not both zero.
    # The two types are: v = (1, m) for m in F_49, and v = (0, 1).
    # So vx and vy are elements of F_49, represented as pairs.
    
    directions = []
    for ma in range(p):
        for mb in range(p):
            # v = (1, m) where 1 = (1,0) and m = (ma, mb)
            directions.append(((1, 0), (ma, mb)))
    # v = (0, 1) where 0 = (0,0) and 1 = (1,0)
    directions.append(((0, 0), (1, 0)))
    
    num_dirs = len(directions)  # 50

    # For each point and direction, track the count of S points on the line through that point
    # in that direction, excluding the point itself.
    # line_count[x_idx][d_idx] = number of points in S on the line through x in direction d, excluding x.
    
    # Initialize with the baseline S
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    S = set()  # set of point indices in the complement
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        S.add(pt_to_idx([[xa, xb], [ya, yb]]))

    # Build line membership: for each direction, and each point, which other points are on the same line?
    # Actually, for each direction d and point x, the line through x in direction d is {x + t*d : t in F_49}.
    # The other points are x + t*d for t != 0.
    
    # Precompute for each point x and direction d, the list of other points on the line.
    # This is expensive: 2401 * 50 * 48 = ~5.7 million entries. Let's precompute.
    
    # Actually, let's precompute line members more efficiently.
    # For each direction d, the lines partition the 2401 points into 49 lines of 49 points each.
    # For each direction d, we can compute the line id for each point, and then for each line,
    # the set of points on it.
    
    # line_membership[d_idx][x_idx] = list of other points on the line through x in direction d
    # This is 50 * 2401 * 48 integers. That's about 5.7M integers, which is ~45 MB. OK.
    
    # Actually, let's just precompute the "other points" for each (x, d) pair as a flat array.
    # Or better: for each direction d, for each point x, compute the 48 other points.
    
    # Let's precompute: for each direction d, for each point x, the list of other points on the line.
    # Store as a 2D array: line_others[d_idx, x_idx, t] for t in 0..47
    # That's 50 * 2401 * 48 = 5,762,400 entries. As int32, that's ~23 MB. OK.
    
    # Actually, let's be smarter. For a given direction d = (vx, vy), the line through x = (px, py)
    # is {(px + t*vx, py + t*vy) : t in F_49}.
    # The other points are for t = 1, 2, ..., 48 (in the field F_49).
    
    # Let's precompute the field elements as indices 0..48 representing t=1..48.
    # Actually, t ranges over all of F_49 except 0, so 48 values.
    
    # Let me just precompute the line others on the fly but cache them.
    
    # For now, let's build the line_count array.
    # line_count[x_idx][d_idx] = count of S points on line through x in direction d, excluding x.
    
    line_count = np.zeros((2401, num_dirs), dtype=np.int32)
    
    # Initialize line_count for the baseline S
    for s_idx in S:
        for d_idx, (vx, vy) in enumerate(directions):
            # For each t in F_49, t != 0: point = s + t*v
            # For each such point x, increment line_count[x][d_idx]
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    # t = (ta, tb)
                    # x = s + t*v = (sx + t*vx, sy + t*vy)
                    s_pt = idx_to_pt(s_idx)
                    sx, sy = s_pt[0], s_pt[1]
                    # t*v = (t*vx, t*vy)
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(sx, tvx)
                    ny = add(sy, tvy)
                    n_idx = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    line_count[n_idx, d_idx] += 1
    
    # Check validity: for each point x, there must exist a direction d with line_count[x][d] == 0
    def is_valid():
        # For each point, check if there's a direction with count 0
        # line_count[x, d] == 0 means the line through x in direction d has no S points other than x
        # We need at least one such d for each x.
        # Note: if x is in S, the line through x in direction d still counts other S points.
        # The constraint is that for each x, exists d such that line_count[x][d] == 0.
        return np.all(np.min(line_count, axis=1) == 0)
    
    # Verify baseline is valid
    assert is_valid(), "Baseline is not valid!"
    
    best_S = set(S)
    best_complement = len(S)
    
    # Local search: try to add points to S
    # For each candidate point p not in S, check if adding p keeps validity.
    # When adding p, for each direction d, all points x on the line through p in direction d (x != p)
    # will have line_count[x][d] incremented by 1.
    # This can only invalidate x if line_count[x][d] was 0 and d was the ONLY direction with count 0 for x.
    
    # Precompute for each point x, the number of directions with line_count[x][d] == 0
    zero_counts = np.sum(line_count == 0, axis=1)
    
    # Now try to add points. Use a priority queue or just iterate.
    candidates = [i for i in range(2401) if i not in S]
    
    # Shuffle candidates for diversity
    np.random.seed(42)
    np.random.shuffle(candidates)
    
    added_this_round = True
    while added_this_round and time.time() < deadline - 30:
        added_this_round = False
        for p_idx in candidates:
            if p_idx in S:
                continue
            # Check if adding p_idx to S keeps validity
            # For each direction d, find all x on line through p in direction d, x != p
            # For each such x, if line_count[x][d] == 0, then after increment, it becomes 1.
            # If zero_counts[x] == 1 before, then after, zero_counts[x] becomes 0 -> invalid.
            valid = True
            affected = []  # (x_idx, d_idx) pairs where line_count[x][d] was 0
            for d_idx, (vx, vy) in enumerate(directions):
                p_pt = idx_to_pt(p_idx)
                px, py = p_pt[0], p_pt[1]
                for ta in range(p):
                    for tb in range(p):
                        if ta == 0 and tb == 0:
                            continue
                        tvx = mul((ta, tb), vx)
                        tvy = mul((ta, tb), vy)
                        nx = add(px, tvx)
                        ny = add(py, tvy)
                        n_idx = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                        if line_count[n_idx, d_idx] == 0:
                            affected.append((n_idx, d_idx))
                            if zero_counts[n_idx] == 1:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Add p_idx to S
                S.add(p_idx)
                for n_idx, d_idx in affected:
                    line_count[n_idx, d_idx] += 1
                    zero_counts[n_idx] -= 1
                added_this_round = True
                if len(S) > best_complement:
                    best_complement = len(S)
                    best_S = set(S)
        
        if time.time() > deadline - 60:
            break
    
    # Use the best S found
    S = best_S
    
    # Build the output: N = F_49^2 \ S
    points = []
    for i in range(2401):
        if i not in S:
            points.append(idx_to_pt(i))
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
