# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.

    Strategy:
      1. Start with a baseline Nikodym set N0 built from the Hermitian unital
         construction (same as the seed program).
      2. Greedily try to remove points from N (i.e., add them to the excluded set S)
         as long as the Nikodym property is preserved.
      3. Use vectorized line precomputation and efficient bad-count updates.

    A point x can be removed from N iff for EVERY direction v (49 slopes + vertical),
      the line L(x, v) = {x + t*v : t in F_49} has at most one point NOT in N
      (namely x itself). Equivalently, before removing x, the punctured line
      L(x,v) \\ {x} must be entirely in N. After removing x, the punctured line
      becomes L(x,v) \\ {x} which still must be in N... wait, that's the same.

    Actually the condition is: for every point x in F_49^2, there exists a direction v
    such that L(x, v) \\ {x} ⊆ N. If we remove x from N, we need to check:
      - For x itself: it still needs SOME direction v where L(x,v)\\{x} ⊆ N.
        But L(x,v)\\{x} doesn't include x, so removing x from N doesn't affect this.
        Wait, L(x,v) = {x + t*v : t in F_49}. The punctured line is {x + t*v : t != 0}.
        These are all points OTHER than x. So removing x from N doesn't change
        whether L(x,v)\\{x} ⊆ N. Good.
      - For every OTHER point y in F_49^2 that lies on a line through x with direction v:
        if y's witnessing direction was v_y and y lies on L(x, v_y) (i.e., x is on the
        line through y in direction v_y), then removing x from N might break y's condition
        IF x was the only point on L(y, v_y)\\{y} that was in N... no wait.

    Let me re-think. The condition for a Nikodym set N is:
      For every point x in F_49^2, there exists a direction v (from the 50 directions:
      49 slopes m giving direction (1,m), plus vertical direction (0,1)) such that
      {x + t*v : t in F_49, t != 0} ⊆ N.

    So each point x needs at least one "witnessing direction" v such that all 48 other
    points on that line are in N.

    If we remove a point p from N, which points' conditions could be broken?
      A point y's condition with direction v is broken if p lies on L(y, v) \\ {y},
      i.e., if p = y + t*v for some t != 0, AND p was in N before removal.
      After removal, p is not in N, so L(y,v)\\{y} is no longer fully in N.

    So removing p breaks the (y, v) pair where p is on the line through y in direction v
    (p != y). If y had ONLY that one witnessing direction, then y loses its Nikodym property.

    Algorithm:
      - Maintain N as a boolean array of size 2401 (or set of indices).
      - For each point y, maintain the set of directions v for which L(y,v)\\{y} ⊆ N.
      - A point y is "safe" if it has at least one such direction.
      - When we remove p from N:
          For each direction v (50 total), find all y such that p is on L(y,v)\\{y}.
          That is, y = p - t*v for t != 0, i.e., y = p + s*v for s != 0 (since -t ranges
          over all nonzero elements). So y ranges over the other 48 points on the line
          through p in direction v.
          For each such y, decrement the count of valid directions for y.
          If y's count drops to 0, y is now "broken".

      - Greedy: at each step, find a point p in N such that removing it breaks the FEWEST
        points (i.e., the number of y whose count would drop to 0 is minimized).
        Actually, we want to remove p if it breaks ZERO points (i.e., no y loses its last
        witnessing direction). If multiple such p exist, pick any. If none exist, we can
        try to remove p that breaks the fewest, but then we'd need to also fix those points
        by adding other points... which gets complicated.

      Simpler greedy: only remove p if it breaks no point. This is safe and monotonic.

    Let's implement this.

    Directions: 50 total.
      - 49 directions: v = (1, m) for m in F_49 (represented as pairs (ma, mb)).
      - 1 direction: v = (0, 1) (vertical).

    For a point p and direction v, the line L(p, v) = {p + t*v : t in F_49}.
    The other 48 points are p + t*v for t != 0.

    We need efficient computation. Let's flatten F_49^2 into indices 0..2400.
    Point index: idx = ((xa * 7 + xb) * 7 + ya) * 7 + yb.
    Or better: idx = xa * 49 + xb * 7 + ya... let me use a consistent scheme.

    Let's use: idx = xa * 343 + xb * 49 + ya * 7 + yb.
    Total: 7^4 = 2401. Good.

    Precompute for each direction v and each point p, the list of other 48 points on L(p,v).
    That's 50 * 2401 * 48 entries, which is manageable.

    Actually, for the greedy removal, when we remove p, we need to know for each direction v,
    the other points on L(p,v). Then for each such point y, we decrement y's count for direction v.

    Let me precompute:
      lines[dir_idx][p_idx] = array of 48 point indices (the other points on the line).

    And maintain:
      valid_count[y_idx] = number of directions v for which L(y,v)\\{y} ⊆ N.
      valid_dir[y_idx, dir_idx] = boolean, whether L(y,v)\\{y} ⊆ N.

    Initially, compute valid_dir for all y and v by checking if all 48 points are in N.

    This is 2401 * 50 * 48 = ~5.7 million checks. With vectorization, this is fast.

    Then greedy loop:
      For each p in N (in some order), check if removing p breaks any point.
      Removing p breaks y (for direction v) if valid_dir[y, v] is True AND p is on L(y,v)\\{y}.
      But p is on L(y,v)\\{y} iff y is on L(p,v)\\{p} (symmetric). So we can precompute
      for each p and v, the set of y on L(p,v)\\{p}. Then removing p affects (y, v) pairs
      where y is in that set.

      For each such (y, v): if valid_dir[y, v] is True, then after removal, valid_dir[y, v]
      becomes False (since p is no longer in N). So valid_count[y] decreases by 1.
      If valid_count[y] becomes 0, y is broken.

      So to check if removing p is safe: for all (y, v) where y is on L(p,v)\\{p} and
      valid_dir[y,v] is True, check if valid_count[y] > 1. If any has valid_count[y] == 1,
      then removing p breaks y.

      We can precompute for each p a list of (y, v) pairs to check. That's 50 * 48 = 2400 pairs per p.
      Total across all p: 2401 * 2400 ≈ 5.7 million. Manageable.

    Let me code this up.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    p = 7
    q = p * p  # 49

    # Field F_49 = F_7[alpha], alpha^2 = 3 (smallest quadratic non-residue mod 7)
    k = 3  # alpha^2 = 3

    # Flattening: idx = xa * 343 + xb * 49 + ya * 7 + yb
    def to_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def from_idx(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb

    # Build all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append((xa, xb, ya, yb))

    n_pts = len(all_points)  # 2401

    # Directions: 50 total
    # dir 0..48: v = (1, m) where m = (ma, mb)
    # dir 49: v = (0, 1)
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, ma, mb))  # v = (1, ma + mb*alpha)
    dirs.append((0, 0, 1))  # vertical: v = (0, 1)
    n_dirs = len(dirs)  # 50

    # Precompute lines: lines[dir_idx][p_idx] = list of 48 other point indices
    lines = np.zeros((n_dirs, n_pts, q - 1), dtype=np.int32)

    for di, (vx, va, vb) in enumerate(dirs):
        for pi in range(n_pts):
            xa, xb, ya, yb = from_idx(pi)
            count = 0
            for t in range(q):
                if t == 0:
                    continue
                # t is in F_49, represented as (ta, tb)
                ta = t // p
                tb = t % p
                # t * v:
                # If vx=1: result = (1*ta, 1*tb) = (ta, tb)
                # If vx=0: result = (0, va*ta + vb*tb*alpha)... wait, need field multiplication.
                
                # Let's do field arithmetic properly.
                # v = (vx, vx_a + vx_b*alpha) where vx is the x-component as a field element.
                # Actually, dirs stores (vx, va, vb) meaning v = (vx + 0*alpha? No...)
                
                # Let me redefine: v is a vector in F_49^2.
                # v = (vx, vy) where vx, vy in F_49.
                # For slope m: v = (1, m) = ((1,0), (ma, mb))
                # For vertical: v = (0, 1) = ((0,0), (0,1))
                
                # So dirs should store (vx_a, vx_b, vy_a, vy_b).
                pass
            # This is getting complicated. Let me redo the direction representation.
    # Let me restart the direction and line computation more carefully.

    # Redo: directions as (vx_a, vx_b, vy_a, vy_b)
    dirs2 = []
    for ma in range(p):
        for mb in range(p):
            # v = (1, ma + mb*alpha) = ((1,0), (ma,mb))
            dirs2.append((1, 0, ma, mb))
    # vertical: v = (0, 1) = ((0,0), (0,1))
    dirs2.append((0, 0, 0, 1))
    n_dirs2 = len(dirs2)  # 50

    # Field multiplication in F_49: (a + b*alpha) * (c + d*alpha) = (ac + bd*k, ad + bc)
    def field_mul(a, b, c, d):
        """Multiply (a+b*alpha) * (c+d*alpha) in F_49."""
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    def field_add(a, b, c, d):
        return ((a + c) % p, (b + d) % p)

    # Precompute lines again
    lines = np.zeros((n_dirs2, n_pts, q - 1), dtype=np.int32)
    for di, (vxa, vxb, vya, vyb) in enumerate(dirs2):
        for pi in range(n_pts):
            xa, xb, ya, yb = from_idx(pi)
            count = 0
            for t in range(q):
                if t == 0:
                    continue
                ta = t // p
                tb = t % p
                # t * v:
                # x-component: (ta + tb*alpha) * (vxa + vxb*alpha)
                nxa, nxb = field_mul(ta, tb, vxa, vxb)
                # y-component: (ta + tb*alpha) * (vya + vyb*alpha)
                nya, nyb = field_mul(ta, tb, vya, vyb)
                # p + t*v:
                nxa, nxb = field_add(xa, xb, nxa, nxb)
                nya, nyb = field_add(ya, yb, nya, nyb)
                idx = to_idx(nxa, nxb, nya, nyb)
                lines[di, pi, count] = idx
                count += 1

    # Baseline Nikodym set from Hermitian unital
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    in_N = np.ones(n_pts, dtype=bool)
    for pi in range(n_pts):
        xa, xb, ya, yb = from_idx(pi)
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            in_N[pi] = False  # excluded

    # Compute valid_dir[y, v]: True if L(y,v)\{y} ⊆ N
    valid_dir = np.zeros((n_pts, n_dirs2), dtype=bool)
    for yi in range(n_pts):
        for di in range(n_dirs2):
            other_pts = lines[di, yi]
            valid_dir[yi, di] = np.all(in_N[other_pts])

    valid_count = valid_dir.sum(axis=1)  # shape (n_pts,)

    # Check baseline is valid
    assert np.all(valid_count >= 1), "Baseline Nikodym set is invalid!"

    # Greedy removal
    # For each p in N, determine if removing it is safe.
    # Precompute for each p, the list of (y, v) pairs affected.
    # affected[p] = list of (y_idx, v_idx) such that y is on L(p,v)\{p}
    # This is symmetric: y on L(p,v)\{p} iff p on L(y,v)\{y}

    # Build affected list
    affected = [[] for _ in range(n_pts)]
    for di in range(n_dirs2):
        for pi in range(n_pts):
            for oi in range(q - 1):
                y = lines[di, pi, oi]
                affected[pi].append((y, di))

    # Greedy loop
    removed = 0
    n_in_N = in_N.sum()
    changed = True
    while changed and (time.time() - start_time) < deadline:
        changed = False
        # Find a point p in N whose removal breaks no point
        for pi in range(n_pts):
            if not in_N[pi]:
                continue
            # Check if removing pi is safe
            safe = True
            for (y, di) in affected[pi]:
                if valid_dir[y, di] and valid_count[y] == 1:
                    safe = False
                    break
            if safe:
                # Remove pi
                in_N[pi] = False
                n_in_N -= 1
                removed += 1
                # Update valid_dir and valid_count for affected (y, di)
                for (y, di) in affected[pi]:
                    if valid_dir[y, di]:
                        valid_dir[y, di] = False
                        valid_count[y] -= 1
                changed = True
                break  # Restart scan after a removal

    # Build output
    points = []
    for pi in range(n_pts):
        if in_N[pi]:
            xa, xb, ya, yb = from_idx(pi)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
