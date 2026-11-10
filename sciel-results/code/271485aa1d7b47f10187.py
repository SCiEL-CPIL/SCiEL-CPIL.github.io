import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580

    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    def to_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def from_idx(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb

    n_pts = q * q  # 2401

    # Directions
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))
    dirs.append((0, 0, 0, 1))
    n_dirs = len(dirs)

    def field_mul(a, b, c, d):
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)

    def field_add(a, b, c, d):
        return ((a + c) % p, (b + d) % p)

    # Precompute lines: lines[di][pi] = array of 48 other point indices
    lines = np.zeros((n_dirs, n_pts, q - 1), dtype=np.int32)
    for di, (vxa, vxb, vya, vyb) in enumerate(dirs):
        for pi in range(n_pts):
            xa, xb, ya, yb = from_idx(pi)
            count = 0
            for t in range(q):
                if t == 0:
                    continue
                ta = t // p
                tb = t % p
                nxa, nxb = field_mul(ta, tb, vxa, vxb)
                nya, nyb = field_mul(ta, tb, vya, vyb)
                nxa, nxb = field_add(xa, xb, nxa, nxb)
                nya, nyb = field_add(ya, yb, nya, nyb)
                idx = to_idx(nxa, nxb, nya, nyb)
                lines[di, pi, count] = idx
                count += 1

    # Baseline: Inspiration 1 logic
    # Exclude points in U where y is NOT in Y_special
    # Y_special: y_a is non-zero square in F_7 => {1, 2, 4}
    y_special = set()
    for u in range(1, p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    in_N = np.ones(n_pts, dtype=bool)
    for pi in range(n_pts):
        xa, xb, ya, yb = from_idx(pi)
        # Check if in Unital U: 2*xa + ya^2 - yb^2*k == 0 mod 7
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            in_N[pi] = False

    # Compute valid_dir and valid_count
    # valid_dir[y, v] is True if L(y, v) \ {y} is fully in N (i.e., no points from S)
    valid_dir = np.zeros((n_pts, n_dirs), dtype=bool)
    for yi in range(n_pts):
        for di in range(n_dirs):
            other_pts = lines[di, yi]
            valid_dir[yi, di] = np.all(in_N[other_pts])
            
    valid_count = valid_dir.sum(axis=1)
    
    # Check baseline validity
    if not np.all(valid_count >= 1):
        # Fallback to a trivial valid set if baseline is broken (shouldn't happen)
        in_N[:] = True
        valid_dir[:] = True
        valid_count[:] = n_dirs

    # Precompute affected pairs for greedy step
    # affected[pi] = list of (y, di) such that y is on L(pi, di) \ {pi}
    affected = [[] for _ in range(n_pts)]
    for di in range(n_dirs):
        for pi in range(n_pts):
            for oi in range(q - 1):
                y = lines[di, pi, oi]
                affected[pi].append((y, di))

    # Greedy expansion
    # We want to add points to S (remove from N).
    # A point pi can be added if for all (y, di) in affected[pi], 
    # it is NOT the case that valid_dir[y, di] is True AND valid_count[y] == 1.
    # If we add pi, then for all (y, di) in affected[pi] where valid_dir[y, di] was True,
    # valid_dir[y, di] becomes False, and valid_count[y] decreases.
    
    changed = True
    while changed and (time.time() - start_time) < deadline:
        changed = False
        # Try to find a point to add
        # Randomize order to avoid getting stuck in local minima? 
        # Or just iterate. 
        # To be efficient, we can check points that are "central" or just iterate 0..N
        for pi in range(n_pts):
            if in_N[pi]:
                # Check if adding pi is safe
                safe = True
                # We need to check if any y loses its last valid direction
                for (y, di) in affected[pi]:
                    if valid_dir[y, di] and valid_count[y] == 1:
                        safe = False
                        break
                if safe:
                    # Add pi to S (remove from N)
                    in_N[pi] = False
                    # Update valid_dir and valid_count
                    for (y, di) in affected[pi]:
                        if valid_dir[y, di]:
                            valid_dir[y, di] = False
                            valid_count[y] -= 1
                    changed = True
                    break # Restart scan

    # Build output
    points = []
    for pi in range(n_pts):
        if in_N[pi]:
            xa, xb, ya, yb = from_idx(pi)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
