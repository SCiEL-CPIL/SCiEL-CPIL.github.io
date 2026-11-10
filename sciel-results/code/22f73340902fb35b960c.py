import numpy as np
import time
import random

DEADLINE = 620.0
START_TIME = time.time()

def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3
    q = 49
    n_total = 2401
    n_dirs = 50

    # Field operations
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

    # Enumerate points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))

    def point_to_int(pt):
        (xa, xb), (ya, yb) = pt
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    # Directions
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))

    # Precompute conflict masks
    # conflict_mask[y][x] is a bitmask of directions d such that y is on the line through x in direction d (y != x)
    conflict_mask = np.zeros((n_total, n_total), dtype=np.uint64)
    
    # For each direction d, group points by line
    # For each line, for each pair (x, y) on the line, set bit d in conflict_mask[y][x] and conflict_mask[x][y]
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        line_map = {}
        for pt_idx, pt in enumerate(all_points):
            x, y = pt
            if vx == (1, 0):
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:
                c = x
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        
        for c, pts in line_map.items():
            n_pts = len(pts)
            for i in range(n_pts):
                for j in range(n_pts):
                    if i != j:
                        x_idx = pts[i]
                        y_idx = pts[j]
                        # y_idx is on the line through x_idx in direction d_idx
                        conflict_mask[y_idx][x_idx] |= (1 << d_idx)

    best_S_size = 0
    best_S_mask = 0
    all_indices = list(range(n_total))

    while _time_left() > 10:
        random.shuffle(all_indices)
        valid_dirs = np.full(n_total, (1 << n_dirs) - 1, dtype=np.uint64)
        S_mask = 0
        S_size = 0
        
        for y in all_indices:
            # Check if y can be added
            # valid_dirs[x] & ~conflict_mask[y][x] must be non-zero for all x
            new_valid_dirs = valid_dirs & ~conflict_mask[y]
            if np.any(new_valid_dirs == 0):
                continue
            
            # Add y
            valid_dirs = new_valid_dirs
            S_mask |= (1 << y)
            S_size += 1
        
        if S_size > best_S_size:
            best_S_size = S_size
            best_S_mask = S_mask

    # Construct result
    points = []
    for pt_idx in range(n_total):
        if not (best_S_mask >> pt_idx) & 1:
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])

    arr = np.array(points, dtype=np.int64)
    return arr
