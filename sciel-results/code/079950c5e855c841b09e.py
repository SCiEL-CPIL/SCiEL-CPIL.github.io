# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 590
    
    p = 7
    k = 3
    
    def deidx(i):
        return (i // 7, i % 7)
    
    def idx(a, b):
        return a * 7 + b
    
    # Build field tables
    mult_table = np.zeros((49, 49), dtype=np.int64)
    add_table = np.zeros((49, 49), dtype=np.int64)
    for i in range(49):
        ai, bi = deidx(i)
        for j in range(49):
            aj, bj = deidx(j)
            mult_table[i, j] = idx((ai*aj + bi*bj*k) % p, (ai*bj + bi*aj) % p)
            add_table[i, j] = idx((ai+aj) % p, (bi+bj) % p)
    
    # Directions
    dirs = [(1, m) for m in range(49)] + [(0, 1)]
    num_dirs = len(dirs)
    num_points = 2401
    
    # Precompute lines
    lines = np.zeros((num_dirs, num_points, 49), dtype=np.int64)
    for d in range(num_dirs):
        vx, vy = dirs[d]
        for pt in range(num_points):
            px, py = deidx(pt)
            for t in range(49):
                tvx = mult_table[t, vx]
                tvy = mult_table[t, vy]
                sx = add_table[px, tvx]
                sy = add_table[py, tvy]
                lines[d, pt, t] = sx * 49 + sy
    
    # other_points[d, p] = 48 points on line through p in direction d, excluding p
    other_points = lines[:, :, 1:]  # t=1..48
    
    # Base Nikodym set from seed
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(idx(u*u % p, v))
    
    in_N = np.zeros(num_points, dtype=bool)
    for pt in range(num_points):
        x = pt // 49
        y = pt % 49
        xa, xb = deidx(x)
        ya, yb = deidx(y)
        in_unital = (2*xa + ya*ya - yb*yb*k) % p == 0
        in_y_special = y in y_special
        if not in_unital or in_y_special:
            in_N[pt] = True
    
    # valid_count[x] = number of directions where line through x has all other 48 in N
    valid_count = np.zeros(num_points, dtype=np.int32)
    for x in range(num_points):
        for d in range(num_dirs):
            # Check if all other 48 points on line are in N
            other = other_points[d, x]
            if np.all(in_N[other]):
                valid_count[x] += 1
    
    # Greedy removal
    for pass_num in range(100):
        if time.time() > deadline:
            break
        any_removed = False
        for p_idx in range(num_points):
            if not in_N[p_idx]:
                continue
            # Check if removing p_idx is safe
            safe = True
            for d in range(num_dirs):
                other = other_points[d, p_idx]
                # For each x in other, valid_count[x] would decrease by 1
                # If any valid_count[x] <= 1, unsafe
                if np.any(valid_count[other] <= 1):
                    safe = False
                    break
            if safe:
                in_N[p_idx] = False
                any_removed = True
                for d in range(num_dirs):
                    other = other_points[d, p_idx]
                    valid_count[other] -= 1
        if not any_removed:
            break
    
    result = []
    for pt in range(num_points):
        if in_N[pt]:
            x = pt // 49
            y = pt % 49
            xa, xb = deidx(x)
            ya, yb = deidx(y)
            result.append([[xa, xb], [ya, yb]])
    
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
