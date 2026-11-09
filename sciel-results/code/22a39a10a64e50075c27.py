# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Strategy: Hermitian unital base + greedy removal with proper revert logic.
    """
    deadline = time.time() + 600  # Safety margin before 630s limit

    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k (smallest quadratic non-residue)
    k_field = 3  # 3 is a quadratic non-residue mod 7

    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables
    f_add_table = np.zeros((q, q), dtype=np.int32)
    f_mul_table = np.zeros((q, q), dtype=np.int32)
    f_neg_table = np.zeros(q, dtype=np.int32)

    for x in range(q):
        xa, xb = to_pair(x)
        f_neg_table[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add_table[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul_table[x, y] = to_int(real, imag)

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    directions = [(1, m) for m in range(q)] + [(0, 1)]
    n_dir = len(directions)

    # Point indexing
    point_index = {}
    idx_to_point = [None] * (q * q)
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            idx_to_point[idx] = (x_idx, y_idx)

    # Build base Nikodym set: complement of excluded points from Hermitian unital
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    in_N = np.ones(q * q, dtype=bool)
    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k_field) % p
            if (trace_x + norm_y) % p == 0 and y_idx not in y_special:
                in_N[x_idx * q + y_idx] = False

    # not_in_N_count[p_idx, d_idx] = # points on line(p, v_d)\{p} not in N
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)
    
    for d_idx, (vx, vy) in enumerate(directions):
        t_v = [(f_mul_table[t, vx], f_mul_table[t, vy]) for t in range(1, q)]
        for p_idx in range(q * q):
            px, py = idx_to_point[p_idx]
            count = 0
            for tx, ty in t_v:
                nx = f_add_table[px, tx]
                ny = f_add_table[py, ty]
                if not in_N[nx * q + ny]:
                    count += 1
            not_in_N_count[p_idx, d_idx] = count

    cover_count = np.sum(not_in_N_count == 0, axis=1).astype(np.int32)

    # Precompute affects: affects[p] = list of (x, d) where p is on line(x, v_d)\{x}
    affects = [[] for _ in range(q * q)]
    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tv_x = f_mul_table[t, vx]
                tv_y = f_mul_table[t, vy]
                nx = f_add_table[px, f_neg_table[tv_x]]
                ny = f_add_table[py, f_neg_table[tv_y]]
                affects[p_idx].append((nx * q + ny, d_idx))

    # Greedy removal with proper tracking
    while time.time() < deadline - 10:
        progress = False
        indices_in_N = [i for i in range(q * q) if in_N[i]]
        np.random.shuffle(indices_in_N)
        
        for p_idx in indices_in_N:
            if not in_N[p_idx]:
                continue
            if time.time() > deadline - 5:
                break
            
            # Collect changes for potential revert
            changes = []  # (x, d, old_count, old_cover)
            valid = True
            
            for (x, d) in affects[p_idx]:
                old_count = not_in_N_count[x, d]
                old_cover = cover_count[x]
                
                if old_count == 0:
                    if old_cover == 1:
                        valid = False
                        break
                    else:
                        cover_count[x] -= 1
                        changes.append((x, d, old_count, old_cover))
                        not_in_N_count[x, d] += 1
                else:
                    changes.append((x, d, old_count, old_cover))
                    not_in_N_count[x, d] += 1
            
            if valid:
                in_N[p_idx] = False
                progress = True
            else:
                # Revert all changes
                for (x, d, old_count, old_cover) in changes:
                    not_in_N_count[x, d] = old_count
                    cover_count[x] = old_cover
        
        if not progress:
            break

    # Build result
    points = []
    for i in range(q * q):
        if in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: reconstruct base set
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                norm_y = (ya * ya - yb * yb * k_field) % p
                if (trace_x + norm_y) % p == 0 and y_idx not in y_special:
                    continue
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
