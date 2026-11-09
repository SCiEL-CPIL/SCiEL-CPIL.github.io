# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    deadline = time.time() + 620
    p = 7
    q = p * p

    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    def f_add(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        return to_int((xa + ya) % p, (xb + yb) % p)

    def f_sub(x, y):
        return f_add(x, f_neg(y))

    def f_neg(x):
        xa, xb = to_pair(x)
        return to_int((-xa) % p, (-xb) % p)

    def f_mul(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        real = (xa * ya + xb * yb * 3) % p
        imag = (xa * yb + xb * ya) % p
        return to_int(real, imag)

    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    n_dir = len(directions)

    all_points = [(x, y) for x in range(q) for y in range(q)]
    point_index = {(x, y): i for i, (x, y) in enumerate(all_points)}
    idx_to_point = {v: k for k, v in point_index.items()}

    in_N = np.ones(q * q, dtype=bool)

    # Initial exclusion: remove points where trace(x) + norm(y) == 0 and y not special
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * 3) % p
            if (trace_x + norm_y) % p == 0:
                if y_idx not in y_special:
                    pt = point_index[(x_idx, y_idx)]
                    in_N[pt] = False

    # Precompute line membership: for each point and direction, which points are on the line
    # not_in_N_count[pt, d] = number of points on line through pt in direction d that are NOT in N
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(q):
            for y_idx in range(q):
                pt = point_index[(x_idx, y_idx)]
                count = 0
                for t in range(1, q):
                    tx = f_add(x_idx, f_mul(t, vx))
                    ty = f_add(y_idx, f_mul(t, vy))
                    t_pt = point_index[(tx, ty)]
                    if not in_N[t_pt]:
                        count += 1
                not_in_N_count[pt, d_idx] = count

    # cover_count[pt] = number of directions for which pt has a valid line (all other points in N)
    cover_count = np.array([np.sum(not_in_N_count[i] == 0) for i in range(q * q)], dtype=np.int32)

    # affects[pt] = list of (other_pt, d_idx) pairs: when pt is removed, these (other_pt, d) pairs
    # gain one more "not in N" point
    affects = [[] for _ in range(q * q)]
    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tx = f_sub(px, f_mul(t, vx))
                ty = f_sub(py, f_mul(t, vy))
                x_idx = point_index[(tx, ty)]
                affects[p_idx].append((x_idx, d_idx))

    # Multi-pass greedy: sort by cover_count descending, try to remove
    max_passes = 8
    for pass_num in range(max_passes):
        if time.time() > deadline - 90:
            break

        n_indices = [i for i in range(q * q) if in_N[i]]
        if not n_indices:
            break

        # Sort by cover_count descending (most redundant first)
        n_indices.sort(key=lambda i: -cover_count[i])

        removed_this_pass = 0
        for p_idx in n_indices:
            if not in_N[p_idx]:
                continue
            if time.time() > deadline - 45:
                break

            # Try to remove this point
            in_N[p_idx] = False
            affected_counts = []
            for (x, d) in affects[p_idx]:
                old_val = not_in_N_count[x, d]
                not_in_N_count[x, d] = old_val + 1
                affected_counts.append((x, d, old_val))

            # Check if any point loses all its cover
            invalid = False
            new_cover_counts = {}
            for (x, d, old_val) in affected_counts:
                if old_val == 0:
                    if x not in new_cover_counts:
                        new_cover_counts[x] = cover_count[x]
                    new_cover_counts[x] -= 1
                    if new_cover_counts[x] <= 0:
                        invalid = True
                        break

            if invalid:
                # Revert
                for (x, d, old_val) in affected_counts:
                    not_in_N_count[x, d] = old_val
                in_N[p_idx] = True
            else:
                # Commit removal
                for (x, d, old_val) in affected_counts:
                    if old_val == 0:
                        cover_count[x] -= 1
                removed_this_pass += 1

        if removed_this_pass == 0:
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
        # Fallback: return all points
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
