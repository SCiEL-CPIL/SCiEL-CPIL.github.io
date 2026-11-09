# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    deadline = time.time() + 600  # 600s safety margin

    # Field F_49 = F_7[alpha], alpha^2 = 3 (smallest QR non-residue mod 7 is 3)
    k = 3

    # Represent each field element as int 0..48: a + b*alpha -> a + b*7
    def to_int(a, b):
        return a + b * p

    def to_pair(z):
        return (z % p, z // p)

    # Precompute field operations
    # Addition: (a1+b1*alpha) + (a2+b2*alpha) = (a1+a2) + (b1+b2)*alpha
    # Multiplication: (a1+b1*alpha)(a2+b2*alpha) = (a1a2 + b1b2*k) + (a1b2 + a2b1)*alpha
    def add(z1, z2):
        a1, b1 = to_pair(z1)
        a2, b2 = to_pair(z2)
        return to_int((a1 + a2) % p, (b1 + b2) % p)

    def sub(z1, z2):
        a1, b1 = to_pair(z1)
        a2, b2 = to_pair(z2)
        return to_int((a1 - a2) % p, (b1 - b2) % p)

    def mul(z1, z2):
        a1, b1 = to_pair(z1)
        a2, b2 = to_pair(z2)
        return to_int((a1 * a2 + b1 * b2 * k) % p, (a1 * b2 + a2 * b1) % p)

    # Build lookup tables for speed
    add_table = np.zeros((q, q), dtype=np.int64)
    sub_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for z1 in range(q):
        for z2 in range(q):
            add_table[z1, z2] = add(z1, z2)
            sub_table[z1, z2] = sub(z1, z2)
            mul_table[z1, z2] = mul(z1, z2)

    # All 2401 points as (x, y) pairs, each in 0..48
    # Point index: x * q + y
    def point_idx(x, y):
        return x * q + y

    # Allowed directions: (1, m) for m in 0..48, and (0, 1)
    # Direction v = (vx, vy)
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    num_dirs = len(directions)  # 50

    # For each point x and direction v, the 49 other points on the line
    # line_points[x_idx, v_idx, t] = point index of x + t*v, t in 1..48
    # x_idx in 0..2400, v_idx in 0..49, t in 0..48 (t=0 is x itself, we use t=1..48)
    # This is 2401 * 50 * 49 = 5,882,450 entries, manageable

    # Actually, let's precompute for each (x, v) the set of other points
    # But storing as a big array might be memory heavy. Let's compute on the fly or use a smarter structure.

    # Alternative: for each point z, precompute which (x, v) pairs are affected by excluding z
    # z is on line x + t*v iff z = x + t*v for some t != 0, i.e., x = z - t*v
    # For each z and each v, the 49 x values are z - t*v for t in 1..48

    # Let's precompute: for each z (2401) and each v (50), the list of 49 x indices
    # affected_x[z_idx, v_idx] = array of 49 x indices
    affected_x = np.zeros((q * q, num_dirs, q - 1), dtype=np.int64)
    for x in range(q):
        for y in range(q):
            z_idx = point_idx(x, y)
            for vi, (vx, vy) in enumerate(directions):
                for t in range(1, q):
                    # x_point = z - t*v
                    # z = (x, y), v = (vx, vy)
                    # t*v = (t*vx, t*vy) in field
                    tvx = mul_table[t, vx]
                    tvy = mul_table[t, vy]
                    px = sub_table[x, tvx]
                    py = sub_table[y, tvy]
                    affected_x[z_idx, vi, t - 1] = point_idx(px, py)

    # Now greedy: start with all points in N (complement empty)
    # count_valid[x_idx] = number of valid directions for x, initially 50
    count_valid = np.full(q * q, num_dirs, dtype=np.int64)
    in_N = np.ones(q * q, dtype=bool)  # True if in N
    excluded = np.zeros(q * q, dtype=bool)  # True if excluded (in complement)

    # valid_dir[x_idx, v_idx] = whether direction v is valid for x
    # Initially all True
    valid_dir = np.ones((q * q, num_dirs), dtype=bool)

    # Greedily exclude points
    # For each candidate z, check if it's safe to exclude
    # z is safe if for every (x, v) such that z is on line x + t*v, 
    #   after removing v from x's valid set, x still has count_valid > 0
    # i.e., for every affected (x, v), valid_dir[x, v] is True (so it would be removed)
    #   and count_valid[x] >= 2 (so x still has another valid dir)
    # Actually: if valid_dir[x, v] is False, removing v doesn't change anything.
    # If valid_dir[x, v] is True, removing it decrements count_valid[x].
    # x becomes uncovered iff count_valid[x] was 1 and we remove its last valid dir.

    # So z is safe iff for every (x, v) affected by z:
    #   NOT (valid_dir[x, v] == True AND count_valid[x] == 1)
    # i.e., for every affected (x, v), if valid_dir[x, v] then count_valid[x] >= 2

    # Let's try excluding points in a random or ordered fashion
    # Use a simple greedy: try each point, if safe, exclude it

    order = np.random.permutation(q * q)
    excluded_count = 0

    for z_idx in order:
        if time.time() > deadline - 60:
            break
        if excluded[z_idx]:
            continue

        # Check if z is safe to exclude
        safe = True
        for vi in range(num_dirs):
            for t in range(q - 1):
                x_idx = affected_x[z_idx, vi, t]
                if valid_dir[x_idx, vi]:
                    if count_valid[x_idx] <= 1:
                        safe = False
                        break
            if not safe:
                break

        if safe:
            # Exclude z
            excluded[z_idx] = True
            in_N[z_idx] = False
            excluded_count += 1
            # Update: for each (x, v) affected by z, if valid_dir[x, v], decrement count_valid[x] and set valid_dir[x, v] = False
            for vi in range(num_dirs):
                for t in range(q - 1):
                    x_idx = affected_x[z_idx, vi, t]
                    if valid_dir[x_idx, vi]:
                        valid_dir[x_idx, vi] = False
                        count_valid[x_idx] -= 1

    # Build result: all points in N
    points = []
    for x in range(q):
        for y in range(q):
            idx = point_idx(x, y)
            if in_N[idx]:
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])

    result = np.array(points, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
