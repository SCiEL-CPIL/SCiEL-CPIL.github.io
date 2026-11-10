# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by searching over blocking set parameters.
    Strategy:
    1. Precompute field arithmetic and line structures.
    2. For various blocking set Y (subsets of F_49 for y-coordinates),
       compute S = {points in unital U where y not in Y}.
    3. Validate each candidate using vectorized checks.
    4. Return the N = F_49^2 \ S with largest complement.
    """
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s kill

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3, non-residue mod 7
    total_points = q * q  # 2401

    # Field element helpers
    def idx(a, b):
        return a + b * p

    def a_of(i):
        return i % p

    def b_of(i):
        return i // p

    # Build multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            res_a = (ai * aj + bi * bj * k) % p
            res_b = (ai * bj + bi * aj) % p
            mul_table[i, j] = idx(res_a, res_b)

    # Build addition table
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            add_table[i, j] = ((ai + aj) % p) + ((bi + bj) % p) * p

    # Precompute line_other[d, x_idx, j] = the j-th other point on line through x in direction d
    # d=0..48: direction (1, m) where m = d
    # d=49: direction (0, 1)
    line_other = np.full((50, total_points, 48), -1, dtype=np.int32)

    for d in range(50):
        for x_idx in range(total_points):
            x = x_idx // q
            y = x_idx % q
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    nx = add_table[x, t]
                    ny = add_table[y, mul_table[t, m]]
                    line_other[d, x_idx, count] = nx * q + ny
                    count += 1
            else:
                for t in range(1, q):
                    nx = x
                    ny = add_table[y, t]
                    line_other[d, x_idx, count] = nx * q + ny
                    count += 1

    # Precompute affected[s_idx, d, j] = j-th point x such that s is on line_d(x), s != x
    affected = np.full((total_points, 50, 48), -1, dtype=np.int32)

    for s_idx in range(total_points):
        sx = s_idx // q
        sy = s_idx % q
        for d in range(50):
            count = 0
            if d < 49:
                m = d
                for t in range(1, q):
                    neg_t = (q - t) % q
                    nx = add_table[sx, neg_t]
                    ny = add_table[sy, mul_table[neg_t, m]]
                    affected[s_idx, d, count] = nx * q + ny
                    count += 1
            else:
                for t in range(1, q):
                    neg_t = (q - t) % q
                    nx = sx
                    ny = add_table[sy, neg_t]
                    affected[s_idx, d, count] = nx * q + ny
                    count += 1

    # Compute unital points: U = {(xa, xb, ya, yb) : 2*xa + ya^2 - k*yb^2 = 0 mod p}
    unital_mask = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        x_idx = xa + xb * p
                        y_idx = ya + yb * p
                        unital_mask[x_idx * q + y_idx] = True

    # Helper: validate a complement set S (returns True if N = complement is valid Nikodym)
    def validate_complement(S_set):
        """Check if F_49^2 \ S_set is a valid Nikodym set."""
        N_mask = np.ones(total_points, dtype=bool)
        for s in S_set:
            N_mask[s] = False

        # valid_dir_mask[x, d] = True if all 48 other points on line_d(x) are in N
        valid_count = np.zeros(total_points, dtype=np.int32)

        for x_idx in range(total_points):
            for d in range(50):
                others = line_other[d, x_idx, :]
                if np.all(N_mask[others]):
                    valid_count[x_idx] += 1

        return np.all(valid_count > 0)

    # Helper: get S for a given y-blocking set Y
    def get_S_for_Y(Y_set):
        """S = {points in unital U where y-coordinate not in Y}"""
        S = set()
        for pt in range(total_points):
            if unital_mask[pt]:
                y_idx = pt % q
                if y_idx not in Y_set:
                    S.add(pt)
        return S

    # Generate candidate Y sets (subsets of F_49 for y-coordinates)
    # The y-coordinate is a field element 0..48
    # We want Y to be "large enough" to block all lines but "small" to maximize |S|

    candidates = []

    # Candidate 1: y_a is a square (known working, |Y| = 28)
    y_sq = set()
    for u in range(p):
        for v in range(p):
            y_sq.add(idx(u * u % p, v))
    candidates.append(('squares', y_sq))

    # Candidate 2: y_a = 0 (|Y| = 7)
    y_zero = set()
    for v in range(p):
        y_zero.add(idx(0, v))
    candidates.append(('y_a=0', y_zero))

    # Candidate 3: y_a in {0, 1} (|Y| = 14)
    y_01 = set()
    for v in range(p):
        y_01.add(idx(0, v))
        y_01.add(idx(1, v))
    candidates.append(('y_a in {0,1}', y_01))

    # Candidate 4: y_a in {0, 1, 2} (|Y| = 21)
    y_012 = set()
    for v in range(p):
        y_012.add(idx(0, v))
        y_012.add(idx(1, v))
        y_012.add(idx(2, v))
    candidates.append(('y_a in {0,1,2}', y_012))

    # Candidate 5: y_a in {0, 1, 2, 3} (|Y| = 28)
    y_0123 = set()
    for v in range(p):
        y_0123.add(idx(0, v))
        y_0123.add(idx(1, v))
        y_0123.add(idx(2, v))
        y_0123.add(idx(3, v))
    candidates.append(('y_a in {0,1,2,3}', y_0123))

    # Candidate 6: y_a in {0, 1, 2, 3, 4} (|Y| = 35)
    y_01234 = set()
    for v in range(p):
        y_01234.add(idx(0, v))
        y_01234.add(idx(1, v))
        y_01234.add(idx(2, v))
        y_01234.add(idx(3, v))
        y_01234.add(idx(4, v))
    candidates.append(('y_a in {0,1,2,3,4}', y_01234))

    # Candidate 7: y_a in {0, 1, 2, 3, 4, 5} (|Y| = 42)
    y_012345 = set()
    for v in range(p):
        y_012345.add(idx(0, v))
        y_012345.add(idx(1, v))
        y_012345.add(idx(2, v))
        y_012345.add(idx(3, v))
        y_012345.add(idx(4, v))
        y_012345.add(idx(5, v))
    candidates.append(('y_a in {0,1,2,3,4,5}', y_012345))

    # Candidate 8: y_a in {0, 1, 2, 3, 4, 5, 6} = all (|Y| = 49)
    y_all = set(range(q))
    candidates.append(('all y', y_all))

    # Candidate 9: y_b = 0 (|Y| = 7)
    yb_zero = set()
    for u in range(p):
        yb_zero.add(idx(u, 0))
    candidates.append(('y_b=0', yb_zero))

    # Candidate 10: y_a=0 or y_b=0 (|Y| = 13)
    y_a0_or_yb0 = set()
    for v in range(p):
        y_a0_or_yb0.add(idx(0, v))
    for u in range(p):
        y_a0_or_yb0.add(idx(u, 0))
    candidates.append(('y_a=0 or y_b=0', y_a0_or_yb0))

    # Try random subsets of various sizes
    np.random.seed(42)
    for size in [7, 14, 21, 28, 35, 42]:
        for trial in range(5):
            random_y = set(np.random.choice(q, size, replace=False))
            candidates.append((f'random_size_{size}_trial_{trial}', random_y))

    # Validate all candidates and find the best (largest complement)
    best_S = None
    best_complement = 0

    for name, Y in candidates:
        if time.time() - start_time > deadline - 30:
            break

        S = get_S_for_Y(Y)
        complement_size = total_points - len(S)

        if complement_size <= best_complement:
            continue  # Can't improve

        if validate_complement(S):
            best_S = S
            best_complement = complement_size

    # If no candidate worked (shouldn't happen since squares is known valid),
    # fall back to the known valid construction
    if best_S is None:
        y_sq = set()
        for u in range(p):
            for v in range(p):
                y_sq.add(idx(u * u % p, v))
        best_S = get_S_for_Y(y_sq)
        best_complement = total_points - len(best_S)

    # Now try greedy expansion from the best S found
    # Add points to S one by one if it doesn't break validity
    if time.time() - start_time < deadline - 60:
        N_mask = np.ones(total_points, dtype=bool)
        for s in best_S:
            N_mask[s] = False

        # Compute valid directions
        valid_dir_mask = np.zeros((total_points, 50), dtype=bool)
        valid_count = np.zeros(total_points, dtype=np.int32)

        for x_idx in range(total_points):
            for d in range(50):
                others = line_other[d, x_idx, :]
                if np.all(N_mask[others]):
                    valid_dir_mask[x_idx, d] = True
                    valid_count[x_idx] += 1

        # Greedily try to add more points to S
        candidates_list = [i for i in range(total_points) if i not in best_S]
        np.random.shuffle(candidates_list)

        improved = True
        while improved and time.time() - start_time < deadline - 30:
            improved = False
            for s in candidates_list:
                if time.time() - start_time > deadline - 30:
                    break
                if s in best_S:
                    continue

                # Check if adding s to S breaks any point
                # Collect all (x, d) that would be invalidated
                to_invalidate = []
                for d in range(50):
                    xs = affected[s, d, :]
                    for x in xs:
                        if x == -1:
                            continue
                        if valid_dir_mask[x, d]:
                            to_invalidate.append((x, d))

                # Check if any x would have 0 valid directions
                x_invalidated_count = {}
                for (x, d) in to_invalidate:
                    x_invalidated_count[x] = x_invalidated_count.get(x, 0) + 1

                valid_addition = True
                for x, cnt in x_invalidated_count.items():
                    if valid_count[x] - cnt <= 0:
                        valid_addition = False
                        break

                if valid_addition:
                    # Add s to S
                    best_S.add(s)
                    N_mask[s] = False
                    for (x, d) in to_invalidate:
                        valid_dir_mask[x, d] = False
                        valid_count[x] -= 1
                    candidates_list.remove(s)
                    improved = True
                    break

    # Build the result: N = all points not in S
    points = []
    for i in range(total_points):
        if i not in best_S:
            x = i // q
            y = i % q
            xa, xb = a_of(x), b_of(x)
            ya, yb = a_of(y), b_of(y)
            points.append([[xa, xb], [ya, yb]])

    result = np.array(points, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
