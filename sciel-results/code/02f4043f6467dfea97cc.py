# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_P = 7
_Q = 49
_Q2 = 2401
_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - time.time() - _MARGIN


def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    k = 3  # alpha^2 = 3

    # Field operations using precomputed tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    def pt_idx(x, y):
        return x * q + y

    def pt_xy(idx):
        return idx // q, idx % q

    # Build all lines
    line_points = []

    # Vertical lines: 49 lines
    for x in range(q):
        pts = np.array([pt_idx(x, y) for y in range(q)], dtype=np.int64)
        line_points.append(pts)

    # Slanted lines: 49*49 = 2401 lines
    for m in range(q):
        for c in range(q):
            pts = np.array([pt_idx(t, int(add_table[mul_table[m, t], c])) for t in range(q)], dtype=np.int64)
            line_points.append(pts)

    n_lines = len(line_points)  # 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(q2)]
    for li in range(n_lines):
        for pi in line_points[li]:
            point_lines[pi].append(li)

    # State variables
    S = np.zeros(q2, dtype=bool)
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    good_line_count = np.zeros(q2, dtype=np.int64)

    # Initialize good_line_count for S = empty
    # All lines have count 0.
    # For z not in S, good if count == 0. All lines are good.
    # Count of lines through any point is q + 1 = 50.
    good_line_count[:] = q + 1

    # Precompute line points as lists for faster iteration in Python
    line_points_list = [line_points[i].tolist() for i in range(n_lines)]
    point_lines_list = [point_lines[i] for i in range(q2)]

    def compute_invalid_count():
        return int(np.sum(good_line_count == 0))

    def get_energy(invalid_count, s_size):
        # We want to minimize invalid_count, then maximize s_size
        # Energy = invalid_count * 10000 - s_size
        return invalid_count * 10000 - s_size

    # Greedy initialization
    def add_point(z):
        """Try to add z to S. Returns True if successful, False if invalid."""
        S[z] = True
        changes = []

        for li in point_lines_list[z]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count

            for w in line_points_list[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                        changes.append((w, 1))
                else:
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
                        changes.append((w, 1))

        z_good = 0
        for li in point_lines_list[z]:
            if line_S_count[li] == 1:
                z_good += 1

        old_z_good = good_line_count[z]
        good_line_count[z] = z_good
        changes.append((z, z_good - old_z_good))

        if np.all(good_line_count > 0):
            return True
        else:
            for w, delta in changes:
                good_line_count[w] -= delta
            for li in point_lines_list[z]:
                line_S_count[li] -= 1
            S[z] = False
            return False

    def remove_point(z):
        if not S[z]:
            return False

        S[z] = False

        for li in point_lines_list[z]:
            old_count = line_S_count[li]
            new_count = old_count - 1
            line_S_count[li] = new_count

            for w in line_points_list[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                else:
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1

        z_good = 0
        for li in point_lines_list[z]:
            if line_S_count[li] == 0:
                z_good += 1

        good_line_count[z] = z_good
        return True

    # 1. Greedy Start
    rng = np.random.default_rng(42)
    points_to_try = np.arange(q2)
    for _pass in range(20):
        if _time_left() <= 0:
            break
        rng.shuffle(points_to_try)
        added_in_pass = 0
        for z in points_to_try:
            if _time_left() <= 0:
                break
            if not S[z]:
                if add_point(z):
                    added_in_pass += 1
        if added_in_pass == 0:
            break

    best_s_size = int(np.sum(S))
    best_S = S.copy()
    best_line_S_count = line_S_count.copy()
    best_good_line_count = good_line_count.copy()
    best_invalid = compute_invalid_count()

    # 2. Simulated Annealing
    # We operate on the current state S, line_S_count, good_line_count
    # We track the best valid state found

    T = 10.0
    cooling = 0.999
    min_T = 0.1
    steps = 0
    max_steps = 1000000 # Cap, but time is the real limit

    rng_sa = np.random.default_rng(123)

    current_invalid = compute_invalid_count()
    current_s_size = int(np.sum(S))
    current_energy = get_energy(current_invalid, current_s_size)

    # Re-implementing with unsafe updates for SA

    def unsafe_add_point(z):
        """Adds z to S. Returns list of changes to revert."""
        S[z] = True
        changes = []
        for li in point_lines_list[z]:
            old_count = line_S_count[li]
            new_count = old_count + 1
            line_S_count[li] = new_count
            for w in line_points_list[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                        changes.append((w, 1))
                else:
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
                        changes.append((w, 1))
        z_good = 0
        for li in point_lines_list[z]:
            if line_S_count[li] == 1:
                z_good += 1
        old_z_good = good_line_count[z]
        good_line_count[z] = z_good
        changes.append((z, z_good - old_z_good))
        return changes

    def unsafe_remove_point(z):
        """Removes z from S. Returns list of changes to revert."""
        S[z] = False
        changes = []
        for li in point_lines_list[z]:
            old_count = line_S_count[li]
            new_count = old_count - 1
            line_S_count[li] = new_count
            for w in line_points_list[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1 and new_count != 1:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 1 and new_count == 1:
                        good_line_count[w] += 1
                        changes.append((w, 1))
                else:
                    if old_count == 0 and new_count != 0:
                        good_line_count[w] -= 1
                        changes.append((w, -1))
                    elif old_count != 0 and new_count == 0:
                        good_line_count[w] += 1
                        changes.append((w, 1))
        z_good = 0
        for li in point_lines_list[z]:
            if line_S_count[li] == 0:
                z_good += 1
        old_z_good = good_line_count[z]
        good_line_count[z] = z_good
        changes.append((z, z_good - old_z_good))
        return changes

    def revert_flip(z, was_adding, changes):
        S[z] = not S[z]
        if was_adding:
            for li in point_lines_list[z]:
                line_S_count[li] -= 1
        else:
            for li in point_lines_list[z]:
                line_S_count[li] += 1
        for w, delta in changes:
            good_line_count[w] -= delta

    # Restart SA Loop
    T = 5.0
    cooling = 0.9995
    min_T = 0.01

    current_invalid = compute_invalid_count()
    current_s_size = int(np.sum(S))
    current_energy = get_energy(current_invalid, current_s_size)

    while _time_left() > 5.0:
        if T < min_T:
            T = min_T

        z = int(rng_sa.integers(0, q2))
        was_adding = not S[z]

        if was_adding:
            changes = unsafe_add_point(z)
        else:
            changes = unsafe_remove_point(z)

        new_invalid = compute_invalid_count()
        new_s_size = int(np.sum(S))
        new_energy = get_energy(new_invalid, new_s_size)

        delta_E = new_energy - current_energy

        if delta_E < 0 or rng_sa.random() < math.exp(-delta_E / T):
            # Accept
            current_invalid = new_invalid
            current_s_size = new_s_size
            current_energy = new_energy

            if current_invalid == 0 and current_s_size > best_s_size:
                best_s_size = current_s_size
                best_S = S.copy()
                best_line_S_count = line_S_count.copy()
                best_good_line_count = good_line_count.copy()
        else:
            # Revert
            revert_flip(z, was_adding, changes)

        T *= cooling

    # Final Greedy Polish
    S[:] = best_S
    line_S_count[:] = best_line_S_count
    good_line_count[:] = best_good_line_count

    for _pass in range(50):
        if _time_left() <= 0:
            break
        improved = False
        for z in range(q2):
            if _time_left() <= 0:
                break
            if not S[z]:
                if add_point(z):
                    improved = True
        if not improved:
            break

    # Build output
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x, y = pt_xy(z)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
