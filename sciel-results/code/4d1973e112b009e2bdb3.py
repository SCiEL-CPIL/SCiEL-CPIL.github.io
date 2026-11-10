import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s limit

    d = 11
    points_list = []

    # 1. All 220 vectors of type (+/-2, +/-2, 0, ...) in Z^11
    # These are 2 * (roots of D_11)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)

    # 2. All 128 vectors of type (+/-1, ..., +/-1, 0, 0, 0) in first 8 dims
    # These are 2 * (Type 2 roots of E_8)
    for signs in product([-1, 1], repeat=8):
        if time.time() > deadline:
            break
        if sum(1 for s in signs if s == -1) % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for idx, val in enumerate(signs):
                p[idx] = val
            points_list.append(p)

    points = np.array(points_list, dtype=np.int64)

    # Verify initial configuration
    norms = np.sum(points**2, axis=1)
    max_norm = np.max(norms)
    assert np.all(norms == 8), f"Not all norms are 8: {np.unique(norms)}"

    min_dist_sq = float('inf')
    n = len(points)
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                if min_dist_sq < max_norm:
                    break
        if min_dist_sq < max_norm:
            break

    assert min_dist_sq >= max_norm, f"Min dist^2 = {min_dist_sq} < {max_norm}"

    # Now try to add more points via local search
    # Candidate vectors must have norm^2 >= max_norm (which is 8)
    # and be at distance^2 >= max_norm from all existing points

    added = []

    # Helper to check if a point can be added
    def can_add(p, existing_points, added_points, max_norm):
        # Check norm
        norm_sq = np.sum(p**2)
        if norm_sq < max_norm:
            return False
        # Check distance to all existing points
        for existing in existing_points:
            diff = p - existing
            dist_sq = np.sum(diff**2)
            if dist_sq < max_norm:
                return False
        # Check distance to all added points
        for ap in added_points:
            diff = p - ap
            dist_sq = np.sum(diff**2)
            if dist_sq < max_norm:
                return False
        return True

    # Try all weight-3 vectors with values (2,2,2)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 2 * si
                    p[j] = 2 * sj
                    p[k] = 2 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-2 vectors with values (3,1)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 3 * si
                p[j] = 1 * sj
                if can_add(p, points, added, max_norm):
                    added.append(p)

    # Try weight-3 vectors with values (2,2,1)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 2 * si
                    p[j] = 2 * sj
                    p[k] = 1 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-2 vectors with values (3,2)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 3 * si
                p[j] = 2 * sj
                if can_add(p, points, added, max_norm):
                    added.append(p)

    # Try weight-3 vectors with values (3,1,1)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 3 * si
                    p[j] = 1 * sj
                    p[k] = 1 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-4 vectors with values (2,1,1,1)
    for i, j, k, l in combinations(range(11), 4):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = 2 * si
                        p[j] = 1 * sj
                        p[k] = 1 * sk
                        p[l] = 1 * sl
                        if can_add(p, points, added, max_norm):
                            added.append(p)

    # Try weight-2 vectors with values (4,0) - i.e., (4,0)
    for i in range(11):
        if time.time() > deadline:
            break
        for si in (1, -1):
            p = np.zeros(d, dtype=np.int64)
            p[i] = 4 * si
            if can_add(p, points, added, max_norm):
                added.append(p)

    # Try weight-2 vectors with values (4,1)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 4 * si
                p[j] = 1 * sj
                if can_add(p, points, added, max_norm):
                    added.append(p)

    # Try weight-2 vectors with values (4,2)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 4 * si
                p[j] = 2 * sj
                if can_add(p, points, added, max_norm):
                    added.append(p)

    # Try weight-3 vectors with values (4,1,1)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 4 * si
                    p[j] = 1 * sj
                    p[k] = 1 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-3 vectors with values (3,2,1)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 3 * si
                    p[j] = 2 * sj
                    p[k] = 1 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-4 vectors with values (2,2,1,1)
    for i, j, k, l in combinations(range(11), 4):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = 2 * si
                        p[j] = 2 * sj
                        p[k] = 1 * sk
                        p[l] = 1 * sl
                        if can_add(p, points, added, max_norm):
                            added.append(p)

    # Try weight-5 vectors with values (1,1,1,1,1)
    for i, j, k, l, m in combinations(range(11), 5):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        for sm in (1, -1):
                            p = np.zeros(d, dtype=np.int64)
                            p[i] = 1 * si
                            p[j] = 1 * sj
                            p[k] = 1 * sk
                            p[l] = 1 * sl
                            p[m] = 1 * sm
                            if can_add(p, points, added, max_norm):
                                added.append(p)

    # Try weight-2 vectors with values (5,0)
    for i in range(11):
        if time.time() > deadline:
            break
        for si in (1, -1):
            p = np.zeros(d, dtype=np.int64)
            p[i] = 5 * si
            if can_add(p, points, added, max_norm):
                added.append(p)

    # Try weight-2 vectors with values (5,1)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 5 * si
                p[j] = 1 * sj
                if can_add(p, points, added, max_norm):
                    added.append(p)

    # Try weight-2 vectors with values (5,2)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 5 * si
                p[j] = 2 * sj
                if can_add(p, points, added, max_norm):
                    added.append(p)

    # Try weight-3 vectors with values (5,1,1)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 5 * si
                    p[j] = 1 * sj
                    p[k] = 1 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-3 vectors with values (4,2,1)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 4 * si
                    p[j] = 2 * sj
                    p[k] = 1 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-4 vectors with values (3,2,1,1)
    for i, j, k, l in combinations(range(11), 4):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = 3 * si
                        p[j] = 2 * sj
                        p[k] = 1 * sk
                        p[l] = 1 * sl
                        if can_add(p, points, added, max_norm):
                            added.append(p)

    # Try weight-4 vectors with values (2,2,2,1)
    for i, j, k, l in combinations(range(11), 4):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = 2 * si
                        p[j] = 2 * sj
                        p[k] = 2 * sk
                        p[l] = 1 * sl
                        if can_add(p, points, added, max_norm):
                            added.append(p)

    # Try weight-5 vectors with values (2,1,1,1,1)
    for i, j, k, l, m in combinations(range(11), 5):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        for sm in (1, -1):
                            p = np.zeros(d, dtype=np.int64)
                            p[i] = 2 * si
                            p[j] = 1 * sj
                            p[k] = 1 * sk
                            p[l] = 1 * sl
                            p[m] = 1 * sm
                            if can_add(p, points, added, max_norm):
                                added.append(p)

    # Try weight-2 vectors with values (6,0)
    for i in range(11):
        if time.time() > deadline:
            break
        for si in (1, -1):
            p = np.zeros(d, dtype=np.int64)
            p[i] = 6 * si
            if can_add(p, points, added, max_norm):
                added.append(p)

    # Try weight-2 vectors with values (6,1)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 6 * si
                p[j] = 1 * sj
                if can_add(p, points, added, max_norm):
                    added.append(p)

    # Try weight-2 vectors with values (6,2)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 6 * si
                p[j] = 2 * sj
                if can_add(p, points, added, max_norm):
                    added.append(p)

    # Try weight-3 vectors with values (6,1,1)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 6 * si
                    p[j] = 1 * sj
                    p[k] = 1 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-3 vectors with values (5,2,1)
    for i, j, k in combinations(range(11), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 5 * si
                    p[j] = 2 * sj
                    p[k] = 1 * sk
                    if can_add(p, points, added, max_norm):
                        added.append(p)

    # Try weight-4 vectors with values (4,2,1,1)
    for i, j, k, l in combinations(range(11), 4):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = 4 * si
                        p[j] = 2 * sj
                        p[k] = 1 * sk
                        p[l] = 1 * sl
                        if can_add(p, points, added, max_norm):
                            added.append(p)

    # Try weight-4 vectors with values (3,3,1,1)
    for i, j, k, l in combinations(range(11), 4):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = 3 * si
                        p[j] = 3 * sj
                        p[k] = 1 * sk
                        p[l] = 1 * sl
                        if can_add(p, points, added, max_norm):
                            added.append(p)

    # Try weight-5 vectors with values (3,1,1,1,1)
    for i, j, k, l, m in combinations(range(11), 5):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        for sm in (1, -1):
                            p = np.zeros(d, dtype=np.int64)
                            p[i] = 3 * si
                            p[j] = 1 * sj
                            p[k] = 1 * sk
                            p[l] = 1 * sl
                            p[m] = 1 * sm
                            if can_add(p, points, added, max_norm):
                                added.append(p)

    # Combine original and added points
    if added:
        all_points = np.vstack([points, np.array(added, dtype=np.int64)])
    else:
        all_points = points

    # Final verification
    final_norms = np.sum(all_points**2, axis=1)
    final_max_norm = np.max(final_norms)
    final_n = len(all_points)

    final_min_dist_sq = float('inf')
    for i in range(final_n):
        if time.time() > deadline:
            break
        for j in range(i + 1, final_n):
            diff = all_points[i] - all_points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < final_min_dist_sq:
                final_min_dist_sq = dist_sq
                if final_min_dist_sq < final_max_norm:
                    break
        if final_min_dist_sq < final_max_norm:
            break

    # If the added points broke the constraint, fall back to original
    if final_min_dist_sq < final_max_norm:
        all_points = points

    # Round and cast
    result = np.around(all_points).astype(np.int64)

    return result
