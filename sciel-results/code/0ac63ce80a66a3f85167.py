import time
import numpy as np
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    # Only optimized for the specific task: q=3, n=8, r=1
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: greedy for generality
        total = q ** n
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q

        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = powers[i]
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)

        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []

        while True:
            if time.time() > deadline - 5:
                break
            max_idx = int(np.argmax(counts))
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(total):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break

        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result

    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    ball_size = 1 + 8 * 2  # 17

    # Precompute balls as lists of integers
    balls = []
    for x in range(total):
        ball = [x]
        digits = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        for i in range(8):
            old_d = digits[i]
            mult = 3 ** i
            for new_d in range(3):
                if new_d != old_d:
                    y = x + (new_d - old_d) * mult
                    ball.append(y)
        balls.append(ball)

    # Build ball_membership matrix: ball_membership[y, x] = 1 iff y in balls[x]
    # This is symmetric (distance is symmetric), so ball_membership == its transpose
    ball_membership = np.zeros((total, total), dtype=np.uint8)
    for x in range(total):
        for y in balls[x]:
            ball_membership[y, x] = 1

    # Greedy initialization
    def greedy_init(noise_scale=0.0):
        counts = np.full(total, ball_size, dtype=np.float64)
        if noise_scale > 0:
            counts += np.random.uniform(0, noise_scale, size=total)
        covered = np.zeros(total, dtype=bool)
        selected = []
        while True:
            if time.time() > deadline - 30:
                break
            max_idx = int(np.argmax(counts))
            if counts[max_idx] <= 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    # Update counts for all x that cover y
                    # ball_membership[y, :] gives all x such that y in balls[x]
                    mask = ball_membership[y, :] == 1
                    counts[mask] -= 1
            if np.all(covered):
                break
        return selected

    # Vectorized 2-opt swap local search
    def swap_local_search(selected):
        current_set = set(selected)
        cover_count = np.zeros(total, dtype=np.int32)
        for cw in current_set:
            cover_count[balls[cw]] += 1

        best_size = len(current_set)
        best_solution = list(current_set)

        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)

            for cw in candidates:
                if cw not in current_set:
                    continue
                # Check if cw is redundant (all its points covered by others)
                if np.min(cover_count[balls[cw]]) <= 1:
                    continue

                # Remove cw
                cover_count[balls[cw]] -= 1
                current_set.remove(cw)
                best_size = len(current_set)
                best_solution = list(current_set)
                improved = True

                # Now try to find a swap: remove another cw2, add a new cw3
                # that covers all points that were only covered by cw2
                # Find points that are now at cover_count == 1 (critical points)
                # Actually, we already removed cw. Now look for another removable cw2.
                # For efficiency, just do another pass of redundancy check.

            # After removing all redundancies, try 2-opt swaps
            if time.time() > deadline - 15:
                break

            # 2-opt: for each pair (cw_a, cw_b) in current_set, check if we can
            # remove one and add a new one that covers the gap
            # More efficient: for each cw_a, find points only covered by cw_a
            # (cover_count == 1 in balls[cw_a]). If none, cw_a is redundant (already handled).
            # For 2-opt: remove cw_a and cw_b, add cw_c such that
            # balls[cw_c] covers all points in (balls[cw_a] ∪ balls[cw_b]) \ (covered by others)
            # This is complex; instead do: remove cw_a, then greedily add best cw_c
            # to restore coverage, check if total size decreased.

            # Simpler 2-opt: for each cw_a, compute critical = points in balls[cw_a]
            # with cover_count == 1. If |critical| > 0, find cw_c that covers all critical.
            # If such cw_c exists and cw_c not in current_set, swap cw_a -> cw_c.
            # But this doesn't reduce size. We need: remove cw_a, and if all critical
            # points are covered by adding one cw_c, but that's size-neutral.
            #
            # True 2-opt for size reduction: remove cw_a AND cw_b, add cw_c.
            # Critical points of cw_a: A = {y in balls[cw_a] : cover_count[y] == 1}
            # Critical points of cw_b: B = {y in balls[cw_b] : cover_count[y] == 1}
            # (computed after removing both)
            # Need cw_c such that balls[cw_c] ⊇ A ∪ B.
            #
            # For efficiency, only try this when |A| and |B| are small.

            # Do a pass of 2-opt swaps
            if time.time() > deadline - 15:
                break

            cw_list = list(current_set)
            random.shuffle(cw_list)

            for i, cw_a in enumerate(cw_list):
                if time.time() > deadline - 15:
                    break
                if cw_a not in current_set:
                    continue

                # Critical points of cw_a (only covered by cw_a)
                crit_a_mask = np.zeros(total, dtype=bool)
                crit_a_mask[balls[cw_a]] = True
                critical_a = crit_a_mask & (cover_count == 1)
                n_crit_a = int(np.sum(critical_a))

                if n_crit_a == 0:
                    # cw_a is redundant, remove it
                    cover_count[balls[cw_a]] -= 1
                    current_set.remove(cw_a)
                    best_size = len(current_set)
                    best_solution = list(current_set)
                    improved = True
                    continue

                if n_crit_a > 4:
                    continue  # Too many critical points, unlikely to find a good swap

                # Try to find cw_b such that removing both cw_a and cw_b
                # allows adding one cw_c
                for j, cw_b in enumerate(cw_list[i+1:]):
                    if time.time() > deadline - 15:
                        break
                    if cw_b not in current_set:
                        continue

                    # Temporarily remove both
                    cover_count[balls[cw_a]] -= 1
                    cover_count[balls[cw_b]] -= 1
                    current_set.remove(cw_a)
                    current_set.remove(cw_b)

                    # Find all points now uncovered (cover_count == 0)
                    uncovered = (cover_count == 0)
                    n_uncovered = int(np.sum(uncovered))

                    if n_uncovered == 0:
                        # We just removed two and need to add none -> size reduced by 2!
                        best_size = len(current_set)
                        best_solution = list(current_set)
                        improved = True
                        # Don't restore, keep the removal
                        continue

                    # Find cw_c that covers all uncovered points
                    # uncovered_mask: boolean array of uncovered points
                    # We need balls[cw_c] ⊇ uncovered set
                    # ball_membership[uncovered, :] gives, for each uncovered point y,
                    # which cw's cover it. We need the intersection to be non-empty.
                    if n_uncovered <= 10:
                        # Get candidate cw's that cover all uncovered points
                        # For each uncovered point y, ball_membership[y, :] == 1 gives candidates
                        # Intersection: all must have 1
                        cand_mask = np.ones(total, dtype=bool)
                        for y in np.where(uncovered)[0]:
                            cand_mask &= (ball_membership[y, :] == 1)
                        candidates_c = np.where(cand_mask)[0]
                        candidates_c = [c for c in candidates_c if c not in current_set]

                        if candidates_c:
                            # Pick one (e.g., first)
                            cw_c = candidates_c[0]
                            current_set.add(cw_c)
                            cover_count[balls[cw_c]] += 1
                            # Size: removed 2, added 1 -> net -1
                            if len(current_set) < best_size:
                                best_size = len(current_set)
                                best_solution = list(current_set)
                                improved = True
                            continue  # Keep the swap

                    # Restore if no good swap found
                    cover_count[balls[cw_a]] += 1
                    cover_count[balls[cw_b]] += 1
                    current_set.add(cw_a)
                    current_set.add(cw_b)

        return best_solution, best_size

    # Main: greedy init + swap local search
    selected = greedy_init()
    best_solution, best_size = swap_local_search(selected)

    # Try a few more greedy restarts with noise if time allows
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        selected2 = greedy_init(noise_scale=0.5)
        sol2, size2 = swap_local_search(selected2)
        if size2 < best_size:
            best_size = size2
            best_solution = sol2

    # Final redundancy removal pass
    current_set = set(best_solution)
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in current_set:
        cover_count[balls[cw]] += 1

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for cw in list(current_set):
            if np.min(cover_count[balls[cw]]) <= 1:
                continue
            cover_count[balls[cw]] -= 1
            current_set.remove(cw)
            improved = True

    best_solution = list(current_set)
    best_size = len(best_solution)

    # Convert to list of digit lists
    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)

    return result
