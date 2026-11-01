import math
import time
import numpy as np

def construct():
    START = time.time()
    DEADLINE = START + 130  # Safety margin before 150s limit

    def elapsed():
        return time.time() - START

    def time_ok(margin=5.0):
        return elapsed() < DEADLINE - margin

    def compute_score(u):
        """Compute the score for set u (must contain 0)."""
        u_arr = np.array(u, dtype=np.int64)
        n = len(u_arr)
        if n == 0:
            return -1.0
        if u_arr[0] != 0:  # u should be sorted with 0 first
            return -1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0

        # Sumset and diffset sizes
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))

        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def is_golomb(u):
        """Check if u is a Golomb ruler (all pairwise differences distinct)."""
        n = len(u)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known Golomb rulers (optimal or near-optimal)
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 45],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 61],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 61, 68],
    ]

    best_set = [0, 1, 3]
    best_score = compute_score(best_set)

    # Evaluate all known Golomb rulers
    for u in golomb_rulers:
        if not time_ok(10.0):
            break
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)

    # Local search with multiple move types
    def local_search(start_u, time_budget=20.0):
        """Hill-climbing local search with add, remove, shift, and swap moves."""
        start_t = time.time()
        current_u = sorted(set(start_u))
        if 0 not in current_u:
            current_u = [0] + current_u
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score

        while time.time() - start_t < time_budget and time_ok(15.0):
            improved = False
            max_u = max(current_u)
            n = len(current_u)

            # Move type 1: Add a new element
            search_hi = max_u + n * n + 20
            for new_val in range(max_u + 1, search_hi + 1):
                candidate = sorted(current_u + [new_val])
                s = compute_score(candidate)
                if s > current_score + 1e-12:
                    current_u = candidate
                    current_score = s
                    if s > best_s:
                        best_s = s
                        best_u = list(candidate)
                    improved = True
                    break

            # Move type 2: Remove an element (if len > 2)
            if not improved and n > 2:
                for i in range(1, n):  # Skip index 0 (which is 0)
                    candidate = current_u[:i] + current_u[i+1:]
                    s = compute_score(candidate)
                    if s > current_score + 1e-12:
                        current_u = candidate
                        current_score = s
                        if s > best_s:
                            best_s = s
                            best_u = list(candidate)
                        improved = True
                        break

            # Move type 3: Shift a non-zero element by small delta
            if not improved:
                for i in range(1, n):
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_val = current_u[i] + delta
                        if new_val < 1:
                            continue
                        candidate = list(current_u)
                        candidate[i] = new_val
                        candidate = sorted(set(candidate))
                        if 0 not in candidate or len(candidate) != n:
                            continue
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
                    if improved:
                        break

            # Move type 4: Swap two non-zero elements
            if not improved and n >= 3:
                for i in range(1, n):
                    for j in range(i + 1, n):
                        candidate = list(current_u)
                        candidate[i], candidate[j] = candidate[j], candidate[i]
                        candidate = sorted(set(candidate))
                        if 0 not in candidate or len(candidate) != n:
                            continue
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
                    if improved:
                        break

            if not improved:
                break

        return best_u, best_s

    # Run local search from best Golomb ruler
    if time_ok(15.0):
        ls_u, ls_s = local_search(best_set, time_budget=25.0)
        if ls_s > best_score:
            best_score = ls_s
            best_set = ls_u

    # Run local search from multiple starting points
    if time_ok(20.0):
        for u in golomb_rulers[:6]:
            if not time_ok(20.0):
                break
            ls_u, ls_s = local_search(u, time_budget=12.0)
            if ls_s > best_score:
                best_score = ls_s
                best_set = ls_u

    # Greedy construction: build up from small sets
    if time_ok(25.0):
        def greedy_build(n_target, time_budget=10.0):
            """Build a set greedily, trying to maximize score at each step."""
            start_t = time.time()
            u = [0, 1]
            while len(u) < n_target and time.time() - start_t < time_budget and time_ok(30.0):
                max_u = max(u)
                n = len(u)
                best_new = None
                best_new_score = compute_score(u)
                search_hi = max_u + n * n + 15
                for new_val in range(max_u + 1, search_hi + 1):
                    candidate = sorted(u + [new_val])
                    s = compute_score(candidate)
                    if s > best_new_score:
                        best_new_score = s
                        best_new = candidate
                    if time.time() - start_t > time_budget:
                        break
                if best_new is not None:
                    u = best_new
                else:
                    break
            return u, compute_score(u)

        for n_target in [5, 7, 9, 11, 13]:
            if not time_ok(30.0):
                break
            g_u, g_s = greedy_build(n_target, time_budget=8.0)
            if g_s > best_score:
                best_score = g_s
                best_set = g_u

    # Final local search polish on the best set found
    if time_ok(15.0):
        ls_u, ls_s = local_search(best_set, time_budget=20.0)
        if ls_s > best_score:
            best_score = ls_s
            best_set = ls_u

    # Ensure valid output
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set

    return best_set
