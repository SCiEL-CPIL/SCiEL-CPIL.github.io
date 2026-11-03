import time
from math import log, floor

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # Search over geometric base b and length m
    # b is the base of the geometric progression
    # m is the number of terms (excluding 0)

    # First, do a coarse grid search to find promising regions
    # b ranges from ~1.1 to ~10, m from 2 to 30

    b_candidates = []
    for b_int in range(12, 101):  # b from 1.2 to 10.0 in steps of 0.1
        b = b_int / 10.0
        b_candidates.append(b)

    # Evaluate on a coarse grid
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 30:
            break
        for m in range(2, 31):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # Now do ternary search on b for several fixed m values near the best
    if time.time() < deadline - 10:
        for m in range(2, 31):
            if time.time() > deadline - 10:
                break
            # Ternary search on b in [1.01, 15.0]
            lo = 1.01
            hi = 15.0
            for _ in range(40):  # 40 iterations of ternary search
                if time.time() > deadline - 5:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                U1 = geometric_set(m1, m)
                U2 = geometric_set(m2, m)
                s1 = score_set(U1)
                s2 = score_set(U2)
                if s1 < s2:
                    lo = m1
                else:
                    hi = m2
            # Evaluate at the best point
            b_opt = (lo + hi) / 2
            U = geometric_set(b_opt, m)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    # Also try some specific well-known constructions
    # Powers of 2
    for n in range(3, 20):
        if time.time() > deadline - 5:
            break
        U = [0] + [2 ** k for k in range(n - 1)]
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U

    # Powers of 3
    for n in range(3, 15):
        if time.time() > deadline - 5:
            break
        U = [0] + [3 ** k for k in range(n - 1)]
        s = score_set(U)
        if s > best_score:
            best_score = s
            best_set = U

    # Local refinement: try small perturbations of the best set
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 3 and iteration < 200:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 3:
                    break
                old_val = current[i]
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            # Try adding an element
            if not improved and n < 20 and time.time() < deadline - 3:
                max_val = max(current)
                for new_val in range(max_val + 1, max_val + 20):
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 3:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set
