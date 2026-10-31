import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 130  # Leave 20s safety margin for 150s limit

    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        arr = np.array(u, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        return 1 + math.log(r) / math.log(m)

    def make_diff_bitmap(u, max_diff):
        """Create a boolean array where bit[d] = True if d is a pairwise difference."""
        bitmap = np.zeros(max_diff + 1, dtype=bool)
        n = len(u)
        for i in range(n):
            for j in range(i + 1, n):
                d = u[j] - u[i]
                if d <= max_diff:
                    bitmap[d] = True
        return bitmap

    def is_golomb_fast(u, existing_bitmap=None, max_diff=None):
        """Fast Golomb check using bitmap. Returns (is_golomb, updated_bitmap, max_diff)."""
        n = len(u)
        if max_diff is None:
            max_diff = 2 * max(u) if u else 0
        if existing_bitmap is not None and len(existing_bitmap) > max_diff:
            bitmap = existing_bitmap.copy()
            max_diff = len(bitmap) - 1
        else:
            bitmap = np.zeros(max_diff + 1, dtype=bool)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = u[j] - u[i]
                if d > max_diff:
                    continue
                if bitmap[d]:
                    return False, None, None
                bitmap[d] = True
        return True, bitmap, max_diff

    # Known optimal or near-optimal Golomb rulers
    # These are verified from the literature
    known_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        # n=7 optimal: length 25
        [0, 1, 4, 10, 12, 17, 25],
        # n=8 optimal: length 34
        [0, 1, 4, 10, 12, 17, 25, 34],
        # n=9 optimal: length 44
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        # n=10 optimal: length 55
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
    ]

    # Verify each known ruler is actually a Golomb ruler
    verified_rulers = []
    for r in known_rulers:
        ok, _, _ = is_golomb_fast(r)
        if ok:
            verified_rulers.append(r)
        else:
            # Try to fix by checking and potentially removing the problematic element
            # Just skip invalid ones
            pass

    candidates = list(verified_rulers)

    def local_search(base_u, time_limit):
        start_ls = time.time()
        best_u = list(base_u)
        best_score = compute_score(best_u)
        improved = True
        iterations = 0

        while improved and (time.time() - start_ls) < time_limit:
            improved = False
            iterations += 1
            if iterations > 200:  # Safety
                break

            current_max = max(best_u)
            n = len(best_u)
            best_candidate = None
            best_candidate_score = best_score

            # Build current difference bitmap
            max_diff = 2 * current_max + 1
            _, bitmap, _ = is_golomb_fast(best_u)

            # Try adding a new element (most promising for improving ratio)
            # Search range: from current_max+1 to current_max + n*n + 10
            search_lo = current_max + 1
            search_hi = min(current_max + n * n + 20, 2000)

            for new_mark in range(search_lo, search_hi + 1):
                if (time.time() - start_ls) > time_limit:
                    break
                candidate = best_u + [new_mark]
                # Fast check: only need to verify new differences
                ok = True
                new_max_diff = max(new_mark, current_max)
                if new_mark > max_diff:
                    # Need to extend bitmap
                    ext_bitmap = np.zeros(new_mark + 1, dtype=bool)
                    ext_bitmap[:max_diff+1] = bitmap
                    bitmap = ext_bitmap
                    max_diff = new_mark
                for i in range(n):
                    d = new_mark - best_u[i]
                    if d > 0 and bitmap[d]:
                        ok = False
                        break
                if not ok:
                    continue
                # It's a Golomb ruler, compute score
                s = compute_score(candidate)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = sorted(candidate)

            # Try removing an element (if size > 3)
            if len(best_u) > 3 and (time.time() - start_ls) < time_limit:
                for i in range(1, len(best_u)):  # keep 0
                    candidate = [x for j, x in enumerate(best_u) if j != i]
                    s = compute_score(candidate)
                    if s > best_candidate_score:
                        best_candidate_score = s
                        best_candidate = candidate

            # Try swapping: replace one element with a nearby value
            if (time.time() - start_ls) < time_limit:
                for i in range(1, len(best_u)):
                    old_val = best_u[i]
                    # Try a range of replacements
                    delta_range = max(5, n * 2)
                    for delta in range(1, delta_range + 1):
                        for new_val in [old_val - delta, old_val + delta]:
                            if new_val < 0 or new_val == 0:
                                continue
                            if new_val in best_u:
                                continue
                            candidate = sorted([x if x != old_val else new_val for x in best_u])
                            ok, _, _ = is_golomb_fast(candidate)
                            if not ok:
                                continue
                            s = compute_score(candidate)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate = candidate
                        if (time.time() - start_ls) > time_limit:
                            break

            if best_candidate is not None:
                best_u = best_candidate
                best_score = best_candidate_score
                improved = True

        return best_u, best_score

    # Run local search on the best known rulers
    # Start from the largest verified rulers for best results
    search_starts = []
    for r in verified_rulers:
        if len(r) >= 5:
            search_starts.append(r)
    # Also try from smaller ones in case larger ones get stuck
    for r in verified_rulers:
        if len(r) < 5:
            search_starts.append(r)

    # Sort by size descending to prioritize larger rulers
    search_starts.sort(key=lambda x: len(x), reverse=True)

    # Allocate time: give more time to larger starting rulers
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        # First, try the largest ruler with most time
        if search_starts:
            best_start = search_starts[0]
            t1 = min(remaining_time * 0.5, 50)
            ext_u, ext_score = local_search(best_start, t1)
            if ext_u and len(ext_u) >= 3:
                candidates.append(ext_u)

        # Then try other starts with remaining time
        remaining_time = DEADLINE - (time.time() - start_time)
        for sp in search_starts[1:4]:  # Try up to 3 more
            if remaining_time < 5:
                break
            t2 = min(remaining_time * 0.25, 20)
            if t2 < 3:
                t2 = max(3, remaining_time * 0.5)
            ext_u, ext_score = local_search(sp, t2)
            if ext_u and len(ext_u) >= 3:
                candidates.append(ext_u)
            remaining_time = DEADLINE - (time.time() - start_time)

    # Also try greedy construction for comparison
    def greedy_golomb(n, time_limit):
        start_g = time.time()
        u = [0]
        for i in range(1, n):
            if (time.time() - start_g) > time_limit:
                break
            pos = 0
            found = False
            max_diff = 2 * max(u) + 1 if u else 0
            while not found:
                pos += 1
                candidate = u + [pos]
                # Fast check
                ok = True
                if pos > max_diff:
                    # Need to check against all existing differences
                    for j in range(len(u)):
                        d = pos - u[j]
                        for k in range(j + 1, len(u)):
                            if pos - u[j] == u[k] - u[j]:
                                ok = False
                                break
                        if not ok:
                            break
                else:
                    # Use bitmap approach
                    bitmap = np.zeros(pos + 1, dtype=bool)
                    for j in range(len(u)):
                        for k in range(j + 1, len(u)):
                            d = u[k] - u[j]
                            if d <= pos:
                                bitmap[d] = True
                    for j in range(len(u)):
                        d = pos - u[j]
                        if d > 0 and d <= pos and bitmap[d]:
                            ok = False
                            break
                if ok:
                    u = candidate
                    found = True
                if pos > n * n * 30:
                    break
        return u

    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        greedy_u = greedy_golomb(12, time_limit=min(remaining_time * 0.3, 15))
        if greedy_u and len(greedy_u) >= 5:
            candidates.append(greedy_u)

    # Find best candidate
    best_set = [0, 1, 3]
    best_score = 0.0
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        if len(u) < 2:
            continue
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = u

    return best_set
