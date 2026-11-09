import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)

    def check_sidon(elements):
        """Fast Sidon check using difference tracking"""
        n = len(elements)
        if n < 2:
            return True
        diffs = set()
        for i in range(n):
            ei = elements[i]
            for j in range(i + 1, n):
                d = elements[j] - ei
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diff_mask(elements):
        """Get bitmask of all positive differences up to 30"""
        mask = 0
        n = len(elements)
        for i in range(n):
            ei = elements[i]
            for j in range(i + 1, n):
                d = elements[j] - ei
                if d <= 30:
                    mask |= (1 << d)
        return mask

    def is_valid(elements):
        """Full validity check"""
        if len(elements) < 2 or len(elements) > 32:
            return False
        if 0 not in elements:
            return False
        if max(elements) > 10**6:
            return False
        if not check_sidon(elements):
            return False
        if not (get_diff_mask(elements) & target_mask) == target_mask:
            return False
        return True

    def fast_backtrack(D, time_limit=None):
        """
        Backtracking with bitset difference tracking.
        We build the set in increasing order, tracking all differences used.
        """
        if time_limit is None:
            time_limit = deadline - 5

        best_result = [None]

        def dfs(elements, used_diff_mask, covered_mask, last_elem):
            if time.time() > time_limit:
                return False

            if (covered_mask & target_mask) == target_mask:
                best_result[0] = elements[:]
                return True

            k = len(elements)
            if k >= 32:
                return False

            missing_mask = target_mask & ~covered_mask
            if missing_mask == 0:
                best_result[0] = elements[:]
                return True

            # Pruning: remaining capacity
            remaining_slots = 32 - k
            missing_count = bin(missing_mask).count('1')
            if missing_count > remaining_slots * k:
                return False

            # Collect candidates with their new coverage count
            candidates = []
            for x in range(last_elem + 1, D + 1):
                valid = True
                new_covered = 0
                for y in elements:
                    d = x - y
                    bit = 1 << d
                    if (used_diff_mask & bit) != 0:
                        valid = False
                        break
                    if d <= 30:
                        if (target_mask & bit) != 0:
                            new_covered += 1

                if not valid:
                    continue
                candidates.append((new_covered, x))

            # Sort: more coverage first, then smaller value
            candidates.sort(key=lambda c: (-c[0], c[1]))

            # Limit candidates to keep search tractable
            max_cands = min(len(candidates), 80)
            for _, x in candidates[:max_cands]:
                new_used = used_diff_mask
                new_covered = covered_mask
                for y in elements:
                    d = x - y
                    new_used |= (1 << d)
                    if d <= 30:
                        new_covered |= (1 << d)

                if dfs(elements + [x], new_used, new_covered, x):
                    return True

            return False

        dfs([0], 0, 0, 0)
        return best_result[0]

    def local_search_minimize(S, time_limit=None):
        """
        Local search to minimize max element.
        Tries to replace each element (except 0) with a smaller value.
        """
        if time_limit is None:
            time_limit = deadline - 10

        if S is None:
            return None

        best_S = sorted(S)
        best_max = max(best_S)
        no_improve = 0
        max_no_improve = 500

        while time.time() < time_limit and no_improve < max_no_improve:
            improved = False
            S_sorted = sorted(best_S)
            n = len(S_sorted)

            for i in range(n - 1, 0, -1):
                if time.time() > time_limit:
                    break

                s_i = S_sorted[i]
                S_rest = S_sorted[:i] + S_sorted[i + 1:]

                lower = S_sorted[i - 1] + 1
                upper = s_i - 1

                if upper <= lower:
                    continue

                # Try all values in descending order (prefer largest that still reduces max)
                for new_val in range(upper, lower - 1, -1):
                    S_new = S_rest + [new_val]
                    S_new.sort()

                    if not check_sidon(S_new):
                        continue
                    if not (get_diff_mask(S_new) & target_mask) == target_mask:
                        continue

                    new_max = max(S_new)
                    if new_max < best_max:
                        best_S = S_new
                        best_max = new_max
                        improved = True
                        break

                if improved:
                    break

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        return best_S

    # Main: try increasing D values
    best_result = None
    best_score = float('inf')

    # Start from a reasonable lower bound
    for D in range(35, 250):
        if time.time() > deadline - 40:
            break

        result = fast_backtrack(D)
        if result is not None:
            if is_valid(result):
                score = max(result)
                if score < best_score:
                    best_score = score
                    best_result = result[:]
                    optimized = local_search_minimize(best_result)
                    if optimized is not None and is_valid(optimized):
                        opt_score = max(optimized)
                        if opt_score < best_score:
                            best_score = opt_score
                            best_result = optimized[:]
                    break

    # If backtracking didn't find anything, use greedy fallback
    if best_result is None:
        S = [0]
        used_diffs = set()
        covered = set()

        for step in range(32):
            if time.time() > deadline - 20:
                break

            missing = target_diffs - covered
            if not missing:
                break

            best_x = None
            best_coverage = -1

            for x in range(1, 1500):
                if x in S:
                    continue
                valid = True
                new_cov = 0
                for y in S:
                    d = abs(x - y)
                    if d in used_diffs:
                        valid = False
                        break
                    if d in missing:
                        new_cov += 1

                if not valid:
                    continue

                if new_cov > best_coverage:
                    best_coverage = new_cov
                    best_x = x

            if best_x is None:
                found = False
                base = max(S) + 1
                for offset in range(1, 300):
                    x = base + offset
                    valid = True
                    for y in S:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        S.append(x)
                        S.sort()
                        used_diffs = set()
                        covered = set()
                        for i in range(len(S)):
                            for j in range(i + 1, len(S)):
                                d = S[j] - S[i]
                                used_diffs.add(d)
                                if d in target_diffs:
                                    covered.add(d)
                        found = True
                        break
                if not found:
                    break
            else:
                S.append(best_x)
                S.sort()
                used_diffs = set()
                covered = set()
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        d = S[j] - S[i]
                        used_diffs.add(d)
                        if d in target_diffs:
                            covered.add(d)

        if is_valid(S):
            best_result = S
            best_score = max(S)

    # Final polish
    if best_result is not None:
        final = local_search_minimize(best_result)
        if final is not None and is_valid(final):
            if max(final) < max(best_result):
                best_result = final

    # Ensure we return a valid set
    if best_result is None or not is_valid(best_result):
        S = [0]
        used_diffs = set()
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            for x in range(1, 5000):
                if x in S:
                    continue
                trial = sorted(S + [x])
                new_diffs = set()
                valid = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        diff = trial[j] - trial[i]
                        if diff in new_diffs:
                            valid = False
                            break
                        new_diffs.add(diff)
                    if not valid:
                        break
                if valid:
                    S = trial
                    used_diffs = new_diffs
                    for dd in new_diffs:
                        if dd in target_diffs:
                            covered.add(dd)
                    found = True
                    break
            if not found:
                y = 2 * max(S) + 3 * d + 10
                while True:
                    trial = sorted(S + [y])
                    new_diffs = set()
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            diff = trial[j] - trial[i]
                            if diff in new_diffs:
                                valid = False
                                break
                            new_diffs.add(diff)
                        if not valid:
                            break
                    if valid:
                        S = trial
                        used_diffs = new_diffs
                        for dd in new_diffs:
                            if dd in target_diffs:
                                covered.add(dd)
                        break
                    y += 1

        best_result = S

    return best_result
