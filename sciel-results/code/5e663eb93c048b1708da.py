import time
import random
import sys
import math

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    def compute_sums(elements):
        """Compute all non-empty subset sums. Returns set or None if duplicates."""
        sums = set()
        for x in elements:
            if x in sums:
                return None
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    return None
                new_sums.add(ns)
            if x in new_sums:
                return None
            new_sums.add(x)
            sums = sums.union(new_sums)
        return sums

    def check_divisibility(sums):
        """Check no two distinct sums divide each other."""
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            si = s_list[i]
            for j in range(i + 1, n):
                sj = s_list[j]
                if sj % si == 0:
                    return False
        return True

    def is_valid(elements):
        if not elements:
            return True
        if len(elements) != len(set(elements)):
            return False
        if any(x < 1 or x > 256 for x in elements):
            return False
        sums = compute_sums(elements)
        if sums is None:
            return False
        return check_divisibility(sums)

    def can_add(x, current_elements, current_sums):
        """Check if x can be added to current_elements."""
        if x in current_elements:
            return None, False
        if x in current_sums:
            return None, False

        new_sums = {x}
        for s in current_sums:
            ns = s + x
            if ns in current_sums or ns in new_sums:
                return None, False
            new_sums.add(ns)

        all_sums = current_sums.union(new_sums)

        # Check divisibility among new_sums
        ns_list = sorted(new_sums)
        for i in range(len(ns_list)):
            si = ns_list[i]
            for j in range(i + 1, len(ns_list)):
                sj = ns_list[j]
                if sj % si == 0:
                    return None, False

        # Check divisibility between new_sums and current_sums
        for sn in new_sums:
            for sc in current_sums:
                if sn == sc:
                    return None, False
                if sn > sc:
                    if sn % sc == 0:
                        return None, False
                else:
                    if sc % sn == 0:
                        return None, False

        return all_sums, True

    def greedy_from_candidates(candidates):
        """Greedy: add elements from candidates in given order."""
        cur = []
        cur_sums = set()
        for x in candidates:
            if time.time() > deadline:
                break
            new_sums, valid = can_add(x, cur, cur_sums)
            if valid:
                cur.append(x)
                cur_sums = new_sums
        return cur, cur_sums

    def extend_set(current_elements, current_sums, time_limit):
        """Try to extend the set by adding more elements."""
        best = list(current_elements)
        best_sums = set(current_sums) if current_sums else set()

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for x in range(256, 1, -1):
                if x not in best:
                    new_sums, valid = can_add(x, best, best_sums)
                    if valid:
                        best.append(x)
                        best_sums = new_sums
                        changed = True
                        break

        return best

    # Main search logic
    best_set = []
    best_size = 0

    # Phase 1: Try windows of large numbers
    if time.time() < deadline - 20:
        for window_start in range(256, 100, -5):
            if time.time() > deadline - 20:
                break
            window_size = random.randint(30, 60)
            candidates = list(range(min(256, window_start), max(1, window_start - window_size), -1))
            random.shuffle(candidates)
            cur, cur_sums = greedy_from_candidates(candidates)
            if len(cur) > best_size:
                best_size = len(cur)
                best_set = list(cur)

    # Phase 2: Extend the best set
    if best_set and time.time() < deadline - 10:
        extended = extend_set(best_set, compute_sums(best_set) if best_set else set(), time.time() + 15)
        if len(extended) > best_size:
            best_set = extended
            best_size = len(extended)

    # Phase 3: More random searches
    if time.time() < deadline - 5:
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            window_start = random.randint(120, 250)
            window_size = random.randint(20, 80)
            candidates = list(range(min(256, window_start), max(1, window_start - window_size), -1))
            random.shuffle(candidates)
            cur, cur_sums = greedy_from_candidates(candidates)
            if len(cur) > best_size:
                best_size = len(cur)
                best_set = list(cur)

            # Quick extend
            if time.time() < deadline - 5:
                extended = extend_set(cur, cur_sums, time.time() + 0.5)
                if len(extended) > best_size:
                    best_size = len(extended)
                    best_set = list(extended)

    # Final validation and cleanup
    if best_set:
        if not is_valid(best_set):
            best_set = [100, 101]

    # Ensure types
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique

    return best_set
