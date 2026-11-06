import time
import random
import bisect

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    def is_valid_addition(s_set, s_sorted, candidate):
        """Check if adding candidate to s_sorted (sorted list) keeps validity."""
        pos = bisect.bisect_left(s_sorted, candidate)
        n = len(s_sorted)

        # candidate as 'a': for each b > candidate, no c >= b in set with (b+c) % candidate == 0
        for i in range(pos, n):
            b = s_sorted[i]
            target = (-b) % candidate
            # Check if any c in s_sorted[i:] has c % candidate == target
            for k in range(i, n):
                if s_sorted[k] % candidate == target:
                    return False

        # candidate as 'b': for each a < candidate, no c >= candidate in set with (candidate+c) % a == 0
        for i in range(pos):
            a = s_sorted[i]
            # Check c = candidate: (candidate + candidate) % a
            if (2 * candidate) % a == 0:
                return False
            # Check c in s_sorted[pos:]
            for k in range(pos, n):
                if (candidate + s_sorted[k]) % a == 0:
                    return False

        # candidate as 'c': for each a < b <= candidate in set, (b + candidate) % a != 0
        for i in range(n):
            a = s_sorted[i]
            for j in range(i + 1, n):
                b = s_sorted[j]
                if b > candidate:
                    break
                if (b + candidate) % a == 0:
                    return False

        return True

    def conflict_count(elem, s_sorted, s_set):
        """Count how many valid triples would be violated if elem were in the set."""
        count = 0
        pos = bisect.bisect_left(s_sorted, elem)
        n = len(s_sorted)

        # elem as 'a'
        for i in range(pos, n):
            b = s_sorted[i]
            target = (-b) % elem
            for k in range(i, n):
                if s_sorted[k] % elem == target:
                    count += 1

        # elem as 'b'
        for i in range(pos):
            a = s_sorted[i]
            if (2 * elem) % a == 0:
                count += 1
            for k in range(pos, n):
                if (elem + s_sorted[k]) % a == 0:
                    count += 1

        # elem as 'c'
        for i in range(n):
            a = s_sorted[i]
            for j in range(i + 1, n):
                b = s_sorted[j]
                if b > elem:
                    break
                if (b + elem) % a == 0:
                    count += 1

        return count

    best_set = []

    # Try different starting points
    for min_start in [128, 144, 160, 171, 180, 192, 200, 210, 220]:
        if time.time() - start_time > deadline:
            break

        # Start with all elements from min_start to N
        s_sorted = list(range(min_start, N + 1))
        s_set = set(s_sorted)

        # Repair: iteratively remove the element with the highest conflict count
        changed = True
        while changed:
            changed = False
            if len(s_sorted) <= 1:
                break
            # Find the element with the most conflicts
            max_conf = 0
            max_idx = -1
            for i, a in enumerate(s_sorted):
                # Temporarily consider a's conflicts with the rest
                # a as 'a': count pairs (b,c) with b,c > a, b<=c, (b+c)%a==0
                pos = bisect.bisect_left(s_sorted, a)
                n = len(s_sorted)
                conf = 0
                # a as 'a'
                for j in range(pos, n):
                    b = s_sorted[j]
                    if b == a:
                        continue
                    target = (-b) % a
                    for k in range(j, n):
                        if s_sorted[k] % a == target:
                            conf += 1
                # a as 'b': pairs (x, a, c) with x < a <= c
                for j in range(pos):
                    x = s_sorted[j]
                    if (2 * a) % x == 0:
                        conf += 1
                    for k in range(pos, n):
                        if (a + s_sorted[k]) % x == 0:
                            conf += 1
                # a as 'c': pairs (x, y, a) with x < y <= a
                for j in range(n):
                    x = s_sorted[j]
                    for k in range(j + 1, n):
                        y = s_sorted[k]
                        if y > a:
                            break
                        if (y + a) % x == 0:
                            conf += 1

                if conf > max_conf:
                    max_conf = conf
                    max_idx = i

            if max_idx >= 0 and max_conf > 0:
                s_sorted.pop(max_idx)
                s_set.discard(s_sorted[max_idx] if max_idx < len(s_sorted) else list(s_set)[0])
                # Recompute s_set
                s_set = set(s_sorted)
                changed = True
            else:
                break

        # Now greedily add elements from 1 to min_start-1
        for candidate in range(1, min_start):
            if candidate in s_set:
                continue
            if is_valid_addition(s_set, s_sorted, candidate):
                pos = bisect.bisect_left(s_sorted, candidate)
                s_sorted.insert(pos, candidate)
                s_set.add(candidate)

        if len(s_sorted) > len(best_set):
            best_set = s_sorted[:]

    # Local search improvement
    s_set = set(best_set)
    for iteration in range(500):
        if time.time() - start_time > deadline:
            break

        if len(best_set) <= 1:
            break

        s = best_set[:]
        s_set_local = set(s)
        idx = random.randint(0, len(s) - 1)
        removed = s.pop(idx)
        s_set_local.discard(removed)

        # Build candidate list: removed element plus random others
        others = [x for x in range(1, N + 1) if x not in s_set_local]
        random.shuffle(others)
        candidates_to_try = [removed] + others[:20]
        random.shuffle(candidates_to_try)

        improved = False
        for candidate in candidates_to_try:
            if candidate in s_set_local:
                continue
            if is_valid_addition(s_set_local, s, candidate):
                pos = bisect.bisect_left(s, candidate)
                s.insert(pos, candidate)
                s_set_local.add(candidate)
                if len(s) > len(best_set):
                    best_set = s[:]
                    s_set = s_set_local
                    improved = True
                    break

    return best_set
