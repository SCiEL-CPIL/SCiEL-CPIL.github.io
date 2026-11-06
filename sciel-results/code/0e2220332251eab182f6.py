import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def check_time():
        return time.time() > deadline

    def is_valid_sidon(a):
        diffs = set()
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))

    def backtrack_solve(max_val, k, time_limit=4.0):
        start = time.time()
        needed = set(range(1, 31))

        def bt(current, diffs, covered):
            if time.time() - start > time_limit:
                return None

            n = len(current)
            if n == k:
                if needed.issubset(covered):
                    return list(current)
                return None

            rem_needed = needed - covered
            rem_slots = k - n
            max_new = sum(n + i for i in range(rem_slots))
            if len(rem_needed) > max_new:
                return None

            candidates = set()
            for d in rem_needed:
                for y in current:
                    x = y + d
                    if x > max_val or x in current:
                        continue
                    candidates.add(x)

            if not candidates:
                start_x = current[-1] + 1
                for x in range(start_x, min(start_x + 20, max_val + 1)):
                    candidates.add(x)

            def score(x):
                cnt = sum(1 for y in current if (x - y) in rem_needed)
                return -cnt

            candidates = sorted(candidates, key=score)

            for x in candidates:
                new_diffs = set()
                valid = True
                for y in current:
                    d = x - y
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    continue

                result = bt(current + [x], diffs | new_diffs, covered | new_diffs)
                if result is not None:
                    return result

            return None

        return bt([0], set(), set())

    best = None

    for mv in range(10, 45):
        if check_time():
            break
        result = backtrack_solve(mv, 9, time_limit=3.0)
        if result is not None:
            best = result
            break

    if best is None and not check_time():
        for mv in range(10, 45):
            if check_time():
                break
            result = backtrack_solve(mv, 10, time_limit=3.0)
            if result is not None:
                best = result
                break

    if best is not None and not check_time():
        random.seed(123)
        best_max = max(best)
        for _ in range(300):
            if check_time():
                break
            improved = False
            cur = list(best)
            for idx in range(len(cur)):
                if check_time():
                    break
                orig = cur[idx]
                for new_val in range(1, orig):
                    if new_val in cur:
                        continue
                    trial = cur[:idx] + [new_val] + cur[idx + 1:]
                    trial = sorted(trial)
                    if len(set(trial)) != len(trial):
                        continue
                    if is_valid_sidon(trial) and max(trial) < best_max:
                        best = trial
                        best_max = max(trial)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    if best is None:
        a = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            for x in range(1, 150):
                if x in a:
                    continue
                trial = sorted(a + [x])
                diffs = set()
                valid = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        dd = trial[j] - trial[i]
                        if dd in diffs:
                            valid = False
                            break
                        diffs.add(dd)
                    if not valid:
                        break
                if valid and d in diffs:
                    a = trial
                    covered = diffs
                    found = True
                    break
            if not found:
                y = 2 * max(a) + d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    diffs = set()
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            dd = trial[j] - trial[i]
                            if dd in diffs:
                                valid = False
                                break
                            diffs.add(dd)
                    if valid:
                        a = trial
                        covered = diffs
                        break
                    y += 1
        best = a

    return best
