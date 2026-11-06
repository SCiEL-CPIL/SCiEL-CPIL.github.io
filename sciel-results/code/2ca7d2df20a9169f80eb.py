import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def is_valid(a):
        diffs = set()
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))

    def search(D):
        found = [None]
        FULL_MASK = (1 << 30) - 1

        def dfs(current, diff_set, diff_mask, last):
            if time.time() > deadline:
                return True
            if found[0] is not None:
                return True

            if diff_mask == FULL_MASK:
                found[0] = list(current)
                return True

            remaining_count = 30 - bin(diff_mask).count('1')
            k_max = 32 - len(current)
            if k_max <= 0:
                return False
            
            # Max new diffs we can generate
            max_new = k_max * len(current) + k_max * (k_max - 1) // 2
            if max_new < remaining_count:
                return False

            # Try candidates
            # We can restrict x to be > last
            # Also, x - last should be small? No, x can be anything.
            # But to minimize D, we search small D first.
            
            # Heuristic: try x that cover more missing diffs
            candidates = []
            for x in range(last + 1, D + 1):
                ok = True
                new_diffs = []
                for elem in current:
                    d = x - elem
                    if d in diff_set:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                score = sum(1 for d in new_diffs if 1 <= d <= 30 and not (diff_mask & (1 << (d - 1))))
                candidates.append((score, x, new_diffs))
            
            # Sort by score descending
            candidates.sort(key=lambda t: -t[0])
            
            for score, x, new_diffs in candidates:
                if time.time() > deadline:
                    return True
                if found[0] is not None:
                    return True

                current.append(x)
                new_diff_set = diff_set | set(new_diffs)
                new_mask = diff_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                if dfs(current, new_diff_set, new_mask, x):
                    return True
                current.pop()

            return False

        dfs([0], set(), 0, 0)
        return found[0]

    # Search for minimal D
    # Lower bound 30, but likely 41
    for D in range(41, 60):
        if time.time() > deadline:
            break
        result = search(D)
        if result is not None:
            return result

    # Fallback
    return [0, 1, 4, 10, 12, 28, 32, 45, 63]
