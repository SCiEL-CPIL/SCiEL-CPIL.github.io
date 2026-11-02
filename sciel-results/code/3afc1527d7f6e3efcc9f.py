def construct(n_rows, k):
    import sys
    import time

    sys.setrecursionlimit(100000)
    start_time = time.time()
    time_limit = 115.0

    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in rows)
    best_rows = [list(r) for r in rows]

    def find_solution(M, n_rows, k, timeout):
        """Find n_rows Sidon sets of size k+1 in [0,M] with pairwise disjoint differences."""
        t0 = time.time()
        candidates = []

        def gen_row(current, row_diffs):
            if time.time() - t0 > timeout:
                return False
            if len(current) == k + 1:
                candidates.append((tuple(current), frozenset(row_diffs)))
                return True
            last = current[-1]
            for x in range(last + 1, M + 1):
                new_diffs = [x - a for a in current]
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                combined = set(row_diffs) | set(new_diffs)
                if len(combined) != len(row_diffs) + len(new_diffs):
                    continue
                current.append(x)
                row_diffs.extend(new_diffs)
                gen_row(current, row_diffs)
                current.pop()
                for _ in new_diffs:
                    row_diffs.pop()
                if time.time() - t0 > timeout:
                    return False
            return True

        gen_row([0], [])
        if not candidates:
            return None

        candidates.sort(key=lambda c: max(c[0]))
        n_cand = len(candidates)
        selected = []
        used_diffs = set()

        def backtrack(idx, count):
            if time.time() - t0 > timeout:
                return False
            if count == n_rows:
                return True
            if n_cand - idx < n_rows - count:
                return False
            for i in range(idx, n_cand):
                row, diffs = candidates[i]
                if not (diffs & used_diffs):
                    selected.append(row)
                    used_diffs.update(diffs)
                    if backtrack(i + 1, count + 1):
                        return True
                    selected.pop()
                    used_diffs.difference_update(diffs)
            return False

        if backtrack(0, 0):
            return [list(r) for r in selected]
        return None

    for target in range(best_max - 1, max(best_max - 20, k), -1):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
        result = find_solution(target, n_rows, k, timeout=min(10.0, remaining))
        if result is not None:
            best_rows = result
            best_max = target
            break

    return best_rows
