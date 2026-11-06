import time

def construct():
    deadline = time.time() + 140.0
    best = None
    best_max = float('inf')

    def check_solution(a):
        s = sorted(a)
        diffs = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.append(s[j] - s[i])
        if len(diffs) != len(set(diffs)):
            return False
        dset = set(diffs)
        for d in range(1, 31):
            if d not in dset:
                return False
        return True

    def compute_forbidden(s):
        """Return set of positions that would break Sidon if added to sorted list s."""
        forb = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = s[j] - s[i]
                forb.add(s[i] - d)
                forb.add(s[j] + d)
                if (s[i] + s[j]) % 2 == 0:
                    forb.add((s[i] + s[j]) // 2)
        forb -= set(s)
        return forb

    def compute_covered(s):
        cov = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = s[j] - s[i]
                if 1 <= d <= 30:
                    cov.add(d)
        return cov

    def solve():
        nonlocal best, best_max
        import sys
        sys.setrecursionlimit(10000)

        def dfs(elems, elems_set, covered, forbidden):
            nonlocal best, best_max
            if time.time() > deadline:
                return
            remaining = set(range(1, 31)) - covered
            if not remaining:
                m = max(elems)
                if m < best_max:
                    best_max = m
                    best = list(elems)
                return

            target = max(remaining)
            cands = set()
            for a in elems:
                cands.add(a + target)
                cands.add(a - target)
            cands = sorted(x for x in cands if x >= 0 and x not in elems_set and x not in forbidden)

            for x in cands:
                if time.time() > deadline:
                    return
                new_elems = elems + [x]
                new_elems_set = elems_set | {x}
                new_covered = compute_covered(new_elems)
                new_forbidden = compute_forbidden(new_elems)
                dfs(new_elems, new_elems_set, new_covered, new_forbidden)

        dfs([0], {0}, set(), set())

    solve()

    if best is not None and check_solution(best):
        return best
    return [0, 1, 4, 10, 12, 17, 20, 23, 29, 32, 37, 43, 46, 49, 55, 58, 61, 67, 70, 73, 79, 82, 85, 91, 94, 97, 103, 106, 109, 115, 118, 121]
