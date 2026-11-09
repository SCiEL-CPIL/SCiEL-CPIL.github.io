import time

def construct():
    start_time = time.time()
    deadline = start_time + 55

    P = 27720
    n = 12
    max_L = 144

    offset = [[0]*13 for _ in range(13)]
    for i in range(1, 13):
        for s in range(i):
            offset[i][s] = (i - 1 - s) % i

    def has_matching(r, L):
        valid = [None] * 13
        for i in range(1, 13):
            c = offset[i][r % i]
            if c >= L:
                return False
            valid[i] = list(range(c, L, i))

        order = tuple(sorted(range(1, 13), key=lambda i: len(valid[i])))

        match_to = [-1] * L

        def dfs(i_curr, visited):
            for j in valid[i_curr]:
                if not visited[j]:
                    visited[j] = True
                    if match_to[j] == -1 or dfs(match_to[j], visited):
                        match_to[j] = i_curr
                        return True
            return False

        for i in order:
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    best_L = 0
    best_r = 0

    for r in range(P):
        if time.time() > deadline:
            break

        if not has_matching(r, max_L):
            if max_L > best_L:
                best_L = max_L
                best_r = r
            continue

        lo, hi = 12, max_L
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1

        if lo > best_L:
            best_L = lo
            best_r = r

    return int(best_r)
