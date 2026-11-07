import time
import random

N = 10

def _cyclic():
    return [[(i + j) % N for j in range(N)] for i in range(N)]

def _cyclic_slope(m):
    return [[(i + m * j) % N for j in range(N)] for i in range(N)]

def _verify(S):
    for sq in S:
        if not isinstance(sq, (list, tuple)):
            return False
        if len(sq) != N:
            return False
        for row in sq:
            if not isinstance(row, (list, tuple)):
                return False
            if len(row) != N:
                return False
            for v in row:
                if type(v) is not int:
                    return False
                if not (0 <= v <= N - 1):
                    return False
        for i in range(N):
            if sorted(sq[i]) != list(range(N)):
                return False
        for j in range(N):
            col = [sq[i][j] for i in range(N)]
            if sorted(col) != list(range(N)):
                return False
    m = len(S)
    for a in range(m):
        for b in range(a + 1, m):
            pairs = set()
            for i in range(N):
                for j in range(N):
                    pairs.add((S[a][i][j], S[b][i][j]))
            if len(pairs) != N * N:
                return False
    return True

def _find_orthogonal(S, time_limit, t_start):
    row_used = [set() for _ in range(N)]
    col_used = [set() for _ in range(N)]
    pair_used = [set() for _ in range(len(S))]
    grid = [[-1] * N for _ in range(N)]
    K = len(S)
    state = {"found": None}

    def get_candidates(i, j):
        res = []
        for s in range(N):
            if s in row_used[i] or s in col_used[j]:
                continue
            ok = True
            for k in range(K):
                if (S[k][i][j], s) in pair_used[k]:
                    ok = False
                    break
            if ok:
                res.append(s)
        return res

    def solve(idx):
        if state["found"] is not None:
            return
        if (time.time() - t_start) > time_limit:
            return
        if idx == N * N:
            state["found"] = [row[:] for row in grid]
            return
        i = idx // N
        j = idx % N
        cands = get_candidates(i, j)
        random.shuffle(cands)
        for s in cands:
            grid[i][j] = s
            row_used[i].add(s)
            col_used[j].add(s)
            for k in range(K):
                pair_used[k].add((S[k][i][j], s))
            solve(idx + 1)
            if state["found"] is not None:
                return
            grid[i][j] = -1
            row_used[i].remove(s)
            col_used[j].remove(s)
            for k in range(K):
                pair_used[k].remove((S[k][i][j], s))

    solve(0)
    return state["found"]

def construct():
    t0 = time.time()
    deadline = t0 + 55.0
    random.seed()
    
    # Start with two known orthogonal cyclic squares
    S = [_cyclic_slope(1), _cyclic_slope(2)]
    
    # Try to add more
    while time.time() < deadline and len(S) < 9:
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        # Try to find next square
        tl = min(5.0, remaining * 0.8)
        newq = _find_orthogonal(S, tl, time.time())
        if newq is not None:
            S.append(newq)
        else:
            # If search fails, try a few more times with different random seeds
            # but don't spend too much time
            found = False
            for _ in range(3):
                if time.time() > deadline - 1.0:
                    break
                tl = min(2.0, (deadline - time.time()) * 0.8)
                newq = _find_orthogonal(S, tl, time.time())
                if newq is not None:
                    S.append(newq)
                    found = True
                    break
            if not found:
                break

    if _verify(S):
        return S
    # Fallback to at least 1
    return [_cyclic_slope(1)]
