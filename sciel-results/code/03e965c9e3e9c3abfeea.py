# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(100000)

_P = 7
_Q = 49
_Q2 = 2401
_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 40.0


def _time_left():
    return _DEADLINE - time.time() - _MARGIN


def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    k = 3  # alpha^2 = 3

    # ---- Field tables ----
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    def pt_idx(x, y):
        return x * q + y

    # ---- Build all lines (2450 total) ----
    # Vertical lines: x = const, 49 lines
    # Slanted lines: y = m*x + c, 2401 lines
    line_points = []
    for x in range(q):
        line_points.append([pt_idx(x, y) for y in range(q)])
    for m in range(q):
        for c in range(q):
            line_points.append([pt_idx(t, int(add_table[mul_table[m, t], c])) for t in range(q)])
    n_lines = len(line_points)  # 2450

    # ---- Incidence: point_lines[z] = list of line indices containing z ----
    point_lines = [[] for _ in range(q2)]
    for li in range(n_lines):
        for pi in line_points[li]:
            point_lines[pi].append(li)

    # ---- Max-flow (Hopcroft-Karp) to find max matching in bipartite graph ----
    # Left: points (q2), Right: lines (n_lines)
    # Edge point z -> line li if z is on line li
    # We want max matching = max |S|

    # Build adjacency: for each point, list of lines
    adj = point_lines  # adj[z] = list of line indices

    # Hopcroft-Karp
    def hopcroft_karp(adj, n_left, n_right):
        from collections import deque
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            dq = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    dq.append(u)
                else:
                    dist[u] = -1
            found = False
            while dq:
                u = dq.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        dq.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, pair_u

    # Run max-flow to find maximum |S|
    max_s, assignment = hopcroft_karp(adj, q2, n_lines)

    # Build S from the matching: S[z] = True if z is matched to some line
    S = np.zeros(q2, dtype=bool)
    for z in range(q2):
        if assignment[z] != -1:
            S[z] = True

    # Verify validity
    # For each point z:
    #   if z in S: need a line through z with exactly 1 S-point (the assigned line)
    #   if z not in S: need a line through z with 0 S-points
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    for z in range(q2):
        if S[z]:
            for li in point_lines[z]:
                line_S_count[li] += 1

    # Check: for z in S, the assigned line should have count 1
    # For z not in S, at least one line through z should have count 0
    valid = True
    for z in range(q2):
        if S[z]:
            # Assigned line must have count 1
            li = assignment[z]
            if line_S_count[li] != 1:
                valid = False
                break
        else:
            # Need at least one line through z with count 0
            found = False
            for li in point_lines[z]:
                if line_S_count[li] == 0:
                    found = True
                    break
            if not found:
                valid = False
                break

    if not valid:
        # Fallback: return all points (trivially valid Nikodym set, |N| = 2401)
        pts_list = []
        for z in range(q2):
            x, y = z // q, z % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])
        return np.array(pts_list, dtype=np.int64)

    # Build output: N = all points not in S
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x, y = z // q, z % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
