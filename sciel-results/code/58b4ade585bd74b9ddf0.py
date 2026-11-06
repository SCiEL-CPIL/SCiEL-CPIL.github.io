import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

_edge_pos = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_pos[(min(_u, _v), max(_u, _v))] = _i

def _edge_idx(u, v):
    return _edge_pos[(min(u, v), max(u, v))]

def construct():
    t0 = time.time()
    deadline = t0 + 85.0

    # Trivial cycle: 0,1,2,...,23
    trivial = list(range(N))
    trivial_score = max(_edge_idx(trivial[i], trivial[(i+1) % N]) for i in range(N))

    best_cycle = trivial
    best_score = trivial_score

    # Precompute adjacency lists with edge positions
    full_adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        full_adj[u].append((v, i))
        full_adj[v].append((u, i))
    for u in range(N):
        full_adj[u].sort(key=lambda x: x[1])

    # Binary search for the minimum K such that a Hamilton cycle exists in G_K
    lo, hi = 24, best_score - 1
    node_limit = 500000

    while lo <= hi and time.time() < deadline - 5.0:
        mid = (lo + hi) // 2
        
        # Build adjacency for prefix of size mid
        adj = [[] for _ in range(N)]
        for u in range(N):
            for v, pos in full_adj[u]:
                if pos <= mid:
                    adj[u].append((v, pos))
        
        # Check if a Hamilton cycle exists using DFS
        start = 0
        path = [start]
        used = bytearray(N)
        used[start] = 1
        found = [False]
        aborted = [False]
        nodes_visited = [0]

        def dfs(cur, depth):
            if found[0] or aborted[0]:
                return
            if time.time() > deadline - 5.0:
                aborted[0] = True
                return
            nodes_visited[0] += 1
            if nodes_visited[0] > node_limit:
                aborted[0] = True
                return
            if depth == N:
                if _edge_idx(cur, start) <= mid:
                    found[0] = True
                return
            for nxt, pos in adj[cur]:
                if not used[nxt]:
                    used[nxt] = 1
                    path.append(nxt)
                    dfs(nxt, depth + 1)
                    if found[0] or aborted[0]:
                        return
                    path.pop()
                    used[nxt] = 0

        dfs(start, 1)

        if found[0]:
            sc = max(_edge_idx(path[i], path[(i+1) % N]) for i in range(N))
            if sc <= mid:
                best_cycle = list(path)
                best_score = sc
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1

    # Local search / hill climbing to improve the score
    # Try swapping two consecutive vertices in the cycle and see if score improves
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(N - 1):
            # Try swapping vertices at positions i and i+1
            old_a = best_cycle[i]
            old_b = best_cycle[i + 1]
            if old_a == old_b:
                continue
            # Swap
            best_cycle[i] = old_b
            best_cycle[i + 1] = old_a
            
            new_score = max(_edge_idx(best_cycle[j], best_cycle[(j+1) % N]) for j in range(N))
            
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                # Revert
                best_cycle[i] = old_a
                best_cycle[i + 1] = old_b

    # Try rotating the cycle to see if it helps (shouldn't change score, but just in case)
    # Actually rotation doesn't change the set of edges, so score stays the same.
    # But maybe a different starting point helps with the search? No, score is invariant.

    # Try 2-opt style moves: reverse a segment
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for i in range(N - 1):
            for j in range(i + 1, N):
                if j == i + 1:
                    continue
                # Reverse segment from i to j
                best_cycle[i:j+1] = reversed(best_cycle[i:j+1])
                new_score = max(_edge_idx(best_cycle[k], best_cycle[(k+1) % N]) for k in range(N))
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    # Revert
                    best_cycle[i:j+1] = reversed(best_cycle[i:j+1])

    return best_cycle
