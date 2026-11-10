"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))

    def time_ok():
        return time.time() < deadline

    # --- Tomita branch-and-bound maximum clique ---
    def max_clique():
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos = {v: i for i, v in enumerate(order)}
        best = []

        def expand(R, P_list):
            nonlocal best
            if not time_ok():
                return
            if not P_list:
                if len(R) > len(best):
                    best = list(R)
                return
            if len(R) + len(P_list) <= len(best):
                return
            # Sort P_list by position (reverse order for Tomita)
            P_list = sorted(P_list, key=lambda x: pos[x], reverse=True)
            i = 0
            while i < len(P_list):
                if not time_ok():
                    return
                v = P_list[i]
                # Upper bound: if even taking all remaining can't beat best, stop
                if len(R) + len(P_list) - i <= len(best):
                    return
                R.append(v)
                # New candidates: those in P_list[0:i] that are adjacent to v
                P_new = [u for u in P_list[:i] if u in adj[v]]
                expand(R, P_new)
                R.pop()
                i += 1

        expand([], list(range(n)))
        return best

    clique = max_clique()

    if not clique or len(clique) < 2:
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # In a clique, every pair is directly adjacent, so path = [u, v]
    B = [int(x) for x in clique]
    pairs = list(itertools.combinations(B, 2))
    result_paths = [[u, v] for u, v in pairs]

    return {'branches': B, 'paths': result_paths}
# EVOLVE-BLOCK-END
