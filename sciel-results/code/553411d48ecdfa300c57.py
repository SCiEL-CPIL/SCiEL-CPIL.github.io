"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))

    def contains_q3(vs, edge_set):
        adj = {v: set() for v in vs}
        for i in range(8):
            for j in range(i + 1, 8):
                u, w = vs[i], vs[j]
                if (min(u, w), max(u, w)) in edge_set:
                    adj[u].add(w)
                    adj[w].add(u)
        for v in vs:
            if len(adj[v]) < 3:
                return False
        assignment = [-1] * 8
        vs_used = [False] * 8

        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for vs_i in range(8):
                if vs_used[vs_i]:
                    continue
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        if vs[vs_i] not in adj[vs[assignment[nb]]]:
                            ok = False
                            break
                if ok:
                    assignment[q3_v] = vs_i
                    vs_used[vs_i] = True
                    if backtrack(q3_v + 1):
                        return True
                    vs_used[vs_i] = False
                    assignment[q3_v] = -1
            return False

        return backtrack(0)

    def is_q3_free(edge_set):
        for vs in all_8sets:
            if contains_q3(vs, edge_set):
                return False
        return True

    # Baseline: K3,13
    k313 = set((a, b) for a in range(3) for b in range(3, 16))
    best_edges = k313
    best_count = len(k313)

    # Greedy: try to add edges to K3,13
    current_edges = set(k313)
    candidate_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                       if (i, j) not in current_edges]

    # Prioritize: edges within {0,1,2} first, then within {3..15}, then cross
    def edge_priority(e):
        u, v = e
        if u < 3 and v < 3:
            return 0
        if u >= 3 and v >= 3:
            return 1
        return 2

    candidate_edges.sort(key=edge_priority)

    for edge in candidate_edges:
        if time.time() > deadline - 25:
            break
        u, v = edge
        creates_q3 = False
        remaining = [x for x in range(16) if x != u and x != v]
        for other6 in combinations(remaining, 6):
            vertex_set = (u, v) + other6
            u_deg = 1
            v_deg = 1
            for x in vertex_set:
                if x == u or x == v:
                    continue
                if (min(u, x), max(u, x)) in current_edges:
                    u_deg += 1
                if (min(v, x), max(v, x)) in current_edges:
                    v_deg += 1
            if u_deg < 3 or v_deg < 3:
                continue
            test_edges = current_edges | {(u, v)}
            if contains_q3(vertex_set, test_edges):
                creates_q3 = True
                break

        if not creates_q3:
            current_edges.add(edge)

    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = current_edges

    # Try random dense graphs
    random.seed(123)
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        p = random.uniform(0.55, 0.75)
        edges = set()
        for i in range(16):
            for j in range(i + 1, 16):
                if random.random() < p:
                    edges.add((i, j))
        if is_q3_free(edges) and len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges

    result = [[u, v] for (u, v) in best_edges]
    return result
# EVOLVE-BLOCK-END
