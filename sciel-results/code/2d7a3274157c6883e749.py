"""C6-free graph on 40 vertices via greedy + local search with fast C6 check."""
import time
import random

N = 40
DEADLINE_MARGIN = 5.0

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    def check_c6_fast(adj_list, u, v):
        """Check if there's a simple path of length 5 from u to v.
        Path: u - a - b - c - d - v, all distinct, each consecutive pair is edge.
        """
        for a in adj_list[u]:
            if a == v:
                continue
            for b in adj_list[a]:
                if b == u or b == v:
                    continue
                for c in adj_list[b]:
                    if c == u or c == a or c == v:
                        continue
                    for d in adj_list[c]:
                        if d == u or d == b or d == v:
                            continue
                        if v in adj_list[d]:
                            return True
        return False

    def greedy_build(order):
        adj_list = [set() for _ in range(N)]
        edges = []
        for (u, v) in order:
            if v in adj_list[u]:
                continue
            adj_list[u].add(v)
            adj_list[v].add(u)
            if check_c6_fast(adj_list, u, v):
                adj_list[u].discard(v)
                adj_list[v].discard(u)
            else:
                edges.append((u, v))
        return edges, adj_list

    def local_search(edges, adj_list):
        edges = list(edges)
        adj_list = [set(s) for s in adj_list]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for (u, v) in all_edges:
                if v not in adj_list[u]:
                    adj_list[u].add(v)
                    adj_list[v].add(u)
                    if not check_c6_fast(adj_list, u, v):
                        edges.append((u, v))
                        improved = True
                    else:
                        adj_list[u].discard(v)
                        adj_list[v].discard(u)
        return edges, adj_list

    best_edges = []
    best_count = 0

    for restart in range(15):
        if time.time() > deadline:
            break

        if restart % 2 == 0:
            edges = []
            adj_list = [set() for _ in range(N)]
            for k in range(8):
                for a in range(5):
                    for b in range(a + 1, 5):
                        u, v = 5 * k + a, 5 * k + b
                        edges.append((u, v))
                        adj_list[u].add(v)
                        adj_list[v].add(u)
            edges, adj_list = local_search(edges, adj_list)
        else:
            order = list(all_edges)
            random.shuffle(order)
            edges, adj_list = greedy_build(order)
            edges, adj_list = local_search(edges, adj_list)

        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)

    return [[u, v] for (u, v) in best_edges]
