import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    adj = [[0] * n for _ in range(n)]
    edges = set()

    def add_edge(u, v):
        if u == v or adj[u][v]:
            return
        adj[u][v] = 1
        adj[v][u] = 1
        edges.add((min(u, v), max(u, v)))

    def remove_edge(u, v):
        if not adj[u][v]:
            return
        adj[u][v] = 0
        adj[v][u] = 0
        edges.discard((min(u, v), max(u, v)))

    # Precompute the 35 unique bipartitions of 8 items into two sets of 4
    all_splits = []
    seen = set()
    for A in combinations(range(8), 4):
        B = tuple(i for i in range(8) if i not in A)
        key = frozenset([frozenset(A), frozenset(B)])
        if key not in seen:
            seen.add(key)
            mask_a = sum(1 << i for i in A)
            mask_b = sum(1 << i for i in B)
            all_splits.append((mask_a, mask_b, A, B))

    def contains_q3_in_subset(v_list):
        """Check if the induced subgraph on v_list (8 vertices) contains Q3."""
        # Build local adjacency bitmasks
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask

        for mask_a, mask_b, A_idx, B_idx in all_splits:
            covered = 0
            is_matching = True
            missing_count = 0

            for i in A_idx:
                for j in B_idx:
                    if not (local_adj[i] & (1 << j)):
                        missing_count += 1
                        if missing_count > 4:
                            is_matching = False
                            break
                        if (covered & (1 << i)) or (covered & (1 << j)):
                            is_matching = False
                            break
                        covered |= (1 << i)
                        covered |= (1 << j)
                if not is_matching:
                    break
            if is_matching:
                return True
        return False

    # Initialize with K3,13 + K3 (42 edges, Q3-free)
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    add_edge(0, 1)
    add_edge(0, 2)
    add_edge(1, 2)

    best_edges = len(edges)
    best_edge_set = set(edges)

    # Local search: try adding random missing edges
    iterations = 0
    while time.time() < deadline and iterations < 100000:
        iterations += 1

        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or adj[u][v]:
            continue

        add_edge(u, v)

        # Check for Q3 containing (u, v)
        found_q3 = False
        other_vertices = [i for i in range(n) if i != u and i != v]

        for combo in combinations(other_vertices, 6):
            vertices = (u, v) + tuple(combo)
            if contains_q3_in_subset(vertices):
                found_q3 = True
                break

        if found_q3:
            remove_edge(u, v)
        else:
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_edge_set = set(edges)

    # Final polish: try all missing edges in a deterministic pass
    if time.time() < deadline:
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    continue
                add_edge(u, v)
                found_q3 = False
                other_vertices = [i for i in range(n) if i != u and i != v]
                for combo in combinations(other_vertices, 6):
                    vertices = (u, v) + tuple(combo)
                    if contains_q3_in_subset(vertices):
                        found_q3 = True
                        break
                if found_q3:
                    remove_edge(u, v)
                else:
                    if len(edges) > best_edges:
                        best_edges = len(edges)
                        best_edge_set = set(edges)

    return [list(e) for e in best_edge_set]

if __name__ == "__main__":
    res = construct()
    print(len(res))
