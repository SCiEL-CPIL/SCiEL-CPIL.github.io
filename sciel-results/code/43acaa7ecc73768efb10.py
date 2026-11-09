import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Base: 8 disjoint copies of K4 -> n=32, alpha=8, C5-free.
    Attempts (within a tight time budget) to reach n=33.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Base 32-vertex graph: 8 x K4
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    # Attempt 33-vertex construction within a strict time budget
    if time.time() < deadline - 5:
        result = _try_33_vertex(start_time, deadline)
        if result is not None:
            return result

    return {'n': n, 'edges': edges}


def _build_adj(n, edge_set):
    adj = [set() for _ in range(n)]
    for e in edge_set:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _has_c5_fast(n, edge_set):
    """Efficient C5 detection: enumerate simple paths of length 4, check if endpoints adjacent."""
    adj = _build_adj(n, edge_set)
    for a in range(n):
        a_nb = adj[a]
        for b in a_nb:
            for c in adj[b]:
                if c == a:
                    continue
                for d in adj[c]:
                    if d == a or d == b:
                        continue
                    for e in adj[d]:
                        if e in (a, b, c):
                            continue
                        if e in a_nb:
                            return True
    return False


def _alpha_ge_fast(n, edge_set, k=9):
    """Check if there is an independent set of size k using branch & bound."""
    adj = _build_adj(n, edge_set)
    best = [0]

    def backtrack(candidates, size):
        if size > best[0]:
            best[0] = size
        if best[0] >= k:
            return
        if not candidates:
            return
        if size + len(candidates) <= best[0]:
            return
        v = candidates[0]
        rest = candidates[1:]
        # Include v
        new_cands = [x for x in rest if x not in adj[v]]
        backtrack(new_cands, size + 1)
        if best[0] >= k:
            return
        # Exclude v
        backtrack(rest, size)

    backtrack(list(range(n)), 0)
    return best[0] >= k


def _try_33_vertex(start_time, deadline):
    """Try to build a 33-vertex C5-free graph with alpha < 9 via local search."""
    base_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                base_edges.append((base + i, base + j))

    random.seed(98765)
    n = 33

    for attempt in range(300):
        if time.time() > deadline - 3:
            break

        edges_set = set(frozenset(e) for e in base_edges)

        # Connect vertex 32 to a random subset
        all_base = list(range(32))
        conn_size = random.randint(1, 6)
        conn = set(random.sample(all_base, conn_size))
        for v in conn:
            edges_set.add(frozenset((32, v)))

        # Local repair
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            if _has_c5_fast(n, edges_set):
                edges_from_32 = [e for e in edges_set if 32 in e]
                if edges_from_32:
                    e = random.choice(edges_from_32)
                    edges_set.remove(e)
                    improved = True
                else:
                    break
            elif _alpha_ge_fast(n, edges_set, 9):
                adj32 = set()
                for e in edges_set:
                    if 32 in e:
                        u, v = tuple(e)
                        other = u if v == 32 else v
                        adj32.add(other)
                non_neighbors = [v for v in range(32) if v not in adj32]
                if non_neighbors:
                    v = random.choice(non_neighbors)
                    edges_set.add(frozenset((32, v)))
                    improved = True
                else:
                    break

        if time.time() > deadline - 3:
            break

        if _has_c5_fast(n, edges_set):
            continue
        if _alpha_ge_fast(n, edges_set, 9):
            continue

        edges_list = [[min(e), max(e)] for e in edges_set]
        return {'n': 33, 'edges': edges_list}

    return None
