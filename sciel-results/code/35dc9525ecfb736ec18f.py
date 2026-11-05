import random
import time

# Q3 (3-cube) edges and adjacency bitmasks
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# For each Q3 edge (qa,qb) and each orientation, precompute the unassigned vertices
# sorted by "constraint strength" (number of assigned neighbors they must connect to)
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))


def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    full_mask = (1 << n) - 1

    # Start with K_{4,4,4,4} complete 4-partite graph (48 edges, Q3-free)
    adj = [0] * n
    parts = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
    for i in range(4):
        for j in range(i+1, 4):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    def has_q3_after_add(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph using bitmask operations."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        result = False
        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)

                def backtrack(idx, used_mask):
                    if time.time() > deadline - 5:
                        return False
                    if idx == len(unassigned):
                        # All assigned, verify all Q3 edges
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
                    candidates = full_mask & ~used_mask

                    # Filter candidates: must be adjacent to all assigned Q3-neighbors of x
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]

                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit

                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if backtrack(idx + 1, new_used):
                            return True
                        assignment[x] = None

                    return False

                if backtrack(0, used_mask):
                    result = True
                    break
            if result:
                break

        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return result

    # Build list of missing edges and try to add them greedily
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                missing.append((i, j))

    random.shuffle(missing)

    # First pass: add all edges that don't create Q3
    for u, v in missing:
        if time.time() > deadline - 20:
            break
        if not has_q3_after_add(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    # Second pass: try again with different orderings to squeeze more edges
    if time.time() < deadline - 20:
        missing2 = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    missing2.append((i, j))
        random.shuffle(missing2)
        for u, v in missing2:
            if time.time() > deadline - 10:
                break
            if not has_q3_after_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # Third pass: try yet another ordering
    if time.time() < deadline - 10:
        missing3 = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    missing3.append((i, j))
        random.shuffle(missing3)
        for u, v in missing3:
            if time.time() > deadline - 5:
                break
            if not has_q3_after_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])

    return edges
