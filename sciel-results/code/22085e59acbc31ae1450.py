import time
import random

# Q3 edges (canonical, vertices 0..7)
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations with sorted unassigned vertices
# For each edge (qa,qb) in Q3, we orient it as (gu,gv) and order the other 6 vertices
# by how many of {gu,gv} they are adjacent to in Q3 (desc) to prune early.
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

def _bits(mask):
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb


def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 120.0  # safety margin before 150s kill

    full_mask = (1 << n) - 1

    def has_q3_after_add(adj, u, v, deadline):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph.
        Uses bitmask candidate intersection for fast pruning."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)

                # Precompute for each unassigned vertex which assigned Q3-neighbors it has
                # and their graph-neighbors, to build initial candidate set
                def bt(idx, used_mask):
                    if time.time() > deadline - 5:
                        return False
                    if idx == 6:
                        # All 6 unassigned are assigned; verify all Q3 edges
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
                    # Candidates: not used, adjacent to all assigned Q3-neighbors of x
                    candidates = full_mask & ~used_mask
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
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False

                if bt(0, used_mask):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return True

        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    best_edges = None
    best_count = -1

    # Try multiple seeds and greedy strategies
    seeds = [
        # K3,13: part0={0,1,2}, part1={3..15}
        lambda adj: [adj[a] | (1 << b) for a in range(3) for b in range(3, 16)],
        # K4,12 minus some edges? K4,12 has 48 edges. Check if Q3-free.
        # Q3 is bipartite 4+4. K4,12 contains any bipartite subgraph with parts <=4,<=12.
        # So K4,12 DOES contain Q3. Bad.
        # Try K3,13 + some intra-part edges
    ]

    # We'll just use K3,13 as base and try different greedy orderings
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)

    cnt = count_edges(adj)
    best_count = cnt
    best_edges = edge_list(adj)

    # Greedy: repeatedly try to add a missing edge in different orders
    # First pass: sorted order
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                missing.append((i, j))

    # Try multiple random orderings
    for attempt in range(5):
        if time.time() > deadline - 10:
            break

        if attempt == 0:
            # sorted
            pass
        else:
            random.shuffle(missing)

        # Reset adj to K3,13
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

        cnt = count_edges(adj)

        # Greedily add edges
        for u, v in missing:
            if time.time() > deadline - 10:
                break
            if not (adj[u] & (1 << v)):
                if not has_q3_after_add(adj, u, v, deadline):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    cnt += 1

        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)

    # Local search: edge swaps to potentially increase edge count
    # Try removing an edge and adding more
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)

    # Rebuild from best found so far? No, let's do local search on the best graph.
    # Load best graph
    adj = [0] * n
    for u, v in best_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    best_count = count_edges(adj)

    # Local search: try swapping edges
    if time.time() < deadline - 20:
        edges_list = [(u, v) for u, v in best_edges]
        random.shuffle(edges_list)

        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            for a, b in edges_list:
                if time.time() > deadline - 20:
                    break
                # Try removing (a,b) and see if we can add more
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)

                # Try to add all missing edges
                added = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if time.time() > deadline - 20:
                            break
                        if not (adj[i] & (1 << j)):
                            if not has_q3_after_add(adj, i, j, deadline):
                                adj[i] |= (1 << j)
                                adj[j] |= (1 << i)
                                added += 1

                if added > 1:
                    improved = True
                    new_count = count_edges(adj)
                    if new_count > best_count:
                        best_count = new_count
                        best_edges = edge_list(adj)
                    # Rebuild edges_list
                    edges_list = [(u, v) for u, v in best_edges]
                    random.shuffle(edges_list)
                else:
                    # Put back (a,b)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    # Remove any edges we added (we need to track them)
                    # For simplicity, just restore from best_edges
                    adj = [0] * n
                    for u, v in best_edges:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

    # Final polish: try to add any remaining edges
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > deadline - 5:
                break
            if not (adj[i] & (1 << j)):
                if not has_q3_after_add(adj, i, j, deadline):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

    final_count = count_edges(adj)
    if final_count > best_count:
        best_count = final_count
        best_edges = edge_list(adj)

    return best_edges
