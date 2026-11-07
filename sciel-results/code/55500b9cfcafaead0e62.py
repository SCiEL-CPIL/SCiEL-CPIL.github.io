"""Densest graph on 15 vertices avoiding equal-degree path endpoints.

Key insight: partition vertices into cliques with distinct degree classes.
Within a clique, paths of length 3 can only have endpoints in the same clique
if the clique is large enough. For small cliques (size <= 4), any 3-edge path
s-u-v-t with all vertices in the same clique forces s and t to be in the same
clique, and since all vertices in a clique have the same degree (within that
clique's subgraph), we need deg[s] != deg[t]. But if s and t are both in the
same clique, they have the same degree! So we must avoid 3-edge paths within
any clique of size >= 4... wait, that's not right.

Actually, let me reconsider. If all vertices in a clique have the same degree
in the full graph, then any path s-u-v-t where s and t are both in that clique
would violate the constraint. So for a clique to be valid, there must be NO
simple 3-edge path with both endpoints in the clique.

In a complete graph K_n, the number of 3-edge paths with endpoints s and t
(s != t) both in the clique: we need s-u-v-t where all are in the clique and
s != t. For n >= 4, such paths exist. For n = 3, s-u-v-t requires 4 distinct
vertices, so no such path exists within a 3-clique. For n <= 3, no violation
within the clique.

Between cliques: if we connect cliques sparsely, paths that go between cliques
will have endpoints in different cliques, and if the degree classes are distinct,
no violation.

Strategy: partition 15 vertices into cliques of size at most 3, each clique
having a distinct degree. Connect cliques with a bipartite structure that
preserves degree distinctness.
"""
import random
import time
import itertools

N = 15


def _check_all_paths(adj, deg):
    """Check all simple 3-edge paths. Returns True if valid."""
    n = N
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True


def _try_add_edge(adj, deg, a, b):
    """Try to add edge (a,b) without it already existing. Returns True if valid."""
    n = N
    da, db = deg[a], deg[b]

    # Case 1: s-a-b-t (a,b is middle edge)
    for s in adj[a]:
        if s == b:
            continue
        for t in adj[b]:
            if t == a:
                continue
            if deg[s] == deg[t]:
                return False

    # Case 2: a-b-u-t (a is endpoint)
    for u in adj[b]:
        if u == a:
            continue
        for t in adj[u]:
            if t == b:
                continue
            if deg[a] == deg[t]:
                return False

    # Case 3: s-u-b-a (a is endpoint)
    for u in adj[b]:
        if u == a:
            continue
        for s in adj[u]:
            if s == b:
                continue
            if deg[s] == deg[a]:
                return False

    # Case 4: b-a-u-t (b is endpoint)
    for u in adj[a]:
        if u == b:
            continue
        for t in adj[u]:
            if t == a:
                continue
            if deg[b] == deg[t]:
                return False

    # Case 5: s-u-a-b (b is endpoint)
    for u in adj[a]:
        if u == b:
            continue
        for s in adj[u]:
            if s == a:
                continue
            if deg[s] == deg[b]:
                return False

    adj[a].add(b)
    adj[b].add(a)
    deg[a] += 1
    deg[b] += 1
    return True


def _remove_edge(adj, deg, a, b):
    adj[a].discard(b)
    adj[b].discard(a)
    deg[a] -= 1
    deg[b] -= 1


def _add_back(adj, deg, a, b):
    adj[a].add(b)
    adj[b].add(a)
    deg[a] += 1
    deg[b] += 1


def _clique_construction(partition, inter_edges):
    """
    Build graph from clique partition and inter-clique edges.
    partition: list of lists, each a list of vertex indices forming a clique.
    inter_edges: list of (u, v) pairs between different cliques.
    """
    n = N
    adj = [set() for _ in range(n)]
    deg = [0] * n

    # Add intra-clique edges
    for clique in partition:
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1

    # Add inter-clique edges
    for u, v in inter_edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    return adj, deg


def _greedy_inter_clique_edges(partition, adj, deg, rng, deadline):
    """Add inter-clique edges greedily."""
    n = N
    count = sum(len(s) for s in adj) // 2

    # Build candidate inter-clique edges
    clique_of = {}
    for ci, clique in enumerate(partition):
        for v in clique:
            clique_of[v] = ci

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if clique_of[i] != clique_of[j] and j not in adj[i]:
                candidates.append((i, j))

    rng.shuffle(candidates)

    for a, b in candidates:
        if time.time() > deadline:
            break
        if _try_add_edge(adj, deg, a, b):
            count += 1

    return adj, deg, count


def _local_search(adj, deg, rng, deadline, iters=200):
    """Local search to improve edge count."""
    n = N
    count = sum(len(s) for s in adj) // 2
    best_adj = [set(s) for s in adj]
    best_deg = list(deg)
    best_count = count

    no_improve = 0
    for it in range(iters):
        if time.time() > deadline:
            break

        # Try adding missing edges
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    missing.append((i, j))
        rng.shuffle(missing)

        improved = False
        for a, b in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, deg, a, b):
                count += 1
                improved = True
                if count > best_count:
                    best_count = count
                    best_adj = [set(s) for s in adj]
                    best_deg = list(deg)
                    no_improve = 0
                break

        if not improved:
            # Try 1-for-1 swaps
            all_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
            rng.shuffle(all_edges)
            swap_found = False
            for ra, rb in all_edges:
                if time.time() > deadline:
                    break
                _remove_edge(adj, deg, ra, rb)
                for a, b in missing:
                    if (a == ra and b == rb) or (a == rb and b == ra):
                        continue
                    if _try_add_edge(adj, deg, a, b):
                        swap_found = True
                        break
                    else:
                        _remove_edge(adj, deg, a, b)
                if swap_found:
                    break
                else:
                    _add_back(adj, deg, ra, rb)
            if not swap_found:
                no_improve += 1
                if no_improve > 15:
                    break
            else:
                no_improve = 0

    return best_adj, best_deg, best_count


def _generate_partitions():
    """Generate various clique partitions of 15 vertices."""
    partitions = []

    # All cliques of size 3: 5 cliques of size 3
    base = list(range(15))
    p1 = [base[i*3:(i+1)*3] for i in range(5)]
    partitions.append(p1)

    # 4 cliques of size 3 + 1 clique of size 3 (same as above)
    # Mix: sizes 3,3,3,3,3
    # Try different orderings
    rng = random.Random(123)
    for _ in range(20):
        shuffled = list(range(15))
        rng.shuffle(shuffled)
        p = [shuffled[i*3:(i+1)*3] for i in range(5)]
        partitions.append(p)

    # Sizes: 4,3,3,3,2
    p2 = [list(range(0,4)), list(range(4,7)), list(range(7,10)), list(range(10,13)), list(range(13,15))]
    partitions.append(p2)

    # Sizes: 3,3,3,2,2,2
    p3 = [list(range(0,3)), list(range(3,6)), list(range(6,9)), list(range(9,11)), list(range(11,13)), list(range(13,15))]
    partitions.append(p3)

    # Sizes: 4,4,3,2,2
    p4 = [list(range(0,4)), list(range(4,8)), list(range(8,11)), list(range(11,13)), list(range(13,15))]
    partitions.append(p4)

    # Sizes: 5,3,3,2,2
    p5 = [list(range(0,5)), list(range(5,8)), list(range(8,11)), list(range(11,13)), list(range(13,15))]
    partitions.append(p5)

    return partitions


def _multi_start(n, rng, deadline):
    """Multiple starts with different partitions and greedy inter-clique edges."""
    best_count = -1
    best_adj = None
    best_deg = None

    partitions = _generate_partitions()

    for start_k in range(40):
        if time.time() > deadline - 12.0:
            break

        # Pick a partition
        part = partitions[start_k % len(partitions)]

        # Build base clique graph
        adj, deg = _clique_construction(part, [])

        # Add inter-clique edges greedily
        adj, deg, cnt = _greedy_inter_clique_edges(part, adj, deg, rng, deadline - 10.0)

        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
            best_deg = list(deg)

        # Local search refinement
        if time.time() < deadline - 8.0:
            adj2, deg2, cnt2 = _local_search(adj, deg, rng, deadline - 6.0, iters=150)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = [set(s) for s in adj2]
                best_deg = list(deg2)

    # Final polish
    if best_adj is not None and time.time() < deadline - 4.0:
        adj3, deg3, cnt3 = _local_search(best_adj, best_deg, rng, deadline - 2.0, iters=300)
        if cnt3 > best_count:
            best_count = cnt3
            best_adj = [set(s) for s in adj3]
            best_deg = list(deg3)

    if best_adj is None:
        best_adj = [set() for _ in range(n)]

    return best_adj


def construct():
    n = N
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s
    rng = random.Random(42)

    adj = _multi_start(n, rng, deadline)

    # Final verification
    deg = [len(adj[i]) for i in range(n)]
    if not _check_all_paths(adj, deg):
        # Fallback: rebuild conservatively
        adj2 = [set() for _ in range(n)]
        deg2 = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng2 = random.Random(777)
        rng2.shuffle(edges)
        for a, b in edges:
            adj2[a].add(b)
            adj2[b].add(a)
            deg2[a] += 1
            deg2[b] += 1
            if not _check_all_paths(adj2, deg2):
                adj2[a].discard(b)
                adj2[b].discard(a)
                deg2[a] -= 1
                deg2[b] -= 1
        adj = adj2

    # Build edge list with proper types
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    return edges
