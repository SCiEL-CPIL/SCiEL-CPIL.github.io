import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    best_edges = []
    best_score = 0

    # Phase 1: Multiple greedy constructions with different strategies.
    # The fast bitmask validity check lets us run many more restarts.
    for seed in range(6000):
        if time.time() > deadline - 25.0:
            break

        if seed % 3 == 0:
            edges = _greedy_random(n, all_edges, seed)
        elif seed % 3 == 1:
            edges = _greedy_degree_biased(n, all_edges, seed)
        else:
            edges = _greedy_random(n, all_edges, seed + 100000)

        score = len(edges)
        if score > best_score:
            best_score = score
            best_edges = edges

    # Phase 2: Local search (additions + 2-edge swaps) on the best solution.
    if time.time() < deadline - 20.0 and best_edges:
        improved = _local_search_swap(n, best_edges, all_edges, deadline - 18.0)
        if len(improved) > best_score:
            best_edges = improved
            best_score = len(improved)

    # Phase 3: Local search on a few other near-best diverse solutions.
    if time.time() < deadline - 15.0:
        for seed in range(1000, 1600):
            if time.time() > deadline - 15.0:
                break
            edges = _greedy_degree_biased(n, all_edges, seed)
            if len(edges) >= best_score - 3:  # Close to best
                improved = _local_search_swap(n, edges, all_edges, deadline - 15.0)
                if len(improved) > best_score:
                    best_edges = improved
                    best_score = len(improved)

    # Phase 4: Simulated-annealing polish (neutral-swap acceptance) for a
    # high-precision final pass. This is a refinement within the same
    # local-search family: it uses the fast validity check and a decaying
    # temperature to escape plateaus the greedy 2-swap cannot.
    if time.time() < deadline - 3.0 and best_edges:
        polished = _anneal_polish(n, best_edges, all_edges, deadline - 2.0)
        if len(polished) > best_score:
            best_edges = polished
            best_score = len(polished)

    return best_edges


def _greedy_random(n, all_edges, seed):
    """Standard random-order greedy."""
    random.seed(seed)
    adj = [0] * n
    edges = []

    shuffled = all_edges[:]
    random.shuffle(shuffled)

    for u, v in shuffled:
        if _can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])

    return edges


def _greedy_degree_biased(n, all_edges, seed):
    """Greedy that prefers edges between lower-degree vertices."""
    random.seed(seed)
    adj = [0] * n
    edges = []
    deg = [0] * n

    shuffled = all_edges[:]
    random.shuffle(shuffled)

    for _pass in range(3):
        remaining = [e for e in shuffled if not ((1 << e[1]) & adj[e[0]])]
        remaining.sort(key=lambda e: (deg[e[0]] + deg[e[1]], random.random()))

        for u, v in remaining:
            if (1 << v) & adj[u]:
                continue
            if _can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])

    return edges


def _can_add(adj, u, v):
    """Fast check whether adding (u,v) keeps the graph K2,3-free.

    Iterates only over the current neighbors of u and v (bit iteration)
    rather than all n vertices, which is much faster for sparse-to-mid
    density graphs and enables more search within the deadline.
    """
    if (1 << v) & adj[u]:
        return False

    # Pair (u, v): common neighbors must be < 3
    if (adj[u] & adj[v]).bit_count() >= 3:
        return False

    # For each neighbor w of v (w != u): common neighbors of u and w gain v
    # iff v is adjacent to w.
    w = adj[v]
    w &= ~(1 << u)
    while w:
        lsb = w & -w
        wi = lsb.bit_length() - 1
        w ^= lsb
        cuw = (adj[u] & adj[wi]).bit_count()
        if adj[wi] & (1 << v):
            cuw += 1
        if cuw >= 3:
            return False

    # For each neighbor w of u (w != v): common neighbors of v and w gain u
    # iff u is adjacent to w.
    w = adj[u]
    w &= ~(1 << v)
    while w:
        lsb = w & -w
        wi = lsb.bit_length() - 1
        w ^= lsb
        cvw = (adj[v] & adj[wi]).bit_count()
        if adj[wi] & (1 << u):
            cvw += 1
        if cvw >= 3:
            return False

    return True


def _local_search_swap(n, edges, all_edges, deadline):
    """Local search: pure additions, then 2-edge swaps (remove 1, add 2)."""
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))

    current_edges = [list(e) for e in edges]
    improved = True

    while improved and time.time() < deadline:
        improved = False

        # Phase A: Try to add any single missing edge.
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if (min(u, v), max(u, v)) in edge_set:
                continue
            if _can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((min(u, v), max(u, v)))
                current_edges.append([u, v])
                improved = True

        if improved:
            continue  # Restart; more additions may now be possible

        # Phase B: Try 2-edge swaps (remove 1, add 2) for net +1.
        edge_list = list(edge_set)
        random.shuffle(edge_list)

        for e in edge_list:
            if time.time() > deadline:
                break

            u1, v1 = e

            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            edge_set.remove(e)

            candidates = []
            for u2, v2 in all_edges:
                if (min(u2, v2), max(u2, v2)) in edge_set:
                    continue
                if _can_add(adj, u2, v2):
                    candidates.append((u2, v2))

            found = False
            if len(candidates) >= 2:
                max_check = min(len(candidates), 40)
                for i in range(max_check):
                    if found:
                        break
                    u2, v2 = candidates[i]
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    edge_set.add((min(u2, v2), max(u2, v2)))

                    for j in range(i + 1, max_check):
                        u3, v3 = candidates[j]
                        if (min(u3, v3), max(u3, v3)) in edge_set:
                            continue
                        if _can_add(adj, u3, v3):
                            adj[u3] |= (1 << v3)
                            adj[v3] |= (1 << u3)
                            edge_set.add((min(u3, v3), max(u3, v3)))
                            found = True
                            break

                    if not found:
                        adj[u2] ^= (1 << v2)
                        adj[v2] ^= (1 << u2)
                        edge_set.remove((min(u2, v2), max(u2, v2)))

                if found:
                    current_edges = [[u, v] for u, v in edge_set]
                    improved = True
                    break
            else:
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edge_set.add(e)
                continue

            if not found:
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edge_set.add(e)

    return current_edges


def _anneal_polish(n, edges, all_edges, deadline):
    """Simulated-annealing polish: accept neutral 1-remove/1-add swaps with
    a decaying temperature to escape local plateaus. Uses the fast validity
    check. Returns the best (max-edge) graph seen during the anneal."""
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))

    best_edge_count = len(edge_set)
    best_adj = [x for x in adj]

    T = 0.5
    cooling = 0.999
    n_edge_list = len(all_edges)

    while time.time() < deadline:
        # First, greedily add any possible edge (0-1 move, always accepted).
        added_any = False
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if (min(u, v), max(u, v)) in edge_set:
                continue
            if _can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((min(u, v), max(u, v)))
                added_any = True
                if len(edge_set) > best_edge_count:
                    best_edge_count = len(edge_set)
                    best_adj = [x for x in adj]
        if added_any:
            continue

        # Neutral 1-remove/1-add swap with annealing acceptance.
        edge_list = list(edge_set)
        random.shuffle(edge_list)

        for e in edge_list:
            if time.time() > deadline:
                break

            u1, v1 = e
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            edge_set.remove(e)

            # Find a single edge that can be added after removal.
            chosen = None
            for u2, v2 in all_edges:
                if (min(u2, v2), max(u2, v2)) in edge_set:
                    continue
                if _can_add(adj, u2, v2):
                    chosen = (u2, v2)
                    break

            if chosen is None:
                # Revert.
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edge_set.add(e)
                continue

            u2, v2 = chosen
            # Net change is 0 (remove 1, add 1). Accept with probability T
            # to allow exploration; this is a neutral move in edge count.
            if random.random() < T:
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                edge_set.add((min(u2, v2), max(u2, v2)))
                # Edge count unchanged; best tracked via additions above.
            else:
                # Revert the removal.
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edge_set.add(e)

        T *= cooling
        if T < 0.01:
            break

    # Reconstruct best graph from best_adj.
    result = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            if u < v:
                result.append([u, v])
    return result
