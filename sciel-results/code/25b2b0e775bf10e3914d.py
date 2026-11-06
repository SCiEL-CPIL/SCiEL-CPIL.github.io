import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin from 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_pairs_set = set(all_pairs)

    def full_check_k33_free(adj, n):
        """Check if graph with adjacency list adj is K3,3-free."""
        for a in range(n):
            na = adj[a]
            for b in range(a + 1, n):
                nb = adj[b]
                for c in range(b + 1, n):
                    nc = adj[c]
                    if len(na & nb & nc) >= 3:
                        return False
        return True

    def can_add_edge(adj, u, v, n):
        """Check if adding edge (u,v) preserves K3,3-freeness.
        Only check triples that include u or v (the only triples whose common-neighbor
        set changes)."""
        # Temporarily add the edge
        adj[u].add(v)
        adj[v].add(u)

        # Check all triples that include u or v
        # Triples with u (but not v): (u, x, y) for x < y, x,y != v
        for x in range(n):
            if x == u or x == v:
                continue
            nx = adj[x]
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                ny = adj[y]
                if len(adj[u] & nx & ny) >= 3:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    return False

        # Triples with v (but not u): (v, x, y) for x < y, x,y != u
        for x in range(n):
            if x == u or x == v:
                continue
            nx = adj[x]
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                ny = adj[y]
                if len(adj[v] & nx & ny) >= 3:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    return False

        return True

    def greedy_build(n, rng):
        """Build a K3,3-free graph greedily with random edge ordering."""
        adj = [set() for _ in range(n)]
        edges = []

        shuffled = all_pairs[:]
        rng.shuffle(shuffled)

        for u, v in shuffled:
            if can_add_edge(adj, u, v, n):
                edges.append([u, v])

        return edges, adj

    def local_search(adj, edges, n, rng, time_limit):
        """Try to add more edges to a valid graph via random attempts."""
        current_edges = list(edges)
        edge_set = set(tuple(e) for e in current_edges)
        adj = [set(s) for s in adj]

        # Build list of missing edges
        missing = [(u, v) for (u, v) in all_pairs if (u, v) not in edge_set]

        # Try adding missing edges in random order
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > time_limit:
                break
            if can_add_edge(adj, u, v, n):
                current_edges.append([u, v])
                edge_set.add((u, v))

        return current_edges, adj

    def local_search_swap(adj, edges, n, rng, time_limit):
        """Try edge swaps: remove one edge, try to add two new edges."""
        current_edges = list(edges)
        edge_set = set(tuple(e) for e in current_edges)
        adj = [set(s) for s in adj]
        improved = True

        while improved and time.time() < time_limit:
            improved = False
            # Try removing a random edge and adding two new ones
            if len(current_edges) < 2:
                break
            for _ in range(min(200, len(current_edges))):
                if time.time() > time_limit:
                    break
                idx = rng.randint(0, len(current_edges) - 1)
                rem_u, rem_v = current_edges[idx]
                # Remove edge
                adj[rem_u].discard(rem_v)
                adj[rem_v].discard(rem_u)
                current_edges.pop(idx)
                edge_set.discard((rem_u, rem_v))

                # Try to add two new edges
                missing = [(u, v) for (u, v) in all_pairs if (u, v) not in edge_set]
                rng.shuffle(missing)
                added = 0
                for u, v in missing[:50]:
                    if added >= 2:
                        break
                    if can_add_edge(adj, u, v, n):
                        current_edges.append([u, v])
                        edge_set.add((u, v))
                        added += 1

                if added > 1:
                    improved = True

        return current_edges, adj

    rng = random.Random(42)

    best_edges = None
    best_len = 0

    # Seed construction: K_{2,18}
    seed_edges = [[a, b] for a in range(2) for b in range(2, 20)]
    best_edges = seed_edges
    best_len = len(seed_edges)

    # Try many randomized greedy builds with local search refinement
    trial = 0
    max_trials = 200

    while trial < max_trials:
        if time.time() > deadline - 15:
            break

        # Randomized greedy build
        edges, adj = greedy_build(n, rng)

        # Local search: try adding more edges
        edges, adj = local_search(adj, edges, n, rng, deadline - 10)

        if len(edges) > best_len:
            best_edges = edges
            best_len = len(edges)

        # Try swap-based local search on the best graph found so far
        if trial % 10 == 0 and time.time() < deadline - 20:
            # Rebuild adjacency from best_edges
            adj_best = [set() for _ in range(n)]
            for u, v in best_edges:
                adj_best[u].add(v)
                adj_best[v].add(u)
            edges_swapped, adj_swapped = local_search_swap(adj_best, best_edges, n, rng, deadline - 10)
            if len(edges_swapped) > best_len:
                best_edges = edges_swapped
                best_len = len(edges_swapped)

        trial += 1

    # Final verification
    if best_edges is None:
        best_edges = seed_edges

    # Verify the result is valid
    adj_final = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_final[u].add(v)
        adj_final[v].add(u)

    if not full_check_k33_free(adj_final, n):
        # Fallback to seed
        best_edges = seed_edges

    return best_edges
