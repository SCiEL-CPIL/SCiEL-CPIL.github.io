import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    E = len(all_edges)
    edge_to_uv = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_uv[i] = (u, v)

    # Precompute for each edge, the list of vertices that are "neighbors" of u or v
    # When we add edge (u,v), the common-neighbor count for pair (u,w) increases by 1 if w~v
    # and for pair (v,w) increases by 1 if w~u. Also pair (u,v) gets +1 for each common neighbor.
    # Actually: adding (u,v) means u and v become neighbors. For any w:
    #   - N(u) gains v, N(v) gains u
    #   - Common neighbors of (u,w): was |N(u)∩N(w)|, now if w~v then +1 (v is new common neighbor)
    #   - Common neighbors of (v,w): was |N(v)∩N(w)|, now if w~u then +1
    #   - Common neighbors of (u,v): was |N(u)∩N(v)|, now unchanged (we're adding the edge, not a common neighbor)
    # Wait, actually the common neighbor count for (u,v) doesn't change when we add (u,v).
    # But for pairs involving u or v and some w:
    #   - (u,w): common neighbors increase by 1 iff v is a neighbor of w (i.e., edge (v,w) exists)
    #   - (v,w): common neighbors increase by 1 iff u is a neighbor of w (i.e., edge (u,w) exists)
    #   - (u,v): no change
    # For pairs not involving u or v: no change.

    def can_add_fast(adj, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        # After adding (u,v):
        # For each w != u,v:
        #   new_common(u,w) = old_common(u,w) + (1 if w~v else 0)
        #   new_common(v,w) = old_common(v,w) + (1 if w~u else 0)
        # We need all new_common <= 2
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[w] >> v) & 1:
                # w is neighbor of v, so common(u,w) increases by 1
                if (adj[u] & adj[w]).bit_count() >= 2:
                    return False
            if (adj[w] >> u) & 1:
                # w is neighbor of u, so common(v,w) increases by 1
                if (adj[v] & adj[w]).bit_count() >= 2:
                    return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        c = 0
        for u in range(n):
            c += adj[u].bit_count()
        return c // 2

    def is_valid(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True

    def greedy_fill(adj):
        """Try to add any missing edge that doesn't violate K2,3."""
        improved = True
        while improved:
            improved = False
            for i in range(E):
                u, v = all_edges[i]
                if (adj[u] >> v) & 1:
                    continue
                if can_add_fast(adj, u, v):
                    add_edge(adj, u, v)
                    improved = True
        return adj

    def count_violations(adj):
        v = 0
        for u in range(n):
            for w in range(u + 1, n):
                if (adj[u] & adj[w]).bit_count() >= 3:
                    v += 1
        return v

    best_adj = [0] * n
    best_count = 0

    def update_best(adj):
        nonlocal best_adj, best_count
        if is_valid(adj):
            c = count_edges(adj)
            if c > best_count:
                best_count = c
                best_adj = adj[:]

    # Run simulated annealing with multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        # Initialize
        adj = [0] * n
        p = random.uniform(0.25, 0.45)
        for i in range(E):
            if random.random() < p:
                u, v = all_edges[i]
                add_edge(adj, u, v)

        # Fix violations by removing edges
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            # Find a violating pair
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    cn = (adj[a] & adj[b]).bit_count()
                    if cn >= 3:
                        # Remove an edge that helps: prefer edges involving a or b
                        # that are common neighbors
                        common = adj[a] & adj[b]
                        # Get list of common neighbors
                        cn_list = []
                        for w in range(n):
                            if (common >> w) & 1:
                                cn_list.append(w)
                        # Remove edge (a, w) or (b, w) for some common neighbor w
                        # Try removing (a, w) for each common neighbor w
                        best_rem = None
                        best_new_viol = count_violations(adj)
                        for w in cn_list:
                            # Try removing (a, w)
                            remove_edge(adj, a, w)
                            nv = count_violations(adj)
                            if nv < best_new_viol:
                                best_new_viol = nv
                                best_rem = (a, w)
                            add_edge(adj, a, w)
                            # Try removing (b, w)
                            remove_edge(adj, b, w)
                            nv = count_violations(adj)
                            if nv < best_new_viol:
                                best_new_viol = nv
                                best_rem = (b, w)
                            add_edge(adj, b, w)
                        if best_rem:
                            remove_edge(adj, best_rem[0], best_rem[1])
                            found = True
                            break
                if found:
                    break
            if not found:
                break

        # Greedy fill
        greedy_fill(adj)
        update_best(adj)

        # Simulated annealing
        violations = count_violations(adj)
        edge_count = count_edges(adj)

        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        steps = 0
        max_steps = 80000

        while T > min_T and steps < max_steps:
            if time.time() > deadline - 5.0:
                break

            idx = random.randint(0, E - 1)
            u, v = all_edges[idx]
            is_present = (adj[u] >> v) & 1

            if is_present:
                # Remove edge
                remove_edge(adj, u, v)
                new_violations = count_violations(adj)
                new_edge_count = edge_count - 1
                old_energy = violations * 1000 - edge_count
                new_energy = new_violations * 1000 - new_edge_count
                delta = new_energy - old_energy
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    violations = new_violations
                    edge_count = new_edge_count
                    if violations == 0:
                        update_best(adj)
                else:
                    add_edge(adj, u, v)
            else:
                # Add edge
                if can_add_fast(adj, u, v):
                    add_edge(adj, u, v)
                    new_violations = violations
                    new_edge_count = edge_count + 1
                    old_energy = violations * 1000 - edge_count
                    new_energy = new_violations * 1000 - new_edge_count
                    delta = new_energy - old_energy
                    if delta <= 0 or random.random() < math.exp(-delta / T):
                        violations = new_violations
                        edge_count = new_edge_count
                        if violations == 0:
                            update_best(adj)
                    else:
                        remove_edge(adj, u, v)
                else:
                    # Can't add without violation, but we might still accept
                    add_edge(adj, u, v)
                    new_violations = count_violations(adj)
                    new_edge_count = edge_count + 1
                    old_energy = violations * 1000 - edge_count
                    new_energy = new_violations * 1000 - new_edge_count
                    delta = new_energy - old_energy
                    if random.random() < math.exp(-delta / T):
                        violations = new_violations
                        edge_count = new_edge_count
                    else:
                        remove_edge(adj, u, v)

            T *= cooling
            steps += 1

        # Final greedy fill
        if count_violations(adj) == 0:
            greedy_fill(adj)
            update_best(adj)

    # Final greedy fill on best
    if best_count > 0:
        greedy_fill(best_adj)
        c = count_edges(best_adj)
        if c > best_count:
            best_count = c

    # Convert to edge list
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
    return result

def _construct_greedy():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.seed(42)
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    adj = [0] * n
    edges = []
    for u, v in shuffled:
        if (adj[u] & adj[v]).bit_count() < 3:
            ok = True
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[u] & adj[w]).bit_count() + (1 if (adj[w] & (1 << v)) else 0) >= 3:
                    ok = False
                    break
                if ok:
                    if (adj[v] & adj[w]).bit_count() + (1 if (adj[w] & (1 << u)) else 0) >= 3:
                        ok = False
                        break
            if ok:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
    return edges

def _greedy_construct(n, all_edges, seed):
    pass

def _can_add_edge_fast(adj, u, v):
    pass

def _local_search(n, edges, all_edges, deadline):
    pass
