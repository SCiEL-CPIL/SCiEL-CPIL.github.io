"""Combined program for R(3,10) triangle-free graph construction."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def has_triangle(adj, n):
        """Check if graph with adjacency bitmasks has a triangle."""
        for u in range(n):
            neighbors_u = adj[u]
            v_mask = neighbors_u
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                if adj[v] & neighbors_u:
                    return True
                v_mask ^= v_bit
        return False

    def has_independent_set(adj, n, target=10, deadline=None):
        """
        Check if there is an independent set of size >= target in G.
        This is equivalent to finding a clique of size target in the complement.
        Uses branch and bound with greedy coloring upper bound.
        Returns True if found, False otherwise.
        """
        comp = [0] * n
        full_mask = (1 << n) - 1
        for u in range(n):
            comp[u] = full_mask ^ adj[u] ^ (1 << u)

        # Precompute vertex ordering by degree in complement (descending)
        # This helps the coloring heuristic
        deg_comp = [bin(comp[u]).count('1') for u in range(n)]
        order = sorted(range(n), key=lambda x: -deg_comp[x])
        # Map: position in order -> vertex
        # We'll use the order for branching priority

        found = [False]

        def color_bound(candidates, order_list):
            """
            Greedy coloring of the subgraph induced by candidates in the complement.
            Returns upper bound on clique size.
            """
            colors = {}
            uncolored = set(candidates)
            color_count = 0
            while uncolored:
                color_count += 1
                # Greedy: pick vertices in order, assign to current color if independent in comp
                color_set = set()
                for v in order_list:
                    if v not in uncolored:
                        continue
                    # v can join current color if no edge in comp to any in color_set
                    can_add = True
                    for w in color_set:
                        if comp[v] & (1 << w):
                            can_add = False
                            break
                    if can_add:
                        color_set.add(v)
                        uncolored.discard(v)
            return color_count

        def dfs(candidates, size, order_list):
            if found[0]:
                return
            if deadline is not None and time.time() > deadline:
                return
            if size >= target:
                found[0] = True
                return
            if not candidates:
                return

            # Upper bound via coloring
            cb = color_bound(candidates, order_list)
            if size + cb < target:
                return

            # Pick vertex with highest degree in candidate subgraph (for better branching)
            # Use the precomputed order
            v = -1
            best_deg = -1
            for u in order_list:
                if candidates & (1 << u):
                    d = bin(candidates & comp[u]).count('1')
                    if d > best_deg:
                        best_deg = d
                        v = u
            if v == -1:
                return

            # Include v
            new_cands = candidates & comp[v]
            dfs(new_cands, size + 1, order_list)

            if found[0]:
                return

            # Exclude v
            dfs(candidates ^ (1 << v), size, order_list)

        dfs(full_mask, 0, order)
        return found[0]

    def is_valid(adj, n, deadline=None):
        if has_triangle(adj, n):
            return False
        if has_independent_set(adj, n, 10, deadline=deadline):
            return False
        return True

    def edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def build_adj_from_edges(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # --- Strategy 1: Systematic circulant search ---
    def try_circulant(n):
        """
        Search for a valid circulant graph on n vertices.
        A circulant graph is defined by a set of offsets S (symmetric: if s in S then n-s in S).
        Triangle-free condition: for all a, b in S, (b-a) mod n not in S and (a+b) mod n not in S.
        """
        max_off = n // 2
        # Try structured offset sets first
        # Pattern: include offsets that are 1 mod 3 (like the seed)
        # But also try other patterns

        # First, try the classic 1 mod 3 pattern
        S = set()
        for off in range(1, max_off + 1):
            if off % 3 == 1:
                S.add(off)
                S.add(n - off)
            elif off % 3 == 2:
                # Don't include 2 mod 3 to avoid triangles in the 1-mod-3 family
                pass

        if S:
            adj = [0] * n
            for u in range(n):
                for s in S:
                    v = (u + s) % n
                    adj[u] |= (1 << v)
            if is_valid(adj, n, deadline=start_time + DEADLINE - 10):
                return edges_from_adj(adj, n)

        # Try random circulant sets with various densities
        for attempt in range(5000):
            if time.time() - start_time > DEADLINE - 15:
                return None

            random.seed(attempt * 7919 + n)
            S = set()
            # Try different density ranges
            density = 0.15 + 0.25 * (attempt % 4) / 4.0
            for off in range(1, max_off + 1):
                if random.random() < density:
                    S.add(off)
                    S.add(n - off)

            if not S:
                continue

            # Check triangle-free condition
            valid_tf = True
            S_list = sorted(S)
            for i, a in enumerate(S_list):
                for b in S_list[i + 1:]:
                    if (b - a) % n in S:
                        valid_tf = False
                        break
                    if (a + b) % n in S:
                        valid_tf = False
                        break
                if not valid_tf:
                    break

            if not valid_tf:
                continue

            # Build adj
            adj = [0] * n
            for u in range(n):
                for s in S:
                    v = (u + s) % n
                    adj[u] |= (1 << v)

            # Check independence
            if not has_independent_set(adj, n, 10, deadline=start_time + DEADLINE - 12):
                return edges_from_adj(adj, n)

        return None

    # --- Strategy 2: Random triangle-free graph with local search ---
    def random_triangle_free(n, seed=None):
        """Generate a random triangle-free graph by greedy edge addition."""
        if seed is not None:
            random.seed(seed)
        adj = [0] * n
        edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges_list)
        for u, v in edges_list:
            if adj[u] & (1 << v):
                continue
            if adj[u] & adj[v]:
                continue  # Would create triangle
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def local_improve(adj, n, max_steps=5000):
        """
        Try to improve a triangle-free graph to reduce its independence number.
        Strategy: try adding edges, if that doesn't work, try edge swaps.
        """
        # First, try to add all possible safe edges
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if not (adj[u] & adj[v]):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

        # Now try local search: remove an edge, then add two edges
        for step in range(max_steps):
            if time.time() - start_time > DEADLINE - 15:
                break

            # Check current alpha
            if not has_independent_set(adj, n, 10, deadline=start_time + DEADLINE - 12):
                return adj

            # Try removing a random edge
            edges_in_graph = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges_in_graph.append((u, v))

            if not edges_in_graph:
                break

            u, v = random.choice(edges_in_graph)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)

            # Try to add new edges
            improved = False
            for x in range(n):
                for y in range(x + 1, n):
                    if x == u and y == v:
                        continue
                    if not (adj[x] & (1 << y)):
                        if not (adj[x] & adj[y]):
                            adj[x] |= (1 << y)
                            adj[y] |= (1 << x)
                            improved = True

            if not improved:
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        return adj

    # --- Main search ---
    best_n = 0
    best_edges = []

    # Try circulant graphs for increasing n
    for n in range(26, 41):
        if time.time() - start_time > DEADLINE - 20:
            break
        edges = try_circulant(n)
        if edges is not None:
            best_n = n
            best_edges = edges
            print(f"Circulant found valid graph at n={n}")

    # Try random search for larger n
    for target_n in range(best_n + 1, 41):
        if time.time() - start_time > DEADLINE - 25:
            break

        found = False
        for trial in range(3000):
            if time.time() - start_time > DEADLINE - 15:
                break

            adj = random_triangle_free(target_n, seed=trial * 104729 + target_n * 31)
            adj = local_improve(adj, target_n, max_steps=200)

            if not has_independent_set(adj, target_n, 10, deadline=start_time + DEADLINE - 12):
                edges = edges_from_adj(adj, target_n)
                best_n = target_n
                best_edges = edges
                found = True
                print(f"Random found valid graph at n={target_n}")
                break

        if found:
            # Try to extend further
            continue
        else:
            # If we couldn't find one at this n, try next
            pass

    # Final: if we have a valid graph, try to extend it by adding vertices
    if best_n > 0 and time.time() - start_time < DEADLINE - 20:
        adj = build_adj_from_edges(best_n, best_edges)
        for new_n in range(best_n + 1, 41):
            if time.time() - start_time > DEADLINE - 10:
                break

            # Add new vertex connected to an independent set
            # Find a large independent set in current graph
            # Greedy independent set
            remaining = set(range(best_n))
            ind_set = set()
            while remaining:
                v = max(remaining, key=lambda x: -bin(adj[x] & ((1 << best_n) - 1) ^ ((1 << best_n) - 1) ^ (1 << x)).count('1'))
                ind_set.add(v)
                # Remove v and its neighbors from remaining
                mask_v = 1 << v
                for w in list(remaining):
                    if (adj[v] >> w) & 1 or w == v:
                        remaining.discard(w)

            if not ind_set:
                break

            # Connect new vertex to all in ind_set
            new_v = best_n
            adj_new = adj + [0]
            for u in ind_set:
                adj_new[new_v] |= (1 << u)
                adj_new[u] |= (1 << new_v)

            if is_valid(adj_new, new_n, deadline=start_time + DEADLINE - 8):
                best_n = new_n
                best_edges = edges_from_adj(adj_new, new_n)
                adj = adj_new
            else:
                break

    if best_n == 0:
        # Fallback: use the 26-vertex circulant
        n1 = 26
        edges1 = [[i, j] for i in range(n1) for j in range(i + 1, n1) if (j - i) % n1 % 3 == 1]
        best_n = n1
        best_edges = edges1

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
