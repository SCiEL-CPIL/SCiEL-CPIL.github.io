import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Fallback: The seed LUW graph
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Get fallback result immediately to have a valid answer
    fallback_n, fallback_edges = luw_construct(k, g)
    best_n = fallback_n
    best_edges = fallback_edges

    # Fast girth check using BFS with early termination
    def check_girth_fast(adj, n, g):
        """Check if graph has girth >= g. Returns True if valid."""
        if n < g:
            return False
        max_depth = (g - 1) // 2

        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        if du + 1 < max_depth:
                            q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Build adjacency list from edge list
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Count short cycles (length < g) in a 3-regular graph
    def count_short_cycles(adj, n, g):
        """Count number of cycles of length < g. Returns count."""
        if n < g:
            return 1000
        max_depth = (g - 1) // 2
        count = 0

        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        if du + 1 < max_depth:
                            q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Only count each cycle once (from smallest vertex)
                            if start == min(u, v):
                                count += 1
        return count

    # Generate random 3-regular graph using configuration model
    def generate_random_3reg(n):
        """Generate a random 3-regular simple graph on n vertices."""
        if n < 4 or n % 2 != 0:
            return None

        # Create stubs: each vertex has 3 stubs
        stubs = []
        for v in range(n):
            for _ in range(3):
                stubs.append(v)

        # Randomly pair stubs, retry if multi-edges or loops
        for attempt in range(100):
            random.shuffle(stubs)
            edges = []
            valid = True
            edge_set = set()

            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))

            if valid and len(edges) == 3 * n // 2:
                return edges

        return None

    # Edge swap: pick two edges (a,b) and (c,d), replace with (a,c) and (b,d) or (a,d) and (b,c)
    def try_edge_swap(edges, adj, n, g):
        """Try a random edge swap. Returns (new_edges, new_adj, improvement) or None."""
        if len(edges) < 2:
            return None

        e1_idx = random.randint(0, len(edges) - 1)
        e2_idx = random.randint(0, len(edges) - 1)
        if e1_idx == e2_idx:
            return None

        a, b = edges[e1_idx]
        c, d = edges[e2_idx]

        # Try both possible reconnections
        for new_e1, new_e2 in [((a, c), (b, d)), ((a, d), (b, c))]:
            # Check for loops and multi-edges
            if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                continue

            key1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
            key2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))

            if key1 == key2:
                continue

            # Check if these edges already exist
            old_keys = set()
            old_keys.add((min(a, b), max(a, b)))
            old_keys.add((min(c, d), max(c, d)))

            if key1 in old_keys or key2 in old_keys:
                continue

            # Build new edge set
            new_edges = list(edges)
            new_edges[e1_idx] = new_e1
            new_edges[e2_idx] = new_e2

            # Build new adjacency
            new_adj = [list(nei) for nei in adj]
            # Remove old edges
            new_adj[a].remove(b)
            new_adj[b].remove(a)
            new_adj[c].remove(d)
            new_adj[d].remove(c)
            # Add new edges
            new_adj[new_e1[0]].append(new_e1[1])
            new_adj[new_e1[1]].append(new_e1[0])
            new_adj[new_e2[0]].append(new_e2[1])
            new_adj[new_e2[1]].append(new_e2[0])

            return new_edges, new_adj

        return None

    # Main search: configuration model + simulated annealing
    def config_model_search(n_target, g):
        """Try to find a 3-regular graph on n_target vertices with girth >= g."""
        # Generate initial graph
        edges = generate_random_3reg(n_target)
        if edges is None:
            return None

        adj = build_adj(edges, n_target)

        # Check if already valid
        if check_girth_fast(adj, n_target, g):
            return n_target, edges

        # Count initial short cycles
        initial_count = count_short_cycles(adj, n_target, g)

        # Simulated annealing
        best_count = initial_count
        best_edges = list(edges)
        best_adj = [list(nei) for nei in adj]

        temperature = 100.0
        cooling = 0.999
        min_temp = 0.01
        max_iterations = 5000

        for it in range(max_iterations):
            if time.time() > deadline:
                break

            # Try a random edge swap
            result = try_edge_swap(edges, adj, n_target, g)
            if result is None:
                continue

            new_edges, new_adj = result

            # Count short cycles in new graph
            new_count = count_short_cycles(new_adj, n_target, g)

            # Accept if better, or with probability exp(-(delta)/T) if worse
            delta = new_count - count_short_cycles(adj, n_target, g)
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                edges = new_edges
                adj = new_adj
                if new_count < best_count:
                    best_count = new_count
                    best_edges = list(edges)
                    best_adj = [list(nei) for nei in adj]

                # If we've eliminated all short cycles, we're done
                if new_count == 0:
                    if check_girth_fast(adj, n_target, g):
                        return n_target, edges

            temperature *= cooling
            if temperature < min_temp:
                break

        # Return best found if valid
        if best_count == 0:
            if check_girth_fast(best_adj, n_target, g):
                return n_target, best_edges

        return None

    # Moore bound for k=3, g=16: n >= 2 * sum_{i=0}^{7} 2^i = 510
    moore_bound = 2 * sum(2**i for i in range(g // 2))

    # Try configuration model search for various n values
    # Start near the Moore bound and go up
    n_candidates = list(range(moore_bound, min(moore_bound + 100, best_n)))
    # Ensure even numbers only (3-regular requires even n)
    n_candidates = [n for n in n_candidates if n % 2 == 0]

    # Also try some larger values
    for extra in [moore_bound + 50, moore_bound + 100, moore_bound + 200, moore_bound + 300]:
        if extra < best_n and extra % 2 == 0:
            n_candidates.append(extra)

    n_candidates = sorted(set(n_candidates))

    for n_target in n_candidates:
        if time.time() > deadline:
            break
        if n_target >= best_n:
            break

        result = config_model_search(n_target, g)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found

    return best_n, best_edges

# EVOLVE-BLOCK-END
