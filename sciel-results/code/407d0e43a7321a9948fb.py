import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, adj):
        """Build conflict as list of bitmasks for fast checking."""
        m = len(edges)
        if m == 0:
            return []
        conflict = [0] * m
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check shared vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    continue
                # Check adjacency (joined by an edge)
                if (u2 in adj[u1] or v2 in adj[u1] or
                        u2 in adj[v1] or v2 in adj[v1]):
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        return conflict

    def greedy_color_bitmask(conflict, m, rng, strategy='largest_first'):
        """Greedy coloring with bitmask conflicts. Returns (colors, num_colors)."""
        if m == 0:
            return [], 0
        color_of_edge = [-1] * m

        if strategy == 'largest_first':
            order = list(range(m))
            order.sort(key=lambda i: bin(conflict[i]).count('1'), reverse=True)
        elif strategy == 'smallest_first':
            order = list(range(m))
            order.sort(key=lambda i: bin(conflict[i]).count('1'))
        elif strategy == 'random':
            order = list(range(m))
            rng.shuffle(order)
        else:
            order = list(range(m))
            order.sort(key=lambda i: bin(conflict[i]).count('1'), reverse=True)

        for i in order:
            forbidden = 0
            for j in range(m):
                if (conflict[i] >> j) & 1:
                    if color_of_edge[j] != -1:
                        forbidden |= (1 << color_of_edge[j])
            c = 0
            while (forbidden >> c) & 1:
                c += 1
            color_of_edge[i] = c

        max_c = max(color_of_edge) if color_of_edge else 0
        return color_of_edge, max_c + 1

    def backtracking_color(conflict, m, k, order, rng, node_limit=200000):
        """
        Try to find a valid coloring with exactly k colors using backtracking
        with forward checking. Returns color array or None.
        """
        if m == 0:
            return [0] * m

        color_of = [-1] * m
        # available[c] = bitmask of edges still available for color c
        available = [(1 << m) - 1] * k
        # For each color, which edges are "free" (not yet colored, no conflict with same-colored edges)
        # We'll track: for color c, the set of uncolored edges that can take color c.

        # Actually, forward checking: when we assign edge i to color c, we remove i and
        # all edges conflicting with i from available[c].

        nodes = [0]

        def bt(pos):
            if nodes[0] > node_limit:
                return False
            if pos == m:
                return True
            nodes[0] += 1
            i = order[pos]
            # Try each color
            for c in range(k):
                # Check if edge i is available for color c
                if not ((available[c] >> i) & 1):
                    continue
                # Assign
                color_of[i] = c
                # Forward check: remove i and its conflicts from available[c]
                removed = (1 << i) | conflict[i]
                new_avail_c = available[c] & ~removed
                available[c] = new_avail_c

                # Check no other uncolored edge has 0 available colors
                valid = True
                for p in range(pos + 1, m):
                    j = order[p]
                    # Is j uncolored? (it is, since we go in order)
                    any_avail = 0
                    for cc in range(k):
                        if (available[cc] >> j) & 1:
                            any_avail = 1
                            break
                    if not any_avail:
                        valid = False
                        break

                if valid:
                    if bt(pos + 1):
                        return True

                # Undo
                available[c] = new_avail_c | removed
                color_of[i] = -1

            return False

        if bt(0):
            return color_of
        return None

    def local_search_improve(conflict, m, colors, max_colors, rng, max_iters=300):
        """Try to reduce number of colors by local search."""
        if m == 0:
            return colors, 0
        current_colors = colors[:]
        current_max = max_colors

        for _ in range(max_iters):
            max_c = max(current_colors) if current_colors else -1
            if max_c < 0:
                break
            high_edges = [i for i in range(m) if current_colors[i] == max_c]
            if not high_edges:
                break
            i = rng.choice(high_edges)
            forbidden = 0
            for j in range(m):
                if (conflict[i] >> j) & 1:
                    if current_colors[j] != -1:
                        forbidden |= (1 << current_colors[j])
            moved = False
            for c in range(max_c):
                if not ((forbidden >> c) & 1):
                    current_colors[i] = c
                    moved = True
                    break
            if not moved:
                # Try swapping
                for j in high_edges:
                    if j == i:
                        continue
                    c_i_new = current_colors[j]
                    c_j_new = max_c
                    ok_i = True
                    for k2 in range(m):
                        if k2 != j and (conflict[i] >> k2) & 1 and current_colors[k2] == c_i_new:
                            ok_i = False
                            break
                    if ok_i:
                        ok_j = True
                        for k2 in range(m):
                            if k2 != i and (conflict[j] >> k2) & 1 and current_colors[k2] == c_j_new:
                                ok_j = False
                                break
                        if ok_j:
                            current_colors[i] = c_i_new
                            current_colors[j] = c_j_new
                            moved = True
                            break
            if not moved:
                continue
            new_max = max(current_colors) if current_colors else 0
            if new_max < current_max:
                current_max = new_max

        return current_colors, (max(current_colors) + 1 if current_colors else 0)

    def solve_instance(edges, adj, rng):
        """Solve one instance: find a valid coloring minimizing colors."""
        m = len(edges)
        if m == 0:
            return []
        conflict = build_conflict_bitmasks(edges, adj)

        # Get a good initial upper bound from greedy
        best_colors = None
        best_score = float('inf')
        for strategy in ['largest_first', 'smallest_first', 'random']:
            colors, score = greedy_color_bitmask(conflict, m, rng, strategy)
            if score < best_score:
                best_score = score
                best_colors = colors

        # Try to improve with backtracking: find smallest k that works
        # Lower bound: max degree in conflict graph + 1, or greedy result
        # We'll try k from a low value up to best_score - 1
        # Order for backtracking: by conflict degree
        order = list(range(m))
        order.sort(key=lambda i: bin(conflict[i]).count('1'), reverse=True)

        # Lower bound: try to find a feasible k below best_score
        # Start from a reasonable lower bound
        # chromatic number lower bound: use greedy as upper, try to go below
        lb = 1
        # A simple lower bound: clique-ish, but let's just try from best_score-1 downward
        # Actually we want the MINIMUM, so try k = lb, lb+1, ... up to best_score
        # But to save time, try k values from low to best_score-1
        found = None
        # Try a range of k values
        max_k_to_try = best_score
        # We'll iterate k from a low bound
        # Lower bound: at least the max number of edges sharing structure
        # Simple LB: 1. But better: use the greedy's structure
        # Let's try k from max(1, best_score - 3) up to best_score - 1, then also lower
        # To be safe, try k from 1 upward but cap by best_score
        # Given time, try a window
        k_start = max(1, best_score - 4)
        for k in range(k_start, best_score):
            if time.time() > deadline - 3.0:
                break
            # Use a fresh rng for determinism within the search
            r2 = random.Random(12345 + k)
            # Try multiple orders if needed
            res = backtracking_color(conflict, m, k, order, r2, node_limit=100000)
            if res is not None:
                found = res
                best_score = k
                break

        if found is not None:
            best_colors = found

        # Final local search polish
        if best_colors is not None:
            improved_colors, improved_score = local_search_improve(
                conflict, m, best_colors, best_score, rng, max_iters=150
            )
            if improved_score < best_score:
                best_score = improved_score
                best_colors = improved_colors

        return best_colors

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    # Pre-generate graphs
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        graphs.append((edges, adj))

    best_results = [[] for _ in graphs]
    best_scores = [float('inf')] * len(graphs)

    # First pass: solve each instance
    for idx, (edges, adj) in enumerate(graphs):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(42 + idx)
        colors = solve_instance(edges, adj, rng)
        if colors is not None:
            best_results[idx] = colors
            best_scores[idx] = max(colors) + 1 if colors else 0

    # Second pass: if time remains, try to improve with different seeds
    if time.time() < deadline - 8.0:
        for idx, (edges, adj) in enumerate(graphs):
            if time.time() > deadline - 5.0:
                break
            # Try a few more seeds to potentially find fewer colors
            for extra_seed in [7, 13, 21, 33]:
                if time.time() > deadline - 4.0:
                    break
                rng = random.Random(extra_seed * 100 + idx)
                colors = solve_instance(edges, adj, rng)
                if colors is not None:
                    sc = max(colors) + 1 if colors else 0
                    if sc < best_scores[idx]:
                        best_scores[idx] = sc
                        best_results[idx] = colors

    # Ensure all results are populated
    for i in range(len(graphs)):
        if not best_results[i] or best_scores[i] == float('inf'):
            edges, adj = graphs[i]
            rng = random.Random(0)
            colors = solve_instance(edges, adj, rng)
            best_results[i] = colors if colors is not None else []

    return best_results
