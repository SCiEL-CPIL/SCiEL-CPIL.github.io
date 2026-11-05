import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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
        
        # Precompute adjacency sets for O(1) lookup
        adj_sets = [set() for _ in range(len(edges[0][0]) + 1 if edges else 1)] 
        # Actually, we need adj for vertices, not edges.
        # adj is passed in as defaultdict(set) from construct scope or built here.
        # Let's build it properly if not passed, but here we assume adj is passed or built.
        # In the current structure, adj is built in construct() and passed.
        
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
                # Check if any endpoint of e1 is adjacent to any endpoint of e2
                # adj[u1] contains all neighbors of u1
                if (u2 in adj[u1] or v2 in adj[u1] or
                    u2 in adj[v1] or v2 in adj[v1]):
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        return conflict

    def get_neighbor_colors(conflict, color, i):
        """Get set of colors used by neighbors of vertex i."""
        used = set()
        mask = conflict[i]
        # Iterate through bits in mask
        # Optimized: iterate through all j, check if bit is set
        # Since m is small (< 200), simple loop is fine
        for j in range(len(color)):
            if (mask >> j) & 1:
                if color[j] != -1:
                    used.add(color[j])
        return used

    def dsatur_greedy(conflict, m):
        """
        DSATUR-inspired greedy coloring using bitmasks.
        """
        if m == 0:
            return [], 0

        color = [-1] * m
        degrees = [bin(conflict[i]).count('1') for i in range(m)]
        
        for _ in range(m):
            best_edge = -1
            best_sat = -1
            best_deg = -1
            
            # Find vertex with max saturation, tie-break by degree
            for i in range(m):
                if color[i] != -1:
                    continue
                sat_colors = get_neighbor_colors(conflict, color, i)
                sat = len(sat_colors)
                deg = degrees[i]
                
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_edge = i
            
            if best_edge == -1:
                break
            
            # Assign smallest available color
            used = get_neighbor_colors(conflict, color, best_edge)
            c = 0
            while c in used:
                c += 1
            color[best_edge] = c
            
        max_c = max(color) if color else 0
        return color, max_c + 1

    def greedy_color(conflict, m, rng, strategy='largest_first'):
        """Greedy coloring with given strategy."""
        if m == 0:
            return [], 0

        color_of_edge = [-1] * m
        degrees = [bin(conflict[i]).count('1') for i in range(m)]

        if strategy == 'largest_first':
            order = list(range(m))
            order.sort(key=lambda i: degrees[i], reverse=True)
        elif strategy == 'random':
            order = list(range(m))
            rng.shuffle(order)
        elif strategy == 'smallest_first':
            order = list(range(m))
            order.sort(key=lambda i: degrees[i])
        elif strategy == 'largest_random_tie':
            order = list(range(m))
            random_vals = [rng.random() for _ in range(m)]
            order.sort(key=lambda i: (-degrees[i], random_vals[i]))
        else:
            order = list(range(m))
            order.sort(key=lambda i: degrees[i], reverse=True)

        for i in order:
            forbidden = get_neighbor_colors(conflict, color_of_edge, i)
            c = 0
            while c in forbidden:
                c += 1
            color_of_edge[i] = c

        max_c = max(color_of_edge) if color_of_edge else 0
        return color_of_edge, max_c + 1

    def try_remove_color(color, conflict, target_color, m):
        """
        Try to eliminate target_color by reassigning its edges to other colors.
        Returns True if successful (mutates color in place).
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return True
        
        # Sort by fewest conflicts first (easier to reassign)
        degrees = [bin(conflict[i]).count('1') for i in target_edges]
        target_edges.sort(key=lambda i: bin(conflict[i]).count('1'))
        
        for idx in target_edges:
            neighbor_colors = get_neighbor_colors(conflict, color, idx)
            
            found = False
            # Try to find a color that is not used by neighbors
            # We can use any color from 0 to max_color-1
            max_c = max(color) + 1
            for c in range(max_c):
                if c == target_color:
                    continue
                if c not in neighbor_colors:
                    color[idx] = c
                    found = True
                    break
            
            if not found:
                return False
        
        return True

    def local_search(color, conflict, m, rng, max_iters=1000):
        """
        Local search: repeatedly try to remove the highest color.
        Uses reassignment and swap moves.
        """
        best = color[:]
        best_nc = max(best) + 1 if best else 0
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            cur_nc = max(best) + 1 if best else 0
            if cur_nc <= 1:
                break
            
            target = cur_nc - 1
            target_edges = [i for i in range(m) if best[i] == target]
            
            if not target_edges:
                continue
            
            # Try direct reassignment
            trial = best[:]
            if try_remove_color(trial, conflict, target, m):
                best = trial
                continue
            
            # Try swap-based moves
            # Pick a vertex in target class
            improved = False
            for v in target_edges:
                # Try to move v to a lower color c
                neighbor_colors = get_neighbor_colors(conflict, best, v)
                for c in range(target):
                    if c not in neighbor_colors:
                        # Can move directly
                        best[v] = c
                        improved = True
                        break
                if improved:
                    break
                
                # If cannot move directly, try swapping with a neighbor u of color c
                # where u can take target color
                if not improved:
                    mask = conflict[v]
                    for u in range(m):
                        if (mask >> u) & 1:
                            c_u = best[u]
                            if c_u == target:
                                continue # Both in target class, swapping doesn't help remove target
                            
                            # Check if v can take c_u
                            if c_u in neighbor_colors:
                                continue
                            
                            # Check if u can take target
                            # u's neighbors (excluding v, since v is moving away)
                            # We need to check if any neighbor of u (other than v) has color target
                            # Since v is the only one we're moving, and v has target color,
                            # we just need to ensure no OTHER neighbor of u has target color.
                            # Actually, if u is adjacent to v, and v has target color,
                            # when v moves away, u can take target color if no other neighbor has it.
                            
                            u_neighbor_colors = get_neighbor_colors(conflict, best, u)
                            # Remove v's color from consideration? No, v is changing.
                            # If we swap: v gets c_u, u gets target.
                            # Conflict check for u taking target:
                            # u must not be adjacent to any vertex (other than v) with color target.
                            # Since v is adjacent to u and has color target, we ignore v.
                            # Are there other vertices with color target adjacent to u?
                            ok_u = True
                            for w in range(m):
                                if w == v: continue
                                if (conflict[u] >> w) & 1:
                                    if best[w] == target:
                                        ok_u = False
                                        break
                            
                            if ok_u:
                                # Perform swap
                                best[v] = c_u
                                best[u] = target
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                # Random perturbation
                if target_edges:
                    v = rng.choice(target_edges)
                    neighbor_colors = get_neighbor_colors(conflict, best, v)
                    available = [c for c in range(max(best)+1) if c != target and c not in neighbor_colors]
                    if available:
                        best[v] = rng.choice(available)
                        # Retry remove
                        trial = best[:]
                        if try_remove_color(trial, conflict, target, m):
                            best = trial

        return best

    def color_graph(edges, adj, conflicts, rng):
        """Try multiple strategies and return the best coloring."""
        m = len(edges)
        if m == 0:
            return [], 0

        best_colors = None
        best_score = float('inf')

        # Try DSATUR
        dsat_colors, dsat_score = dsatur_greedy(conflicts, m)
        if dsat_score < best_score:
            best_score = dsat_score
            best_colors = dsat_colors

        # Try greedy strategies
        strategies = ['largest_first', 'random', 'smallest_first', 'largest_random_tie']
        for strategy in strategies:
            colors, score = greedy_color(conflicts, m, rng, strategy)
            if score < best_score:
                best_score = score
                best_colors = colors

        # Local search improvement
        if best_colors is not None:
            improved_colors = local_search(best_colors, conflicts, m, rng, max_iters=200)
            improved_score = max(improved_colors) + 1 if improved_colors else 0
            if improved_score < best_score:
                best_score = improved_score
                best_colors = improved_colors

        return best_colors, best_score

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
        conflicts = build_conflict_bitmasks(edges, adj)
        graphs.append((edges, adj, conflicts))

    best_results = [[] for _ in range(len(graphs))]
    best_scores = [float('inf')] * len(graphs)

    # Run multiple iterations with different random seeds
    num_iterations = 0
    max_iterations = 5000  # Increased due to speedup

    while time.time() < deadline - 5.0:
        num_iterations += 1
        if num_iterations > max_iterations:
            break

        iter_seed = 10000 + num_iterations
        rng = random.Random(iter_seed)

        for idx, (edges, adj, conflicts) in enumerate(graphs):
            if time.time() > deadline - 2.0:
                break

            colors, score = color_graph(edges, adj, conflicts, rng)

            if score < best_scores[idx]:
                best_scores[idx] = score
                best_results[idx] = colors

    # Ensure all results are populated
    for i in range(len(graphs)):
        if not best_results[i] or best_scores[i] == float('inf'):
            edges, adj, conflicts = graphs[i]
            rng = random.Random(0)
            colors, _ = color_graph(edges, adj, conflicts, rng)
            best_results[i] = colors

    return best_results

