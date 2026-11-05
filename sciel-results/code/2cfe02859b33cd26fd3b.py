import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        
        # Build Hyperedges
        hyperedges = []
        V_list = list(V_set)
        for i, a in enumerate(V_list):
            # a + a = 2a
            if 2 * a in V_set:
                hyperedges.append((a, a, 2 * a))
            for j in range(i + 1, len(V_list)):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    hyperedges.append((a, b, c))
        
        # Build adjacency: adj[v] = list of (u, w) such that {v, u, w} is a hyperedge.
        # For (a, a, 2a):
        #   v=a: (a, 2a)
        #   v=2a: (a, a)
        # For (a, b, c) with a!=b:
        #   v=a: (b, c)
        #   v=b: (a, c)
        #   v=c: (a, b)
        
        adj = {v: [] for v in V_set}
        for (a, b, c) in hyperedges:
            if a == b:
                adj[a].append((a, 2 * a))
                adj[2 * a].append((a, a))
            else:
                adj[a].append((b, c))
                adj[b].append((a, c))
                adj[c].append((a, b))
        
        # Precompute degree
        deg = {v: 0 for v in V_set}
        for (a, b, c) in hyperedges:
            deg[a] += 1
            deg[b] += 1
            deg[c] += 1

        def min_vertex_cover(graph_edges, vertices):
            """
            Find minimum vertex cover for a small graph.
            graph_edges: list of tuples (u, w). If u==w, it's a loop.
            vertices: set of vertices involved.
            Returns the set of vertices in the minimum cover.
            """
            # Convert to a set of edges for faster lookup
            # We represent the graph by its edges.
            # We use a recursive branching approach.
            
            # Base case: no edges
            if not graph_edges:
                return set()
            
            # If there is a loop (u, u), u MUST be in the cover.
            # Remove u and all edges incident to u.
            loops = [e[0] for e in graph_edges if e[0] == e[1]]
            if loops:
                # Pick one loop vertex
                v = loops[0]
                # v is in cover
                # Remove v from vertices
                # Remove all edges involving v
                new_edges = []
                for u, w in graph_edges:
                    if u != v and w != v:
                        new_edges.append((u, w))
                # Recurse
                cover = min_vertex_cover(new_edges, vertices - {v})
                cover.add(v)
                return cover
            
            # Pick a vertex with highest degree
            deg_in_graph = {v: 0 for v in vertices}
            for u, w in graph_edges:
                deg_in_graph[u] += 1
                deg_in_graph[w] += 1
                
            # If no vertices have edges (shouldn't happen if edges exist), return empty
            if not deg_in_graph or max(deg_in_graph.values()) == 0:
                return set()
                
            v = max(deg_in_graph, key=lambda x: deg_in_graph[x])
            
            # Branch 1: Include v in cover
            # Remove v and all edges incident to v
            edges_without_v = []
            for u, w in graph_edges:
                if u != v and w != v:
                    edges_without_v.append((u, w))
            vertices_without_v = vertices - {v}
            cover1 = min_vertex_cover(edges_without_v, vertices_without_v)
            cover1.add(v)
            
            # Branch 2: Exclude v from cover
            # Then all neighbors of v MUST be in cover
            neighbors = set()
            for u, w in graph_edges:
                if u == v:
                    neighbors.add(w)
                elif w == v:
                    neighbors.add(u)
            
            # Add neighbors to cover
            # Remove neighbors and v from graph
            # Edges remaining are those not incident to v or any neighbor
            removed = {v} | neighbors
            edges_remaining = []
            for u, w in graph_edges:
                if u not in removed and w not in removed:
                    edges_remaining.append((u, w))
            
            vertices_remaining = vertices - removed
            cover2 = min_vertex_cover(edges_remaining, vertices_remaining)
            cover2.update(neighbors)
            
            # Return the smaller cover
            if len(cover1) < len(cover2):
                return cover1
            else:
                return cover2

        def local_search(S, V_set, budget, rng):
            t_start = time.time()
            S = set(S)
            best_S = set(S)
            best_size = len(S)
            
            # Temperature for simulated annealing
            temp = 1.0
            
            while time.time() - t_start < budget:
                # Pick random v not in S
                remaining = [x for x in V_set if x not in S]
                if not remaining:
                    break
                v = rng.choice(remaining)
                
                # Get conflicts
                conflicts = []
                for (u, w) in adj[v]:
                    if u == v:
                        # v + v = w. Need w in S.
                        if w in S:
                            conflicts.append((w, w))
                    else:
                        # v + u = w or u + v = w. Need u in S and w in S.
                        if u in S and w in S:
                            conflicts.append((u, w))
                
                if not conflicts:
                    S.add(v)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    continue
                
                # Find minimum vertex cover for the conflict graph
                # Vertices in conflict graph are those in S that appear in conflicts
                conflict_vertices = set()
                for u, w in conflicts:
                    conflict_vertices.add(u)
                    conflict_vertices.add(w)
                
                # Build graph edges
                # Note: conflicts are already pairs (u, w)
                # We need to pass them to min_vertex_cover
                min_cover = min_vertex_cover(conflicts, conflict_vertices)
                
                # min_cover is the set of elements to remove from S
                # New set: S - min_cover + {v}
                new_S = set(S)
                new_S.difference_update(min_cover)
                new_S.add(v)
                
                gain = len(new_S) - len(S)
                
                # Acceptance criteria
                if gain > 0:
                    S = new_S
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                elif gain == 0:
                    # Accept with probability 0.3
                    if rng.random() < 0.3:
                        S = new_S
                else:
                    # Gain < 0. Accept with small probability (annealing)
                    # p = exp(-|gain| / temp)
                    p = random.expovariate(1.0) # Simple random check? No, use standard SA
                    # Let's use a fixed small probability for now
                    if rng.random() < 0.01:
                        S = new_S
                        
                # Cool down
                temp *= 0.999

            return best_S

        # Initial Solutions
        best_S = set()
        best_size = 0
        rng = random.Random(seed_off)
        
        # 1. Odds
        S_odds = set(x for x in V_set if x % 2 != 0)
        if len(S_odds) > best_size:
            best_size = len(S_odds)
            best_S = S_odds
            
        # 2. Greedy by Degree (fewest conflicts first)
        S_greedy = set()
        for v in sorted(V_set, key=lambda x: (deg[x], abs(x))):
            # Check if v can be added
            can_add = True
            for (u, w) in adj[v]:
                if u == v:
                    if w in S_greedy:
                        can_add = False
                        break
                else:
                    if u in S_greedy and w in S_greedy:
                        can_add = False
                        break
            if can_add:
                S_greedy.add(v)
                
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_S = S_greedy
            
        # 3. Local Search from best_S
        budget_ls = budget * 0.8
        best_S = local_search(best_S, V_set, budget_ls, rng)
        
        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [12, 30, 60]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        results.append(res)

    return results
