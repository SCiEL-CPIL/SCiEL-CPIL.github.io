import time
import random
import heapq
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def max_clique(graph, vertices):
        """Find a maximum clique in the induced subgraph of 'vertices' using Bron-Kerbosch with pivot."""
        best = []
        
        def bronk(r, p, x):
            nonlocal best
            if not p and not x:
                if len(r) > len(best):
                    best = r.copy()
                return
            # Choose pivot u from p U x with max degree in p
            # For simplicity and speed in small graphs, just pick first or use heuristic
            # Here we use a simple heuristic: pick u in p U x that maximizes len(p & neighbors(u))
            all_pivot = list(p) + list(x)
            if not all_pivot:
                return
            
            # Simple pivot selection: first vertex in all_pivot
            # Better: pick u that maximizes intersection size
            # Given time constraints, a simple pick is okay, but let's try to be smart.
            # Since we are in a B&B, speed matters.
            # Let's just iterate over p in reverse order (heuristic: last added is often "central"?)
            # Actually, standard BK is fine for n=90.
            
            # Optimization: if len(r) + len(p) <= len(best), prune
            if len(r) + len(p) <= len(best):
                return

            # Pivot selection
            u = None
            max_deg = -1
            for v in all_pivot:
                # Count neighbors of v in p
                deg = sum(1 for w in p if w in graph[v])
                if deg > max_deg:
                    max_deg = deg
                    u = v
            
            # Iterate over p \ N(u)
            # If u is None, iterate all p
            if u is None:
                candidates = list(p)
            else:
                candidates = [v for v in p if v not in graph[u]]
            
            for v in candidates:
                bronk(r + [v], p & graph[v], x & graph[v])
                p.remove(v)
                x.add(v)

        bronk([], set(vertices), set())
        return best

    def greedy_cover(edges, graph, n, k):
        """
        Greedy heuristic: repeatedly pick a large clique, remove its edges.
        Returns list of cliques and count.
        """
        remaining_edges = set(edges)
        cliques = []
        vertices = set(range(n))
        
        while remaining_edges:
            # Find max clique in the graph induced by vertices that are endpoints of remaining edges
            # To be efficient, we can just find max clique in the whole graph, 
            # but it might not cover any remaining edges if we are not careful.
            # Better: Find max clique in the subgraph formed by endpoints of remaining edges.
            endpoints = set()
            for u, v in remaining_edges:
                endpoints.add(u)
                endpoints.add(v)
            
            if not endpoints:
                break
            
            # Find max clique in induced subgraph on endpoints
            # We need to restrict max_clique to only consider edges in remaining_edges?
            # No, a clique in the original graph is a clique in any subgraph.
            # But we want a clique that covers many remaining edges.
            # A clique C covers an edge (u,v) if u,v in C.
            # So we want a clique C that maximizes the number of edges in C that are in remaining_edges.
            
            # Heuristic: Find max clique in the "residual graph" where edges are remaining_edges.
            # Build residual graph
            res_graph = {v: set() for v in endpoints}
            for u, v in remaining_edges:
                res_graph[u].add(v)
                res_graph[v].add(u)
            
            # Run Max Clique on res_graph
            # Modify max_clique to take a graph dict
            best_c = []
            def bronk_res(r, p, x):
                nonlocal best_c
                if not p and not x:
                    if len(r) > len(best_c):
                        best_c = r.copy()
                    return
                if len(r) + len(p) <= len(best_c):
                    return
                
                all_pivot = list(p) + list(x)
                if not all_pivot:
                    return
                
                u = None
                max_deg = -1
                for v in all_pivot:
                    deg = len(p & res_graph[v])
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                
                if u is None:
                    candidates = list(p)
                else:
                    candidates = [v for v in p if v not in res_graph[u]]
                
                for v in candidates:
                    bronk_res(r + [v], p & res_graph[v], x & res_graph[v])
                    p.remove(v)
                    x.add(v)
            
            bronk_res([], set(endpoints), set())
            
            if not best_c:
                # Fallback: pick any edge
                u, v = next(iter(remaining_edges))
                cliques.append([u, v])
                remaining_edges.discard((u, v))
                remaining_edges.discard((v, u))
            else:
                cliques.append(best_c)
                # Remove all edges in this clique
                for i in range(len(best_c)):
                    for j in range(i + 1, len(best_c)):
                        u, v = best_c[i], best_c[j]
                        remaining_edges.discard((u, v))
                        remaining_edges.discard((v, u))
                        
        return cliques

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Initial solution via greedy
        init_cliques = greedy_cover(edges, adj, n, k)
        best_cliques = init_cliques
        best_count = len(best_cliques)
        
        # Branch and Bound
        # State: (uncovered_edges_set)
        # We will use a simple recursive B&B
        # To avoid deep recursion issues, we can use a stack if needed, but n=90 is small enough for recursion depth if we prune well.
        
        uncovered = set(edges)
        
        def bb(uncovered_edges, current_cliques):
            nonlocal best_cliques, best_count
            
            if time.time() > deadline:
                return
                
            if not uncovered_edges:
                if len(current_cliques) < best_count:
                    best_count = len(current_cliques)
                    best_cliques = current_cliques.copy()
                return
            
            # Lower Bound: Greedy packing of remaining edges
            # Simple LB: number of edges / max_edges_in_any_clique
            # Max edges in any clique is bounded by max_clique_size * (max_clique_size - 1) / 2
            # But we can just run a fast greedy to get a LB
            # Actually, a simple LB is len(uncovered_edges) / 1 (each clique covers at least 1 edge)
            # Better LB: Greedy cover of remaining edges
            # This is expensive. Let's use a simpler LB:
            # The maximum number of edges a single clique can cover is bounded by the size of the max clique in the residual graph.
            # Let's find max clique in residual graph to get a tight LB.
            
            endpoints = set()
            for u, v in uncovered_edges:
                endpoints.add(u)
                endpoints.add(v)
                
            if not endpoints:
                return
            
            # Build residual graph
            res_adj = {v: set() for v in endpoints}
            for u, v in uncovered_edges:
                res_adj[u].add(v)
                res_adj[v].add(u)
                
            # Find max clique in residual graph
            max_c = []
            def bronk_bb(r, p, x):
                nonlocal max_c
                if not p and not x:
                    if len(r) > len(max_c):
                        max_c = r.copy()
                    return
                if len(r) + len(p) <= len(max_c):
                    return
                all_pivot = list(p) + list(x)
                if not all_pivot:
                    return
                u = None
                max_deg = -1
                for v in all_pivot:
                    deg = len(p & res_adj[v])
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                if u is None:
                    candidates = list(p)
                else:
                    candidates = [v for v in p if v not in res_adj[u]]
                for v in candidates:
                    bronk_bb(r + [v], p & res_adj[v], x & res_adj[v])
                    p.remove(v)
                    x.add(v)
            
            bronk_bb([], set(endpoints), set())
            
            if not max_c:
                # No edges? Should have been handled
                return
                
            max_c_size = len(max_c)
            max_edges_per_clique = max_c_size * (max_c_size - 1) // 2
            if max_edges_per_clique == 0:
                max_edges_per_clique = 1 # At least one edge
            
            lb = (len(uncovered_edges) + max_edges_per_clique - 1) // max_edges_per_clique
            
            if len(current_cliques) + lb >= best_count:
                return
                
            # Branch: Pick a max clique from the residual graph
            # We already have max_c
            # But maybe a slightly smaller clique is better?
            # For simplicity, branch on the found max clique.
            # To improve, we could branch on multiple top cliques.
            
            new_cliques = current_cliques + [max_c]
            new_uncovered = set(uncovered_edges)
            for i in range(len(max_c)):
                for j in range(i + 1, len(max_c)):
                    u, v = max_c[i], max_c[j]
                    new_uncovered.discard((u, v))
                    new_uncovered.discard((v, u))
                    
            bb(new_uncovered, new_cliques)
            
            # Optional: Branch on other cliques if time permits
            # For now, single branch might be too greedy. 
            # We need to ensure we explore alternatives.
            # However, finding a max clique is a strong move.
            # If this doesn't work, we might need to branch on "include vertex v in next clique" style.
            # Given time limits, let's stick to the strong heuristic branch first.
            # If it's too slow or suboptimal, we might just return the greedy solution.
            
            # Actually, pure B&B on cliques is hard. 
            # Let's rely on the greedy initial solution being quite good 
            # and use B&B to try to beat it slightly.
            # If B&B is too slow, we return greedy.

        # Run B&B
        # To avoid hanging, we limit the depth or time.
        # The deadline guard inside bb handles time.
        bb(uncovered, [])
        
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback to seed if out of time
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        cliques = solve_instance(*spec)
        # Ensure types are correct
        # Vertices must be int
        # Cliques must be lists of ints
        # Every edge in exactly one clique
        # Every clique size >= 2
        
        # Validation
        valid = True
        edge_count = 0
        seen_edges = set()
        for c in cliques:
            if len(c) < 2:
                valid = False
                break
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    e = (c[i], c[j])
                    if e in seen_edges:
                        valid = False
                        break
                    seen_edges.add(e)
                    edge_count += 1
            if not valid:
                break
        
        # Check if all original edges are covered
        orig_edges = set(instance(*spec))
        if not valid or seen_edges != orig_edges:
            # Fallback to seed
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            results.append(cliques)

    return results

# Keep the required interface
def construct():
    start_time = time.time()
    deadline = 110.0
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def greedy_cover(edges, n):
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            endpoints = set()
            for u, v in remaining_edges:
                endpoints.add(u)
                endpoints.add(v)
            
            if not endpoints:
                break
            
            res_adj = {v: set() for v in endpoints}
            for u, v in remaining_edges:
                res_adj[u].add(v)
                res_adj[v].add(u)
            
            best_c = []
            def bronk_res(r, p, x):
                nonlocal best_c
                if not p and not x:
                    if len(r) > len(best_c):
                        best_c = r.copy()
                    return
                if len(r) + len(p) <= len(best_c):
                    return
                all_pivot = list(p) + list(x)
                if not all_pivot:
                    return
                u = None
                max_deg = -1
                for v in all_pivot:
                    deg = len(p & res_adj[v])
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                if u is None:
                    candidates = list(p)
                else:
                    candidates = [v for v in p if v not in res_adj[u]]
                for v in candidates:
                    bronk_res(r + [v], p & res_adj[v], x & res_adj[v])
                    p.remove(v)
                    x.add(v)
            
            bronk_res([], set(endpoints), set())
            
            if not best_c:
                u, v = next(iter(remaining_edges))
                cliques.append([u, v])
                remaining_edges.discard((u, v))
                remaining_edges.discard((v, u))
            else:
                cliques.append(best_c)
                for i in range(len(best_c)):
                    for j in range(i + 1, len(best_c)):
                        u, v = best_c[i], best_c[j]
                        remaining_edges.discard((u, v))
                        remaining_edges.discard((v, u))
                        
        return cliques

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Greedy solution
        init_cliques = greedy_cover(edges, n)
        best_cliques = init_cliques
        best_count = len(best_cliques)
        
        # Try to improve with a simple local search or B&B if time permits
        # Given the complexity, we might just stick to the greedy solution 
        # which is already much better than the edge-by-edge baseline.
        # The baseline is sum of edges. Greedy covers many edges per clique.
        
        return best_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        cliques = solve_instance(*spec)
        
        # Validation
        valid = True
        seen_edges = set()
        orig_edges = set(instance(*spec))
        
        for c in cliques:
            if len(c) < 2:
                valid = False
                break
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    e = (c[i], c[j])
                    if e in seen_edges:
                        valid = False
                        break
                    seen_edges.add(e)
            if not valid:
                break
        
        if not valid or seen_edges != orig_edges:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            results.append(cliques)

    return results
