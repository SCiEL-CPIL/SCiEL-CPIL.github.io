import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve_mvc(n, edges):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        active_vertices = set()
        for u, v in edges:
            active_vertices.add(u)
            active_vertices.add(v)
            
        if not active_vertices:
            return set()

        best_cover = set(active_vertices) # Initial upper bound: all vertices
        best_size = len(active_vertices)
        timed_out = False

        def greedy_matching_size(vertices, adj):
            if not vertices:
                return 0
            used = set()
            count = 0
            # Iterate vertices to find edges
            # To be fast, iterate edges of the subgraph
            # We don't have a subgraph edge list, so we iterate vertices and check neighbors
            for u in vertices:
                if u in used:
                    continue
                # Find an unused neighbor
                for v in adj[u]:
                    if v in vertices and v not in used:
                        used.add(u)
                        used.add(v)
                        count += 1
                        break
            return count

        def dfs(vertices, current_size, cover_so_far):
            nonlocal best_cover, best_size, timed_out
            
            if timed_out:
                return
            
            if time.time() - start_time > deadline:
                timed_out = True
                return

            if not vertices:
                if current_size < best_size:
                    best_size = current_size
                    best_cover = set(cover_so_far)
                return
            
            lb = greedy_matching_size(vertices, adj)
            if current_size + lb >= best_size:
                return
            
            # Pick vertex with max degree
            max_deg = -1
            v = None
            for u in vertices:
                deg = len(adj[u] & vertices)
                if deg > max_deg:
                    max_deg = deg
                    v = u
            
            if v is None:
                if current_size < best_size:
                    best_size = current_size
                    best_cover = set(cover_so_far)
                return

            neighbors = adj[v] & vertices
            
            # Branch 1: Include v
            new_vertices = vertices - {v} - neighbors
            new_cover = set(cover_so_far)
            new_cover.add(v)
            dfs(new_vertices, current_size + 1, new_cover)
            
            if timed_out:
                return

            # Branch 2: Exclude v
            new_vertices2 = vertices - {v} - neighbors
            new_cover2 = set(cover_so_far)
            new_cover2.update(neighbors)
            dfs(new_vertices2, current_size + len(neighbors), new_cover2)

        dfs(active_vertices, 0, set())
        
        return best_cover

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
        
        edges = instance(n, p, seed)
        cover = solve_mvc(n, edges)
        results.append(sorted(cover))
        
    return results
