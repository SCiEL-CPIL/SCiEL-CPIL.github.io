import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_odd_vertices(adj):
        return [u for u in adj if len(adj[u]) % 2 != 0]

    def min_cost_flow_path_cover(edges, time_limit):
        """
        Formulate edge path cover as min-cost flow.
        
        For each undirected edge {u,v}, create two directed arcs u->v and v->u with capacity 1 and cost 1.
        Add a super-source S and super-sink T.
        For each vertex v:
          - If deg(v) is odd, we need exactly one path to start or end at v.
          - If deg(v) is even, v can be internal (or start/end of same path).
        
        The key insight: in a path decomposition, the number of paths equals 
        (number of odd-degree vertices) / 2 if we allow trails, but for simple paths
        it's more complex. However, min-cost flow can find a good solution.
        
        Actually, let me use a different approach: model this as finding minimum number
        of paths by using flow where each path corresponds to a unit of flow from S to T.
        
        Simpler approach: Use the fact that in any path cover, the number of paths k satisfies:
        k >= max(1, ceil(odd_count / 2)) where odd_count is the number of odd-degree vertices.
        
        Let me implement a practical heuristic based on flow-like ideas but simpler:
        1. Find all odd-degree vertices
        2. Pair them up and find paths between pairs (using BFS/DFS)
        3. For remaining edges, find cycles and break them into paths
        """
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Get odd-degree vertices
        odd_verts = [u for u in adj if len(adj[u]) % 2 != 0]
        n_odd = len(odd_verts)
        
        # Theoretical lower bound on number of paths
        if n_odd == 0:
            # Eulerian graph: can be one cycle, so one path (if we break cycle)
            # Actually one cycle = one path if we allow it
            min_paths = 1 if edges else 0
        else:
            # At least n_odd/2 paths needed
            min_paths = n_odd // 2
        
        # Build path cover using a strategy:
        # 1. If there are odd vertices, pair them and find paths between pairs
        # 2. Remaining edges form even-degree subgraph -> decompose into cycles
        # 3. Break cycles into paths as needed
        
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        # Step 1: Handle odd-degree vertices by finding paths between pairs
        if odd_verts:
            # Try to pair odd vertices and find paths between them
            # Use greedy pairing with shortest paths
            odd_set = list(odd_verts)
            rng = random.Random(42)
            rng.shuffle(odd_set)
            
            used_vertices_in_paths = set()
            
            for i in range(0, len(odd_set) - 1, 2):
                if time.time() > time_limit:
                    break
                u, v = odd_set[i], odd_set[i+1]
                
                # Find a simple path from u to v in remaining graph
                path = find_simple_path(u, v, adj, remaining_edges)
                if path:
                    paths.append(path)
                    # Remove edges from remaining
                    for j in range(len(path) - 1):
                        a, b = path[j], path[j+1]
                        edge = (min(a, b), max(a, b))
                        if edge in remaining_edges:
                            remaining_edges.remove(edge)
                            adj[a].remove(b)
                            adj[b].remove(a)
                    used_vertices_in_paths.update(path)
        
        # Step 2: Decompose remaining edges (which should have all even degrees) into cycles/paths
        while remaining_edges:
            if time.time() > time_limit:
                break
            
            # Find any edge
            start_edge = next(iter(remaining_edges))
            current = start_edge[0]
            path = [current]
            visited = {current}
            
            # Walk along edges to form a cycle or path
            while True:
                # Find an unused edge from current
                next_vertex = None
                for neighbor in list(adj[current]):
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in remaining_edges and neighbor not in visited:
                        next_vertex = neighbor
                        break
                
                if next_vertex is None:
                    # No more edges, check if we can close the cycle
                    for neighbor in list(adj[current]):
                        edge = (min(current, neighbor), max(current, neighbor))
                        if edge in remaining_edges and neighbor == path[0]:
                            next_vertex = neighbor
                            break
                
                if next_vertex is None:
                    break
                
                path.append(next_vertex)
                visited.add(next_vertex)
                
                # Remove the edge
                a, b = current, next_vertex
                edge = (min(a, b), max(a, b))
                remaining_edges.remove(edge)
                adj[a].remove(b)
                adj[b].remove(a)
                
                current = next_vertex
                
            if len(path) > 1:
                paths.append(path)
        
        return paths

    def find_simple_path(start, end, adj, remaining_edges):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]
        
        # BFS
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in adj[current]:
                edge = (min(current, neighbor), max(current, neighbor))
                if edge not in remaining_edges:
                    continue
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None

    def greedy_path_cover(edges, rng):
        """Fallback greedy path cover."""
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while remaining_edges:
            odd_verts = get_odd_vertices(adj)
            if odd_verts:
                start = rng.choice(odd_verts)
            else:
                candidates = [u for u in adj if adj[u]]
                if not candidates:
                    break
                start = rng.choice(candidates)
                
            path = [start]
            visited_vertices = {start}
            current = start
            
            while True:
                valid_neighbors = []
                for neighbor in adj[current]:
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in remaining_edges and neighbor not in visited_vertices:
                        valid_neighbors.append(neighbor)
                        
                if not valid_neighbors:
                    break
                
                next_v = rng.choice(valid_neighbors)
                path.append(next_v)
                visited_vertices.add(next_v)
                current = next_v
                
            if len(path) > 1:
                paths.append(path)
                
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
            if time.time() > deadline:
                break
                
        return paths

    def try_merge_paths(paths, edges_set):
        """Try to merge two paths if they share a vertex (making one longer path)."""
        # This is complex for simple paths; skip for now
        return paths

    def local_search_improve(paths, edges, time_limit):
        """Try to improve the path cover by local operations."""
        # For now, just return the input
        return paths

    def solve_graph(edges):
        best_cover = None
        best_count = float('inf')
        
        # Try min-cost flow based approach
        if time.time() < deadline - 5:
            cover = min_cost_flow_path_cover(edges, deadline - 10)
            if cover and len(cover) < best_count:
                best_count = len(cover)
                best_cover = cover
        
        # Try greedy with multiple seeds
        for seed in range(30):
            if time.time() > deadline - 5:
                break
            rng = random.Random(seed)
            cover = greedy_path_cover(edges, rng)
            
            count = len(cover)
            if count < best_count:
                best_count = count
                best_cover = cover
        
        if best_cover is None:
            # Fallback: trivial cover (each edge is a path)
            best_cover = [[list(e)] for e in edges]
            
        return best_cover

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        edges = instance(*spec)
        cover = solve_graph(edges)
        results.append(cover)
        
    return results
