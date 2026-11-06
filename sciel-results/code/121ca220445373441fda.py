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

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]
    
    def decompose_graph(edges):
        if not edges:
            return []
        
        remaining_edges = set(edges)
        pieces = []
        
        # Pre-compute adjacency for the full graph to speed up BFS?
        # No, the graph changes. We need to rebuild or update.
        # Given the small size, rebuilding adjacency each time or using the set directly is fine.
        
        def find_shortest_cycle():
            if not remaining_edges:
                return None
                
            best_cycle = None
            best_len = float('inf')
            
            # For each edge, check if it's part of a cycle
            for u, v in list(remaining_edges):
                # Temporarily remove edge (u, v)
                remaining_edges.discard((u, v))
                
                # BFS from u to v
                # If path found, cycle is path + (v, u)
                queue = deque([(u, [u])])
                visited = {u}
                found_path = None
                
                while queue:
                    node, path = queue.popleft()
                    if node == v:
                        found_path = path
                        break
                    
                    # Find neighbors of node in remaining_edges
                    # We need to iterate through remaining_edges to find neighbors?
                    # Or maintain an adjacency set.
                    # Let's maintain an adjacency set for speed.
                    pass # Placeholder for logic below
                    
                # Restore edge (u, v)
                remaining_edges.add((u, v))
                
                if found_path:
                    cycle = found_path + [v] # Wait, path ends at v?
                    # path is [u, ..., v]. Cycle is [u, ..., v, u].
                    # The edges are (u, p1), ..., (pk, v), (v, u).
                    # The cycle vertices are [u, p1, ..., v].
                    # Length is len(found_path) + 1 (for the closing edge)?
                    # No, the cycle is the list of vertices.
                    # If path is [u, a, v], cycle is [u, a, v].
                    # Edges: (u,a), (a,v), (v,u).
                    # Number of vertices = 3.
                    # So cycle = found_path.
                    # But wait, found_path ends at v.
                    # The cycle is [u, ..., v]. The closing edge is (v, u).
                    # So the cycle vertices are found_path.
                    # Length (number of edges) = len(found_path).
                    
                    if len(found_path) < best_len:
                        best_len = len(found_path)
                        best_cycle = list(found_path)
                        
            return best_cycle

        # The BFS above is inefficient because it scans remaining_edges to find neighbors.
        # Let's build an adjacency set for the current remaining_edges.
        
        def get_adjacency(edge_set):
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        while remaining_edges:
            if time.time() > deadline:
                break
                
            adj = get_adjacency(remaining_edges)
            
            best_cycle = None
            best_len = float('inf')
            
            for u, v in list(remaining_edges):
                # Remove edge (u, v) temporarily
                remaining_edges.discard((u, v))
                adj[u].discard(v)
                adj[v].discard(u)
                
                # BFS from u to v
                if v in adj[u]: # Should be false since we removed it, but check just in case
                    # Direct edge? No, we removed it.
                    pass
                    
                queue = deque([(u, [u])])
                visited = {u}
                found_path = None
                
                while queue:
                    node, path = queue.popleft()
                    if node == v:
                        found_path = path
                        break
                        
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))
                            
                # Restore edge (u, v)
                remaining_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
                if found_path:
                    # The cycle is found_path + [u]?
                    # path: u -> ... -> v
                    # Cycle: u -> ... -> v -> u
                    # Vertices in cycle: u, ..., v
                    # So the cycle list is found_path.
                    # But we need to ensure the closing edge (v, u) is valid.
                    # It was in remaining_edges, so yes.
                    
                    # Length is len(found_path)
                    # Wait, if path is [u, v], length 2?
                    # But (u, v) is an edge. A cycle of length 2 is not allowed in simple graphs?
                    # The problem says "cycle given by at least three distinct vertices".
                    # So we must ignore cycles of length 2 (which would mean parallel edges, not present here)
                    # or length 1 (self-loop, not present).
                    # The shortest possible cycle is 3.
                    
                    if len(found_path) >= 3:
                        if len(found_path) < best_len:
                            best_len = len(found_path)
                            best_cycle = list(found_path)
                            
            if best_cycle:
                pieces.append(best_cycle)
                # Remove edges of best_cycle
                for i in range(len(best_cycle) - 1):
                    u, v = best_cycle[i], best_cycle[i+1]
                    if (u, v) in remaining_edges:
                        remaining_edges.remove((u, v))
                    elif (v, u) in remaining_edges:
                        remaining_edges.remove((v, u))
                u, v = best_cycle[-1], best_cycle[0]
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                elif (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
            else:
                # No cycle found, remove all remaining edges
                for edge in remaining_edges:
                    pieces.append(list(edge))
                remaining_edges.clear()
                
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        # Time limit per graph
        time_limit = deadline - (deadline - start_time) * (i / len(graphs)) - 5.0 # Rough estimate
        # Simpler: just check global deadline inside the loop
        
        pieces = decompose_graph(edges)
        results.append(pieces)
        
        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break
            
    return results

if __name__ == "__main__":
    res = construct()
    # Quick sanity check
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)
        
        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break
                
        if valid and used_edges != edge_set:
            valid = False
            
        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
