import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        paths = []
        
        # Helper to find a simple path greedily
        def find_greedy_path(start_node):
            path = [start_node]
            visited_nodes = {start_node}
            while True:
                current = path[-1]
                # Find neighbors with available edges
                candidates = []
                for neighbor in adj[current]:
                    edge = tuple(sorted((current, neighbor)))
                    if edge in remaining_edges:
                        candidates.append(neighbor)
                
                if not candidates:
                    break
                
                # Choose neighbor with highest degree in remaining graph
                # to keep options open (Heuristic)
                def degree_count(node):
                    count = 0
                    for nb in adj[node]:
                        e = tuple(sorted((node, nb)))
                        if e in remaining_edges:
                            count += 1
                    return count
                
                candidates.sort(key=lambda x: degree_count(x), reverse=True)
                next_node = candidates[0]
                
                # Check if adding next_node keeps path simple
                # Since we track visited_nodes, we just ensure next_node is not visited
                if next_node in visited_nodes:
                    # If the only option is a visited node, we are stuck or must backtrack?
                    # For simple path, we cannot go to visited node.
                    # So if all candidates are visited, we stop.
                    # Wait, if next_node is visited, we can't go there.
                    # We should filter candidates to only unvisited nodes.
                    pass
                
                # Refine: Filter candidates to unvisited nodes
                valid_candidates = [c for c in candidates if c not in visited_nodes]
                if not valid_candidates:
                    break
                    
                next_node = valid_candidates[0]
                path.append(next_node)
                visited_nodes.add(next_node)
                
                # Remove the edge used
                edge = tuple(sorted((current, next_node)))
                remaining_edges.discard(edge)
            
            return path

        # Main loop
        while remaining_edges:
            if time.time() - start_time > time_limit:
                # Return best effort (though this shouldn't happen for small graphs)
                # We must return valid paths covering remaining edges?
                # If we timeout, we might have uncovered edges.
                # To be safe, add remaining edges as individual paths.
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            # Try to start from each node to find the longest path?
            # For efficiency, just pick the node with highest remaining degree
            best_path = []
            best_len = 0
            # Get nodes with remaining edges
            nodes_with_edges = set()
            for u, v in remaining_edges:
                nodes_with_edges.add(u)
                nodes_with_edges.add(v)
            
            # Sort nodes by degree to prioritize starting at high-degree nodes
            def get_degree(node):
                return sum(1 for e in remaining_edges if node in e)
            
            sorted_nodes = sorted(nodes_with_edges, key=get_degree, reverse=True)
            
            # Try a few start nodes to find a good path
            for start_node in sorted_nodes[:5]: # Try top 5
                # We need to temporarily remove edges to simulate the path?
                # Or just simulate.
                # Let's simulate on a copy of remaining_edges? No, too slow.
                # Let's just run the greedy algorithm which modifies remaining_edges?
                # No, we need to compare paths.
                
                # Let's just pick the best start node and run one greedy path.
                # If we want to optimize, we can try multiple starts.
                # Given the small size, let's try all start nodes if time permits.
                # But let's start with the highest degree node.
                pass
                
            # Simplified: Just start from the highest degree node
            if sorted_nodes:
                start_node = sorted_nodes[0]
                path = find_greedy_path(start_node)
                if len(path) > 1:
                    paths.append(path)
                else:
                    # If no path found (shouldn't happen if edges exist), 
                    # take one edge
                    u, v = next(iter(remaining_edges))
                    paths.append([u, v])
                    remaining_edges.discard((u, v))
            else:
                break
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
