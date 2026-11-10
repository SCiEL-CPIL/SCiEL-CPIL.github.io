"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque

def construct():
    # Replicate the verifier's instance generator exactly.
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Decompose the edge set into cycles and single edges, minimizing the number of pieces.
        
        Strategy:
        1. Build a multigraph representation.
        2. Repeatedly find and remove cycles. We use a greedy heuristic to find long cycles
           by following paths in the remaining graph, preferring to close cycles when possible.
           This is equivalent to finding a maximum set of edge-disjoint cycles in a greedy manner.
           For dense random graphs, this should find many cycles.
        3. Remaining edges become single-edge pieces.
        
        To improve on pure greedy, we can try to find a 2-factor or near 2-factor decomposition
        by repeatedly finding cycles in the residual graph.
        """
        if not edges:
            return []
        
        # Build adjacency list with edge counts
        # Since edges are unique (simple graph), we can just track which edges remain
        remaining = set(edges)
        pieces = []
        
        # Build adjacency list for the current graph
        # We'll maintain a dynamic adjacency structure
        
        def build_adj(rem_edges):
            adj = defaultdict(set)
            for u, v in rem_edges:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        # Greedy cycle finding:
        # We repeatedly try to find a cycle in the remaining graph.
        # Use a heuristic: pick a random edge, then try to extend a path to form a cycle.
        
        adj = build_adj(remaining)
        
        # To make it deterministic and reproducible, we use a fixed random seed for the heuristic
        rng = random.Random(42)
        
        while remaining:
            # Find a cycle in the remaining graph
            cycle = None
            
            # Try to find a cycle by starting from various edges
            # For efficiency, we try to find long cycles
            edges_list = list(remaining)
            rng.shuffle(edges_list)
            
            for start_edge in edges_list:
                u, v = start_edge
                # Try to find a path from v back to u using edges other than (u,v)
                # This is a path-finding problem: find a path from v to u in G - {(u,v)}
                
                # Use BFS/DFS to find a path from v to u
                # We want the path to be as long as possible to maximize cycle length
                # But for simplicity, let's first find any cycle, then try to find longer ones
                
                # Actually, let's use a simpler approach: 
                # Repeatedly find any cycle using a modified DFS
                # Start from a random vertex, do DFS, if we find a back edge, we have a cycle
                
                # Let's try a different approach: 
                # Use the fact that in a connected component with edges, we can find a cycle
                # by doing DFS and detecting back edges.
                
                # For each connected component, find a cycle
                # Then remove it and repeat
                
                # Let's implement: find a cycle in the graph
                # We'll use DFS from each unvisited vertex
                
                visited = set()
                parent = {}
                cycle_found = None
                
                def dfs_cycle(node, start_node, adj_map):
                    visited.add(node)
                    for neighbor in sorted(adj_map[node]):  # Sorted for determinism
                        if neighbor == start_node and len(visited) >= 3:
                            # Found a cycle: node -> ... -> start_node
                            # Reconstruct the path
                            path = [node]
                            cur = node
                            while cur != start_node:
                                cur = parent[cur]
                                path.append(cur)
                            path.reverse()
                            return path
                        if neighbor not in visited:
                            parent[neighbor] = node
                            result = dfs_cycle(neighbor, start_node, adj_map)
                            if result:
                                return result
                    return None
                
                # Try to find a cycle starting from various vertices
                for start in sorted(adj.keys()):
                    if start in visited:
                        continue
                    # Reset for this DFS tree
                    visited = set()
                    parent = {}
                    cycle_found = dfs_cycle(start, start, adj)
                    if cycle_found:
                        break
                
                if cycle_found:
                    cycle = cycle_found
                    break
            
            if cycle and len(cycle) >= 3:
                # Extract the cycle edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    cycle_edges.add(edge)
                
                # Remove these edges from remaining
                for e in cycle_edges:
                    if e in remaining:
                        remaining.remove(e)
                
                # Update adjacency
                adj = build_adj(remaining)
                
                # Add the cycle as a piece
                pieces.append(cycle)
            else:
                # No more cycles; all remaining edges are single pieces
                for e in remaining:
                    pieces.append(list(e))
                break
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
