"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Map edge to index
    edge_set = set()
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_set.add((u, v))
        edge_set.add((v, u))
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    # We want to find a decomposition into edge-disjoint cycles.
    # Since the graph is 5-regular (degree 5 at each vertex), the maximum
    # number of edge-disjoint cycles is at most floor(5/2) = 2 if we require
    # each cycle to use every vertex (Hamiltonian). But the problem says
    # "All cycles must use exactly the same vertex set of size at least three".
    # This means each cycle must be a Hamiltonian cycle on all 21 vertices.
    # 
    # Wait, let me re-read: "All cycles must use exactly the same vertex set of size at least three in the pinned 21-vertex host."
    # This means all cycles have the same vertex set. Since the host has 21 vertices,
    # each cycle must be a Hamiltonian cycle on all 21 vertices.
    #
    # The graph is 5-regular. A Hamiltonian cycle uses 2 edges per vertex.
    # So we can have at most floor(5/2) = 2 edge-disjoint Hamiltonian cycles?
    # No, 2 cycles use 4 edges per vertex, leaving 1 edge per vertex.
    # Can we have a 3rd cycle? No, because 3 cycles would need 6 edges per vertex,
    # but we only have 5.
    #
    # So the maximum is 2 edge-disjoint Hamiltonian cycles.
    #
    # Let's verify: the seed returns [list(range(21))] which is 1 cycle.
    # If we can find 2 edge-disjoint Hamiltonian cycles, the score would be 2/1 = 2.0.
    
    # Strategy: Find two edge-disjoint Hamiltonian cycles.
    # 
    # Approach: 
    # 1. Find one Hamiltonian cycle.
    # 2. Remove its edges.
    # 3. Find a Hamiltonian cycle in the remaining graph (which should be 3-regular).
    
    # Let's use a backtracking approach to find Hamiltonian cycles.
    
    def find_hamiltonian_cycle(graph_adj, n, start=0):
        """Find a Hamiltonian cycle in the given graph using backtracking."""
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        def backtrack():
            if len(path) == n:
                # Check if last vertex is connected to start
                if start in graph_adj[path[-1]]:
                    return True
                return False
            
            last = path[-1]
            # Try neighbors in order
            for neighbor in sorted(graph_adj[last]):
                if not visited[neighbor]:
                    if len(path) == n - 1:
                        # If this is the last vertex, it must connect to start
                        if start not in graph_adj[neighbor]:
                            continue
                    
                    visited[neighbor] = True
                    path.append(neighbor)
                    if backtrack():
                        return True
                    path.pop()
                    visited[neighbor] = False
            
            return False
        
        if backtrack():
            return path
        return None
    
    # Find first Hamiltonian cycle
    adj1 = {v: set(neighs) for v, neighs in adj.items()}
    cycle1 = find_hamiltonian_cycle(adj1, n)
    
    if cycle1 is None:
        # Fallback: return the seed solution
        return [list(range(n))]
    
    # Remove edges of cycle1
    adj2 = {v: set(neighs) for v, neighs in adj.items()}
    for i in range(len(cycle1)):
        u = cycle1[i]
        v = cycle1[(i + 1) % len(cycle1)]
        adj2[u].discard(v)
        adj2[v].discard(u)
    
    # Find second Hamiltonian cycle in adj2
    cycle2 = find_hamiltonian_cycle(adj2, n)
    
    if cycle2 is None:
        # If we can't find a second cycle, return just the first one
        return [cycle1]
    
    # Return both cycles
    return [cycle1, cycle2]

# EVOLVE-BLOCK-END
