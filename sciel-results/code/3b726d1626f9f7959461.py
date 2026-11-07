import sys
import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Set of forbidden edges (tuples of sorted vertices)
    forbidden_edges = set()
    
    def find_cycle():
        # DFS to find a Hamiltonian cycle avoiding forbidden_edges
        # Start at 0
        start = 0
        path = [start]
        visited = {start}
        
        def dfs(u):
            if len(path) == n:
                # Check if u connects to start
                edge = tuple(sorted((u, start)))
                if edge not in forbidden_edges:
                    return True
                return False
            
            for v in adj[u]:
                if v in visited:
                    continue
                edge = tuple(sorted((u, v)))
                if edge in forbidden_edges:
                    continue
                
                # Heuristic: If v is not the start and v has no other unvisited neighbors
                # and we are not at the last step, this might be a dead end?
                # Simple pruning: check if v is isolated in the remaining graph
                # (excluding start if we are at the end)
                # For simplicity, just proceed. N=21 is small.
                
                visited.add(v)
                path.append(v)
                
                if dfs(v):
                    return True
                
                visited.remove(v)
                path.pop()
            
            return False
        
        if dfs(start):
            return path[:]
        return None

    cycles = []
    # We can have at most floor(70/21) = 3 Hamiltonian cycles
    max_cycles = 3
    
    for _ in range(max_cycles):
        cycle = find_cycle()
        if cycle is None:
            break
        cycles.append(cycle)
        # Mark edges in this cycle as forbidden
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            forbidden_edges.add(tuple(sorted((u, v))))

    if not cycles:
        # Fallback: return a single Hamiltonian cycle if the search failed for some reason?
        # The seed guarantees at least 1 exists.
        # But if our DFS is buggy or too slow, we should ensure we return at least 1 valid cycle.
        # Let's assume the DFS works. If it returns None immediately, something is wrong.
        # However, the seed returns range(n) which is a valid cycle.
        # Let's just return whatever we found. If 0, it's invalid.
        # To be safe, if 0 found, try a simple greedy or return the seed solution?
        # The seed solution is [0..20]. Let's verify if it's a valid cycle in the graph.
        # 0-1, 1-2, ..., 20-0.
        # Edge 20-0 exists.
        # So [0..20] is a valid cycle.
        # If our search finds 0, we return the seed solution to guarantee score >= 1.
        return [list(range(n))]
        
    return cycles
# EVOLVE-BLOCK-END
