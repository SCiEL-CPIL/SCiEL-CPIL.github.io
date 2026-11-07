"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try with 3 colors first, then 4, etc.
    for num_colors in [3, 4, 5, 6]:
        colors = list(range(num_colors))
        assignment = [-1] * n
        
        # Helper to check if adding edge (v, u) creates a cycle in subgraph of colors {c_v, c_u}
        # v is the newly assigned vertex with color c_v
        # u is a neighbor with already assigned color c_u
        def creates_cycle(v, c_v, u, c_u):
            # DFS from v to u in induced subgraph of {c_v, c_u}
            # We traverse only edges between vertices with colors in {c_v, c_u}
            # If u is reachable from v, a cycle is formed by adding (v,u)
            target = u
            start = v
            visited = set()
            stack = [start]
            visited.add(start)
            
            while stack:
                node = stack.pop()
                if node == target:
                    return True
                for nb in adj[node]:
                    if nb in visited:
                        continue
                    # Check if nb has a color in {c_v, c_u}
                    # Note: assignment[nb] must be valid (not -1) and in the set
                    if assignment[nb] in (c_v, c_u):
                        visited.add(nb)
                        stack.append(nb)
            return False

        def backtrack():
            # MRV: pick unassigned vertex with smallest domain
            min_domain_size = float('inf')
            min_vertex = -1
            for i in range(n):
                if assignment[i] == -1:
                    if len(domains[i]) < min_domain_size:
                        min_domain_size = len(domains[i])
                        min_vertex = i
                        if min_domain_size == 0:
                            return False
            
            if min_vertex == -1:
                return True
            
            v = min_vertex
            # Value Ordering: try colors that cause the fewest cycles/conflicts? 
            # For now, just iterate. 
            for c in domains[v]:
                # Try assigning color c to v
                assignment[v] = c
                
                # Check for bicolored cycles involving v
                valid = True
                for nb in adj[v]:
                    if assignment[nb] != -1:
                        # Check if adding edge (v, nb) creates a cycle in {c, assignment[nb]}
                        if creates_cycle(v, c, nb, assignment[nb]):
                            valid = False
                            break
                
                if not valid:
                    assignment[v] = -1
                    continue
                
                # Forward checking: update domains of unassigned neighbors
                saved_domains = {}
                for nb in adj[v]:
                    if assignment[nb] == -1:
                        saved_domains[nb] = domains[nb][:]
                        if c in domains[nb]:
                            domains[nb] = [x for x in domains[nb] if x != c]
                            if len(domains[nb]) == 0:
                                valid = False
                                break
                
                if not valid:
                    # Restore
                    for nb, d in saved_domains.items():
                        domains[nb] = d
                    assignment[v] = -1
                    continue
                
                if backtrack():
                    return True
                
                # Undo
                for nb, d in saved_domains.items():
                    domains[nb] = d
                assignment[v] = -1
            
            return False
        
        # Initialize domains
        domains = [list(colors) for _ in range(n)]
        
        # Symmetry breaking: fix first vertex to color 0
        assignment[0] = 0
        domains[0] = [0]
        for nb in adj[0]:
            domains[nb] = [c for c in domains[nb] if c != 0]
        
        if backtrack():
            # Verify solution
            result = list(assignment)
            for i in range(n):
                assert isinstance(result[i], int)
                assert 0 <= result[i] < 50
            return result
    
    # Fallback: use all distinct colors
    return list(range(n))
# EVOLVE-BLOCK-END
