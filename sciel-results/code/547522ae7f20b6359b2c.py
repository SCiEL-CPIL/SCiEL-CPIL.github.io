"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Map edges to indices
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Adjacency list with edge indices
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Colors: 0, 1, 2
    # State: color_of_edge[i] = -1 if unassigned, else 0, 1, 2
    color_of_edge = [-1] * m
    
    # Degree count per vertex per color
    deg = [[0] * 3 for _ in range(n)]
    
    # Track components for each color to check for premature cycles
    # We can maintain a union-find or just check connectivity of the subgraph for each color
    # Since n=21, we can just do a BFS/DFS on the subgraph of a specific color when needed.
    
    def check_color_validity(color):
        """
        Check if the subgraph of edges with the given color is a single Hamiltonian cycle.
        Returns True if it is valid or can still become valid (i.e., not yet closed into a smaller cycle).
        Actually, if it's not fully assigned, we just need to ensure no component is a closed cycle of length < n.
        """
        # Build adjacency for this color
        adj_color = [[] for _ in range(n)]
        for i in range(m):
            if color_of_edge[i] == color:
                u, v = edges[i]
                adj_color[u].append(v)
                adj_color[v].append(u)
                
        visited = [False] * n
        for start in range(n):
            if not visited[start]:
                # BFS/DFS to find component
                if deg[start][color] == 0:
                    # Isolated vertex, might be OK if not all edges assigned yet?
                    # But eventually it must have degree 2.
                    # If all edges incident to start are assigned and deg is 0, it's invalid.
                    # But we check deg constraints separately.
                    continue
                    
                queue = [start]
                visited[start] = True
                comp_nodes = [start]
                
                while queue:
                    curr = queue.pop(0)
                    for nb in adj_color[curr]:
                        if not visited[nb]:
                            visited[nb] = True
                            queue.append(nb)
                            comp_nodes.append(nb)
                            
                # Check if this component is a cycle
                # A cycle component has size > 1 and all nodes in it have degree 2 within the component
                # If size < n, and it's a cycle, it's invalid (can't extend to Hamiltonian)
                if len(comp_nodes) < n:
                    is_cycle = True
                    for node in comp_nodes:
                        if len(adj_color[node]) != 2:
                            is_cycle = False
                            break
                    if is_cycle:
                        return False
        return True

    def solve(idx):
        if idx == m:
            # Check if all colors form Hamiltonian cycles
            for c in range(3):
                if not check_color_validity(c):
                    return None
            # Also verify degrees are exactly 2
            for v in range(n):
                for c in range(3):
                    if deg[v][c] != 2:
                        return None
            # Extract cycles
            cycles = []
            for c in range(3):
                # Reconstruct cycle from edges of color c
                # Start at 0
                curr = 0
                cycle = [0]
                visited_edges = set()
                while True:
                    # Find next edge
                    next_node = -1
                    for nb, e_idx in adj[curr]:
                        if color_of_edge[e_idx] == c and e_idx not in visited_edges:
                            next_node = nb
                            visited_edges.add(e_idx)
                            break
                    if next_node == -1 or next_node == 0:
                        break
                    cycle.append(next_node)
                    curr = next_node
                if len(cycle) == n:
                    cycles.append(cycle)
                else:
                    return None
            return cycles

        u, v = edges[idx]
        
        # Symmetry breaking: For the first few edges, fix colors to reduce search space
        # Edge 0: (0,1). Fix color 0.
        if idx == 0:
            candidates = [0]
        elif idx == 1:
            # Edge 1: (0,2). Vertex 0 already has 1 edge of color 0.
            # Can be 0, 1, or 2?
            # If we fix edge 0 to 0, edge 1 can be 0, 1, 2.
            # Let's just iterate 0,1,2 but prune based on deg.
            candidates = [0, 1, 2]
        else:
            candidates = [0, 1, 2]
            
        for c in candidates:
            # Check degree constraints
            if deg[u][c] >= 2 or deg[v][c] >= 2:
                continue
                
            # Assign
            color_of_edge[idx] = c
            deg[u][c] += 1
            deg[v][c] += 1
            
            # Prune: Check if any color has formed a premature cycle
            valid = True
            # Only check the color we just assigned? Or all?
            # Checking all is safer but slower. Checking just c is usually sufficient for premature cycle detection.
            if not check_color_validity(c):
                valid = False
            else:
                # Also check if any other color is now "stuck" or invalid?
                # The degree constraints for other colors haven't changed, so their structural validity hasn't changed.
                pass
            
            if valid:
                result = solve(idx + 1)
                if result is not None:
                    return result
            
            # Unassign
            color_of_edge[idx] = -1
            deg[u][c] -= 1
            deg[v][c] -= 1
            
        return None

    # Optimization: Order edges to maximize pruning
    # Sort edges? The current order is fine, but maybe grouping by vertex helps.
    # Let's stick to the given order for simplicity, but the recursion depth is 63.
    # Python recursion limit might be an issue? 63 is small.
    
    # To speed up, we can add a heuristic:
    # If a vertex has degree 1 in a color, and only one edge left to assign for that vertex,
    # that edge MUST be that color.
    
    # Let's add a forward checking step before branching.
    # This might be complex. Let's try the basic CSP first.
    # Given the symmetry and small size, it should be fast.
    
    # One optimization: Fix the first 3 edges to break symmetry completely?
    # Edges incident to 0: (0,1), (0,2), (0,5), (0,16), (0,19), (0,20)
    # We need 2 edges of each color incident to 0.
    # So among these 6 edges, 2 are color 0, 2 are color 1, 2 are color 2.
    
    # Let's use a simpler symmetry break:
    # Edge (0,1) is color 0.
    # Edge (0,2) is color 0 or 1 or 2.
    
    # Actually, let's just run the solver.
    
    result = solve(0)
    
    if result:
        return [[int(v) for v in cycle] for cycle in result]
    else:
        # Fallback
        return [list(range(n))]

# EVOLVE-BLOCK-END
