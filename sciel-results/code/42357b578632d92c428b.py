"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 50  # 50 second deadline guard
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_solution = None
    best_colors = n  # worst case: all different colors
    
    # Try to find solution with k colors, starting from small k
    for k in range(3, n + 1):
        if time.time() > deadline:
            break
        solution = _solve_with_k_colors(n, adj, k, deadline)
        if solution is not None:
            distinct = len(set(solution))
            if distinct < best_colors:
                best_colors = distinct
                best_solution = solution
            break  # Found with k colors, try fewer? No, we're increasing k
    
    if best_solution is None:
        # Fallback: assign unique colors
        best_solution = list(range(n))
    
    return best_solution


def _solve_with_k_colors(n, adj, k, deadline):
    """Try to color graph with at most k colors satisfying constraints."""
    color = [-1] * n
    
    # DSU for each color pair to detect cycles
    # pair_dsu[(c1, c2)] = DSU for the bipartite subgraph of colors c1, c2
    # We'll use a dict mapping (c1,c2) -> dict of parent pointers
    
    def make_dsu():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False  # cycle detected
            parent[rx] = ry
            return True
        return find, union
    
    pair_dsus = {}
    
    def get_dsu(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in pair_dsus:
            pair_dsus[key] = make_dsu()
        return pair_dsus[key]
    
    # Order vertices by degree (all 4 here), then by index for determinism
    # Use a heuristic: process vertices in order of increasing flexibility
    vertex_order = list(range(n))
    
    # Backtracking search
    def backtrack(idx):
        if time.time() > deadline:
            return None
        
        if idx == n:
            return list(color)
        
        v = vertex_order[idx]
        
        # Try each color
        for c in range(k):
            # Check: no neighbor has same color
            valid = True
            for nb in adj[v]:
                if color[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check: for each neighbor with different color, adding edge (v, nb) 
            # must not create a cycle in the pair subgraph
            color[v] = c
            dsu_changes = []
            cycle_found = False
            for nb in adj[v]:
                cn = color[nb]
                if cn == c:
                    continue
                key = (min(c, cn), max(c, cn))
                find, union = get_dsu(key)
                # In the DSU, we track vertices. We need to check if adding
                # edge between v and nb creates a cycle.
                # Represent each vertex in the DSU by its id.
                if not union(v, nb):
                    cycle_found = True
                    break
                dsu_changes.append((key, v, nb))
            
            if cycle_found:
                # Undo DSU changes (simplified: we don't truly undo, but we 
                # need to track this. For simplicity, let's use a different approach)
                color[v] = -1
                continue
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            color[v] = -1
            # Undo DSU changes - this is complex. Let's use a simpler approach:
            # Instead of undoing, we'll use a different strategy.
        
        return None
    
    # The above DSU undo is complex. Let's use a simpler validation approach:
    # After assigning all colors, validate the entire solution.
    # Use greedy with lookahead.
    
    # Simpler approach: greedy assignment with validation at the end
    # and backtracking on conflicts
    
    # Let's use a cleaner backtracking with proper state management
    # Store DSU state as a snapshot
    
    def solve():
        color = [-1] * n
        # For cycle detection, we'll maintain for each pair of colors, 
        # the connected components using a simple approach
        
        # Actually, let's use a different strategy:
        # Greedily assign colors, and validate at the end.
        # If validation fails, try different assignments.
        
        # Vertex ordering: use degeneracy-like ordering
        # For 4-regular, all have same degree, so use natural order
        # but with some heuristic
        
        # Try multiple orderings
        orderings = [
            list(range(n)),
            sorted(range(n), key=lambda v: -len(adj[v])),  # all same
        ]
        
        for order in orderings:
            if time.time() > deadline:
                return None
            
            color = [-1] * n
            # Assign colors greedily
            for v in order:
                assigned = set(color[nb] for nb in adj[v] if color[nb] >= 0)
                for c in range(k):
                    if c not in assigned:
                        color[v] = c
                        break
                else:
                    # All colors used by neighbors, try next ordering
                    break
            else:
                # All vertices colored, validate
                if _validate(color, adj, n):
                    return color
        
        return None
    
    result = solve()
    if result is not None:
        return result
    return None


def _validate(color, adj, n):
    """Validate that the coloring satisfies all constraints."""
    # Check 1: adjacent vertices have different colors
    for u, v in P['edges']:
        if color[u] == color[v]:
            return False
    
    # Check 2: for each pair of colors, the induced subgraph is a forest
    # Build subgraph for each pair
    from collections import defaultdict
    
    # For each pair of colors (c1, c2), collect edges between vertices of those colors
    # and check if it's a forest (no cycles)
    
    # Group vertices by color
    color_to_vertices = defaultdict(list)
    for v in range(n):
        color_to_vertices[color[v]].append(v)
    
    # For each pair of colors, check if the bipartite subgraph is a forest
    colors_used = list(color_to_vertices.keys())
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            c1, c2 = colors_used[i], colors_used[j]
            v1_set = set(color_to_vertices[c1])
            v2_set = set(color_to_vertices[c2])
            
            # Build DSU for this pair
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                parent[rx] = ry
                return True
            
            # Add all edges between vertices of color c1 and c2
            for u, v in P['edges']:
                if (color[u] == c1 and color[v] == c2) or (color[u] == c2 and color[v] == c1):
                    if not union(u, v):
                        return False  # Cycle found
    
    return True


# EVOLVE-BLOCK-END
