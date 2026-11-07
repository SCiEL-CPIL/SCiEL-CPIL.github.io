import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort vertices by degree (all degree 4 here, but for generality)
    # For this graph all have degree 4, so use natural order or a better heuristic
    # Let's try ordering by a greedy heuristic: process in an order that helps
    # Actually, let's use a fixed ordering based on the graph structure
    vertex_order = list(range(n))
    
    def solve(num_colors):
        """Try to find a valid coloring with num_colors colors."""
        if time.time() > deadline:
            return None
        
        # DSU for each color pair to detect cycles
        # pair_dsu[(c1,c2)] maps vertex -> parent in DSU
        # We'll track for each pair of colors, the DSU state
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Use iterative backtracking with forward checking
        # State: color assignment for each vertex
        assignment = [-1] * n
        
        # For each color pair, maintain DSU
        # dsu[pair_key] = {vertex: parent}
        # pair_key = (min(c1,c2), max(c1,c2))
        
        # Precompute all color pairs
        all_pairs = [(i, j) for i in range(num_colors) for j in range(i+1, num_colors)]
        
        # DSU implementation using dictionaries
        class DSU:
            def __init__(self):
                self.parent = {}
            
            def find(self, x):
                if x not in self.parent:
                    self.parent[x] = x
                    return x
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                rx = self.find(x)
                ry = self.find(y)
                if rx == ry:
                    return False  # Would create cycle
                self.parent[rx] = ry
                return True
            
            def copy(self):
                new_dsu = DSU()
                new_dsu.parent = dict(self.parent)
                return new_dsu
        
        # Backtracking search
        # Order vertices to improve search: use a heuristic
        # Try different orderings
        
        # For this specific graph, let's try a systematic approach
        # The graph is C(50, {1, 7}), which has a nice structure
        
        # Try a simple modular approach first for small num_colors
        if num_colors <= 10:
            # Try modular colorings: color[i] = (a*i + b) % num_colors
            for a in range(1, num_colors):
                for b in range(num_colors):
                    coloring = [(a * i + b) % num_colors for i in range(n)]
                    if _check_valid(coloring, edges, adj, n):
                        return coloring
            # Try other patterns
            # Try: color[i] = (i * a) % num_colors with different a
            # Already covered above
        
        # Full backtracking
        def backtrack(idx, dsu_states, assignment):
            if time.time() > deadline:
                return None
            
            if idx == n:
                return assignment[:]
            
            v = vertex_order[idx]
            
            for c in range(num_colors):
                # Check proper coloring: no neighbor has same color
                valid = True
                for nb in adj[v]:
                    nb_idx = vertex_order.index(nb)  # This is slow, let's precompute
                    if nb_idx < idx and assignment[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check forest constraint for each pair involving color c
                # For each other color c2 that is used, check pair (c, c2)
                new_dsu_states = {}
                can_assign = True
                for pair in all_pairs:
                    c1, c2 = pair
                    if c not in (c1, c2):
                        new_dsu_states[pair] = dsu_states[pair]
                        continue
                    other_c = c2 if c1 == c else c1
                    # Only need to check if other_c is actually used
                    # But we haven't assigned all vertices yet, so we check edges to already-colored neighbors
                    dsu_copy = dsu_states[pair].copy()
                    
                    # For each neighbor of v that has color other_c, we need to union v and nb in this pair's DSU
                    for nb in adj[v]:
                        if assignment[nb] == other_c:
                            if not dsu_copy.union(v, nb):
                                can_assign = False
                                break
                    if not can_assign:
                        break
                    new_dsu_states[pair] = dsu_copy
                
                if not can_assign:
                    continue
                
                assignment[v] = c
                result = backtrack(idx + 1, new_dsu_states, assignment)
                if result is not None:
                    return result
                assignment[v] = -1
            
            return None
        
        # Initialize DSU states
        initial_dsu = {pair: DSU() for pair in all_pairs}
        
        # Precompute index lookup
        idx_lookup = {v: i for i, v in enumerate(vertex_order)}
        
        # Modify backtrack to use idx_lookup
        def backtrack2(idx, dsu_states, assignment):
            if time.time() > deadline:
                return None
            
            if idx == n:
                return assignment[:]
            
            v = vertex_order[idx]
            
            for c in range(num_colors):
                # Check proper coloring
                valid = True
                for nb in adj[v]:
                    nb_idx = idx_lookup[nb]
                    if nb_idx < idx and assignment[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check forest constraint
                new_dsu_states = {}
                can_assign = True
                for pair in all_pairs:
                    c1, c2 = pair
                    if c not in (c1, c2):
                        new_dsu_states[pair] = dsu_states[pair]
                        continue
                    other_c = c2 if c1 == c else c1
                    dsu_copy = dsu_states[pair].copy()
                    
                    for nb in adj[v]:
                        if assignment[nb] == other_c:
                            if not dsu_copy.union(v, nb):
                                can_assign = False
                                break
                    if not can_assign:
                        break
                    new_dsu_states[pair] = dsu_copy
                
                if not can_assign:
                    continue
                
                assignment[v] = c
                result = backtrack2(idx + 1, new_dsu_states, assignment)
                if result is not None:
                    return result
                assignment[v] = -1
            
            return None
        
        result = backtrack2(0, initial_dsu, assignment)
        return result
    
    def _check_valid(coloring, edges, adj, n):
        """Check if a coloring is valid."""
        # Check proper
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check forest constraint for each color pair
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                # Build subgraph induced by vertices with color c1 or c2
                # Check if it's a forest
                # Use DSU
                dsu = {}
                def find(x):
                    if x not in dsu:
                        dsu[x] = x
                        return x
                    if dsu[x] != x:
                        dsu[x] = find(dsu[x])
                    return dsu[x]
                
                def union(x, y):
                    rx = find(x)
                    ry = find(y)
                    if rx == ry:
                        return False
                    dsu[rx] = ry
                    return True
                
                for u, v in edges:
                    if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                        if not union(u, v):
                            return False
        return True
    
    # Try increasing number of colors
    best_result = None
    for num_colors in range(2, 51):
        if time.time() > deadline:
            break
        result = solve(num_colors)
        if result is not None:
            # Verify
            if _check_valid(result, edges, adj, n):
                best_result = result
                break
    
    if best_result is None:
        # Fallback: use the trivial 50-color solution
        best_result = list(range(n))
    
    return [int(c) for c in best_result]
