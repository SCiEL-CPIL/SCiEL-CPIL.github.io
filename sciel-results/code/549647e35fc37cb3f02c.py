"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Global variables for backtracking to avoid re-creation overhead in closure
    vertex_order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def check_forest_constraint(coloring, v, c):
        """
        Checks if assigning color c to vertex v creates a cycle in any bicolored subgraph.
        Returns True if valid, False if cycle created.
        """
        for d in range(len(set(coloring))): # We iterate over colors 0..k-1, but only assigned ones matter
            if d == c:
                continue
            
            # Get neighbors of v with color c or d
            neigh_c = [u for u in adj[v] if coloring[u] == c]
            neigh_d = [u for u in adj[v] if coloring[u] == d]
            
            if not neigh_c and not neigh_d:
                continue
            
            # We need to check connectivity among these neighbors in the subgraph induced by {c, d}
            # excluding v for now, then see if adding v connects two components.
            # Actually, we check if any two neighbors are in the same component of G[{c,d} \ {v}]
            
            # Build set of vertices with color c or d
            cd_verts = set()
            for i in range(n):
                if coloring[i] == c or coloring[i] == d:
                    cd_verts.add(i)
            
            if len(cd_verts) <= 1:
                continue
            
            parent = {x: x for x in cd_verts}
            rank = {x: 0 for x in cd_verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # Add edges
            for u in cd_verts:
                for w in adj[u]:
                    if w in cd_verts and u < w:
                        union(u, w)
            
            # Check neighbors
            components = set()
            for u in neigh_c + neigh_d:
                if u in parent:
                    components.add(find(u))
                else:
                    # This shouldn't happen if u is in neigh_c/d
                    pass
            
            # If two neighbors are in the same component, adding v creates a cycle
            if len(components) < len(neigh_c + neigh_d):
                return False
        
        return True

    def backtrack(k, idx, coloring, node_limit):
        if idx == n:
            return True, coloring
        
        v = vertex_order[idx]
        
        # Try each color from 0 to k-1
        # Heuristic: try colors that appear least often in neighbors?
        for c in range(k):
            # Check proper coloring
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check forest constraint
            if not check_forest_constraint(coloring, v, c):
                continue
            
            coloring[v] = c
            res, col = backtrack(k, idx + 1, coloring, node_limit)
            if res:
                return True, col
            coloring[v] = -1
        
        return False, None

    # Try k=3 first with a node limit to avoid hanging if impossible
    # If 3 fails quickly, try 4
    node_limit_3 = 5000
    node_limit_4 = 100000
    
    # Implementing node count in backtrack requires modification. 
    # Simpler: Just run it. If it's too slow, it will timeout.
    # But we can add a simple counter.
    
    class State:
        def __init__(self):
            self.nodes = 0
            self.limit = 0
            self.found = False
            self.coloring = None

    def bt_with_limit(k, limit):
        state = State()
        state.limit = limit
        coloring = [-1] * n
        
        def bt(idx):
            if state.found:
                return
            state.nodes += 1
            if state.nodes > state.limit:
                return
            
            if idx == n:
                state.found = True
                state.coloring = coloring[:]
                return
            
            v = vertex_order[idx]
            
            for c in range(k):
                if state.found or state.nodes > state.limit:
                    return
                
                # Proper coloring check
                valid = True
                for u in adj[v]:
                    if coloring[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Forest constraint check
                if not check_forest_constraint(coloring, v, c):
                    continue
                
                coloring[v] = c
                bt(idx + 1)
                if not state.found:
                    coloring[v] = -1
        
        bt(0)
        if state.found:
            return state.coloring
        return None

    # Try 3 colors
    result = bt_with_limit(3, 10000)
    if result:
        return [int(c) for c in result]
        
    # Try 4 colors
    result = bt_with_limit(4, 50000)
    if result:
        return [int(c) for c in result]

    # Fallback: Greedy 5-coloring
    coloring = [-1] * n
    for v in vertex_order:
        used = set(coloring[u] for u in adj[v])
        for c in range(5):
            if c not in used:
                coloring[v] = c
                break
    
    # Verify and if invalid, use n colors
    # For safety, just return list(range(n)) if greedy fails (unlikely)
    # But greedy with 5 colors on 4-regular graph should work for proper coloring.
    # We need to check forest constraint.
    # If it fails, we might need more colors or a different strategy.
    # Given time, let's assume 4 works. If not, 5 likely works.
    
    # Simple validation for the result we return
    def is_valid(coloring):
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 < c2:
                    verts = [i for i in range(n) if coloring[i] in (c1, c2)]
                    if len(verts) <= 1: continue
                    vert_set = set(verts)
                    parent = {v: v for v in verts}
                    rank = {v: 0 for v in verts}
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    def union(x, y):
                        rx, ry = find(x), find(y)
                        if rx == ry: return False
                        if rank[rx] < rank[ry]: rx, ry = ry, rx
                        parent[ry] = rx
                        if rank[rx] == rank[ry]: rank[rx] += 1
                        return True
                    for u in verts:
                        for w in adj[u]:
                            if w in vert_set and u < w:
                                if not union(u, w): return False
        return True

    if result:
        if is_valid(result):
            return [int(c) for c in result]
            
    # If all else fails, use n colors
    return list(range(n))

# EVOLVE-BLOCK-END
