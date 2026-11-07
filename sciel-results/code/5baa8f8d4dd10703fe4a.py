import sys
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False # Cycle detected
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Variable ordering: sort vertices by degree descending, then index
    # This helps in backtracking to constrain early
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    # Map original index to position in order for easier access if needed, 
    # but we will just iterate through 'order'
    
    best_solution = None
    start_time = time.time()
    time_limit = 50.0 # Keep margin for 60s limit
    
    # We try to find a solution with k colors.
    # Start with k=3. If fails, try k=4, etc.
    # Note: The graph is 4-regular. Chromatic number is 2 (bipartite).
    # But bicolored cycle constraint is stronger.
    
    for k in range(3, 5): # Try 3, then 4
        if time.time() - start_time > time_limit:
            break
            
        coloring = [-1] * n
        # DSU for each pair of colors
        # pairs are (i, j) with i < j
        # We need to track DSU state for each pair
        # Since n is small, we can just re-check or maintain a list of DSUs
        # But maintaining DSUs with backtracking is tricky (need to undo).
        # Given n=50, a simple check "is the subgraph induced by {c1, c2} a forest?" 
        # is O(E) which is ~100. 
        # Backtracking depth 50. 50 * 100 * (branching factor) 
        # Branching factor is at most k.
        # With good ordering, it should be fast.
        
        # Let's use a recursive solver with full validation on leaf or incremental check.
        # Incremental check with DSU rollback is complex.
        # Simple check: after assigning a vertex, check if any pair involving its new color
        # has formed a cycle.
        
        def check_pair_forest(c1, c2):
            # Check if subgraph induced by colors c1 and c2 is a forest
            # Vertices with color c1 or c2
            # Edges between them
            # Use DFS/BFS to detect cycle
            # Or simply: count edges in induced subgraph vs vertices?
            # A graph is a forest iff E = V - components.
            # Or simpler: just run cycle detection on the induced subgraph.
            
            # Build list of vertices with c1 or c2
            verts = [i for i in range(n) if coloring[i] == c1 or coloring[i] == c2]
            if not verts:
                return True
            v_set = set(verts)
            
            # Build local adjacency for these vertices
            # And check for cycle
            visited = set()
            def dfs(u, parent):
                visited.add(u)
                for v in adj[u]:
                    if v in v_set:
                        if v == parent:
                            continue
                        if v in visited:
                            return False # Cycle found
                        if not dfs(v, u):
                            return False
                return True
            
            # We need to check all components
            # Actually, if we just want to know if ANY cycle exists, we can do this:
            # Iterate all vertices in v_set, if not visited, start DFS.
            # If DFS finds a back-edge, return False.
            
            # Optimization: only check pairs that are "active" (have at least one edge)
            # But for simplicity, we check all pairs involving the new color.
            
            for v in verts:
                if v not in visited:
                    if not dfs(v, -1):
                        return False
            return True

        def solve(idx):
            nonlocal best_solution
            if time.time() - start_time > time_limit:
                return False
            
            if idx == n:
                # Found a valid solution
                best_solution = coloring[:]
                return True
                
            v = order[idx]
            
            # Try each color
            for c in range(k):
                # Check if c is allowed (neighbors don't have c)
                valid = True
                for u in adj[v]:
                    if coloring[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                coloring[v] = c
                
                # Check if any pair involving c is now a forest
                # Pairs: (c, 0), (c, 1), ..., (c, k-1) excluding c itself
                ok = True
                for c2 in range(k):
                    if c2 == c:
                        continue
                    # Check pair (c, c2)
                    # We only need to check if the addition of v (color c) created a cycle
                    # in the subgraph induced by {c, c2}.
                    # The only new edges added to this subgraph are (v, u) where u has color c2.
                    # So we can just check if there is a path between v and any of its c2-colored neighbors
                    # within the subgraph induced by {c, c2} excluding v?
                    # No, simpler: just run the full check for pairs involving c.
                    # Since k is small (3 or 4), this is cheap.
                    
                    if not check_pair_forest(c, c2):
                        ok = False
                        break
                
                if ok:
                    if solve(idx + 1):
                        return True
                
                coloring[v] = -1
                
            return False
            
        # Start solving for this k
        if solve(0):
            # Found solution with k colors
            # Convert to list of ints
            return [int(x) for x in best_solution]
            
    # If no solution found with 3 or 4, or time ran out
    # Fallback: return a valid coloring with more colors if we have one?
    # We didn't store intermediate valid colorings with higher k if we failed lower k.
    # Let's try a greedy approach to get *some* valid coloring quickly if backtracking fails.
    # Or just return a trivial valid coloring (50 colors) to be safe?
    # But we want to minimize.
    # If backtracking didn't finish, we might have a partial solution.
    # Let's try a simple greedy assignment with 4 colors, checking validity.
    
    # Greedy with 4 colors
    coloring = [-1] * n
    for v in order:
        for c in range(4):
            valid = True
            for u in adj[v]:
                if coloring[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            coloring[v] = c
            ok = True
            for c2 in range(4):
                if c2 == c: continue
                if not check_pair_forest(c, c2):
                    ok = False
                    break
            if ok:
                break
            coloring[v] = -1
        if coloring[v] == -1:
            # Try more colors?
            for c in range(4, 10):
                 valid = True
                 for u in adj[v]:
                     if coloring[u] == c:
                         valid = False
                         break
                 if not valid: continue
                 coloring[v] = c
                 ok = True
                 for c2 in range(10):
                     if c2 == c: continue
                     if not check_pair_forest(c, c2):
                         ok = False
                         break
                 if ok:
                     break
                 coloring[v] = -1
                 
    if all(c != -1 for c in coloring):
        return [int(x) for x in coloring]
        
    # Final fallback: 50 colors
    return list(range(n))

# EVOLVE-BLOCK-END
