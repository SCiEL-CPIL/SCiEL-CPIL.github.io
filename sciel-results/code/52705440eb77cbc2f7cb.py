"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Union-Find for bicolored cycle detection
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True
    
    def is_valid(coloring):
        """Check: adjacent vertices differ, and each pair-of-colors induced subgraph is a forest."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Collect all color pairs present in edges
        color_pairs = set()
        for u, v in edges:
            c1, c2 = coloring[u], coloring[v]
            if c1 < c2:
                color_pairs.add((c1, c2))
            elif c2 < c1:
                color_pairs.add((c2, c1))
        
        for c1, c2 in color_pairs:
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not uf.union(u, v):
                        return False
        return True
    
    def count_colors(coloring):
        return len(set(coloring))
    
    # Precompute a good vertex ordering for backtracking
    # Use degeneracy ordering or simple degree-based ordering
    # For this graph, let's use a simple heuristic: order by degree, then index
    degree = [len(adj[v]) for v in range(n)]
    
    # Try multiple ordering strategies
    def get_order(strategy):
        if strategy == 0:
            # By degree ascending
            return sorted(range(n), key=lambda v: (degree[v], v))
        elif strategy == 1:
            # By degree descending
            return sorted(range(n), key=lambda v: (-degree[v], v))
        elif strategy == 2:
            # Natural order
            return list(range(n))
        elif strategy == 3:
            # Reverse natural order
            return list(range(n-1, -1, -1))
        else:
            # BFS order from vertex 0
            visited = [False] * n
            order = []
            q = deque([0])
            visited[0] = True
            while q:
                u = q.popleft()
                order.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            return order
    
    # Backtracking search for k-coloring
    def backtrack_search(k, order):
        """Search for a valid k-coloring using backtracking with the given vertex order."""
        coloring = [-1] * n
        # For each color pair, track which vertices are connected
        # We'll use a simpler approach: just check validity at the end of each complete assignment
        # But that's too slow. Instead, we'll do incremental checking.
        
        # For efficiency, track for each pair of colors, the edges that use that pair
        # and use union-find incrementally
        # This is complex, so let's use a simpler but effective approach:
        # Assign colors in order, checking only local constraints (adjacency)
        # and doing full validation at the end.
        
        def valid_partial(coloring, v):
            """Check if assigning coloring[v] to vertex v is locally valid."""
            c = coloring[v]
            if c < 0:
                return True
            for u in adj[v]:
                if coloring[u] == c:
                    return False
            return True
        
        def solve(idx):
            if time.time() - start_time > DEADLINE:
                return False
            if idx == len(order):
                # Full solution found, validate bicolored forest constraint
                return is_valid(coloring)
            
            v = order[idx]
            for c in range(k):
                if not valid_partial(coloring, v) and coloring[v] != -1:
                    continue
                # Check if color c is already used by a neighbor
                neighbor_colors = set()
                for u in adj[v]:
                    if coloring[u] >= 0:
                        neighbor_colors.add(coloring[u])
                if c in neighbor_colors:
                    continue
                
                coloring[v] = c
                if solve(idx + 1):
                    return True
                coloring[v] = -1
            
            return False
        
        if solve(0):
            return coloring
        return None
    
    # Try to find the minimum number of colors
    best_coloring = None
    best_count = n + 1
    
    # First, try small k values with different orderings
    for k in range(2, 8):
        if time.time() - start_time > DEADLINE:
            break
        for strategy in range(5):
            if time.time() - start_time > DEADLINE:
                break
            order = get_order(strategy)
            coloring = backtrack_search(k, order)
            if coloring is not None:
                cnt = count_colors(coloring)
                if cnt < best_count:
                    best_count = cnt
                    best_coloring = coloring[:]
                # If we found a valid coloring with k colors, no need to try other strategies for this k
                break
        # If we found a valid k-coloring, we can stop (k is minimal)
        if best_count <= k:
            break
    
    # If backtracking didn't find a good solution, try heuristic approaches
    if best_count > 5:
        # Try greedy coloring with different orderings
        def greedy_coloring(max_colors, order):
            coloring = [-1] * n
            for v in order:
                used = set()
                for u in adj[v]:
                    if coloring[u] >= 0:
                        used.add(coloring[u])
                for c in range(max_colors):
                    if c not in used:
                        coloring[v] = c
                        break
                if coloring[v] == -1:
                    return None
            return coloring
        
        for max_colors in range(3, 8):
            if time.time() - start_time > DEADLINE:
                break
            for strategy in range(5):
                if time.time() - start_time > DEADLINE:
                    break
                order = get_order(strategy)
                coloring = greedy_coloring(max_colors, order)
                if coloring is not None and is_valid(coloring):
                    cnt = count_colors(coloring)
                    if cnt < best_count:
                        best_count = cnt
                        best_coloring = coloring[:]
                    break
    
    # If still no good solution, use a guaranteed valid coloring
    if best_coloring is None:
        # Color by vertex index (guaranteed valid: all adjacent vertices have different colors)
        best_coloring = [int(v) for v in range(n)]
    
    # Ensure all values are proper Python ints in 0..49
    result = [int(c) for c in best_coloring]
    for i in range(len(result)):
        if result[i] < 0:
            result[i] = 0
        if result[i] > 49:
            result[i] = 49
    
    # Final verification
    if not is_valid(result):
        # Last resort: unique color per vertex
        result = [int(v) for v in range(n)]
    
    return result
# EVOLVE-BLOCK-END
