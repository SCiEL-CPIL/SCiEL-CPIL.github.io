"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Degrees
    deg = [len(adj[i]) for i in range(n)]
    
    # Order vertices by decreasing degree, then by index for determinism
    order = list(range(n))
    order.sort(key=lambda x: (-deg[x], x))
    
    # Reverse order index for quick lookup
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i
    
    def solve():
        # Try with increasing number of colors
        for num_colors in range(3, 11):
            result = _solve_with_k_colors(num_colors)
            if result is not None:
                return result
        return None
    
    def _solve_with_k_colors(k):
        # DSU with undo stack for each color pair
        # Pair index: (c1, c2) -> c1 * k + c2 for c1 < c2
        # We'll use a dictionary mapping (c1, c2) -> DSU state
        
        # For each color pair, maintain:
        # parent: dict vertex -> parent vertex
        # rank: dict vertex -> rank
        # We need undo capability, so we'll log operations
        
        class PairDSU:
            def __init__(self):
                self.parent = {}
                self.rank = {}
                self.log = []  # list of (operation, data) for undo
            
            def find(self, x):
                if x not in self.parent:
                    self.parent[x] = x
                    self.rank[x] = 0
                    return x
                # Path compression with logging would be complex
                # Use simple find without compression for undo-ability
                root = x
                while self.parent[root] != root:
                    root = self.parent[root]
                return root
            
            def union(self, x, y):
                rx = self.find(x)
                ry = self.find(y)
                if rx == ry:
                    return False  # Already in same component -> cycle
                # Union by rank
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                # Log the operation
                self.log.append(('union', ry, rx, self.parent[ry], self.rank[rx]))
                self.parent[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1
                return True
            
            def undo(self, count):
                for _ in range(count):
                    op = self.log.pop()
                    if op[0] == 'union':
                        _, ry, rx, old_parent_ry, old_rank_rx = op
                        self.parent[ry] = old_parent_ry
                        self.rank[rx] = old_rank_rx
            
            def same(self, x, y):
                return self.find(x) == self.find(y)
        
        # Create DSU for each color pair
        pair_dsus = {}
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                pair_dsus[(c1, c2)] = PairDSU()
        
        # Assignment array: -1 means unassigned
        assign = [-1] * n
        
        # Track how many unions were done per pair for undo
        pair_union_count = defaultdict(int)
        
        def assign_vertex(v, c):
            """Assign vertex v to color c, return list of (pair, success) for undo."""
            assign[v] = c
            unions_done = []
            for u in adj[v]:
                if assign[u] != -1:
                    cu = assign[u]
                    if cu == c:
                        # Conflict: adjacent same color
                        assign[v] = -1
                        return None
                    # Get pair
                    c_min = min(c, cu)
                    c_max = max(c, cu)
                    pair = (c_min, c_max)
                    dsu = pair_dsus[pair]
                    if dsu.union(v, u):
                        pair_union_count[pair] += 1
                        unions_done.append(pair)
                    else:
                        # Cycle detected
                        # Undo all unions
                        for p in reversed(unions_done):
                            pair_dsus[p].undo(1)
                            pair_union_count[p] -= 1
                        assign[v] = -1
                        return None
            return unions_done
        
        def unassign_vertex(v, unions_done):
            """Undo assignment of vertex v."""
            for p in reversed(unions_done):
                pair_dsus[p].undo(1)
                pair_union_count[p] -= 1
            assign[v] = -1
        
        # Backtracking with forward checking
        def backtrack(idx):
            if idx == len(order):
                return True
            
            v = order[idx]
            
            # Try colors in order of least constraining value
            # Sort colors by how many neighbors already have that color (ascending)
            color_counts = [0] * k
            for u in adj[v]:
                if assign[u] != -1:
                    color_counts[assign[u]] += 1
            
            # Try colors with fewest conflicts first
            color_order = sorted(range(k), key=lambda c: color_counts[c])
            
            for c in color_order:
                # Check if color c is valid (no neighbor has color c)
                valid = True
                for u in adj[v]:
                    if assign[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Try assigning
                unions_done = assign_vertex(v, c)
                if unions_done is not None:
                    if backtrack(idx + 1):
                        return True
                    unassign_vertex(v, unions_done)
            
            return False
        
        if backtrack(0):
            return assign
        return None
    
    result = solve()
    if result is None:
        # Fallback: return unique colors (always valid but not optimal)
        result = list(range(n))
    
    # Ensure all are Python ints
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
