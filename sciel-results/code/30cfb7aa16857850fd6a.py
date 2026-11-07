"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _solve_with_k_colors(n, edges, k, deadline):
    """
    Fast backtracking solver with MRV dynamic vertex ordering and
    incremental bicolored-forest checking via union-find with rollback.
    """
    adj = _build_adjacency(n, edges)
    colors = [-1] * n
    deadline = deadline
    
    num_pairs = k * (k - 1) // 2
    
    pair_to_idx = {}
    pair_idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            pair_to_idx[(i, j)] = pair_idx
            pair_idx += 1
    
    class RollbackDSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.history = []
        
        def find(self, x):
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.history.append((ry, rx, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def rollback(self, count):
            for _ in range(count):
                child, parent, rank_parent_before = self.history.pop()
                self.parent[child] = child
                self.rank[parent] = rank_parent_before
    
    dsus = [RollbackDSU(n) for _ in range(num_pairs)]
    
    def get_pair_idx(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return pair_to_idx[(c1, c2)]
    
    def try_assign(v, color):
        assignments = []
        for nb in adj[v]:
            if colors[nb] == -1:
                continue
            if colors[nb] == color:
                return None
            pi = get_pair_idx(color, colors[nb])
            if dsus[pi].union(v, nb):
                assignments.append((pi, v, nb))
            else:
                for (pi2, u2, v2) in assignments:
                    dsus[pi2].rollback(1)
                return None
        return assignments
    
    def undo_assign(v, color, assignments):
        for (pi, u, v_edge) in assignments:
            dsus[pi].rollback(1)
    
    max_color_used = -1
    colored_count = [0]  # mutable counter for colored vertices
    
    def get_candidates(v, max_c):
        """Get valid color candidates for vertex v given current coloring."""
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        candidates = []
        for c in range(k):
            if c in forbidden:
                continue
            if max_color_used >= 0:
                if c <= max_color_used or c == max_color_used + 1:
                    candidates.append(c)
            else:
                if c == 0:
                    candidates.append(c)
        return candidates
    
    def backtrack():
        nonlocal max_color_used
        if time.time() > deadline:
            return False
        
        if colored_count[0] == n:
            return True
        
        # MRV: find uncolored vertex with fewest valid candidates
        best_v = -1
        best_cands = None
        best_count = k + 1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            cands = get_candidates(v, max_color_used)
            cnt = len(cands)
            if cnt < best_count:
                best_count = cnt
                best_v = v
                best_cands = cands
                if cnt <= 1:
                    break
        
        if best_v == -1:
            return True  # all colored
        
        if best_count == 0:
            return False  # dead end
        
        # Sort candidates: prefer colors already used (lower index first)
        best_cands.sort()
        
        for color in best_cands:
            assignments = try_assign(best_v, color)
            if assignments is None:
                continue
            
            colors[best_v] = color
            colored_count[0] += 1
            old_max = max_color_used
            if color > max_color_used:
                max_color_used = color
            
            if backtrack():
                return True
            
            colors[best_v] = -1
            colored_count[0] -= 1
            undo_assign(best_v, color, assignments)
            max_color_used = old_max
        
        return False
    
    if backtrack():
        return colors
    return None

def _verify_coloring(colors, n, edges):
    """Verify that the coloring is valid."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forest constraint
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        def find(x):
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
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    # Try increasing number of colors
    for k in range(3, 10):
        if time.time() > deadline:
            break
        result = _solve_with_k_colors(n, edges, k, deadline)
        if result is not None:
            # Verify
            if _verify_coloring(result, n, edges):
                return [int(c) for c in result]
    
    # Fallback: use a simple greedy with many colors
    # Assign unique colors to ensure validity
    return list(range(n))
# EVOLVE-BLOCK-END
