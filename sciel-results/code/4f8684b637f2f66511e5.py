"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
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
    
    # Determine color pair indices
    def pair_idx(c1, c2, max_colors):
        a, b = min(c1, c2), max(c1, c2)
        return a * max_colors + b
    
    def check_validity(coloring, max_colors):
        """Check if coloring is valid."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        
        # Check bicolored subgraphs are forests
        color_pairs = set()
        for u, v in edges:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                if c1 < c2:
                    color_pairs.add((c1, c2))
                else:
                    color_pairs.add((c2, c1))
        
        for c1, c2 in color_pairs:
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not uf.union(u, v):
                        return False
        
        return True
    
    def backtrack(max_colors, order, deadline, rng):
        """
        Backtracking search with incremental DSU and checkpoint-based undo.
        Returns a valid coloring or None.
        """
        coloring = [-1] * n
        # For each color pair, maintain a DSU
        num_pairs = max_colors * (max_colors - 1) // 2
        dsus = [UnionFind(n) for _ in range(num_pairs)]
        
        def pair_to_idx(c1, c2):
            a, b = min(c1, c2), max(c1, c2)
            return a * max_colors + b
        
        # Precompute: for each vertex and each color, which pairs are affected
        # and which neighbors need to be checked
        # We'll compute this on the fly for simplicity
        
        def try_assign(v, c, depth):
            """
            Try to assign color c to vertex v.
            Returns list of (pair_idx, u, v) edges that were added, or None if conflict.
            """
            added_edges = []
            for u in adj[v]:
                d = coloring[u]
                if d != -1 and d != c:
                    pi = pair_to_idx(c, d)
                    # Check if adding edge (u, v) creates a cycle
                    if not dsus[pi].union(u, v):
                        # Undo what we've done so far
                        for (pi2, u2, v2) in added_edges:
                            # Can't easily undo union-find, so we need checkpoint approach
                            pass
                        return None
                    added_edges.append((pi, u, v))
            return added_edges
        
        # Since undoing union-find is hard, use checkpoint approach:
        # Save DSU state before each assignment, restore on backtrack.
        # For 50 vertices and up to 6 pairs, this is manageable.
        
        def save_dsu_state():
            """Save current DSU state."""
            state = []
            for uf in dsus:
                state.append((uf.parent[:], uf.rank[:]))
            return state
        
        def restore_dsu_state(state):
            """Restore DSU state from checkpoint."""
            for i, (parent, rank) in enumerate(state):
                dsus[i].parent = parent[:]
                dsus[i].rank = rank[:]
        
        def search(v_idx):
            """Recursive backtracking."""
            if time.time() > deadline:
                return False
            
            if v_idx == n:
                return True
            
            v = order[v_idx]
            
            # Determine available colors (not used by neighbors)
            neighbor_colors = set()
            for u in adj[v]:
                if coloring[u] != -1:
                    neighbor_colors.add(coloring[u])
            
            available = [c for c in range(max_colors) if c not in neighbor_colors]
            
            # Try colors in a heuristic order
            # Prefer colors that appear less in neighbors (less constraint)
            rng.shuffle(available)
            
            for c in available:
                # Save state
                saved = save_dsu_state()
                
                # Try assignment
                coloring[v] = c
                conflict = False
                added = []
                
                for u in adj[v]:
                    d = coloring[u]
                    if d != -1 and d != c:
                        pi = pair_to_idx(c, d)
                        if not dsus[pi].union(u, v):
                            conflict = True
                            break
                        added.append((pi, u, v))
                
                if not conflict:
                    if search(v_idx + 1):
                        return True
                
                # Restore state
                coloring[v] = -1
                restore_dsu_state(saved)
            
            return False
        
        if search(0):
            return coloring
        return None
    
    def greedy_coloring(order, max_colors, rng):
        """Greedy coloring in given order."""
        coloring = [-1] * n
        for v in order:
            neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
            available = [c for c in range(max_colors) if c not in neighbor_colors]
            if not available:
                return None
            coloring[v] = rng.choice(available)
        return coloring
    
    best_coloring = None
    best_count = n
    
    # Try backtracking with 3 colors
    for max_colors in [3, 4]:
        if time.time() > deadline - 5:
            break
        
        found = False
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(attempt * 7919 + max_colors)
            order = list(range(n))
            rng.shuffle(order)
            
            coloring = backtrack(max_colors, order, deadline - 5, rng)
            if coloring is not None:
                if check_validity(coloring, max_colors):
                    used = len(set(coloring))
                    if used < best_count:
                        best_count = used
                        best_coloring = coloring[:]
                    found = True
                    break
        
        if found:
            break
    
    # If backtracking didn't find a solution, try random greedy
    if best_coloring is None or best_count > 4:
        for max_colors in [3, 4, 5]:
            if time.time() > deadline - 2:
                break
            
            for attempt in range(2000):
                if time.time() > deadline - 2:
                    break
                
                rng = random.Random(100000 + attempt * 31 + max_colors)
                order = list(range(n))
                rng.shuffle(order)
                
                coloring = greedy_coloring(order, max_colors, rng)
                if coloring is not None and check_validity(coloring, max_colors):
                    used = len(set(coloring))
                    if used < best_count:
                        best_count = used
                        best_coloring = coloring[:]
                    if used <= max_colors:
                        break
            
            if best_coloring is not None and best_count <= max_colors:
                break
    
    # Fallback: use a valid coloring with more colors
    if best_coloring is None:
        # Use greedy with increasing colors
        for max_colors in [5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]:
            coloring = greedy_coloring(list(range(n)), max_colors, random.Random(42))
            if coloring is not None and check_validity(coloring, max_colors):
                best_coloring = coloring
                best_count = len(set(coloring))
                break
        
        if best_coloring is None:
            # Worst case: use all different colors
            best_coloring = list(range(n))
            best_count = n
    
    # Convert to proper Python ints
    result = [int(c) for c in best_coloring]
    
    # Verify one final time
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    
    # Final validity check
    for u, v in edges:
        assert result[u] != result[v], f"Adjacent vertices {u},{v} have same color {result[u]}"
    
    # Check bicolored forests
    for c1 in range(best_count):
        for c2 in range(c1 + 1, best_count):
            uf = UnionFind(n)
            for u, v in edges:
                cu, cv = result[u], result[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    assert uf.union(u, v), f"Bicolored cycle in colors {c1},{c2}"
    
    return result
# EVOLVE-BLOCK-END
