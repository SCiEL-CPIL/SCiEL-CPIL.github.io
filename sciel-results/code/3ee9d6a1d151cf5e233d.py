"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge list as tuples for faster access
    edge_list = [(u, v) for u, v in edges]
    
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
    
    def check_validity(coloring):
        """Check if coloring is valid: adjacent vertices differ and bicolored subgraphs are forests."""
        for u, v in edge_list:
            if coloring[u] == coloring[v]:
                return False
        
        # Check bicolored subgraphs are forests
        color_pairs = set()
        for u, v in edge_list:
            c1, c2 = coloring[u], coloring[v]
            if c1 != c2:
                if c1 < c2:
                    color_pairs.add((c1, c2))
                else:
                    color_pairs.add((c2, c1))
        
        for c1, c2 in color_pairs:
            uf = UnionFind(n)
            for u, v in edge_list:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not uf.union(u, v):
                        return False
        
        return True
    
    def mrv_backtrack(max_colors, timeout=10.0):
        """
        MRV (Minimum Remaining Values) backtracking with incremental DSU cycle detection.
        Returns a valid coloring or None.
        """
        deadline_local = start_time + timeout
        coloring = [-1] * n
        assigned_count = 0
        
        # For each color pair, maintain DSU state
        # Key: (c1, c2) where c1 < c2
        dsu_parent = {}
        dsu_rank = {}
        
        def find(parent_arr, x):
            while parent_arr[x] != x:
                parent_arr[x] = parent_arr[parent_arr[x]]
                x = parent_arr[x]
            return x
        
        def union(parent_arr, rank_arr, x, y):
            px, py = find(parent_arr, x), find(parent_arr, y)
            if px == py:
                return False
            if rank_arr[px] < rank_arr[py]:
                px, py = py, px
            parent_arr[py] = px
            if rank_arr[px] == rank_arr[py]:
                rank_arr[px] += 1
            return True
        
        def get_dsu_key(c1, c2):
            return (min(c1, c2), max(c1, c2))
        
        def ensure_dsu(c1, c2):
            key = get_dsu_key(c1, c2)
            if key not in dsu_parent:
                dsu_parent[key] = list(range(n))
                dsu_rank[key] = [0] * n
        
        def backtrack():
            nonlocal assigned_count
            if time.time() > deadline_local:
                return False
            
            if assigned_count == n:
                return True
            
            # Find unassigned vertex with fewest available colors (MRV)
            best_v = -1
            best_available = None
            best_count = max_colors + 1
            
            for v in range(n):
                if coloring[v] != -1:
                    continue
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if len(available) < best_count:
                    best_count = len(available)
                    best_v = v
                    best_available = available
                    if best_count <= 1:
                        break
            
            if best_v == -1:
                # All assigned
                return True
            
            if not best_available:
                return False
            
            v = best_v
            
            for c in best_available:
                coloring[v] = c
                assigned_count += 1
                
                # Check if assigning color c to v creates any bicolored cycle
                conflict = False
                modified_keys = set()
                
                for u in adj[v]:
                    d = coloring[u]
                    if d != -1 and d != c:
                        key = get_dsu_key(c, d)
                        ensure_dsu(c, d)
                        modified_keys.add(key)
                        
                        if not union(dsu_parent[key], dsu_rank[key], u, v):
                            conflict = True
                            break
                
                if not conflict:
                    if backtrack():
                        return True
                
                # Undo
                coloring[v] = -1
                assigned_count -= 1
                # DSU undo is complex; we'll rely on the fact that if we found a solution,
                # we keep it. For failed branches, we need to restore DSU state.
                # Simplification: save/restore DSU state for modified keys
                # Actually, let's use a different approach: save full DSU state before modification
                
            return False
        
        # Use a simpler approach: save/restore entire DSU state
        def backtrack_v2():
            nonlocal assigned_count
            if time.time() > deadline_local:
                return False
            
            if assigned_count == n:
                return True
            
            # Find unassigned vertex with fewest available colors (MRV)
            best_v = -1
            best_available = None
            best_count = max_colors + 1
            
            for v in range(n):
                if coloring[v] != -1:
                    continue
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if len(available) < best_count:
                    best_count = len(available)
                    best_v = v
                    best_available = available
                    if best_count <= 1:
                        break
            
            if best_v == -1:
                return True
            
            if not best_available:
                return False
            
            v = best_v
            
            for c in best_available:
                coloring[v] = c
                assigned_count += 1
                
                # Save DSU state for all potentially affected keys
                affected_keys = set()
                for u in adj[v]:
                    d = coloring[u]
                    if d != -1 and d != c:
                        key = get_dsu_key(c, d)
                        ensure_dsu(c, d)
                        affected_keys.add(key)
                
                saved_state = {}
                for key in affected_keys:
                    saved_state[key] = (dsu_parent[key][:], dsu_rank[key][:])
                
                # Check for cycles
                conflict = False
                for u in adj[v]:
                    d = coloring[u]
                    if d != -1 and d != c:
                        key = get_dsu_key(c, d)
                        if not union(dsu_parent[key], dsu_rank[key], u, v):
                            conflict = True
                            break
                
                if not conflict:
                    if backtrack_v2():
                        return True
                
                # Undo
                coloring[v] = -1
                assigned_count -= 1
                for key, (sp, sr) in saved_state.items():
                    dsu_parent[key] = sp
                    dsu_rank[key] = sr
            
            return False
        
        if backtrack_v2():
            return coloring[:]
        return None
    
    def local_search(coloring, max_colors, timeout=5.0):
        """
        Local search: try to reduce number of distinct colors by recoloring vertices.
        """
        deadline_local = start_time + timeout
        current = coloring[:]
        current_colors = set(current)
        
        for _ in range(10000):
            if time.time() > deadline_local:
                break
            
            # Try to recolor a vertex to a color not currently used
            # to reduce the total number of distinct colors
            target_color = -1
            for c in range(max_colors):
                if c not in current_colors:
                    target_color = c
                    break
            
            if target_color == -1:
                break
            
            # Find a vertex that can be recolored to target_color
            improved = False
            for v in range(n):
                if current[v] == target_color:
                    continue
                
                # Check if v can be recolored to target_color
                neighbor_colors = set(current[u] for u in adj[v])
                if target_color in neighbor_colors:
                    continue
                
                # Check bicolored cycle constraint
                old_color = current[v]
                current[v] = target_color
                
                valid = True
                # Check all color pairs involving target_color
                for c in current_colors:
                    if c == target_color:
                        continue
                    uf = UnionFind(n)
                    for u, w in edge_list:
                        cu, cw = current[u], current[w]
                        if (cu == target_color and cw == c) or (cu == c and cw == target_color):
                            if not uf.union(u, w):
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    improved = True
                    current_colors = set(current)
                    break
                else:
                    current[v] = old_color
            
            if not improved:
                break
        
        return current
    
    best_coloring = None
    best_count = n + 1
    
    # Try MRV backtracking with 3 colors
    if time.time() < deadline:
        coloring = mrv_backtrack(3, timeout=15.0)
        if coloring is not None and check_validity(coloring):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Try MRV backtracking with 4 colors
    if best_count > 3 and time.time() < deadline:
        coloring = mrv_backtrack(4, timeout=15.0)
        if coloring is not None and check_validity(coloring):
            used = len(set(coloring))
            if used < best_count:
                best_count = used
                best_coloring = coloring[:]
    
    # Try local search to reduce colors from a valid 4-coloring
    if best_count > 3 and best_coloring is not None and time.time() < deadline:
        improved = local_search(best_coloring, 4, timeout=10.0)
        if check_validity(improved):
            used = len(set(improved))
            if used < best_count:
                best_count = used
                best_coloring = improved[:]
    
    # Try random greedy for 3 colors with MRV-like ordering
    if best_count > 3 and time.time() < deadline:
        for attempt in range(200):
            if time.time() > deadline:
                break
            random.seed(100000 + attempt)
            coloring = [-1] * n
            valid = True
            
            # Use dynamic ordering based on available colors
            remaining = set(range(n))
            while remaining:
                # Find vertex with fewest available colors
                best_v = None
                best_avail = None
                best_cnt = 4
                for v in remaining:
                    neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                    available = [c for c in range(3) if c not in neighbor_colors]
                    if len(available) < best_cnt:
                        best_cnt = len(available)
                        best_v = v
                        best_avail = available
                        if best_cnt <= 1:
                            break
                if best_v is None or not best_avail:
                    valid = False
                    break
                coloring[best_v] = random.choice(best_avail)
                remaining.discard(best_v)
            
            if valid and check_validity(coloring):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                if best_count <= 3:
                    break
    
    # Fallback: use greedy with more colors
    if best_coloring is None:
        for max_colors in [5, 6, 7, 8]:
            coloring = [-1] * n
            valid = True
            for v in range(n):
                neighbor_colors = set(coloring[u] for u in adj[v] if coloring[u] != -1)
                available = [c for c in range(max_colors) if c not in neighbor_colors]
                if not available:
                    valid = False
                    break
                coloring[v] = available[0]
            if valid and check_validity(coloring):
                used = len(set(coloring))
                if used < best_count:
                    best_count = used
                    best_coloring = coloring[:]
                break
    
    # Final fallback: assign unique colors
    if best_coloring is None:
        best_coloring = list(range(n))
        best_count = n
    
    # Convert to proper Python ints
    result = [int(c) for c in best_coloring]
    
    # Verify one final time
    for i, c in enumerate(result):
        assert isinstance(c, int) and not isinstance(c, bool)
        assert 0 <= c <= 49
    
    # Final validity check
    for u, v in edge_list:
        assert result[u] != result[v], f"Adjacent vertices {u},{v} have same color {result[u]}"
    
    # Check bicolored forests
    for c1 in range(best_count):
        for c2 in range(c1 + 1, best_count):
            uf = UnionFind(n)
            for u, v in edge_list:
                cu, cv = result[u], result[v]
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    assert uf.union(u, v), f"Bicolored cycle in colors {c1},{c2}"
    
    return result
# EVOLVE-BLOCK-END
