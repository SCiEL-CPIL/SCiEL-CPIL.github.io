"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_forest_full(colors):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    n = len(colors)
    color_to_verts = defaultdict(list)
    for i, c in enumerate(colors):
        color_to_verts[c].append(i)
    
    adj = _build_adj()
    colors_used = set(colors)
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            verts = set(color_to_verts[c1] + color_to_verts[c2])
            if len(verts) <= 1:
                continue
            parent = {v: v for v in verts}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            for u in verts:
                for v in adj[u]:
                    if v in verts and u < v:
                        if not union(u, v):
                            return False
    return True

class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []
        
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        # Record state for rollback
        # We need to restore parent[px] and rank[px] or parent[py] and rank[py]
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.history.append((px, py, self.parent[px], self.rank[px], self.rank[py]))
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def rollback(self, steps=1):
        for _ in range(steps):
            px, py, old_parent_px, old_rank_px, old_rank_py = self.history.pop()
            self.parent[py] = py # Reset py's parent to itself? No, it was changed to px.
            # Actually, we stored the state of px and py before the union.
            # But we only changed parent[py] and potentially rank[px].
            # So we restore parent[py] to py? No, it was py before? No, find(py) returned py.
            # So parent[py] was py. We set it to px.
            # So restoring parent[py] = py is correct.
            # And restoring rank[px] is correct.
            # What about rank[py]? It wasn't changed.
            self.parent[py] = py
            self.rank[px] = old_rank_px
            # self.rank[py] is unchanged, so no need to restore.
            # But wait, if we swapped px and py, the logic holds.
            # The history stores the "loser" py and the "winner" px.
            # So restoring parent[py] = py and rank[px] = old_rank_px is sufficient.

def greedy_with_reuse():
    n = P['n']
    adj = _build_adj()
    colors = [-1] * n
    ufs = {}
    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            ufs[(c1, c2)] = UF(n)
            
    for v in range(n):
        assigned = False
        for c in range(n):
            # Check if c is valid
            valid = True
            # Check adjacency
            for u in adj[v]:
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check forest property
            for c2 in range(n):
                if c2 == c:
                    continue
                key = (min(c, c2), max(c, c2))
                uf = ufs[key]
                neighbors_c2 = [u for u in adj[v] if colors[u] == c2]
                if len(neighbors_c2) >= 2:
                    roots = set()
                    for u in neighbors_c2:
                        root = uf.find(u)
                        if root in roots:
                            valid = False
                            break
                        roots.add(root)
                    if not valid:
                        break
                        
            if valid:
                # Assign c
                colors[v] = c
                for c2 in range(n):
                    if c2 == c:
                        continue
                    key = (min(c, c2), max(c, c2))
                    uf = ufs[key]
                    neighbors_c2 = [u for u in adj[v] if colors[u] == c2]
                    for u in neighbors_c2:
                        uf.union(v, u)
                assigned = True
                break
        
        if not assigned:
            # Assign a new unique color
            used_colors = set(colors)
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            
    # Remap
    color_map = {}
    remapped = []
    for c in colors:
        if c not in color_map:
            color_map[c] = len(color_map)
        remapped.append(color_map[c])
    return remapped

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    adj = _build_adj()
    
    # Order: Row-major is good for grids
    order = list(range(n))
    
    def try_coloring(k, order_seq):
        colors = [-1] * n
        pos_in_order = {v: i for i, v in enumerate(order_seq)}
        
        # Initialize UFs for all pairs of colors
        ufs = {}
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                ufs[(c1, c2)] = UF(n)
        
        def check_and_assign(v, c):
            # Check validity first
            for c2 in range(k):
                if c2 == c:
                    continue
                key = (min(c, c2), max(c, c2))
                uf = ufs[key]
                neighbors_c2 = [u for u in adj[v] if colors[u] == c2]
                if len(neighbors_c2) >= 2:
                    roots = set()
                    for u in neighbors_c2:
                        root = uf.find(u)
                        if root in roots:
                            return False
                        roots.add(root)
            
            # Perform unions
            for c2 in range(k):
                if c2 == c:
                    continue
                key = (min(c, c2), max(c, c2))
                uf = ufs[key]
                neighbors_c2 = [u for u in adj[v] if colors[u] == c2]
                for u in neighbors_c2:
                    uf.union(v, u)
            
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return colors[:]
            
            v = order_seq[idx]
            
            neighbor_colors = set()
            for u in adj[v]:
                if pos_in_order[u] < idx:
                    neighbor_colors.add(colors[u])
            
            for c in range(k):
                if c in neighbor_colors:
                    continue
                
                uf_hist_lengths = {key: len(uf.history) for key, uf in ufs.items()}
                
                if check_and_assign(v, c):
                    colors[v] = c
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    colors[v] = -1
                    
                    for key, uf in ufs.items():
                        while len(uf.history) > uf_hist_lengths[key]:
                            uf.rollback()
            
            return None
        
        return backtrack(0)

    # Try k=3, 4, 5 with time limits
    for k in [3, 4, 5]:
        if time.time() > deadline - 10:
            break
        result = try_coloring(k, order)
        if result is not None:
            final_colors = [0] * n
            for i, v in enumerate(order):
                final_colors[v] = result[i]
            # Verify
            if _check_bicolored_forest_full(final_colors):
                valid = True
                for u, v in P['edges']:
                    if final_colors[u] == final_colors[v]:
                        valid = False
                        break
                if valid:
                    color_map = {}
                    remapped = []
                    for c in final_colors:
                        if c not in color_map:
                            color_map[c] = len(color_map)
                        remapped.append(color_map[c])
                    return [int(x) for x in remapped]
            
    # Fallback: Greedy
    return greedy_with_reuse()
# EVOLVE-BLOCK-END
