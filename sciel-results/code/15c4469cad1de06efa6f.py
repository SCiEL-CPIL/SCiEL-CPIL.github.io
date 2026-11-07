"""Rewritten program with CSP backtracking for bicolored-forest coloring."""
import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)


class UnionFind:
    """Union-Find with path compression and union by rank."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n
    
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


def check_bicolored_forest(colors, k):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    color_to_verts = [[] for _ in range(k)]
    for i, c in enumerate(colors):
        if c != -1:
            color_to_verts[c].append(i)
    
    for a in range(k):
        for b in range(a + 1, k):
            verts = set(color_to_verts[a]) | set(color_to_verts[b])
            if len(verts) <= 1:
                continue
            
            # Build local index mapping
            vert_list = list(verts)
            vert_to_idx = {v: i for i, v in enumerate(vert_list)}
            uf = UnionFind(len(vert_list))
            
            for u in vert_list:
                for v in adj[u]:
                    if v in vert_to_idx and u < v:
                        if not uf.union(vert_to_idx[u], vert_to_idx[v]):
                            return False
    
    return True


def valid_coloring(colors):
    """Check all constraints: adjacent differ, and bicolored subgraphs are forests."""
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    
    k = max(colors) + 1 if colors else 0
    return check_bicolored_forest(colors, k)


def csp_solve(k, time_limit=5.0):
    """
    CSP backtracking with MRV heuristic and forward checking.
    Returns a valid coloring with at most k colors, or None.
    """
    start_time = time.time()
    colors = [-1] * N
    
    # Precompute edges for quick access
    def get_forbidden(v, colors):
        """Get forbidden colors for vertex v based on already-colored neighbors."""
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        return forbidden
    
    def check_partial(colors, v, c):
        """
        Check if assigning color c to vertex v creates a bicolored cycle.
        Only checks pairs involving c.
        """
        for other in range(k):
            if other == c:
                continue
            
            # Get vertices with color c or other
            verts = set()
            for i in range(N):
                if colors[i] in (c, other):
                    verts.add(i)
            verts.add(v)  # v is being assigned c
            
            if len(verts) <= 1:
                continue
            
            vert_list = list(verts)
            vert_to_idx = {v2: i for i, v2 in enumerate(vert_list)}
            uf = UnionFind(len(vert_list))
            
            for u in vert_list:
                for w in adj[u]:
                    if w in vert_to_idx and u < w:
                        if not uf.union(vert_to_idx[u], vert_to_idx[w]):
                            return False
        return True
    
    def backtrack(unassigned, depth):
        if time.time() - start_time > time_limit:
            return False
        
        if not unassigned:
            return True
        
        # MRV: pick unassigned vertex with fewest available colors
        best_v = None
        best_options = None
        best_count = float('inf')
        
        for v in unassigned:
            forbidden = get_forbidden(v, colors)
            options = [c for c in range(k) if c not in forbidden]
            if len(options) < best_count:
                best_count = len(options)
                best_v = v
                best_options = options
                if best_count == 0:
                    return False
        
        if best_v is None:
            return True
        
        # Try each option
        for c in best_options:
            colors[best_v] = c
            
            # Check bicolored forest constraint
            if check_partial(colors, best_v, c):
                new_unassigned = [v for v in unassigned if v != best_v]
                if backtrack(new_unassigned, depth + 1):
                    return True
            
            colors[best_v] = -1
        
        return False
    
    unassigned = list(range(N))
    if backtrack(unassigned, 0):
        return list(colors)
    
    return None


def local_search_eliminate(colors, time_limit=5.0):
    """Try to eliminate colors one by one using local search."""
    start_time = time.time()
    current = list(colors)
    current_max = max(current) + 1
    
    for target_color in range(current_max - 1, -1, -1):
        if time.time() - start_time > time_limit:
            break
        if target_color not in set(current):
            continue
        
        verts_to_recolor = [i for i in range(N) if current[i] == target_color]
        if not verts_to_recolor:
            continue
        
        # Try multiple orderings
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            new_current = list(current)
            verts_copy = list(verts_to_recolor)
            import random
            random.shuffle(verts_copy)
            
            success = True
            for v in verts_copy:
                recolored = False
                for c in range(target_color):
                    new_current[v] = c
                    if valid_coloring(new_current):
                        recolored = True
                        break
                if not recolored:
                    success = False
                    break
            
            if success:
                current = new_current
                current_max = target_color
                break
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_colors = list(range(N))  # Fallback: 50 colors
    best_count = N
    
    # Try to find colorings with few colors
    for k in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            colors = csp_solve(k, time_limit=5.0)
            if colors is not None:
                count = len(set(colors))
                if count < best_count:
                    best_count = count
                    best_colors = list(colors)
                
                # Try to eliminate more colors
                if time.time() < deadline - 15:
                    improved = local_search_eliminate(colors, time_limit=3.0)
                    improved_count = len(set(improved))
                    if improved_count < best_count:
                        best_count = improved_count
                        best_colors = list(improved)
                
                found = True
                if best_count <= 3:
                    break
        
        if best_count <= 3:
            break
    
    # Final verification
    if valid_coloring(best_colors):
        result = [int(c) for c in best_colors]
        return result
    else:
        # Fallback: return 50 distinct colors
        return list(range(N))
