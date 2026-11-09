"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to lists for speed
    adj_list = [sorted(s) for s in adj]
    
    # Build edge list for bicolored cycle checking
    # For each pair of colors, we need to track which edges use that pair
    # A bicolored cycle exists if the subgraph induced by two colors has a cycle
    # i.e., edges >= vertices in that induced subgraph (for a connected component)
    # More precisely: for the subgraph with only vertices of color a or b,
    # a cycle exists iff #edges >= #vertices (in any connected component)
    
    # Strategy: backtracking with incremental bicolored forest check
    # When we assign color c to vertex v, we need to check all pairs (c, d)
    # where d is a color already used by a neighbor of v, to ensure no cycle forms
    
    # For efficiency, maintain for each color pair (a,b):
    # - the set of edges between vertices colored a or b
    # - detect cycle using union-find on those edges
    
    # Actually, let's use a simpler incremental approach:
    # When assigning color c to vertex v, for each neighbor u with color d != c:
    #   The edge (u,v) is now part of the (c,d) bicolored subgraph
    #   We need to check if adding this edge creates a cycle in the (c,d) subgraph
    #   This is equivalent to: are u and v already connected in the (c,d) subgraph?
    
    # Maintain union-find per color pair
    # color_pairs: dict mapping (min(a,b), max(a,b)) -> union-find structure
    # But union-find per pair is expensive to maintain incrementally
    
    # Alternative: For each color pair, maintain the edges and check connectivity
    # When we add an edge (u,v) to pair (c,d), check if u and v are connected
    # in the existing (c,d) subgraph. If yes, cycle formed -> backtrack.
    
    # Let's use a different approach: 
    # For each pair of colors (a,b), maintain a DSU (disjoint set union)
    # When we color vertex v with color c, for each neighbor u with color d:
    #   pair = (min(c,d), max(c,d))
    #   If u and v are already connected in DSU[pair], adding edge (u,v) creates cycle
    #   Otherwise, union u and v in DSU[pair]
    
    # But we also need to handle the case where v gets color c and we need to
    # connect v to all its neighbors with other colors in the appropriate DSUs
    
    # Let me think more carefully:
    # The bicolored subgraph for colors (a,b) consists of all edges where both
    # endpoints have color a or b. A cycle exists iff there's a cycle in this subgraph.
    # 
    # When we assign color c to vertex v:
    # For each neighbor u of v:
    #   If color[u] = d, then edge (u,v) is in the (min(c,d), max(c,d)) subgraph
    #   We need to check if this edge creates a cycle in that subgraph
    #   i.e., are u and v already connected via other edges in that subgraph?
    
    # For this, I'll maintain for each color pair a DSU over vertices.
    # When an edge (u,v) is added to pair (a,b)'s subgraph, we union u and v.
    # Before unioning, if u and v are already in the same set, a cycle is formed.
    
    # Data structures:
    # color[v] = current color of vertex v, or -1 if uncolored
    # pair_dsu: dict from (a,b) -> DSU structure
    # pair_edge_count: dict from (a,b) -> number of edges in that pair's subgraph
    
    # DSU implementation
    class DSU:
        def __init__(self):
            self.parent = {}
            self.rank = {}
        
        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                self.rank[x] = 0
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False  # already connected, would create cycle
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
    
    # Backtracking search
    color = [-1] * n
    pair_dsu = {}  # (a,b) -> DSU
    
    # Vertex ordering: use degree-based ordering for better pruning
    # Higher degree vertices first (they constrain more)
    order = sorted(range(n), key=lambda v: -len(adj_list[v]))
    # Map: position in order -> vertex, and vertex -> position
    pos_of = [0] * n
    for i, v in enumerate(order):
        pos_of[v] = i
    
    def get_pair_dsu(a, b):
        if a > b:
            a, b = b, a
        if (a, b) not in pair_dsu:
            pair_dsu[(a, b)] = DSU()
        return pair_dsu[(a, b)]
    
    def check_and_add_edge(u, v, cu, cv):
        """Check if adding edge (u,v) with colors (cu,cv) creates a bicolored cycle.
        Returns True if OK, False if cycle would be created."""
        if cu == cv:
            return False  # adjacent vertices can't have same color
        a, b = min(cu, cv), max(cu, cv)
        dsu = get_pair_dsu(a, b)
        # Check if u and v are already connected in this pair's subgraph
        ru, rv = dsu.find(u), dsu.find(v)
        if ru == rv:
            return False  # adding this edge would create a cycle
        # Union them (we'll undo this on backtrack)
        dsu.union(u, v)
        return True
    
    def undo_edge(u, v, cu, cv):
        """Undo the union for edge (u,v) with colors (cu,cv)."""
        if cu == cv:
            return
        a, b = min(cu, cv), max(cu, cv)
        dsu = get_pair_dsu(a, b)
        # We need to undo the union. This is tricky with standard DSU.
        # Alternative: don't actually union, just check connectivity.
        # But then we'd need to maintain connectivity differently.
        # 
        # Actually, for backtracking, it's hard to undo DSU operations.
        # Let's use a different approach: instead of incremental DSU,
        # we'll check for cycles by doing a BFS/DFS on the bicolored subgraph
        # when needed. But that's expensive.
        #
        # Better approach: use a DSU with rollback (persistent DSU)
        # Or: since n=50 is small, we can afford to rebuild check each time.
        #
        # Let me reconsider: for n=50, the bicolored subgraph for any pair
        # has at most 50 vertices and at most ~100 edges. Checking for a cycle
        # via BFS is O(V+E) which is very fast.
        #
        # New approach: when assigning color c to vertex v, for each neighbor
        # u with color d, check if the (c,d) bicolored subgraph has a cycle
        # that includes the new edge (u,v). This is equivalent to checking
        # if u and v are connected in the (c,d) subgraph WITHOUT the edge (u,v).
        #
        # For n=50, we can just do a BFS on the (c,d) subgraph each time.
        pass
    
    # Given the small size (n=50), let's use a simpler approach:
    # When assigning color c to vertex v, for each neighbor u with color d != c:
    #   Build the (c,d) bicolored subgraph and check if it has a cycle
    #   (specifically, check if adding edge (u,v) creates a cycle)
    #
    # Actually, even simpler: just check if the entire (c,d) subgraph (including
    # the new edge) has a cycle. We can do this by counting edges vs vertices
    # in each connected component, or by doing a DFS cycle detection.
    
    # For n=50, let's just do a full check after each assignment:
    # For all pairs of colors used, check if the induced subgraph has a cycle.
    # With at most 5 colors and 50 vertices, this is fast.
    
    def has_bicolored_cycle(coloring):
        """Check if any pair of colors induces a subgraph with a cycle."""
        used_colors = set(c for c in coloring if c >= 0)
        for i, ca in enumerate(used_colors):
            for cb in list(used_colors)[i+1:]:
                # Get all vertices with color ca or cb
                verts = [v for v in range(n) if coloring[v] in (ca, cb)]
                if len(verts) < 3:
                    continue  # need at least 3 vertices for a cycle
                # Build subgraph
                vert_set = set(verts)
                # Count edges in subgraph
                edge_count = 0
                for u in verts:
                    for v in adj_list[u]:
                        if v in vert_set and u < v:
                            edge_count += 1
                # A graph with V vertices and E edges has a cycle iff E >= V
                # (for the whole graph, if it's connected)
                # More precisely, a graph has a cycle iff E >= V for some component
                # Simple check: if E >= V, there's a cycle (necessary condition for forest is E < V)
                # Actually for a forest: E = V - (#components) <= V - 1 if connected
                # So if E >= V, definitely has a cycle
                # If E < V, might still have a cycle in a disconnected graph? No.
                # A forest has exactly V - C edges where C is number of components.
                # So E <= V - 1 always for a forest (when V >= 1).
                # If E >= V, there's definitely a cycle.
                # If E < V, we need to check more carefully? 
                # Actually no: a graph has no cycle iff it's a forest iff E = V - C for some C >= 1.
                # The maximum E for a forest on V vertices is V-1 (when C=1, i.e., tree).
                # So if E >= V, there's a cycle. If E < V, there might not be.
                # But E < V doesn't guarantee no cycle! Example: two disjoint cycles.
                # So we need to actually check for cycles.
                # 
                # Let's do a proper cycle check using BFS/DFS.
                # Build adjacency for the subgraph
                sub_adj = {v: [] for v in verts}
                for u in verts:
                    for v in adj_list[u]:
                        if v in vert_set:
                            sub_adj[u].append(v)
                # BFS to detect cycle
                visited = set()
                for start in verts:
                    if start in visited:
                        continue
                    # BFS
                    queue = [start]
                    visited.add(start)
                    parent = {start: -1}
                    has_cycle = False
                    while queue:
                        node = queue.pop(0)
                        for neighbor in sub_adj[node]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                parent[neighbor] = node
                                queue.append(neighbor)
                            elif neighbor != parent[node]:
                                has_cycle = True
                                break
                        if has_cycle:
                            break
                    if has_cycle:
                        return True
        return False
    
    # This full check is O(k^2 * (V+E)) where k is number of colors used.
    # For k <= 5 and V=50, E~100, this is very fast.
    
    # Now let's do backtracking with this check:
    best_result = None
    
    def backtrack(pos, colors_used):
        nonlocal best_result
        if time.time() > deadline:
            return False
        if pos == n:
            # All vertices colored
            distinct = len(set(color))
            if best_result is None or distinct < len(set(best_result)):
                best_result = list(color)
            return True
        
        v = order[pos]
        
        # Try colors in order of preference: try unused colors first, then used ones
        # Actually, for minimizing distinct colors, try used colors first
        candidate_colors = list(colors_used)
        # Add some new colors to try (to allow expanding)
        for c in range(10):  # try colors 0-9
            if c not in colors_used:
                candidate_colors.append(c)
        
        # Sort: try colors that lead to fewer conflicts first
        # Simple heuristic: try colors that are already used first
        for c in candidate_colors:
            if time.time() > deadline:
                return False
            
            # Check if c is valid for v (no neighbor has color c)
            valid = True
            for u in adj_list[v]:
                if color[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Assign
            color[v] = c
            new_colors_used = colors_used | {c}
            
            # Check for bicolored cycles
            if not has_bicolored_cycle(color):
                if backtrack(pos + 1, new_colors_used):
                    return True
            
            # Undo
            color[v] = -1
        
        return False
    
    # Start backtracking
    backtrack(0, set())
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback: return the trivial coloring (shouldn't happen)
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
