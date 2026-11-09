"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _has_bicolored_cycle(adj, colors, k):
    """
    Returns True if any pair of colors induces a cycle.
    Uses union-find with early exit.
    """
    n = P['n']
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # Collect nodes with color c1 or c2
            nodes = set()
            for i in range(n):
                if colors[i] == c1 or colors[i] == c2:
                    nodes.add(i)
            if len(nodes) < 3:
                continue
            # Union-find
            parent = {i: i for i in nodes}
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
            for u in nodes:
                for v in adj[u]:
                    if v in nodes and u < v:
                        if not union(u, v):
                            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85
    adj = _build_adjacency()
    n = P['n']
    
    # Build edge list for quick access
    edges = P['edges']
    
    # Try k from 3 upward
    for k in range(3, 8):
        if time.time() > deadline - 5:
            break
        
        # Backtracking with MRV and union-find undo journal
        colors = [-1] * n
        # For each color pair, maintain union-find state
        # uf_state[(c1,c2)] = (parent_dict, rank_dict, journal)
        # But we'll use a simpler approach: maintain per-pair DSU with undo
        
        # DSU with undo for each color pair
        # We'll track which edges are "active" for each pair
        # Actually, let's use a cleaner approach:
        # For each pair (c1,c2), maintain a DSU that only includes vertices colored c1 or c2
        # When we assign a vertex a color, we add edges to relevant DSUs
        
        # Simpler: use a global approach with careful bookkeeping
        # For each color pair, maintain:
        #   - set of vertices with that color pair
        #   - DSU parent/rank for those vertices
        #   - list of edges added (for undo)
        
        # Let's use a different strategy: 
        # Maintain for each pair (c1,c2) a DSU
        # When assigning vertex v to color c:
        #   For each neighbor u with color c2 != c:
        #     Add edge (u,v) to DSU for pair (c,c2)
        #     If this creates a cycle, it's invalid
        
        # DSU implementation with undo
        class DSUWithUndo:
            def __init__(self):
                self.parent = {}
                self.rank = {}
                self.journal = []  # list of (child, parent, rank_parent_before)
            
            def find(self, x):
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]
                    x = self.parent[x]
                return x
            
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return False
                # Union by rank
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                self.journal.append((ry, rx, self.rank[rx]))
                self.parent[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1
                return True
            
            def undo(self, n_steps=1):
                for _ in range(n_steps):
                    if not self.journal:
                        break
                    child, parent, rank_parent_before = self.journal.pop()
                    self.parent[child] = child
                    self.rank[parent] = rank_parent_before
            
            def add_node(self, x):
                if x not in self.parent:
                    self.parent[x] = x
                    self.rank[x] = 0
            
            def remove_node(self, x):
                # This is tricky; we'll handle via undo instead
                pass
        
        # For each color pair, we need a DSU
        # But we only have k colors, so at most k*(k-1)/2 pairs
        # Let's pre-create DSUs for all pairs
        
        # Actually, let's use a cleaner backtracking approach:
        # We'll maintain the coloring and check validity incrementally
        
        def backtrack(v_idx, remaining):
            if time.time() > deadline - 3:
                return False
            
            if v_idx == n:
                return True
            
            # MRV: pick the uncolored vertex with fewest available colors
            # For simplicity, use the next vertex in order but with some heuristic
            
            # Get available colors
            used_colors = set()
            for u in adj[v_idx]:
                if colors[u] != -1:
                    used_colors.add(colors[u])
            
            available = [c for c in range(k) if c not in used_colors]
            
            if not available:
                return False
            
            # Try each available color
            for c in available:
                # Check if assigning v_idx color c creates a bicolored cycle
                # For each neighbor u with color c2 != c:
                #   In the subgraph induced by colors {c, c2}, adding edge (v_idx, u)
                #   must not create a cycle
                
                valid = True
                affected_pairs = []
                
                for u in adj[v_idx]:
                    c2 = colors[u]
                    if c2 == -1 or c2 == c:
                        continue
                    # Check pair (c, c2)
                    # We need to verify that adding v_idx to this pair doesn't create a cycle
                    # This requires checking connectivity in the induced subgraph
                    
                    # For efficiency, we'll do a quick BFS/DFS check
                    # Actually, for small k this should be fast
                    
                    # Collect all vertices with color c or c2
                    # and check if v_idx and u are already connected in the subgraph
                    # induced by colors {c, c2} excluding the edge (v_idx, u)
                    
                    # Simpler: just do the full check for this pair
                    nodes = set()
                    for i in range(n):
                        ci = colors[i]
                        if ci == c or ci == c2:
                            nodes.add(i)
                    
                    # Add v_idx to nodes (it will have color c)
                    nodes.add(v_idx)
                    
                    # Check if there's a path from v_idx to u in the subgraph
                    # induced by nodes, NOT using the edge (v_idx, u)
                    # If there is, then adding edge (v_idx, u) creates a cycle
                    
                    # BFS from v_idx
                    visited = {v_idx}
                    queue = [v_idx]
                    found_path = False
                    while queue and not found_path:
                        cur = queue.pop(0)
                        for nb in adj[cur]:
                            if nb not in nodes:
                                continue
                            # Skip the edge (v_idx, u) since we're checking
                            # if there's an alternative path
                            if cur == v_idx and nb == u:
                                continue
                            if cur == u and nb == v_idx:
                                continue
                            if nb not in visited:
                                visited.add(nb)
                                if nb == u:
                                    found_path = True
                                    break
                                queue.append(nb)
                    
                    if found_path:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign color
                colors[v_idx] = c
                
                if backtrack(v_idx + 1, remaining - 1):
                    return True
                
                # Undo
                colors[v_idx] = -1
            
            return False
        
        # Start backtracking
        # Use a vertex ordering that might help
        # For now, use natural order
        if backtrack(0, n):
            # Verify the solution
            if all(colors[u] != colors[v] for u, v in edges):
                if not _has_bicolored_cycle(adj, colors, k):
                    return [int(c) for c in colors]
        
        # If backtracking failed or timed out, try random restarts with local search
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Local search
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Check if valid
                proper = all(colors[u] != colors[v] for u, v in edges)
                if proper and not _has_bicolored_cycle(adj, colors, k):
                    return [int(c) for c in colors]
                
                # Pick a random vertex
                v = random.randint(0, n - 1)
                old_color = colors[v]
                new_color = random.randint(0, k - 1)
                while new_color == old_color:
                    new_color = random.randint(0, k - 1)
                
                colors[v] = new_color
                
                # Accept if it doesn't make things worse
                # (simple hill climbing)
                # For now, just accept randomly with some probability
                if random.random() < 0.5:
                    colors[v] = old_color
    
    # Final fallback: use 4 colors with a pattern
    # Based on the graph structure, try a periodic coloring
    colors = [i % 4 for i in range(n)]
    # Fix any conflicts
    for _ in range(1000):
        if time.time() > deadline - 2:
            break
        proper = all(colors[u] != colors[v] for u, v in edges)
        if proper and not _has_bicolored_cycle(adj, colors, 4):
            return [int(c) for c in colors]
        # Find a conflict and fix it
        fixed = False
        for u, v in edges:
            if colors[u] == colors[v]:
                # Change v's color
                new_c = (colors[v] + 1) % 4
                if new_c != colors[u]:
                    colors[v] = new_c
                    fixed = True
                    break
        if not fixed:
            # Random perturbation
            v = random.randint(0, n - 1)
            colors[v] = random.randint(0, 3)
    
    # Ultimate fallback
    return list(range(n))
# EVOLVE-BLOCK-END
