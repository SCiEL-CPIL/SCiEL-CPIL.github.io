"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 140.0

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_2sat(n_vars, clauses):
        """Solve 2-SAT. Variables are 0..n_vars-1. Clauses are (lit1, lit2) where lit = 2*var or 2*var+1 (True/False)."""
        # Build implication graph
        # Node 2*var = var is False, node 2*var+1 = var is True
        # Clause (a, b) means: if not a then b, if not b then a
        # neg(lit) = lit ^ 1
        nodes = 2 * n_vars
        adj = [[] for _ in range(nodes)]
        
        for a, b in clauses:
            # (a OR b) => (not a => b) and (not b => a)
            adj[a ^ 1].append(b)
            adj[b ^ 1].append(a)
        
        # Kosaraju's algorithm for SCC
        visited = [False] * nodes
        order = []
        
        def dfs1(v):
            visited[v] = True
            for u in adj[v]:
                if not visited[u]:
                    dfs1(u)
            order.append(v)
        
        for i in range(nodes):
            if not visited[i]:
                dfs1(i)
        
        # Build reverse graph
        radj = [[] for _ in range(nodes)]
        for v in range(nodes):
            for u in adj[v]:
                radj[u].append(v)
        
        comp = [-1] * nodes
        comp_id = 0
        
        def dfs2(v, cid):
            comp[v] = cid
            for u in radj[v]:
                if comp[u] == -1:
                    dfs2(u, cid)
        
        for v in reversed(order):
            if comp[v] == -1:
                dfs2(v, comp_id)
                comp_id += 1
        
        # Check for contradiction
        for var in range(n_vars):
            if comp[2 * var] == comp[2 * var + 1]:
                return None  # Unsatisfiable
        
        # Extract assignment: var is True if comp[2*var+1] > comp[2*var]
        # (in Kosaraju, later components have higher IDs and come "later" in topological order)
        assignment = [False] * n_vars
        for var in range(n_vars):
            assignment[var] = comp[2 * var + 1] > comp[2 * var]
        
        return assignment

    def solve_instance(n, edges, lists, deadline):
        if time.time() > deadline:
            return None
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Bipartite coloring: color vertices as 0 or 1 based on bipartition
        # Use BFS from vertex 0
        side = [-1] * n
        side[0] = 0
        queue = [0]
        for i in range(len(queue)):
            v = queue[i]
            for u in adj[v]:
                if side[u] == -1:
                    side[u] = 1 - side[v]
                    queue.append(u)
        
        # Side 0 and Side 1 are the two bipartition sets
        side0 = [v for v in range(n) if side[v] == 0]
        side1 = [v for v in range(n) if side[v] == 1]
        
        # Greedily assign colors to side 0 first
        colour = [-1] * n
        for v in side0:
            # Pick the first available color from list
            colour[v] = lists[v][0]
        
        # For side 1 vertices, determine available colors
        # Each side1 vertex's neighbors are all in side0, which have been colored
        var_to_vertex = []  # variable index -> vertex in side1
        vertex_to_var = {}  # vertex -> variable index
        var_colors = []  # for each var, the two possible colors
        
        for v in side1:
            # Find colors in lists[v] not used by any neighbor
            used = set()
            for u in adj[v]:
                if colour[u] != -1:
                    used.add(colour[u])
            avail = [c for c in lists[v] if c not in used]
            
            if len(avail) == 0:
                # No valid color - should not happen by Alon-Tarsi, but handle gracefully
                return None
            elif len(avail) == 1:
                colour[v] = avail[0]
            else:
                # Two choices - create a variable
                var_idx = len(var_to_vertex)
                var_to_vertex.append(v)
                vertex_to_var[v] = var_idx
                var_colors.append(avail)  # [color_if_False, color_if_True]
        
        n_vars = len(var_to_vertex)
        
        if n_vars == 0:
            return colour
        
        # Build 2-SAT clauses
        # For each edge (u, v) where both u and v are in side1 (wait, no - side1 vertices only connect to side0)
        # Actually, in a bipartite graph, edges only go between side0 and side1
        # So two side1 vertices are never adjacent
        # 
        # Wait - let me reconsider. The constraints are:
        # For each edge (u, v): colour[u] != colour[v]
        # If u is in side0 and v is in side1:
        #   If v has a variable, we need: colour[v] != colour[u]
        #   colour[u] is fixed, so this eliminates one of v's two choices
        #
        # But we already filtered avail based on side0 neighbors!
        # So if a side1 vertex has 2 available colors, it means neither is used by its side0 neighbors.
        # 
        # However, we also need to check constraints between side1 vertices... but there are none!
        # In a bipartite graph, all edges are between the two sides.
        #
        # So actually, if we've already assigned side0 and filtered side1 domains based on side0,
        # then each side1 vertex is independent! There are no edges between side1 vertices.
        #
        # This means we don't even need 2-SAT! We can just pick any valid color for each side1 vertex.
        
        # But wait - what if a side0 vertex's color conflicts with multiple side1 vertices?
        # No, because we fixed side0 first, and each side1 vertex independently checks its neighbors.
        
        # So the problem decomposes! Just pick any available color for each side1 vertex.
        for v in side1:
            if colour[v] == -1:
                var_idx = vertex_to_var[v]
                # Pick the first available color
                colour[v] = var_colors[var_idx][0]
        
        return colour

    # Fallback: backtracking with forward checking
    def solve_backtrack(n, edges, lists, deadline):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        assign = [-1] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            v = order[idx]
            for c in lists[v]:
                conflict = False
                for nb in adj[v]:
                    if assign[nb] == c:
                        conflict = True
                        break
                if not conflict:
                    assign[v] = c
                    if backtrack(idx + 1):
                        return True
                    assign[v] = -1
            return False
        
        if backtrack(0):
            return assign
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Try the bipartite decomposition approach first
        result = solve_instance(n, edges, lists, deadline)
        
        if result is None:
            # Fallback to backtracking
            result = solve_backtrack(n, edges, lists, deadline)
        
        if result is None:
            # Final fallback: greedy
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            order = sorted(range(n), key=lambda v: -len(adj[v]))
            colour = [0] * n
            for v in order:
                taken = set()
                for nb in adj[v]:
                    taken.add(colour[nb])
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            result = colour
        
        out.append([int(c) for c in result])
    
    return out
# EVOLVE-BLOCK-END
