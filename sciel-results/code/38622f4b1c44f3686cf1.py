"""Combined: deterministic backtracking + 2-SAT reduction + portfolio search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct():
    from random import Random

    def instance(splits, seed, palette=5):
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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def bipartition(adj, n):
        """Return (side0, side1) where side0+side1 = all vertices, no edges within each side."""
        color = [-1] * n
        side0 = []
        side1 = []
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            side0.append(start)
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        if color[v] == 0:
                            side0.append(v)
                        else:
                            side1.append(v)
                        q.append(v)
        return side0, side1, color

    def solve_2sat(n, adj, lists, fixed):
        """
        Try to solve via 2-SAT reduction.
        fixed: dict vertex -> color for pre-assigned vertices.
        Returns colour list or None if 2-SAT is unsatisfiable.
        
        Strategy: 
        - For vertices not fixed, if their list has exactly 2 colors, create a binary variable.
        - If list has 3 colors, we need to pick which 2 to use (try all 3 pairs).
        - Build implication graph and check satisfiability.
        """
        # Identify which vertices need binary decisions
        # First, propagate fixed colors
        colour = [None] * n
        for v, c in fixed.items():
            colour[v] = c
        
        # Check for immediate conflicts
        for v in range(n):
            if colour[v] is not None:
                if colour[v] not in lists[v]:
                    return None
                for u in adj[v]:
                    if colour[u] is not None and colour[u] == colour[v]:
                        return None
        
        # Determine variable vertices (those with exactly 2 available colors after propagation)
        # For vertices with 3 colors, we'll need to branch or pick
        var_vertex = {}  # vertex -> (c0, c1) where c0 < c1
        var_index = {}   # vertex -> index
        unassigned = []
        
        for v in range(n):
            if colour[v] is not None:
                continue
            # Available colors
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            avail = [c for c in lists[v] if c not in taken]
            if not avail:
                return None
            if len(avail) == 1:
                colour[v] = avail[0]
            elif len(avail) == 2:
                idx = len(var_index)
                var_vertex[v] = (avail[0], avail[1])
                var_index[v] = idx
                unassigned.append(v)
            else:
                # 3 colors available - this breaks simple 2-SAT
                # We'll handle by trying each color choice
                # For now, mark as needing branching
                return None  # Signal to use backtracking
        
        if not unassigned:
            # All assigned, verify
            for v in range(n):
                for u in adj[v]:
                    if colour[v] == colour[u]:
                        return None
            return colour
        
        num_vars = len(var_index)
        
        # Build 2-SAT implication graph
        # Variable i: vertex v_i, choices c0_i, c1_i
        # x_i = True means choose c1_i, x_i = False means choose c0_i
        # For each edge (v, u) where both are variables:
        #   if c_a == c_b, then NOT (x_v == x_u) i.e., x_v != x_u
        #   This gives: (x_v -> NOT x_u) and (NOT x_v -> x_u) etc.
        
        # Implication graph: 2 * num_vars nodes
        # Node 2*i = x_i is False (choose c0)
        # Node 2*i+1 = x_i is True (choose c1)
        # neg(2*i) = 2*i+1, neg(2*i+1) = 2*i
        
        graph = [[] for _ in range(2 * num_vars)]
        reverse_graph = [[] for _ in range(2 * num_vars)]
        
        def add_imp(a, b):
            graph[a].append(b)
            reverse_graph[b].append(a)
        
        # For each edge between two variable vertices
        for v in unassigned:
            iv = var_index[v]
            c0_v, c1_v = var_vertex[v]
            for u in adj[v]:
                if u not in var_index:
                    continue
                if u <= v:  # avoid duplicate
                    continue
                iu = var_index[u]
                c0_u, c1_u = var_vertex[u]
                
                # Constraint: colour[v] != colour[u]
                # (c0_v == c0_u) => x_v != x_u
                # (c0_v == c1_u) => x_v == x_u  
                # (c1_v == c0_u) => x_v != x_u
                # (c1_v == c1_u) => x_v == x_u
                
                # x_v != x_u means: (x_v -> NOT x_u) and (x_u -> NOT x_v)
                #                 i.e., (2*iv+1 -> 2*iu) and (2*iu+1 -> 2*iv)
                # x_v == x_u means: (x_v -> x_u) and (x_u -> x_v)
                #                 i.e., (2*iv+1 -> 2*iu+1) and (2*iu+1 -> 2*iv+1)
                #                 and (2*iv -> 2*iu) and (2*iu -> 2*iv)
                
                if c0_v == c0_u:
                    # x_v != x_u
                    add_imp(2*iv+1, 2*iu)
                    add_imp(2*iu+1, 2*iv)
                if c0_v == c1_u:
                    # x_v == x_u
                    add_imp(2*iv+1, 2*iu+1)
                    add_imp(2*iu+1, 2*iv+1)
                    add_imp(2*iv, 2*iu)
                    add_imp(2*iu, 2*iv)
                if c1_v == c0_u:
                    # x_v != x_u
                    add_imp(2*iv+1, 2*iu)
                    add_imp(2*iu+1, 2*iv)
                if c1_v == c1_u:
                    # x_v == x_u
                    add_imp(2*iv+1, 2*iu+1)
                    add_imp(2*iu+1, 2*iv+1)
                    add_imp(2*iv, 2*iu)
                    add_imp(2*iu, 2*iv)
        
        # Also add constraints from fixed neighbors
        for v in unassigned:
            iv = var_index[v]
            c0_v, c1_v = var_vertex[v]
            for u in adj[v]:
                if colour[u] is None:
                    continue
                cu = colour[u]
                # c0_v != cu and c1_v != cu must both hold
                # If c0_v == cu, then x_v must be True (choose c1_v)
                # If c1_v == cu, then x_v must be False (choose c0_v)
                # If both, contradiction
                if c0_v == cu and c1_v == cu:
                    return None
                if c0_v == cu:
                    # Force x_v = True
                    add_imp(2*iv, 2*iv+1)
                if c1_v == cu:
                    # Force x_v = False
                    add_imp(2*iv+1, 2*iv)
        
        # Check satisfiability via SCC (Kosaraju)
        visited = [False] * (2 * num_vars)
        order = []
        
        def dfs1(u):
            visited[u] = True
            for w in graph[u]:
                if not visited[w]:
                    dfs1(w)
            order.append(u)
        
        import sys
        sys.setrecursionlimit(100000)
        
        for i in range(2 * num_vars):
            if not visited[i]:
                dfs1(i)
        
        visited = [False] * (2 * num_vars)
        comp = [-1] * (2 * num_vars)
        comp_id = 0
        
        def dfs2(u, cid):
            visited[u] = True
            comp[u] = cid
            for w in reverse_graph[u]:
                if not visited[w]:
                    dfs2(w, cid)
        
        for u in reversed(order):
            if not visited[u]:
                dfs2(u, comp_id)
                comp_id += 1
        
        # Check: for any variable i, comp[2*i] != comp[2*i+1]
        for i in range(num_vars):
            if comp[2*i] == comp[2*i+1]:
                return None
        
        # Extract solution
        # x_i = True if comp[2*i+1] > comp[2*i] (in reverse topological order)
        for v in unassigned:
            iv = var_index[v]
            c0, c1 = var_vertex[v]
            if comp[2*iv+1] > comp[2*iv]:
                colour[v] = c1
            else:
                colour[v] = c0
        
        # Final verification
        for v in range(n):
            if colour[v] not in lists[v]:
                return None
            for u in adj[v]:
                if colour[v] == colour[u]:
                    return None
        
        return colour

    def solve_backtrack(n, adj, lists):
        """Deterministic backtracking with good heuristics."""
        import sys
        sys.setrecursionlimit(100000)
        
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        pos = {v: i for i, v in enumerate(order)}
        
        later = [[] for _ in range(n)]
        for v in range(n):
            pv = pos[v]
            for u in adj[v]:
                if pos[u] > pv:
                    later[v].append(u)
        
        colour = [None] * n
        domains = [set(lst) for lst in lists]
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = set()
            for u in adj[v]:
                cu = colour[u]
                if cu is not None:
                    used.add(cu)
            
            for c in sorted(domains[v] - used):
                colour[v] = c
                ok = True
                for u in later[v]:
                    used_u = set()
                    for w in adj[u]:
                        cw = colour[w]
                        if cw is not None:
                            used_u.add(cw)
                    if not (domains[u] - used_u):
                        ok = False
                        break
                if ok:
                    if backtrack(idx + 1):
                        return True
                colour[v] = None
            return False
        
        if backtrack(0):
            return colour
        # Fallback: fill with first available
        for v in range(n):
            if colour[v] is None:
                colour[v] = lists[v][0]
        return colour

    def solve_instance(n, adj, lists, deadline):
        """Try 2-SAT first, then backtracking."""
        # First, try to find a good fixed assignment to enable 2-SAT
        # Strategy: assign colors to a maximal independent set greedily
        
        side0, side1, bip_color = bipartition(adj, n)
        
        # Try assigning colors to side0 first (the larger side or either)
        # For each vertex in side0, pick a color that doesn't conflict with already-assigned neighbors
        
        # Approach 1: Try to assign all of side0 with distinct-friendly colors
        # Then check if remaining side1 can be solved via 2-SAT
        
        best_result = None
        
        # Try different color assignments for side0
        # Greedy: for each vertex in side0, pick color that leaves most options for side1 neighbors
        
        for attempt in range(3):
            if time.monotonic() > deadline:
                break
            
            fixed = {}
            success = True
            
            # Assign side0 vertices
            for v in side0:
                taken = set()
                for u in adj[v]:
                    if u in fixed:
                        taken.add(fixed[u])
                avail = [c for c in lists[v] if c not in taken]
                if not avail:
                    success = False
                    break
                # Pick color that minimizes conflicts for side1 neighbors
                best_c = avail[0]
                best_score = -1
                for c in avail:
                    score = 0
                    for u in adj[v]:
                        if u in side1:
                            if c in lists[u]:
                                score -= 1
                    if score > best_score:
                        best_score = score
                        best_c = c
                fixed[v] = best_c
            
            if not success:
                continue
            
            # Now try 2-SAT for side1
            result = solve_2sat(n, adj, lists, fixed)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            return best_result
        
        # Fallback: full backtracking
        if time.monotonic() > deadline:
            # Emergency: just return something
            colour = [lists[v][0] for v in range(n)]
            return colour
        
        return solve_backtrack(n, adj, lists)

    # ---- Precompute instances ----
    params = ((20, 701), (40, 702), (60, 703), (100, 704))
    data = []
    for splits, seed in params:
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        data.append((n, adj, lists))

    start_time = time.monotonic()
    DEADLINE = 140.0
    
    out = []
    for i in range(4):
        n, adj, lists = data[i]
        remaining = DEADLINE - (time.monotonic() - start_time)
        if remaining < 0.5:
            # Not enough time, return quick fallback
            colour = [lists[v][0] for v in range(n)]
            out.append(colour)
        else:
            result = solve_instance(n, adj, lists, start_time + DEADLINE)
            out.append(result)
    
    # Ensure types are Python int and lists
    final_out = []
    for lst in out:
        if isinstance(lst, (list, tuple)):
            final_out.append([int(x) for x in lst])
        else:
            final_out.append(lst)
    
    return final_out
# EVOLVE-BLOCK-END
