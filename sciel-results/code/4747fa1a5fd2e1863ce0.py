import numpy as np
import time
import random

def count_edges(adj):
    return np.sum(adj != -1) // 2

def construct():
    n = 20
    deadline = time.time() + 115
    random.seed(42)
    np.random.seed(42)
    
    # Precompute 4-cycle witnesses for each edge (u,v)
    witness = {}
    for u in range(n):
        for v in range(u+1, n):
            wl = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in range(n):
                    if y == u or y == v or y == x:
                        continue
                    wl.append((x, y))
            witness[(u, v)] = wl
            witness[(v, u)] = wl
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_forbidden(u, v, adj):
        forbidden = set()
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue
            
            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]
            
            if c1 == c2 or c1 == c3 or c2 == c3:
                return set(range(n))
            
            forbidden.add(c1)
            forbidden.add(c2)
            forbidden.add(c3)
        
        return forbidden
    
    def is_recolor_valid(u, v, new_c, adj):
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1:
                continue
            if adj[x, y] == -1:
                continue
            if adj[y, u] == -1:
                continue
            
            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]
            
            if new_c == c1 or new_c == c2 or new_c == c3:
                return False
        return True
    
    def get_color_frequencies(adj):
        freq = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i, j]
                if c != -1:
                    freq[c] += 1
        return freq
    
    def find_min_color_change_set(missing_edge, adj, max_depth=4):
        """
        Backtracking search for a minimal set of edge recolorings that would
        allow adding the missing edge. Returns list of (u, v, new_color) or None.
        """
        u, v = missing_edge
        forb = get_forbidden(u, v, adj)
        if len(forb) < n:
            return []  # Already can add
        
        # Find which 4-cycles are causing conflicts
        conflicting_cycles = []
        for (x, y) in witness[(u, v)]:
            if adj[v, x] == -1 or adj[x, y] == -1 or adj[y, u] == -1:
                continue
            c1 = adj[v, x]
            c2 = adj[x, y]
            c3 = adj[y, u]
            if c1 in forb or c2 in forb or c3 in forb:
                # Check if this cycle actually contributes to the blocking
                if c1 == c2 or c1 == c3 or c2 == c3:
                    return None  # Invalid state
                conflicting_cycles.append(((v, x, c1), (x, y, c2), (y, u, c3)))
        
        if not conflicting_cycles:
            return None
        
        # Try to find recolorings that break all conflicts
        # Use a simple backtracking approach
        def backtrack(changes, visited_edges, depth):
            if depth > max_depth:
                return None
            
            # Check if current changes would allow adding (u,v)
            adj_test = adj.copy()
            for (a, b, c) in changes:
                adj_test[a, b] = c
                adj_test[b, a] = c
            
            forb_test = get_forbidden(u, v, adj_test)
            if len(forb_test) < n:
                return changes
            
            # Find a conflict that needs to be resolved
            for (e1, e2, e3) in conflicting_cycles:
                (a1, b1, c1), (a2, b2, c2), (a3, b3, c3) = e1, e2, e3
                # Check if this cycle is still conflicting
                if adj_test[b1, a1] != c1 or adj_test[b2, a2] != c2 or adj_test[b3, a3] != c3:
                    continue
                
                # Try to recolor one of the three edges
                for (a, b, c) in [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)]:
                    if (a, b) in visited_edges or (b, a) in visited_edges:
                        continue
                    
                    # Try new colors
                    colors = list(range(n))
                    random.shuffle(colors)
                    for new_c in colors:
                        if new_c == c:
                            continue
                        if not is_recolor_valid(a, b, new_c, adj_test):
                            continue
                        
                        new_changes = changes + [(a, b, new_c)]
                        new_visited = visited_edges | {(a, b), (b, a)}
                        result = backtrack(new_changes, new_visited, depth + 1)
                        if result is not None:
                            return result
            
            return None
        
        return backtrack([], set(), 0)
    
    def greedy_add_edges(adj, edges_list):
        added = 0
        for u, v in edges_list:
            if time.time() > deadline:
                break
            if adj[u, v] != -1:
                continue
            
            forb = get_forbidden(u, v, adj)
            if len(forb) < n:
                available = [c for c in range(n) if c not in forb]
                if available:
                    freq = get_color_frequencies(adj)
                    min_freq = min(freq[c] for c in available)
                    best_colors = [c for c in available if freq[c] == min_freq]
                    c = random.choice(best_colors)
                    adj[u, v] = c
                    adj[v, u] = c
                    added += 1
        return added
    
    def structured_init():
        """Initialize with a more balanced structure."""
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        
        # Create a near-regular graph structure
        # Start with a cycle and add chords strategically
        vertices = list(range(n))
        random.shuffle(vertices)
        
        # Add cycle edges
        for i in range(n):
            u = vertices[i]
            v = vertices[(i+1) % n]
            c = i % n
            adj[u, v] = c
            adj[v, u] = c
        
        # Add some chords with careful color assignment
        for i in range(n):
            if time.time() > deadline:
                break
            u = vertices[i]
            # Connect to vertex 3 steps ahead
            v = vertices[(i+3) % n]
            if adj[u, v] == -1:
                forb = get_forbidden(u, v, adj)
                if len(forb) < n:
                    available = [c for c in range(n) if c not in forb]
                    if available:
                        freq = get_color_frequencies(adj)
                        min_freq = min(freq[c] for c in available)
                        best_colors = [c for c in available if freq[c] == min_freq]
                        c = random.choice(best_colors)
                        adj[u, v] = c
                        adj[v, u] = c
        
        return adj
    
    best_count = 0
    best_adj = None
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Alternate between random and structured initialization
        if restart % 3 == 0:
            adj = structured_init()
        else:
            adj = -np.ones((n, n), dtype=int)
            np.fill_diagonal(adj, -1)
            
            vertices = list(range(n))
            random.shuffle(vertices)
            for i in range(1, n):
                u = vertices[i]
                v = vertices[random.randint(0, i-1)]
                c = random.randint(0, n-1)
                adj[u, v] = c
                adj[v, u] = c
        
        # Greedy edge addition
        edges = all_edges[:]
        random.shuffle(edges)
        greedy_add_edges(adj, edges)
        
        # Local search with enhanced recoloring
        for iteration in range(1500):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding missing edges
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == -1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline:
                    break
                if adj[u, v] != -1:
                    continue
                
                forb = get_forbidden(u, v, adj)
                if len(forb) < n:
                    available = [c for c in range(n) if c not in forb]
                    if available:
                        freq = get_color_frequencies(adj)
                        min_freq = min(freq[c] for c in available)
                        best_colors = [c for c in available if freq[c] == min_freq]
                        c = random.choice(best_colors)
                        adj[u, v] = c
                        adj[v, u] = c
                        improved = True
            
            if improved:
                continue
            
            # 2. Backtracking-based repair for a blocked edge
            if missing:
                u, v = random.choice(missing)
                changes = find_min_color_change_set((u, v), adj, max_depth=3)
                if changes:
                    # Apply the changes
                    for (a, b, new_c) in changes:
                        if adj[a, b] != -1:
                            adj[a, b] = new_c
                            adj[b, a] = new_c
                    improved = True
                    
                    # Try to add the edge
                    if adj[u, v] == -1:
                        forb = get_forbidden(u, v, adj)
                        if len(forb) < n:
                            available = [c for c in range(n) if c not in forb]
                            if available:
                                freq = get_color_frequencies(adj)
                                min_freq = min(freq[c] for c in available)
                                best_colors = [c for c in available if freq[c] == min_freq]
                                c_add = random.choice(best_colors)
                                adj[u, v] = c_add
                                adj[v, u] = c_add
            
            # 3. Targeted single-edge recoloring
            if not improved and missing:
                u, v = random.choice(missing)
                # Find a blocking edge
                for (x, y) in witness[(u, v)]:
                    if adj[v, x] == -1 or adj[x, y] == -1 or adj[y, u] == -1:
                        continue
                    
                    c1 = adj[v, x]
                    c2 = adj[x, y]
                    c3 = adj[y, u]
                    
                    if c1 == c2 or c1 == c3 or c2 == c3:
                        continue
                    
                    # Try recoloring one of the path edges
                    for (a, b) in [(v, x), (x, y), (y, u)]:
                        old_c = adj[a, b]
                        colors = list(range(n))
                        random.shuffle(colors)
                        for new_c in colors:
                            if new_c == old_c:
                                continue
                            if is_recolor_valid(a, b, new_c, adj):
                                adj[a, b] = new_c
                                adj[b, a] = new_c
                                improved = True
                                # Try adding
                                if adj[u, v] == -1:
                                    forb = get_forbidden(u, v, adj)
                                    if len(forb) < n:
                                        available = [c for c in range(n) if c not in forb]
                                        if available:
                                            freq = get_color_frequencies(adj)
                                            min_freq = min(freq[c] for c in available)
                                            best_colors = [c for c in available if freq[c] == min_freq]
                                            c_add = random.choice(best_colors)
                                            adj[u, v] = c_add
                                            adj[v, u] = c_add
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # 4. Random recoloring to escape local optima
            if not improved:
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] != -1]
                random.shuffle(existing)
                for u, v in existing[:15]:
                    if time.time() > deadline:
                        break
                    old_c = adj[u, v]
                    colors = list(range(n))
                    random.shuffle(colors)
                    for new_c in colors:
                        if new_c == old_c:
                            continue
                        if is_recolor_valid(u, v, new_c, adj):
                            adj[u, v] = new_c
                            adj[v, u] = new_c
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj.copy()
        
        if best_count > 175:
            break
    
    # Final polish with backtracking repair
    if best_adj is not None and time.time() < deadline - 5:
        adj = best_adj.copy()
        for polish_iter in range(800):
            if time.time() > deadline - 2:
                break
            
            improved = False
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j] == -1]
            random.shuffle(missing)
            
            # Try adding
            for u, v in missing:
                if time.time() > deadline - 2:
                    break
                if adj[u, v] != -1:
                    continue
                forb = get_forbidden(u, v, adj)
                if len(forb) < n:
                    available = [c for c in range(n) if c not in forb]
                    if available:
                        freq = get_color_frequencies(adj)
                        min_freq = min(freq[c] for c in available)
                        best_colors = [c for c in available if freq[c] == min_freq]
                        c = random.choice(best_colors)
                        adj[u, v] = c
                        adj[v, u] = c
                        improved = True
            
            if not improved:
                # Use backtracking repair
                if missing:
                    u, v = random.choice(missing)
                    changes = find_min_color_change_set((u, v), adj, max_depth=4)
                    if changes:
                        for (a, b, new_c) in changes:
                            if adj[a, b] != -1:
                                adj[a, b] = new_c
                                adj[b, a] = new_c
                        improved = True
                        
                        if adj[u, v] == -1:
                            forb = get_forbidden(u, v, adj)
                            if len(forb) < n:
                                available = [c for c in range(n) if c not in forb]
                                if available:
                                    freq = get_color_frequencies(adj)
                                    min_freq = min(freq[c] for c in available)
                                    best_colors = [c for c in available if freq[c] == min_freq]
                                    c_add = random.choice(best_colors)
                                    adj[u, v] = c_add
                                    adj[v, u] = c_add
            
            if not improved:
                break
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Convert to result
    res = []
    if best_adj is None:
        for v in range(1, n):
            res.append([0, v, 0])
        return res
    
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] != -1:
                res.append([i, j, int(best_adj[i, j])])
    return res
