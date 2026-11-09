import numpy as np
import time
import random
import itertools

def construct():
    n = 20
    num_colors = 20
    deadline = time.time() + 115.0
    
    # 1. Precompute edge indices and 4-cycles
    edge_index = {}
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            idx = len(all_edges)
            all_edges.append((u, v))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            
    m = len(all_edges) # 190
    
    # Precompute all 4-cycles as tuples of 4 edge indices
    cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = [edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)]]
        cycles.append(tuple(e1))
        # Cycle 2: a-b-d-c-a
        e2 = [edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)]]
        cycles.append(tuple(e2))
        # Cycle 3: a-c-b-d-a
        e3 = [edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)]]
        cycles.append(tuple(e3))
        
    num_cycles = len(cycles)
    
    # Precompute edge_to_cycles
    edge_to_cycles = [[] for _ in range(m)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)

    # State
    adj = -np.ones((n, n), dtype=int)
    np.fill_diagonal(adj, -1)
    active = np.zeros(m, dtype=bool)
    colors = -np.ones(m, dtype=int)
    
    def get_forbidden_colors_inactive(edge_idx):
        """
        Returns a set of colors that are forbidden for the edge at edge_idx,
        assuming the edge is NOT yet active.
        """
        forbidden = set()
        for ci in edge_to_cycles[edge_idx]:
            cyc = cycles[ci]
            other_edges = [e for e in cyc if e != edge_idx]
            
            if not all(active[e] for e in other_edges):
                continue
                
            c1 = colors[other_edges[0]]
            c2 = colors[other_edges[1]]
            c3 = colors[other_edges[2]]
            
            forbidden.add(c1)
            forbidden.add(c2)
            forbidden.add(c3)
            
        return forbidden

    def is_recolor_valid(edge_idx, new_color):
        """
        Check if recoloring the active edge at edge_idx to new_color
        keeps all active 4-cycles rainbow.
        """
        if new_color == colors[edge_idx]:
            return True
            
        for ci in edge_to_cycles[edge_idx]:
            cyc = cycles[ci]
            if not all(active[e] for e in cyc):
                continue
                
            other_colors = []
            for e in cyc:
                if e == edge_idx:
                    other_colors.append(new_color)
                else:
                    other_colors.append(colors[e])
                    
            if len(set(other_colors)) < 4:
                return False
                
        return True

    best_edges = 0
    best_adj_copy = None
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        
        # Reset state
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)
        active = np.zeros(m, dtype=bool)
        colors = -np.ones(m, dtype=int)
        
        # Initialize with a random spanning tree
        vertices = list(range(n))
        random.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[random.randint(0, i-1)]
            idx = edge_index[(u, v)]
            c = random.randint(0, num_colors - 1)
            adj[u, v] = c
            adj[v, u] = c
            active[idx] = True
            colors[idx] = c
            
        # Greedy Phase
        inactive_indices = [i for i in range(m) if not active[i]]
        random.shuffle(inactive_indices)
        
        for idx in inactive_indices:
            if time.time() > deadline:
                break
            if active[idx]:
                continue
                
            forb = get_forbidden_colors_inactive(idx)
            if len(forb) < num_colors:
                available = [c for c in range(num_colors) if c not in forb]
                c = random.choice(available)
                u, v = all_edges[idx]
                adj[u, v] = c
                adj[v, u] = c
                active[idx] = True
                colors[idx] = c

        # Repair Phase
        # Try to add more edges by recoloring
        repair_attempts = 0
        max_repair_attempts = 1000
        while repair_attempts < max_repair_attempts:
            if time.time() > deadline:
                break
            repair_attempts += 1
            
            inactive_now = [i for i in range(m) if not active[i]]
            if not inactive_now:
                break
                
            target_idx = random.choice(inactive_now)
            forb = get_forbidden_colors_inactive(target_idx)
            
            if len(forb) < num_colors:
                available = [c for c in range(num_colors) if c not in forb]
                c = random.choice(available)
                u, v = all_edges[target_idx]
                adj[u, v] = c
                adj[v, u] = c
                active[target_idx] = True
                colors[target_idx] = c
                continue
            else:
                # Stuck. Try to recolor an edge to unblock target_idx.
                # Pick a random forbidden color
                target_color = random.choice(list(forb))
                
                # Find a cycle where target_color is forbidden
                found_recolor = False
                # Shuffle cycles for randomness
                cyc_indices = edge_to_cycles[target_idx][:]
                random.shuffle(cyc_indices)
                
                for ci in cyc_indices:
                    if found_recolor:
                        break
                    cyc = cycles[ci]
                    other_edges = [e for e in cyc if e != target_idx]
                    if not all(active[e] for e in other_edges):
                        continue
                        
                    # Check if any other edge has target_color
                    for e in other_edges:
                        if colors[e] == target_color:
                            # Try to recolor e
                            new_colors = [c for c in range(num_colors) if c != colors[e]]
                            random.shuffle(new_colors)
                            
                            for c_new in new_colors:
                                if is_recolor_valid(e, c_new):
                                    colors[e] = c_new
                                    u_e, v_e = all_edges[e]
                                    adj[u_e, v_e] = c_new
                                    adj[v_e, u_e] = c_new
                                    found_recolor = True
                                    break
                            if found_recolor:
                                break
                    if found_recolor:
                        break
                
                if not found_recolor:
                    # If we can't unblock this edge, try another random edge in the next iteration
                    # This is handled by the while loop continuing
                    pass

        current_edges = np.sum(active)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_copy = adj.copy()
            
        if best_edges == m:
            break

    if best_adj_copy is None:
        adj = -np.ones((n, n), dtype=int)
        for v in range(1, n):
            c = v % num_colors
            adj[0, v] = c
            adj[v, 0] = c
        best_adj_copy = adj

    res = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj_copy[i, j] != -1:
                res.append([i, j, int(best_adj_copy[i, j])])
    return res
