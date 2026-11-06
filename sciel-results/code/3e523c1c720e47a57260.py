"""Constructive greedy + local search for rainbow 4-cycle edge coloring."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    N = 20
    C = 20
    
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    def get_color(adj, u, v):
        if u in adj and v in adj[u]:
            return adj[u][v]
        if v in adj and u in adj[v]:
            return adj[v][u]
        return None
    
    def set_color(adj, u, v, c):
        adj.setdefault(u, {})[v] = c
        adj.setdefault(v, {})[u] = c
    
    def remove_edge(adj, u, v):
        if u in adj and v in adj[u]:
            del adj[u][v]
        if v in adj and u in adj[v]:
            del adj[v][u]
    
    def check_edge_valid(adj, u, v, c):
        """Check if adding edge (u,v) with colour c maintains rainbow 4-cycles."""
        # 4-cycles containing (u,v): u-x-y-v where x~u, y~v, x~y, x!=v, y!=u
        for x in adj.get(u, {}):
            if x == v:
                continue
            for y in adj.get(v, {}):
                if y == u or y == x:
                    continue
                if get_color(adj, x, y) is not None:
                    c_vy = get_color(adj, v, y)
                    c_xy = get_color(adj, x, y)
                    c_xu = get_color(adj, x, u)
                    if c_vy is not None and c_xy is not None and c_xu is not None:
                        if len({c, c_vy, c_xy, c_xu}) < 4:
                            return False
        return True
    
    def full_validate(adj):
        """Check all 4-cycles are rainbow."""
        for a in range(N):
            for b in range(a+1, N):
                if get_color(adj, a, b) is None:
                    continue
                for x in adj.get(a, {}):
                    if x == b:
                        continue
                    for y in adj.get(b, {}):
                        if y == a or y == x:
                            continue
                        if get_color(adj, x, y) is not None:
                            c_ab = get_color(adj, a, b)
                            c_bx = get_color(adj, b, x)
                            c_xy = get_color(adj, x, y)
                            c_ya = get_color(adj, y, a)
                            if None in (c_ab, c_bx, c_xy, c_ya):
                                continue
                            if len({c_ab, c_bx, c_xy, c_ya}) < 4:
                                return False
        return True
    
    def adj_to_result(adj):
        result = []
        for u in range(N):
            for v in adj.get(u, {}):
                if v > u:
                    result.append([u, v, adj[u][v]])
        return result
    
    def edge_count(adj):
        return sum(len(v) for v in adj.values()) // 2
    
    random.seed(123)
    best_adj = {}
    best_count = 0
    
    # Phase 1: Multiple greedy attempts
    for attempt in range(8):
        if time.time() - start_time > DEADLINE - 40:
            break
        adj = {}
        
        # Different ordering strategies
        if attempt == 0:
            # Hadamard-inspired: use modular arithmetic for initial structure
            # Assign colours based on (u*v) mod C for a base
            order = all_edges[:]
        elif attempt == 1:
            # Sort by vertex sum (prefer low-index vertices first)
            order = sorted(all_edges, key=lambda e: e[0] + e[1])
        elif attempt == 2:
            order = sorted(all_edges, key=lambda e: e[1] - e[0])
        else:
            order = all_edges[:]
            random.shuffle(order)
        
        for (u, v) in order:
            if time.time() - start_time > DEADLINE - 40:
                break
            # Try colours in a structured order
            if attempt == 0:
                colour_order = [(u * v + i) % C for i in range(C)]
            elif attempt == 1:
                colour_order = [(u + v + i) % C for i in range(C)]
            else:
                colour_order = list(range(C))
                random.shuffle(colour_order)
            
            added = False
            for c in colour_order:
                if check_edge_valid(adj, u, v, c):
                    set_color(adj, u, v, c)
                    added = True
                    break
            if not added:
                # Try random colours
                for _ in range(10):
                    c = random.randint(0, C - 1)
                    if check_edge_valid(adj, u, v, c):
                        set_color(adj, u, v, c)
                        added = True
                        break
        
        if full_validate(adj):
            cnt = edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = {k: dict(v) for k, v in adj.items()}
    
    current_result = adj_to_result(best_adj)
    
    # Phase 2: Local search improvement
    if time.time() - start_time < DEADLINE - 5:
        adj = {k: dict(v) for k, v in best_adj.items()}
        
        # Iteratively try to add edges
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            for (u, v) in all_edges:
                if get_color(adj, u, v) is not None:
                    continue
                for c in range(C):
                    if check_edge_valid(adj, u, v, c):
                        set_color(adj, u, v, c)
                        improved = True
                        break
        
        # Recoloring local search
        if time.time() - start_time < DEADLINE - 5:
            for iteration in range(30):
                if time.time() - start_time > DEADLINE - 5:
                    break
                progress = False
                # Try recoloring random edges
                edges_in_adj = [(u, v) for u in range(N) for v in adj.get(u, {}) if v > u]
                random.shuffle(edges_in_adj)
                for (u, v) in edges_in_adj:
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    cur_c = adj[u][v]
                    for new_c in range(C):
                        if new_c == cur_c:
                            continue
                        # Temporarily recolor
                        adj[u][v] = new_c
                        adj[v][u] = new_c
                        if full_validate(adj):
                            progress = True
                            break
                        else:
                            adj[u][v] = cur_c
                            adj[v][u] = cur_c
                
                # After recoloring, try adding more edges
                for (u, v) in all_edges:
                    if get_color(adj, u, v) is not None:
                        continue
                    for c in range(C):
                        if check_edge_valid(adj, u, v, c):
                            set_color(adj, u, v, c)
                            break
                
                if not progress:
                    break
        
        if full_validate(adj):
            final_result = adj_to_result(adj)
            if len(final_result) > len(current_result):
                current_result = final_result
    
    return current_result
