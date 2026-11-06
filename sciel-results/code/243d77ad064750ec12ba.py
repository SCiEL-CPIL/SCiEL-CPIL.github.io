import time
import random

def construct():
    deadline = time.time() + 115.0
    
    def get_edges(n):
        return [(i, j) for i in range(n) for j in range(i+1, n)]

    def build_adj(n, colors):
        adj = [[set() for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        return adj

    def count_c6_using_edge(adj, u, v):
        # adj is a list of sets (for one color class)
        N_u = adj[u]
        N_v = adj[v]
        count = 0
        for a in N_u:
            if a == v:
                continue
            N_a = adj[a]
            for c in N_v:
                if c == u or c == a:
                    continue
                N_c = adj[c]
                common = N_a & N_c
                for b in common:
                    if b != u and b != v:
                        count += 1
        return count

    def total_c6_count(n, colors):
        adj = build_adj(n, colors)
        total = 0
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                total += count_c6_using_edge(adj[c], i, j)
                idx += 1
        return total // 6

    def solve_for_n(n):
        edges = get_edges(n)
        m = len(edges)
        
        colors = [random.randint(0, 3) for _ in range(m)]
        
        edge_idx = {}
        for k, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = k
            edge_idx[(v, u)] = k
            
        adj = build_adj(n, colors)
        
        total_cost = 0
        for k, (u, v) in enumerate(edges):
            c = colors[k]
            total_cost += count_c6_using_edge(adj[c], u, v)
        total_cost //= 6
        
        if total_cost == 0:
            return colors
            
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() < deadline:
            if T < min_T:
                colors = [random.randint(0, 3) for _ in range(m)]
                adj = build_adj(n, colors)
                total_cost = 0
                for k, (u, v) in enumerate(edges):
                    c = colors[k]
                    total_cost += count_c6_using_edge(adj[c], u, v)
                total_cost //= 6
                T = 1.0
                if total_cost == 0:
                    return colors

            k = random.randint(0, m - 1)
            u, v = edges[k]
            old_c = colors[k]
            
            new_c = random.randint(0, 3)
            while new_c == old_c:
                new_c = random.randint(0, 3)
            
            # Remove from old
            adj[old_c][u].remove(v)
            adj[old_c][v].remove(u)
            
            k_old = count_c6_using_edge(adj[old_c], u, v)
            
            # Add to new
            adj[new_c][u].add(v)
            adj[new_c][v].add(u)
            
            k_new = count_c6_using_edge(adj[new_c], u, v)
            
            delta = k_new - k_old
            
            if delta <= 0 or random.random() < pow(2.71828, -delta / T):
                colors[k] = new_c
                total_cost += delta
                if total_cost == 0:
                    return colors
            else:
                adj[new_c][u].remove(v)
                adj[new_c][v].remove(u)
                adj[old_c][u].add(v)
                adj[old_c][v].add(u)
            
            T *= cooling
            
        return None

    best_n = 9
    n_seed = 9
    best_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
        res = solve_for_n(n)
        if res is not None:
            if total_c6_count(n, res) == 0:
                best_n = n
                best_colors = res
                break
                
    return {'n': best_n, 'colors': best_colors}
