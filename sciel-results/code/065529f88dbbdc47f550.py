import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n = 16
    m_target = 65
    
    def calc_score(adj, deg):
        max_sum = 0
        for u in range(n):
            adj_u = adj[u]
            deg_u = deg[u]
            for v in range(u + 1, n):
                if (adj_u >> v) & 1:
                    common = adj_u & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = deg_u + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    def get_critical_edges(adj, deg, max_sum):
        critical = set()
        for u in range(n):
            adj_u = adj[u]
            deg_u = deg[u]
            for v in range(u + 1, n):
                if (adj_u >> v) & 1:
                    common = adj_u & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = deg_u + deg[v] + deg[w]
                        if s == max_sum:
                            critical.add((u, v))
                            critical.add((u, w))
                            critical.add((v, w))
                        common ^= lsb
        return critical

    # Initial randomization
    best_score = float('inf')
    best_adj = None
    best_deg = None
    best_edges_set = None
    
    for _ in range(20):
        if time.time() > deadline: break
        edges = set()
        while len(edges) < m_target:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                edges.add((min(u,v), max(u,v)))
        
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
        score = calc_score(adj, deg)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            best_deg = deg[:]
            best_edges_set = edges.copy()
            
    adj = best_adj
    deg = best_deg
    edges_set = best_edges_set
    current_score = best_score
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        critical_edges = get_critical_edges(adj, deg, current_score)
        
        if not critical_edges:
            break
            
        best_new_score = current_score
        best_move = None
        
        for e_out in critical_edges:
            u, v = e_out
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            deg[u] -= 1
            deg[v] -= 1
            edges_set.remove(e_out)
            
            best_candidate = None
            best_candidate_score = best_new_score
            
            for e_in in all_pairs:
                if e_in in edges_set:
                    continue
                x, y = e_in
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                deg[x] += 1
                deg[y] += 1
                edges_set.add(e_in)
                
                new_score = calc_score(adj, deg)
                
                if new_score < best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate = e_in
                
                adj[x] ^= (1 << y)
                adj[y] ^= (1 << x)
                deg[x] -= 1
                deg[y] -= 1
                edges_set.remove(e_in)
                
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            edges_set.add(e_out)
            
            if best_candidate and best_candidate_score < best_new_score:
                best_new_score = best_candidate_score
                best_move = (e_out, best_candidate)
                
        if best_move:
            e_out, e_in = best_move
            u, v = e_out
            x, y = e_in
            
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            deg[u] -= 1
            deg[v] -= 1
            edges_set.remove(e_out)
            
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            deg[x] += 1
            deg[y] += 1
            edges_set.add(e_in)
            
            current_score = best_new_score
        else:
            # Random perturbation
            e_out = random.choice(list(edges_set))
            while True:
                e_in = random.choice(all_pairs)
                if e_in not in edges_set and e_in != e_out:
                    break
            
            u, v = e_out
            x, y = e_in
            
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            deg[u] -= 1
            deg[v] -= 1
            edges_set.remove(e_out)
            
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            deg[x] += 1
            deg[y] += 1
            edges_set.add(e_in)
            
            new_score = calc_score(adj, deg)
            
            if new_score <= current_score:
                current_score = new_score
            else:
                adj[x] ^= (1 << y)
                adj[y] ^= (1 << x)
                deg[x] -= 1
                deg[y] -= 1
                edges_set.remove(e_in)
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges_set.add(e_out)

    edges_list = [list(e) for e in edges_set]
    return {'n': n, 'edges': edges_list}
