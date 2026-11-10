import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def bronk_max_clique(P, X, adj, deadline):
        best = [0]
        best_count = [0]
        
        def bronk(P, X):
            if time.time() > deadline:
                return
            if not P:
                cnt = X.bit_count()
                if cnt > best_count[0]:
                    best_count[0] = cnt
                    best[0] = X
                return
            if X.bit_count() + P.bit_count() <= best_count[0]:
                return
            
            R = P | X
            best_pivot = -1
            best_pivot_count = -1
            r = R
            while r:
                u = (r & -r).bit_length() - 1
                r &= r - 1
                cnt = (P & adj[u]).bit_count()
                if cnt > best_pivot_count:
                    best_pivot_count = cnt
                    best_pivot = u
            
            if best_pivot >= 0:
                candidates = P & ~adj[best_pivot]
            else:
                candidates = P
            
            c = candidates
            while c:
                v_bit = c & -c
                v = v_bit.bit_length() - 1
                c &= c - 1
                bronk(P & adj[v], X & adj[v])
                P &= ~v_bit
                X |= v_bit
        
        bronk(P, 0)
        return best[0]

    def greedy_clique_cover_large(n, edges, adj, rng, deadline):
        remaining = set(edges)
        if not remaining:
            return []
        
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        rem_deg = [rem_adj[i].bit_count() for i in range(n)]
        cliques = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            active_vertices = [v for v in range(n) if rem_deg[v] > 0]
            if not active_vertices:
                break
            
            if len(active_vertices) > 50:
                sample_size = min(50, len(active_vertices))
                sample = rng.sample(active_vertices, sample_size)
            else:
                sample = active_vertices
            
            best_vertex = max(sample, key=lambda v: rem_deg[v])
            clique = [best_vertex]
            common = rem_adj[best_vertex]
            
            added = True
            while added:
                added = False
                if time.time() > deadline:
                    return None
                
                best_w = -1
                best_w_deg = -1
                cm = common
                while cm:
                    w = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    if w not in clique and rem_deg[w] > best_w_deg:
                        best_w_deg = rem_deg[w]
                        best_w = w
                
                if best_w < 0:
                    break
                
                clique.append(best_w)
                common &= rem_adj[best_w]
                added = True
            
            clique_set = set(clique)
            new_remaining = set()
            for u2, v2 in remaining:
                if u2 in clique_set and v2 in clique_set:
                    rem_adj[u2] &= ~(1 << v2)
                    rem_adj[v2] &= ~(1 << u2)
                    rem_deg[u2] -= 1
                    rem_deg[v2] -= 1
                else:
                    new_remaining.add((u2, v2))
            
            cliques.append(clique)
            remaining = new_remaining
        
        return cliques

    def greedy_clique_cover_min_common(n, edges, adj, rng, deadline):
        remaining = set(edges)
        if not remaining:
            return []
        
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            edge_list = list(remaining)
            if len(edge_list) > 100:
                sample_size = min(100, len(edge_list))
                sample = rng.sample(edge_list, sample_size)
            else:
                sample = edge_list
            
            best_edge = None
            best_common_mask = None
            best_count = float('inf')
            for u, v in sample:
                common = rem_adj[u] & rem_adj[v]
                count = common.bit_count()
                if count < best_count:
                    best_count = count
                    best_edge = (u, v)
                    best_common_mask = common
            
            u, v = best_edge
            clique = [u, v]
            common = best_common_mask
            
            added = True
            while added:
                added = False
                if time.time() > deadline:
                    return None
                
                best_w = -1
                best_w_count = -1
                cm = common
                while cm:
                    w = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    if w not in clique:
                        w_count = (common & rem_adj[w]).bit_count()
                        if w_count > best_w_count:
                            best_w_count = w_count
                            best_w = w
                
                if best_w < 0:
                    break
                
                clique.append(best_w)
                common &= rem_adj[best_w]
                added = True
            
            clique_set = set(clique)
            new_remaining = set()
            for u2, v2 in remaining:
                if u2 in clique_set and v2 in clique_set:
                    rem_adj[u2] &= ~(1 << v2)
                    rem_adj[v2] &= ~(1 << u2)
                else:
                    new_remaining.add((u2, v2))
            
            cliques.append(clique)
            remaining = new_remaining
        
        return cliques

    def greedy_clique_cover_bronk(n, edges, adj, rng, deadline):
        remaining = set(edges)
        if not remaining:
            return []
        
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        
        while remaining:
            if time.time() > deadline:
                return None
            
            active_vertices = [v for v in range(n) if rem_adj[v] != 0]
            if not active_vertices:
                break
            
            if len(active_vertices) > 30:
                sample_size = min(30, len(active_vertices))
                sample = rng.sample(active_vertices, sample_size)
            else:
                sample = active_vertices
            
            best_vertex = max(sample, key=lambda v: rem_adj[v].bit_count())
            P = rem_adj[best_vertex]
            
            clique_mask = bronk_max_clique(P, 0, rem_adj, deadline)
            clique_mask |= (1 << best_vertex)
            
            clique = []
            cm = clique_mask
            while cm:
                w = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                clique.append(w)
            
            if len(clique) < 2:
                nb = rem_adj[best_vertex]
                if nb:
                    w = (nb & -nb).bit_length() - 1
                    clique = [best_vertex, w]
                else:
                    break
            
            clique_set = set(clique)
            new_remaining = set()
            for u2, v2 in remaining:
                if u2 in clique_set and v2 in clique_set:
                    rem_adj[u2] &= ~(1 << v2)
                    rem_adj[v2] &= ~(1 << u2)
                else:
                    new_remaining.add((u2, v2))
            
            cliques.append(clique)
            remaining = new_remaining
        
        return cliques

    def solve_instance(n, p, seed, deadline):
        adj, edges = gen_graph(n, p, seed)
        
        if not edges:
            return []
        
        best_cliques = [[u, v] for u, v in edges]
        best_count = len(best_cliques)
        
        rng_master = random.Random(12345)
        
        elapsed = time.time() - start_time
        remaining_time = deadline - start_time - elapsed
        if n <= 32:
            num_restarts = min(500, max(10, int(remaining_time / 0.03)))
        else:
            num_restarts = min(300, max(5, int(remaining_time / 0.08)))
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            rng = random.Random(rng_master.randint(0, 2**31))
            
            strategy = restart % 3
            if strategy == 0:
                result = greedy_clique_cover_large(n, edges, adj, rng, deadline)
            elif strategy == 1:
                result = greedy_clique_cover_min_common(n, edges, adj, rng, deadline)
            else:
                result = greedy_clique_cover_bronk(n, edges, adj, rng, deadline)
            
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
        
        return best_cliques

    result1 = solve_instance(32, 0.6, 601, DEADLINE)
    
    if time.time() > DEADLINE:
        adj2, edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, DEADLINE)
    
    return [result1, result2]
