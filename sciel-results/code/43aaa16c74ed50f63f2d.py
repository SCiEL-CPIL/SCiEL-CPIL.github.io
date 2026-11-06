import random
import time

def _find_k4(adj):
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    if adj[w] & common:
                        cw = adj[w] & common
                        lsb2 = cw & -cw
                        x = lsb2.bit_length() - 1
                        return (u, v, w, x)
    return None

def _find_indep6(adj):
    n = len(adj)
    mask = (1 << n) - 1
    
    def dfs(candidates, size):
        if size == 0:
            return []
        if candidates.bit_count() < size:
            return None
        
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            comp_neigh = (~adj[v]) & mask & ~(1 << v)
            new_cands = candidates & comp_neigh
            res = dfs(new_cands, size - 1)
            if res is not None:
                return [v] + res
        return None

    return dfs(mask, 6)

def _tabu_search(adj, n, time_limit):
    start_time = time.time()
    tabu_list = []
    tabu_len = 100
    
    while time.time() - start_time < time_limit:
        k4 = _find_k4(adj)
        if k4:
            u, v, w, x = k4
            edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
            # Filter tabu
            valid_edges = [e for e in edges if e not in tabu_list]
            if not valid_edges:
                valid_edges = edges
            ru, rv = random.choice(valid_edges)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
            tabu_list.append((ru, rv))
            if len(tabu_list) > tabu_len:
                tabu_list.pop(0)
        else:
            indep = _find_indep6(adj)
            if indep:
                pairs = [(indep[i], indep[j]) for i in range(len(indep)) for j in range(i+1, len(indep))]
                valid_pairs = [e for e in pairs if e not in tabu_list and (e[1], e[0]) not in tabu_list]
                if not valid_pairs:
                    valid_pairs = pairs
                u, v = random.choice(valid_pairs)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                tabu_list.append((u, v))
                if len(tabu_list) > tabu_len:
                    tabu_list.pop(0)
            else:
                return True
                
    return False

def construct():
    deadline = time.time() + 110
    
    for n in range(36, 24, -1):
        if time.time() > deadline - 5:
            break
            
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
                
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            time_per_attempt = min(5.0, (deadline - time.time() - 5) / 20)
            
            if _tabu_search(adj, n, time_per_attempt):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    return {'n': 25, 'edges': []}
