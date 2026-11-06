import random
import time
import itertools

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

def _structured_init(n):
    adj = [0] * n
    group_size = n // 5
    remainder = n % 5
    groups = []
    start = 0
    for i in range(5):
        size = group_size + (1 if i < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size
    
    for g in groups:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if random.random() < 0.2:
                    u, v = g[i], g[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for a, b in c5_edges:
        for u in groups[a]:
            for v in groups[b]:
                if random.random() < 0.4:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    return adj

def _walksat(adj, time_limit):
    n = len(adj)
    start_time = time.time()
    steps = 0
    max_steps = 30000
    perturb_every = 150
    k4_weights = {}
    indep_weights = {}
    
    while time.time() - start_time < time_limit:
        if steps >= max_steps:
            break
        
        k4 = _find_k4(adj)
        
        if k4:
            u, v, w, x = k4
            k4_key = frozenset([u, v, w, x])
            k4_weights[k4_key] = k4_weights.get(k4_key, 0) + 1
            
            edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
            edge_scores = []
            for e in edges:
                score = 0
                for k4k, wt in k4_weights.items():
                    if e[0] in k4k and e[1] in k4k:
                        score += wt
                edge_scores.append((score, e))
            edge_scores.sort(reverse=True)
            if random.random() < 0.7:
                ru, rv = edge_scores[0][1]
            else:
                ru, rv = random.choice(edges)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
        else:
            indep = _find_indep6(adj)
            if indep:
                indep_key = frozenset(indep)
                indep_weights[indep_key] = indep_weights.get(indep_key, 0) + 1
                
                best_pair = None
                best_score = -1
                pairs = list(itertools.combinations(indep, 2))
                random.shuffle(pairs)
                
                for i, j in pairs[:10]:
                    k4_count = 0
                    for u in range(n):
                        if u == i or u == j:
                            continue
                        if (adj[i] & (1 << u)) and (adj[j] & (1 << u)):
                            common = adj[i] & adj[j] & adj[u]
                            if common:
                                k4_count += 1
                    score = -k4_count * 10 + random.random()
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
                
                if best_pair is None:
                    best_pair = pairs[0]
                
                i, j = best_pair
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            else:
                return True
        
        steps += 1
        
        if steps % perturb_every == 0:
            num_flips = random.choice([2, 3])
            for _ in range(num_flips):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
    return False

def construct():
    deadline = time.time() + 110
    best_n = 0
    best_adj = None
    
    for n in range(35, 24, -1):
        if time.time() > deadline - 5:
            break
            
        attempts = 0
        max_attempts = 25
        while time.time() < deadline - 5:
            attempts += 1
            
            if attempts % 3 == 0:
                adj = _structured_init(n)
            else:
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            time_per_attempt = min(2.5, (deadline - time.time() - 5) / max(1, max_attempts - attempts))
            
            if _walksat(adj, time_per_attempt):
                best_n = n
                best_adj = adj
                break
            if attempts >= max_attempts:
                break
    
    if best_n == 0:
        for n in range(24, 15, -1):
            if time.time() > deadline - 1:
                break
            for _ in range(15):
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                if _walksat(adj, 0.5):
                    best_n = n
                    best_adj = adj
                    break
            if best_n > 0:
                break

    if best_n == 0:
        best_n = 4
        best_adj = [9, 3, 6, 12]

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
