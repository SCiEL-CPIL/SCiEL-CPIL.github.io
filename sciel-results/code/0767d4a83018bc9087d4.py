import random
import time

def _popcount(x):
    return bin(x).count('1')

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
                        if cw:
                            lsb2 = cw & -cw
                            x = lsb2.bit_length() - 1
                            return (u, v, w, x)
    return None

def _find_indep6(adj):
    n = len(adj)
    def find_clique(candidates, size):
        if size == 0:
            return []
        if len(candidates) < size:
            return None
        # Heuristic: pick vertex with highest degree in complement?
        # Or random. Random is simpler.
        idx = random.randint(0, len(candidates) - 1)
        v = candidates[idx]
        candidates = candidates[:idx] + candidates[idx+1:]
        
        new_cands = []
        for u in candidates:
            if (adj[u] & (1 << v)) == 0:
                new_cands.append(u)
                
        res = find_clique(new_cands, size - 1)
        if res is not None:
            return [v] + res
        return None
        
    candidates = list(range(n))
    return find_clique(candidates, 6)

def _is_valid(adj):
    return _find_k4(adj) is None and _find_indep6(adj) is None

def _repair(adj, time_limit):
    n = len(adj)
    start_time = time.time()
    while time.time() - start_time < time_limit:
        if _is_valid(adj):
            return True
        
        k4 = _find_k4(adj)
        if k4:
            u, v, w, x = k4
            edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
            ru, rv = random.choice(edges)
            adj[ru] &= ~(1 << rv)
            adj[rv] &= ~(1 << ru)
        else:
            indep = _find_indep6(adj)
            if indep:
                i, j = random.sample(indep, 2)
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            else:
                return True
    return _is_valid(adj)

def _random_graph(n, p):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    deadline = time.time() + 110
    best_n = 0
    best_adj = None
    
    # Try to find a graph for n=35 down to 20
    for n in range(35, 19, -1):
        if time.time() > deadline - 5:
            break
            
        # Allocate time for this n
        # If n is large, give it more time?
        # Let's just try a few restarts.
        attempts = 0
        while time.time() < deadline - 5:
            attempts += 1
            adj = _random_graph(n, 0.5)
            # Give repair a small budget
            if _repair(adj, 0.5):
                best_n = n
                best_adj = adj
                break
            if attempts > 10:
                break
    
    if best_n == 0:
        # Fallback
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_n = 17
        best_adj = adj

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
