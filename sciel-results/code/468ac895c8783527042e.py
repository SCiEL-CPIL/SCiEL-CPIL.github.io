import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try n=40, 39, 38...
    # Allocate time proportionally or fixed chunks
    # Let's try 40 for 40s, 39 for 40s, etc.
    
    for n in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27]:
        if time.time() > deadline - 5.0: # 5s buffer
            break
            
        time_limit_for_n = min(30.0, deadline - time.time())
        
        result = search_for_graph(n, time_limit_for_n)
        if result is not None:
            return result
            
    # Fallback to seed
    return {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }

def search_for_graph(n, time_budget):
    best_alpha = float('inf')
    best_S = None
    start_time = time.time()
    
    # Number of restarts depends on how fast alpha calc is
    # Let's just loop until time runs out
    while time.time() - start_time < time_budget:
        S = random_maximal_sum_free_set(n)
        alpha = calc_alpha_circulant(S, n)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_S = S[:]
            
        if best_alpha < 10:
            return build_graph_from_S(best_S, n)
            
    return None

def random_maximal_sum_free_set(n):
    S = set()
    candidates = list(range(1, n // 2 + 1))
    random.shuffle(candidates)
    T = set()
    
    for s in candidates:
        if is_sum_free_addition(s, T, n):
            S.add(s)
            T.add(s % n)
            T.add((-s) % n)
            
    return list(S)

def is_sum_free_addition(s, T, n):
    new_T = T | {s % n, (-s) % n}
    for t in T:
        if (s + t) % n in T: return False
        if (s + t) % n == s % n: return False
        if (s + t) % n == (-s) % n: return False
        if (-s + t) % n in T: return False
        if (-s + t) % n == s % n: return False
        if (-s + t) % n == (-s) % n: return False
    if (2 * s) % n in T: return False
    if (2 * s) % n == s % n: return False
    if (2 * s) % n == (-s) % n: return False
    if (-2 * s) % n in T: return False
    if (-2 * s) % n == s % n: return False
    if (-2 * s) % n == (-s) % n: return False
    return True

def calc_alpha_circulant(S, n):
    adj = [0] * n
    for s in S:
        s1 = s % n
        s2 = (-s) % n
        for u in range(n):
            adj[u] |= 1 << ((u + s1) % n)
            adj[u] |= 1 << ((u + s2) % n)
    for u in range(n):
        adj[u] &= ~(1 << u)
        
    best = [0]
    
    def dfs(size, mask):
        if size + bin(mask).count('1') <= best[0]:
            return
        if mask == 0:
            if size > best[0]:
                best[0] = size
            return
            
        max_deg = -1
        v = -1
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = bin(adj[idx] & mask).count('1')
            if deg > max_deg:
                max_deg = deg
                v = idx
            m ^= lsb
            
        new_mask = mask & ~adj[v] & ~(1 << v)
        dfs(size + 1, new_mask)
        
        dfs(size, mask & ~(1 << v))
        
    initial_mask = (1 << n) - 1
    dfs(0, initial_mask)
    return best[0]

def build_graph_from_S(S, n):
    edges = []
    S_set = set(S)
    for u in range(n):
        for v in range(u + 1, n):
            diff = (v - u) % n
            if diff in S_set or (-diff) % n in S_set:
                edges.append([u, v])
    return {'n': n, 'edges': edges}
