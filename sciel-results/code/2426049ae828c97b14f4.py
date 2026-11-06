import time
import random

def paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = []
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                edges.append([i, j])
    return {'n': 17, 'edges': edges}

def construct():
    start_time = time.time()
    deadline = 115.0
    
    # Try n from 35 down to 18
    # If we find a valid graph, return it.
    for n in range(35, 17, -1):
        if time.time() > deadline:
            break
        
        result = solve_n(n, deadline)
        if result is not None:
            return result
            
    # Fallback
    return paley_17()

def solve_n(n, deadline):
    # ILS
    for restart in range(50):
        if time.time() > deadline:
            return None
            
        # Init
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local Search
        for step in range(20000):
            if time.time() > deadline:
                return None
            
            # Check K4
            k4 = find_k4(adj, n)
            if k4:
                u, v, w, x = k4
                # Remove random edge
                e = random.choice([(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)])
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
                continue
            
            # Check I6
            i6 = find_i6(adj, n)
            if i6:
                u, v = random.sample(i6, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
            
            # Success
            return convert_to_edges(adj, n)
            
        # Perturb
        for _ in range(10):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    return None

def find_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                # Find two connected vertices in common
                # Iterate w in common
                w = (common & -common).bit_length() - 1 # Get first set bit index?
                # Actually, just loop
                common_list = [w for w in range(n) if (common >> w) & 1]
                for i in range(len(common_list)):
                    w = common_list[i]
                    for j in range(i + 1, len(common_list)):
                        x = common_list[j]
                        if (adj[w] >> x) & 1:
                            return (u, v, w, x)
    return None

def find_i6(adj, n):
    def get_degree(v):
        return bin(adj[v]).count('1')
    
    candidates = list(range(n))
    
    def dfs(current, available):
        if len(current) == 6:
            return current
        if not available:
            return None
        
        # Pick min degree vertex in available
        min_deg = float('inf')
        min_idx = -1
        for i, v in enumerate(available):
            d = get_degree(v)
            if d < min_deg:
                min_deg = d
                min_idx = i
        
        v = available[min_idx]
        
        # Check independence from current
        is_indep = True
        for u in current:
            if (adj[u] >> v) & 1:
                is_indep = False
                break
        
        if is_indep:
            new_current = current + [v]
            new_available = []
            for w in available:
                if w == v: continue
                if not ((adj[v] >> w) & 1):
                    new_available.append(w)
            
            res = dfs(new_current, new_available)
            if res:
                return res
        return None

    return dfs([], candidates)

def convert_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
