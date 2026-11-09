import time
import random

def _check_validity(colors):
    n = len(colors)
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if len({colors[a], colors[a+d], colors[a+2*d], colors[a+3*d]}) < 3:
                return False
    return True

def _search_k(k, time_limit=5.0):
    n = 128
    colors = [-1] * n
    nodes_visited = 0
    limit = 100000 # Node limit per k
    
    def dfs(i):
        nonlocal nodes_visited
        if i == n:
            return True
        nodes_visited += 1
        if nodes_visited > limit:
            return False
        
        # Heuristic: try colors that appear least in previous?
        # Or just 0..k-1
        for c in range(k):
            colors[i] = c
            valid = True
            # Check APs ending at i
            for d in range(1, i//3 + 1):
                a = i - 3*d
                b = a + d
                c_idx = a + 2*d
                if len({colors[a], colors[b], colors[c_idx], colors[i]}) < 3:
                    valid = False
                    break
            
            if valid:
                if dfs(i+1):
                    return True
            colors[i] = -1
        return False

    start = time.time()
    # Try a few random orders
    for _ in range(5):
        nodes_visited = 0
        # Shuffle color order
        order = list(range(k))
        random.shuffle(order)
        
        # Monkey patch dfs to use order?
        # Easier: just use standard order but randomize start?
        # Or just run standard DFS.
        if dfs(0):
            if _check_validity(colors):
                return colors[:]
        
        if time.time() - start > time_limit:
            break
            
    return None

def construct():
    # Try k=2
    # Simple random restart local search for k=2
    n = 128
    aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
            
    def count_violations(c):
        v = 0
        for a, b, c_idx, d_idx in aps:
            if len({c[a], c[b], c[c_idx], c[d_idx]}) < 3:
                v += 1
        return v

    # Try a few random seeds for local search
    for _ in range(100):
        c = [random.randint(0, 1) for _ in range(n)]
        v = count_violations(c)
        if v == 0:
            return c
        # Hill climbing
        improved = True
        while improved:
            improved = False
            for i in range(n):
                old = c[i]
                c[i] = 1 - old
                new_v = count_violations(c)
                if new_v < v:
                    v = new_v
                    improved = True
                    if v == 0:
                        return c
                else:
                    c[i] = old
        if v == 0:
            return c

    # Try k=3, 4, 5 with DFS
    for k in range(3, 10):
        res = _search_k(k, time_limit=2.0)
        if res:
            return res

    # Fallback
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
