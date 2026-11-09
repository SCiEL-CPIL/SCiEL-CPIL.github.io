import numpy as np
import random
import time

def construct():
    max_n = 35
    min_n = 17
    
    # Allocate most time to the hardest (largest) cases
    # We will try 35, 34, 33...
    # If 35 takes 100s and fails, we have 20s left for 34, etc.
    
    start_total = time.time()
    total_time_limit = 115.0 # Leave 5s for overhead/return
    
    for n in range(max_n, min_n - 1, -1):
        remaining = total_time_limit - (time.time() - start_total)
        if remaining < 1.0:
            break
            
        # Allocate time: 80% of remaining for the current largest n, rest for smaller
        # But if we fail, we move on.
        # Let's give 10 seconds for n < 35, and 100 seconds for n=35.
        if n == 35:
            timeout = min(remaining, 100.0)
        elif n == 34:
            timeout = min(remaining, 20.0)
        else:
            timeout = min(remaining, 5.0)
            
        if timeout < 1.0:
            continue

        found = _find_valid_graph(n, timeout)
        if found is not None:
            return found
            
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def _find_valid_graph(n, timeout):
    start_time = time.time()
    mask_all = (1 << n) - 1
    
    # Try multiple restarts
    # Use a better initialization: Random graph with p tuned
    # For R(4,6), p ~ 0.5 is standard
    
    for attempt in range(500): # More restarts if we have time
        if time.time() - start_time > timeout:
            break
        
        # Initialize
        p = 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        non_adj = [0] * n
        for i in range(n):
            non_adj[i] = mask_all ^ (1 << i) ^ adj[i]
        
        # Local Search with Simulated Annealing-like acceptance
        # But simple greedy is faster. Let's stick to greedy but with random tie-breaking
        # and more steps.
        
        current_energy = _calc_energy(adj, non_adj, n, mask_all)
        
        for step in range(5000):
            if time.time() - start_time > timeout:
                break
            
            if current_energy == 0:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # Find a violation to fix
            # Instead of finding ANY, find one that is "easy" to fix?
            # Or just pick random edge to flip if energy is high?
            
            # Strategy: If K4 exists, remove edge from it. If Indep6 exists, add edge to it.
            k4 = _find_k4(adj, n, mask_all)
            if k4:
                u, v, w, x = k4
                pair = random.choice([(u, v), (u, w), (u, x), (v, w), (v, x), (w, x)])
                a, b = pair
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                non_adj[a] |= (1 << b)
                non_adj[b] |= (1 << a)
                # Recalc energy only if needed? Or just continue?
                # Recalculating energy is expensive. 
                # We can just loop again.
                continue
            
            indep6 = _find_indep6(non_adj, n, mask_all)
            if indep6:
                i, j = random.sample(range(6), 2)
                a, b = indep6[i], indep6[j]
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                non_adj[a] &= ~(1 << b)
                non_adj[b] &= ~(1 << a)
                continue
                
            # If no violation found by _find_k4/_find_indep6, it must be valid
            # (These functions return None if not found)
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return None

def _calc_energy(adj, non_adj, n, mask_all):
    # This is a placeholder, we don't actually use it in the loop above to save time
    return 0

def _find_k4(adj, n, mask_all):
    for u in range(n):
        vs = adj[u] & (mask_all << (u + 1))
        while vs:
            lsb_v = vs & -vs
            v = lsb_v.bit_length() - 1
            vs ^= lsb_v
            
            ws = adj[u] & adj[v] & (mask_all << (v + 1))
            while ws:
                lsb_w = ws & -ws
                w = lsb_w.bit_length() - 1
                ws ^= lsb_w
                
                xs = adj[u] & adj[v] & adj[w] & (mask_all << (w + 1))
                if xs:
                    lsb_x = xs & -xs
                    x = lsb_x.bit_length() - 1
                    return (u, v, w, x)
    return None

def _find_indep6(non_adj, n, mask_all):
    def _find_clique_k(k, mask):
        if k == 0:
            return 1
        if k == 1:
            if mask:
                return mask & -mask
            return 0
        
        c = mask
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            
            next_cands = mask & non_adj[v] & (mask_all << (v + 1))
            if next_cands == 0:
                continue
            res = _find_clique_k(k - 1, next_cands)
            if res:
                return res | (1 << v)
        return 0

    for v1 in range(n):
        cands = non_adj[v1] & (mask_all << (v1 + 1))
        if cands == 0:
            continue
        
        res = _find_clique_k(5, cands)
        if res:
            full_mask = res | (1 << v1)
            verts = []
            m = full_mask
            while m:
                lsb = m & -m
                verts.append(lsb.bit_length() - 1)
                m ^= lsb
            return verts
    return None
