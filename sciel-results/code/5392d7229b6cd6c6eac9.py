import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best = {'n': seed_n, 'colors': seed_colors}
    
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
            
        time_limit = 30.0 if n == 17 else (20.0 if n == 16 else (15.0 if n == 15 else 10.0))
        target_end = time.time() + time_limit
        
        result = _search(n, target_end)
        if result:
            return result
            
    return best

def _search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    colors = [-1] * num_edges
    adj = [[0] * n for _ in range(4)]
    
    def check_c6(u, v, c):
        vis = [0] * 6
        vis[0] = 1 << u
        cur = 1 << u
        all_vis = 1 << u
        
        for d in range(1, 6):
            nxt = 0
            m = cur
            while m:
                lsb = m & -m
                node = lsb.bit_length() - 1
                m ^= lsb
                nxt |= adj[c][node]
            
            nxt &= ~all_vis
            vis[d] = nxt
            all_vis |= nxt
            cur = nxt
            
            if not cur:
                return False
            if (1 << v) & cur:
                return True
                
        return False

    def backtrack(idx):
        if time.time() > deadline:
            return False
            
        if idx == num_edges:
            return True
            
        u, v = edges[idx]
        
        if idx == 0:
            c_vals = [0]
        elif idx == 1:
            c_vals = [0, 1]
        elif idx == 2:
            c_vals = [0, 1, 2]
        else:
            c_vals = [0, 1, 2, 3]
            # random.shuffle(c_vals) # Might help, but symmetry breaking is stronger
            
        used_mask = 0
        for i in range(idx):
            if colors[i] != -1:
                used_mask |= (1 << colors[i])
                
        for c in c_vals:
            if idx > 2:
                if not (used_mask & (1 << c)):
                    is_smallest_unused = True
                    for cc in range(c):
                        if not (used_mask & (1 << cc)):
                            is_smallest_unused = False
                            break
                    if not is_smallest_unused:
                        continue
                        
            if check_c6(u, v, c):
                continue
                
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            if backtrack(idx + 1):
                return True
                
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1
            
        return False

    if backtrack(0):
        return {'n': n, 'colors': colors}
        
    return None
