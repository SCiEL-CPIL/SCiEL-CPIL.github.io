import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def check_valid_fast(adj, n, min_deg):
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() > 1:
                    return False
        return True

    def local_search(n, target_deg, seed, max_time=1.0):
        random.seed(seed)
        
        p = (target_deg + 2) / n
        p = min(p, 0.5)
        
        adj = [0] * n
        deg = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    deg[i] += 1
                    deg[j] += 1
        
        best_min_deg = min(deg)
        best_adj = adj[:]
        
        start = time.time()
        stuck_count = 0
        
        while time.time() - start < max_time:
            
            # 1. Repair C4s
            c4_found = False
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[i] & adj[j]
                    if common.bit_count() > 1:
                        common_list = []
                        mask = common
                        k = 0
                        while mask:
                            if mask & 1:
                                common_list.append(k)
                            mask >>= 1
                            k += 1
                        
                        w = random.choice(common_list)
                        
                        if deg[i] >= deg[j]:
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg[i] -= 1
                            deg[w] -= 1
                        else:
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg[j] -= 1
                            deg[w] -= 1
                        
                        c4_found = True
                        stuck_count = 0
                        break
                if c4_found:
                    break
            
            if c4_found:
                continue
            
            # 2. No C4s. Check min degree.
            current_min_deg = min(deg)
            if current_min_deg > best_min_deg:
                best_min_deg = current_min_deg
                best_adj = adj[:]
            
            if current_min_deg >= target_deg:
                return adj, current_min_deg
            
            # 3. Boost min degree
            u = min(range(n), key=lambda x: deg[x])
            
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                if (adj[u] & adj[v]) == 0:
                    candidates.append(v)
            
            if candidates:
                v = random.choice(candidates)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > 50:
                    # Kick: remove a random edge
                    edge_found = False
                    for i in range(n):
                        mask = adj[i]
                        if mask:
                            neighbors = []
                            m = mask
                            k = 0
                            while m:
                                if m & 1:
                                    neighbors.append(k)
                                m >>= 1
                                k += 1
                            if neighbors:
                                w = random.choice(neighbors)
                                adj[i] &= ~(1 << w)
                                adj[w] &= ~(1 << i)
                                deg[i] -= 1
                                deg[w] -= 1
                                edge_found = True
                                break
                    if not edge_found:
                        break
                    stuck_count = 0
        
        return best_adj, best_min_deg

    best_n = 0
    best_edges = []
    
    # Try n=46 first
    n_target = 46
    min_deg_target = n_target - 39
    for attempt in range(1000):
        if time.time() > deadline - 30.0: # Leave 30s for n=47
            break
        seed = attempt * 104729 + 13
        adj, min_deg_found = local_search(n_target, min_deg_target, seed, max_time=1.0)
        if min_deg_found >= min_deg_target:
            if check_valid_fast(adj, n_target, min_deg_target):
                best_n = n_target
                best_edges = get_edges(adj, n_target)
                # Don't return yet, try to improve with n=47
                break # Move to n=47 phase
    
    # Try n=47
    n_target = 47
    min_deg_target = n_target - 39
    for attempt in range(1000):
        if time.time() > deadline - 5.0:
            break
        seed = attempt * 104729 + 7919
        adj, min_deg_found = local_search(n_target, min_deg_target, seed, max_time=1.0)
        if min_deg_found >= min_deg_target:
            if check_valid_fast(adj, n_target, min_deg_target):
                return {'n': n_target, 'edges': get_edges(adj, n_target)}
    
    # Return best found (likely n=46 if found, else fallback)
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid_fast(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
