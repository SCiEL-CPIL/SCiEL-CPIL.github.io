import random
import time
import itertools

_cache = {}

def get_subsets(n, k):
    key = (n, k)
    if key not in _cache:
        _cache[key] = []
        for comb in itertools.combinations(range(n), k):
            mask = 0
            verts = []
            for v in comb:
                mask |= (1 << v)
                verts.append(v)
            _cache[key].append((mask, verts))
    return _cache[key]

def get_edge_subsets(n, k):
    key = (n, k)
    if key not in _cache:
        _cache[key] = {}
        for mask, verts in get_subsets(n, k):
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u = verts[i]
                    v = verts[j]
                    if u > v: u, v = v, u
                    ekey = (u, v)
                    if ekey not in _cache[key]:
                        _cache[key][ekey] = []
                    _cache[key][ekey].append((mask, verts))
    return _cache[key]

def is_regular_mask_verts(mask, verts, adj):
    degs = []
    for v in verts:
        deg = (adj[v] & mask).bit_count()
        degs.append(deg)
    return all(d == degs[0] for d in degs)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]:
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            masks_6 = get_subsets(n, 6)
            edge_masks_6 = get_edge_subsets(n, 6)
            
            cost = 0
            for mask, verts in masks_6:
                if is_regular_mask_verts(mask, verts, adj):
                    cost += 1
            
            steps_since_improvement = 0
            max_steps = 2000
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                if cost == 0:
                    valid = True
                    for k in range(7, n+1):
                        masks_k = get_subsets(n, k)
                        for mask, verts in masks_k:
                            if is_regular_mask_verts(mask, verts, adj):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        break
                
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                ekey = (u, v)
                if ekey not in edge_masks_6:
                    continue
                
                delta = 0
                for mask, verts in edge_masks_6[ekey]:
                    was_reg = is_regular_mask_verts(mask, verts, adj)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    is_reg_now = is_regular_mask_verts(mask, verts, adj)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    if was_reg and not is_reg_now:
                        delta -= 1
                    elif not was_reg and is_reg_now:
                        delta += 1
                
                if delta < 0:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    cost += delta
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
                    if steps_since_improvement > 200:
                        break

    return {'n': 5, 'edges': []}
