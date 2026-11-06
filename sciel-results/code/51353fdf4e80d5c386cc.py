import random
import time

def _find_k4(adj):
    n = len(adj)
    # Iterate edges
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    c ^= lsb
                    # Check if w and any other in common are connected
                    # common includes w? No, common is neighbors of u and v.
                    # w is in common.
                    # We need an edge between w and x where x is in common \ {w}
                    # and x is a neighbor of w.
                    # adj[w] & common gives neighbors of w that are also neighbors of u and v.
                    # If this is non-zero, we have a K4.
                    if adj[w] & common:
                        # Find x
                        cw = adj[w] & common
                        lsb2 = cw & -cw
                        x = lsb2.bit_length() - 1
                        return (u, v, w, x)
    return None

def _find_indep6(adj):
    n = len(adj)
    mask = (1 << n) - 1
    
    # Heuristic: sort vertices by degree in complement (ascending)
    # deg_comp(v) = n - 1 - popcount(adj[v])
    # We want to pick vertices with few complement neighbors first.
    order = sorted(range(n), key=lambda v: bin(adj[v]).count('1'), reverse=True)
    # Wait, high degree in original = low degree in complement.
    # So we want to pick high degree original vertices first?
    # If a vertex has high degree in original, it has few non-neighbors.
    # So it's less likely to be in an independent set?
    # Actually, if we are looking for an independent set, we want vertices that are NOT connected.
    # So we want vertices with few complement edges.
    # So yes, pick high degree original vertices first?
    # No, if I pick a vertex with high original degree, it has few complement neighbors.
    # So the candidate set shrinks fast.
    
    # Let's just use the order 0..n-1 for simplicity, but maybe randomize order?
    # Randomizing the order in the DFS can help escape local minima in the search for the set.
    
    def dfs(candidates, size, order):
        if size == 0:
            return []
        if candidates.bit_count() < size:
            return None
        
        # Pick next vertex from order that is in candidates
        # To make it heuristic, we can try the first one in the sorted order that is available
        # Or just iterate through candidates.
        
        c = candidates
        # To add randomness/heuristic, we can shuffle the iteration order of candidates
        # But that's expensive.
        
        # Standard DFS
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            # Neighbors in complement
            comp_neigh = (~adj[v]) & mask & ~(1 << v)
            new_cands = candidates & comp_neigh
            res = dfs(new_cands, size - 1, order)
            if res is not None:
                return [v] + res
        return None

    return dfs(mask, 6, order)

def _local_search_with_perturbation(adj, time_limit):
    n = len(adj)
    start_time = time.time()
    steps = 0
    max_steps = 20000
    perturb_every = 100
    
    while time.time() - start_time < time_limit:
        if steps >= max_steps:
            break
            
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
                # No K4 and no Ind6
                return True
                
        steps += 1
        
        if steps % perturb_every == 0:
            # Perturbation: flip 3 random edges
            for _ in range(3):
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
    
    # Try n=35 down to 25
    for n in range(35, 24, -1):
        if time.time() > deadline - 5:
            break
            
        attempts = 0
        max_attempts = 50
        while time.time() < deadline - 5:
            attempts += 1
            # Random initialization
            # Bias towards 0.5
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Time per attempt
            time_per_attempt = min(1.0, (deadline - time.time() - 5) / max(1, max_attempts - attempts))
            
            if _local_search_with_perturbation(adj, time_per_attempt):
                best_n = n
                best_adj = adj
                break
            if attempts >= max_attempts:
                break
    
    if best_n == 0:
        # Fallback to smaller n
        for n in range(24, 15, -1):
            if time.time() > deadline - 1:
                break
            for _ in range(20):
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                if _local_search_with_perturbation(adj, 0.5):
                    best_n = n
                    best_adj = adj
                    break
            if best_n > 0:
                break

    if best_n == 0:
        # Absolute fallback: n=4 C4
        best_n = 4
        best_adj = [9, 3, 6, 12]

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
