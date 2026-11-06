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
            # Neighbors of v are forbidden
            # We want non-neighbors
            # In bitmask, ~adj[v] gives non-neighbors, but need to mask to n bits
            # Also exclude v itself
            comp_neigh = (~adj[v]) & mask & ~(1 << v)
            new_cands = candidates & comp_neigh
            res = dfs(new_cands, size - 1)
            if res is not None:
                return [v] + res
        return None

    return dfs(mask, 6)

def _is_safe_to_add(adj, u, v, n):
    """
    Check if adding edge (u, v) creates a K4.
    A K4 is formed if there exist w, x such that:
    1. w, x are common neighbors of u and v.
    2. w and x are adjacent.
    """
    common = adj[u] & adj[v]
    # If there is an edge between any two vertices in 'common', adding (u,v) creates a K4.
    c = common
    while c:
        lsb = c & -c
        w = lsb.bit_length() - 1
        c ^= lsb
        # Check if w is connected to any other vertex in common
        # Remove w from common to check against others
        rest = common & ~(1 << w)
        if adj[w] & rest:
            return False
    return True

def _walksat_smart(adj, time_limit):
    n = len(adj)
    start_time = time.time()
    steps = 0
    max_steps = 10000
    perturb_every = 100
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
            # Prefer removing edges that are in many K4s
            edge_scores = []
            for e in edges:
                score = 0
                for k4k, wt in k4_weights.items():
                    if e[0] in k4k and e[1] in k4k:
                        score += wt
                edge_scores.append((score, e))
            edge_scores.sort(reverse=True)
            
            if random.random() < 0.8:
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
                
                pairs = list(itertools.combinations(indep, 2))
                random.shuffle(pairs)
                
                best_pair = None
                best_score = -1
                
                # Try to find a pair that is safe to add (doesn't create K4)
                # If multiple, pick one that breaks the most indep sets?
                # Or just pick the first safe one.
                for i, j in pairs:
                    if _is_safe_to_add(adj, i, j, n):
                        best_pair = (i, j)
                        best_score = 100
                        break
                
                if best_pair is None:
                    # If no safe pair, pick the one that creates the "least" K4 potential
                    # i.e., min number of edges in common neighbors
                    for i, j in pairs:
                        common = adj[i] & adj[j]
                        # Count edges in common subgraph
                        edge_count = 0
                        c = common
                        while c:
                            lsb = c & -c
                            w = lsb.bit_length() - 1
                            c ^= lsb
                            if adj[w] & common:
                                edge_count += 1
                        # Lower edge_count is better
                        score = -edge_count
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
            # Perturb: flip a few random edges
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
    deadline = time.time() + 115
    best_n = 0
    best_adj = None
    
    # Try to find n=35 first
    # Use structured initialization: random regular graph
    # Degree ~ 17 or 18
    
    for n in [35, 34, 33]:
        if time.time() > deadline - 10:
            break
            
        found = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Generate random regular graph
            adj = [0] * n
            degree = n // 2 + 1 # e.g., 18 for n=35
            
            # Simple random regular generation (may fail for small n/odd degree)
            # Use configuration model or just random init and force degree
            # For speed, just random init with p=0.5
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Check if it's close to regular? No, just run WalkSAT
            time_per_attempt = min(2.0, (deadline - time.time() - 5) / max(1, max_attempts - attempts))
            
            if _walksat_smart(adj, time_per_attempt):
                best_n = n
                best_adj = adj
                found = True
                break
        
        if found:
            break

    if best_n == 0:
        # Fallback to smaller n
        for n in range(32, 15, -1):
            if time.time() > deadline - 2:
                break
            for _ in range(10):
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                if _walksat_smart(adj, 1.0):
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
