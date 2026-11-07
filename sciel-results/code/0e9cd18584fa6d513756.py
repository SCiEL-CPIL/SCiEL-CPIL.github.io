"""Combined approach: degree-sequence guided construction with bitmask validity."""
import time
import random
import itertools

N = 15
DEADLINE = 55.0  # Safety margin under the 90s limit

def _adjacency(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _degrees(adj):
    return [bin(adj[i]).count('1') for i in range(N)]

def _neighbors(adj):
    """Precompute neighbor lists for faster iteration."""
    nbrs = []
    for i in range(N):
        nbrs.append([j for j in range(N) if (adj[i] >> j) & 1])
    return nbrs

def _is_valid(adj, deg):
    # For every edge (u, v) with u < v, check all simple 3-edge paths s-u-v-t
    nbrs = _neighbors(adj)
    for u in range(N):
        for v in nbrs[u]:
            if v <= u:
                continue
            for s in nbrs[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in nbrs[v]:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _valid(edges):
    adj = _adjacency(edges)
    deg = _degrees(adj)
    return _is_valid(adj, deg)

def _greedy_add(order, start_edges):
    edges = list(start_edges)
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    adj = _adjacency(edges)
    deg = _degrees(adj)
    for (u, v) in order:
        key = (min(u, v), max(u, v))
        if key in present:
            continue
        # Quick degree check: adding edge increases deg[u] and deg[v]
        trial_adj = adj[:]
        trial_adj[u] |= (1 << v)
        trial_adj[v] |= (1 << u)
        trial_degs = deg[:]
        trial_degs[u] += 1
        trial_degs[v] += 1
        if _is_valid(trial_adj, trial_degs):
            edges.append([u, v])
            present.add(key)
            adj = trial_adj
            deg = trial_degs
    return edges

def _erdos_gallai(degs):
    """Check if degree sequence is graphical using Erdős–Gallai theorem."""
    n = len(degs)
    if sum(degs) % 2 != 0:
        return False
    if any(d < 0 or d > n - 1 for d in degs):
        return False
    s = sorted(degs, reverse=True)
    total = sum(s)
    for k in range(1, n + 1):
        left = sum(s[:k])
        right = k * (k - 1) + sum(min(d, k) for d in s[k:])
        if left > right:
            return False
    return True

def _try_degree_partition(target_degs, rng, t0):
    """Try to construct a graph with the given degree sequence that is valid."""
    best = []
    best_len = 0
    
    for attempt in range(30):
        if time.time() - t0 > DEADLINE:
            break
        
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(all_edges)
        
        cur_degs = [0] * N
        edges = []
        adj = [0] * N
        
        for u, v in all_edges:
            if time.time() - t0 > DEADLINE:
                break
            if cur_degs[u] >= target_degs[u] or cur_degs[v] >= target_degs[v]:
                continue
            
            trial_adj = adj[:]
            trial_adj[u] |= (1 << v)
            trial_adj[v] |= (1 << u)
            trial_degs = cur_degs[:]
            trial_degs[u] += 1
            trial_degs[v] += 1
            
            if _is_valid(trial_adj, trial_degs):
                edges.append([u, v])
                adj = trial_adj
                cur_degs = trial_degs
        
        if len(edges) > best_len:
            best_len = len(edges)
            best = edges
    
    return best, best_len

def _local_search(adj, deg, t0, rng, max_iter=200):
    """Local search: try to add missing edges, remove problematic ones."""
    nbrs = _neighbors(adj)
    improved = True
    iters = 0
    while improved and iters < max_iter and time.time() - t0 < DEADLINE:
        improved = False
        iters += 1
        # Try adding each missing edge
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    continue
                trial_adj = adj[:]
                trial_adj[u] |= (1 << v)
                trial_adj[v] |= (1 << u)
                trial_degs = deg[:]
                trial_degs[u] += 1
                trial_degs[v] += 1
                if _is_valid(trial_adj, trial_degs):
                    adj = trial_adj
                    deg = trial_degs
                    improved = True
                    return adj, deg
        # If no edge can be added, try removing one to enable others
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    continue
                trial_adj = adj[:]
                trial_adj[u] &= ~(1 << v)
                trial_adj[v] &= ~(1 << u)
                trial_degs = deg[:]
                trial_degs[u] -= 1
                trial_degs[v] -= 1
                if _is_valid(trial_adj, trial_degs):
                    # Check if we can then add more edges
                    adj2, deg2 = _local_search(trial_adj, trial_degs, t0, rng, max_iter=50)
                    if bin(adj2[0]).count('1') + bin(adj2[1]).count('1') > bin(adj[0]).count('1') + bin(adj[1]).count('1'):
                        adj = adj2
                        deg = deg2
                        improved = True
                        return adj, deg
    return adj, deg

def construct():
    t0 = time.time()
    rng = random.Random(20240607)
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Strong baseline: K_{7,8} has 56 edges and is valid
    k78 = [[a, b] for a in range(7) for b in range(7, 15)]
    best = k78
    best_len = len(k78)
    
    def consider(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand
    
    # Phase 1: Greedy from K_{7,8} with multiple random orders
    for _ in range(15):
        if time.time() - t0 > DEADLINE:
            break
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy_add(order, k78)
        consider(cand)
    
    # Phase 2: Local search from best
    if time.time() - t0 < DEADLINE - 5:
        adj = _adjacency(best)
        deg = _degrees(adj)
        adj2, deg2 = _local_search(adj, deg, t0, rng)
        edges2 = [[u, v] for u in range(N) for v in range(u + 1, N) if (adj2[u] >> v) & 1]
        consider(edges2)
    
    # Phase 3: Try different degree sequences
    candidate_degs = []
    for _ in range(100):
        if time.time() - t0 > DEADLINE:
            break
        degs = [8] * 7 + [7] * 8
        for _ in range(5):
            i = rng.randint(0, N - 1)
            if degs[i] < N - 1:
                degs[i] += 1
        for _ in range(3):
            i = rng.randint(0, N - 1)
            if degs[i] > 1:
                degs[i] -= 1
        
        if _erdos_gallai(degs):
            candidate_degs.append(degs)
    
    for degs in candidate_degs[:30]:
        if time.time() - t0 > DEADLINE:
            break
        if sum(degs) // 2 <= best_len:
            continue
        cand, cl = _try_degree_partition(degs, rng, t0)
        consider(cand)
    
    # Phase 4: Local search from best
    if time.time() - t0 < DEADLINE - 5:
        adj = _adjacency(best)
        deg = _degrees(adj)
        adj2, deg2 = _local_search(adj, deg, t0, rng)
        edges2 = [[u, v] for u in range(N) for v in range(u + 1, N) if (adj2[u] >> v) & 1]
        consider(edges2)
    
    # Phase 5: Random restarts with repair from dense random graphs
    for _ in range(20):
        if time.time() - t0 > DEADLINE:
            break
        
        adj = [0] * N
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.85:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        deg = _degrees(adj)
        for _ in range(50):
            if time.time() - t0 > DEADLINE:
                break
            if _is_valid(adj, deg):
                break
            found = False
            for u in range(N):
                if found:
                    break
                for v in range(u + 1, N):
                    if not (adj[u] >> v) & 1:
                        continue
                    for s in range(N):
                        if s == u or s == v or not (adj[u] >> s) & 1:
                            continue
                        for t in range(N):
                            if t == u or t == s or not (adj[v] >> t) & 1:
                                continue
                            if deg[s] == deg[t]:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                deg[u] -= 1
                                deg[v] -= 1
                                edges = [[u2, v2] for u2 in range(N) for v2 in range(u2 + 1, N) if (adj[u2] >> v2) & 1]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        consider(edges)
    
    # Final local search polish
    if time.time() - t0 < DEADLINE - 3:
        adj = _adjacency(best)
        deg = _degrees(adj)
        adj2, deg2 = _local_search(adj, deg, t0, rng)
        edges2 = [[u, v] for u in range(N) for v in range(u + 1, N) if (adj2[u] >> v) & 1]
        consider(edges2)
    
    result = [[int(u), int(v)] for u, v in best]
    return result
