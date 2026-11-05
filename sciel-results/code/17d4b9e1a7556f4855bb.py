import time
import random

_START = time.time()
_DEADLINE = 145  # seconds, leave 5s margin for the 150s limit

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    """Check if graph has a K4. Returns the K4 vertices if found, else None."""
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            # Common neighbors of i and j, with index > j
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            if common == 0:
                continue
            # Check if any two common neighbors are connected
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                c ^= lsb
                # Check if k is connected to any other common neighbor
                if adj[k] & common:
                    # Find the specific l
                    c2 = adj[k] & common
                    lsb2 = c2 & -c2
                    l = lsb2.bit_length() - 1
                    return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    """Check if graph has an independent set of size 6. Returns the set if found, else None."""
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        
        # Pick vertex that minimizes remaining candidates after inclusion
        best_v = -1
        best_count = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            new_cand = candidates & ~adj[v]
            cnt = new_cand.bit_count()
            if cnt < best_count:
                best_count = cnt
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return None
            
        v = best_v
        new_candidates = candidates & ~((1 << v) | adj[v])
        
        # Branch 1: Include v
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        
        # Branch 2: Exclude v
        res = backtrack(candidates & ~(1 << v), current_set)
        if res:
            return res
        
        return None

    return backtrack((1 << n) - 1, [])

def _try_local_search(n, adj_init, time_limit, seed=None):
    """Try to find a valid graph on n vertices using local search from initial adj."""
    if seed is not None:
        random.seed(seed)
    start = time.time()
    
    # Copy initial adjacency
    adj = list(adj_init)
    
    # Local search loop
    while time.time() - start < time_limit:
        # Check for K4
        k4 = _has_k4(adj, n)
        if k4:
            # Remove a random edge from the K4
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        # Check for Ind 6
        ind6 = _has_ind_6(adj, n)
        if ind6:
            # Add a random edge between two vertices in the independent set
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        # If no K4 and no Ind 6, we found a valid graph
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    return None

def _random_adj(n, p_edge):
    """Create a random adjacency list with edge probability p_edge."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p_edge:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    # Start with Paley 17 as baseline
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    best_n = p
    
    # Try to find larger graphs from n=35 down to 18
    for n in range(35, 17, -1):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Time allocation: give a slice of time to each n
        # 18 values of n. 140s / 18 ~ 7.7s
        time_slice = 7.0
        
        if _elapsed() + time_slice > _DEADLINE - 3:
            time_slice = _DEADLINE - 3 - _elapsed()
            if time_slice < 0.5:
                break
        
        found = False
        attempts = 5
        
        for attempt in range(attempts):
            if _elapsed() > _DEADLINE - 3:
                break
            
            time_left = _DEADLINE - 3 - _elapsed()
            if time_left <= 0: break
            alloc = time_left / (attempts - attempt)
            
            # Use different densities and seeds
            p_edge = 0.4 + 0.05 * attempt
            seed = 42 + attempt * 100 + n
            adj_init = _random_adj(n, p_edge)
            edges = _try_local_search(n, adj_init, alloc, seed=seed)
            if edges is not None:
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            return {'n': best_n, 'edges': best_edges}
    
    return {'n': best_n, 'edges': best_edges}
