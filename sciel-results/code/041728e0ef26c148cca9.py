import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for overhead/return
    
    # 1. Generate baseline LUW graph
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Try to find a smaller graph using random local search
    # Moore bound lower limit
    if g % 2 == 0:
        h = g // 2
        moore = k + k * (k-1) * ( (k-1)**(h-1) - 1 ) // (k-2)
    else:
        h = (g - 1) // 2
        moore = 2 * k * ( (k-1)**h - 1 ) // (k-2)
        
    # We know n must be at least moore.
    # We also know n must allow a k-regular graph (k*n even).
    start_n = max(moore, 10)
    
    # We will try n from start_n up to n_luw - 1
    # But n_luw might be huge. We should prioritize n close to the theoretical random threshold.
    # Expected girth ~ 2 log_k n.
    # For girth g, n ~ k^(g/2).
    # Let's start near that estimate.
    n_estimate = int(k ** (g / 2.0))
    n_estimate = max(n_estimate, start_n)
    
    # We will test n values in a range around the estimate and up to n_luw.
    # To save time, we'll test a few specific n values.
    # If n_luw is not much larger than n_estimate, we can scan.
    
    # Strategy: Try n = n_estimate, n_estimate + 10, ... up to n_luw
    # Or just try n_estimate and n_luw - 100?
    # Actually, if n_luw is 2000 and n_estimate is 1100, we should try 1100, 1150, 1200...
    
    # Let's define a list of candidate n's to try.
    candidates = []
    # Add n_estimate
    candidates.append(n_estimate)
    # Add some steps up to n_luw
    step = max(10, (n_luw - n_estimate) // 10)
    for i in range(1, 11):
        n_cand = n_estimate + i * step
        if n_cand >= n_luw:
            break
        candidates.append(n_cand)
        
    # Ensure candidates are sorted and valid (k*n even)
    unique_candidates = sorted(set(candidates))
    # Filter for k*n even
    valid_candidates = [n for n in unique_candidates if (n * k) % 2 == 0]
    
    # If n_luw is small, just scan from moore to n_luw
    if n_luw - start_n < 100:
        valid_candidates = [n for n in range(start_n, n_luw) if (n * k) % 2 == 0]
        
    for n in valid_candidates:
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a graph of size n
        # Time budget per n: proportional to remaining time / remaining candidates
        remaining_candidates = len([c for c in valid_candidates if c >= n])
        time_per_candidate = (deadline - time.time() - 2.0) / max(1, remaining_candidates)
        time_per_candidate = max(0.1, min(5.0, time_per_candidate)) # Cap between 0.1 and 5s
        
        success, edges = _random_local_search(n, k, g, time_per_candidate)
        if success:
            best_n = n
            best_edges = edges
            # Since we are scanning upwards from estimate, the first success is likely good.
            # But we want the MINIMUM n. 
            # If we found one at n, we should check if a smaller n works?
            # We are scanning from n_estimate upwards. 
            # If n_estimate is too low, we might fail.
            # If we succeed at n, it's a valid upper bound.
            # To minimize n, we should have scanned from low to high.
            # My candidate list starts at n_estimate. 
            # If n_estimate < true_min, we fail. Then we try higher.
            # If we succeed at n, it's the first success in this upward scan.
            # Is it the global minimum? Not necessarily, if n_estimate was > true_min.
            # But n_estimate is based on random graph theory, which is a good lower bound proxy.
            # So the first success is likely close to optimal.
            # We can break here to save time for return.
            break
            
    return best_n, best_edges

def _luw_construct(k, g):
    # Copy from seed
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _random_local_search(n, k, g, time_limit):
    """
    Try to find a k-regular graph of girth >= g on n vertices.
    Returns (True, edges) if found, else (False, None).
    """
    if (n * k) % 2 != 0:
        return False, None
    
    start_time = time.time()
    
    # 1. Generate random k-regular graph
    # Using configuration model
    # List of stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    # Shuffle stubs
    random.shuffle(stubs)
    
    # Pair stubs to form edges
    # Use a dict to track adjacency
    adj = [set() for _ in range(n)]
    
    # We need to ensure simple graph (no loops, no multi-edges)
    # Configuration model can produce loops/multi-edges.
    # We will retry generation if invalid, or repair.
    # For simplicity, we'll generate and then repair loops/multi-edges via swaps.
    
    # Initial pairing
    edges = []
    valid = True
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            valid = False
            break
        if v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
        
    if not valid:
        # Retry a few times
        for _ in range(10):
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            edges = []
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
            if valid:
                break
        
        if not valid:
            return False, None
            
    # 2. Check girth
    def get_girth(adj, n):
        # BFS from each node to find shortest cycle
        # Shortest cycle through node u is min over neighbors v of (dist(u, v) + 1)
        # where dist is shortest path not using edge (u,v) directly? 
        # Actually, standard BFS from u, record dist.
        # For each edge (u, v), cycle length is dist[u->v] + 1.
        # We need dist from u to v in graph without edge (u,v)?
        # No, BFS from u gives shortest path to v. If the path doesn't use (u,v), cycle is dist+1.
        # If the path uses (u,v), it's not a simple cycle back to u via that edge.
        # Standard method: BFS from u. For each neighbor v, if we find a path to v of length L, 
        # and the first step was u->v, we must be careful.
        # Better: For each edge (u, v), compute shortest path from u to v in G - (u,v). 
        # Cycle length = dist + 1.
        # This is O(E * (V+E)) which is too slow for n=1000.
        
        # Faster: BFS from each node u. 
        # dist[u] = 0.
        # For each neighbor v of u, BFS from u. 
        # When we reach a neighbor w of u, if w != v (start node), we have a cycle of length dist[w] + 1.
        # We want the min such cycle over all u and all w.
        
        min_girth = float('inf')
        for u in range(n):
            # BFS from u
            dist = [-1] * n
            parent = [-1] * n
            queue = [u]
            dist[u] = 0
            idx = 0
            found_cycle = False
            while idx < len(queue):
                curr = queue[idx]
                idx += 1
                for nb in adj[curr]:
                    if dist[nb] == -1:
                        dist[nb] = dist[curr] + 1
                        parent[nb] = curr
                        queue.append(nb)
                    elif nb != u and parent[curr] != nb: 
                        # Found a cycle
                        # Cycle length: dist[curr] + dist[nb] + 1
                        # But this finds ANY cycle. 
                        # We want the shortest cycle. 
                        # Since BFS explores in order of distance, the first time we see a back-edge,
                        # is it the shortest?
                        # Yes, for the specific start node u, the shortest cycle through u 
                        # is found when we first encounter a neighbor of u (other than the start parent) 
                        # OR when we encounter a cross edge?
                        # Actually, standard BFS from u:
                        # If we find an edge (curr, nb) where dist[nb] < dist[curr], it's a back edge.
                        # If dist[nb] == dist[curr], it's a cross edge (in undirected, this implies a cycle).
                        # Cycle length = dist[curr] + dist[nb] + 1.
                        # We track the min.
                        cycle_len = dist[curr] + dist[nb] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth >= g:
                                # We can stop early if we already satisfy girth? 
                                # No, we need to ensure NO cycle is shorter than g.
                                # So we must find the GLOBAL minimum girth.
                                pass
            # Optimization: If min_girth is already < g, we know we fail.
            # We can break early if min_girth < g? 
            # No, we need the exact min to know if it's >= g.
            # But if we find a cycle of length < g, we know the girth is < g.
            # So we can return early with girth < g.
            if min_girth < g:
                return min_girth
                
        return min_girth
    
    # Initial girth check
    girth = get_girth(adj, n)
    if girth >= g:
        return True, edges
        
    # 3. Local Search
    # While time remains:
    #   Find a short cycle
    #   Swap edges to break it
    #   Re-check girth
    
    while time.time() < start_time + time_limit:
        # Find shortest cycle
        # We need to identify the cycle to break it.
        # The get_girth function above doesn't return the cycle.
        # Let's modify it to return the cycle edges or nodes.
        
        # For efficiency, we'll just do random swaps and check girth.
        # Random 2-opt swap:
        # Pick two random edges (u, v) and (w, x).
        # Remove them.
        # Try adding (u, w) and (v, x).
        # Check if valid (no loops, no multi-edges).
        # If valid, perform swap.
        # Check new girth.
        # If girth improved, keep. Else revert.
        
        # To bias towards breaking short cycles, we could pick edges from short cycles,
        # but finding them is expensive. Random swaps are effective in practice.
        
        # Pick random edges
        # We have a list of edges.
        # But adj is a set of sets. 
        # Let's maintain a list of edges for easy picking.
        # Note: edges list might get out of sync if we don't update it carefully.
        # Let's just pick random u, v from adj.
        
        u = random.randint(0, n-1)
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        
        w = random.randint(0, n-1)
        if not adj[w]:
            continue
        x = random.choice(list(adj[w]))
        
        # Ensure edges are distinct and not sharing nodes in a way that causes issues?
        # If u==w or v==x etc, swap might be trivial or invalid.
        if u == w or u == x or v == w or v == x:
            continue
            
        # Check if (u, w) and (v, x) are already edges
        if w in adj[u] or x in adj[v]:
            continue
        # Check if (u, x) and (v, w) are already edges (for the other swap type)
        # We'll try swap 1: (u,v)->(u,w), (w,x)->(v,x) is not right.
        # Remove (u,v) and (w,x).
        # Add (u,w) and (v,x).
        
        # Check validity of new edges
        if w == u or x == v: # Loops
            continue
        if w in adj[u] or x in adj[v]: # Multi edges
            continue
            
        # Perform swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[w].remove(x)
        adj[x].remove(w)
        
        adj[u].add(w)
        adj[w].add(u)
        adj[v].add(x)
        adj[x].add(v)
        
        # Check girth
        new_girth = get_girth(adj, n)
        if new_girth >= g:
            # Success
            # Reconstruct edges list
            final_edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        final_edges.append((i, j))
            return True, final_edges
        else:
            # Revert
            adj[u].remove(w)
            adj[w].remove(u)
            adj[v].remove(x)
            adj[x].remove(v)
            
            adj[u].add(v)
            adj[v].add(u)
            adj[w].add(x)
            adj[x].add(w)
            
    return False, None
