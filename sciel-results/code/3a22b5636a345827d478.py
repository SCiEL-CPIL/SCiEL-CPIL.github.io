import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
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

    def check_valid(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg."""
        # Check min degree
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        # Check C4: any two vertices share at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common.bit_count() > 1:
                    return False
        return True

    def greedy_build(n, target_deg):
        """
        Greedily construct a C4-free graph on n vertices aiming for min degree >= target_deg.
        Returns adj list of bitmasks or None if failed.
        """
        adj = [0] * n
        deg = [0] * n
        
        # We want to add edges such that we don't create C4.
        # A C4 is created if we add edge (u,v) and there exists w such that 
        # u-w-v is a path (i.e., w is common neighbor of u and v).
        # So, we can only add (u,v) if common_neighbors(u,v) == 0.
        
        # To maximize min degree, we should prioritize connecting low-degree vertices.
        # However, we must maintain the property.
        
        # Randomized greedy:
        # 1. Create a list of all potential edges.
        # 2. Shuffle them? No, that's slow.
        # 3. Iterate vertices, try to connect to others.
        
        # Better approach: 
        # Maintain a list of "available" edges.
        # Start with empty graph.
        # While there are low-degree vertices:
        #   Pick a vertex u with low degree.
        #   Find all v such that (u,v) not in graph and common_neighbors(u,v)==0.
        #   Add one such edge.
        
        # To make it fast, we can just iterate u from 0 to n-1 and try to add edges to v > u.
        # But this might leave some vertices isolated if we're not careful.
        
        # Let's try a multi-pass approach.
        # Pass 1: Connect each vertex to as many as possible.
        
        # We'll use a randomized order to break symmetries.
        order = list(range(n))
        random.shuffle(order)
        
        # We want to add edges.
        # For each u in order:
        #   Find candidates v (v != u) such that:
        #     1. Not adjacent
        #     2. No common neighbor
        #   If found, add edge.
        
        # This is O(n^3) if done naively.
        # For n=47, n^3 ~ 100k. This is fast.
        
        # We want to ensure min degree is high.
        # So, we should prioritize adding edges to vertices with currently lowest degree.
        
        # Let's simulate:
        # 1. Initialize adj, deg.
        # 2. Loop while we can add edges:
        #    Find u with min degree.
        #    Find v such that (u,v) can be added.
        #    Add (u,v).
        #    If no such v for min-degree u, break? Or pick next min?
        
        # To avoid getting stuck, we can allow some randomness.
        
        # Let's try a simpler, robust heuristic:
        # Iterate u from 0 to n-1.
        # For each u, try to connect to v (v > u) if possible.
        # To boost min degree, we might need to go back and connect earlier vertices to later ones?
        # Actually, if we process u in order, and try to connect to ALL possible v, we get a maximal C4-free graph.
        # Maximal C4-free graphs often have high min degree.
        
        # Let's try:
        for u in range(n):
            # Candidates for u
            # We want to connect u to v such that common_neighbors(u,v) == 0.
            # Note: if we add (u,v), it might block future edges for other vertices.
            # But we just want A valid graph with high min degree.
            
            # To maximize min degree, we should connect u to vertices that are currently "starving".
            # However, checking "starving" dynamically is complex.
            
            # Let's just try to add as many edges as possible for u.
            # Sort candidates by degree of v? Connect to low degree v first?
            
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                # Check common neighbors
                # common = adj[u] & adj[v]
                # if common.bit_count() == 0:
                #    candidates.append(v)
                
                # Optimization: if adj[u] is dense, common is likely non-zero.
                # But n is small.
                
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            # Sort candidates by degree (ascending) to help low-degree vertices?
            # Or random?
            # Let's sort by degree to balance.
            candidates.sort(key=lambda x: deg[x])
            
            # Add all?
            # If we add all, we might create C4s with future vertices?
            # No, C4 is local to the pair (u,v) and their common neighbors.
            # Adding (u,v) only creates C4 if there was a common neighbor.
            # So adding all valid candidates is safe for C4-freeness *at this step*.
            # However, it might reduce the options for other vertices.
            
            # To maximize min degree, adding all is good.
            for v in candidates:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        # Check if min degree is sufficient
        if min(deg) >= target_deg:
            return adj
        
        return None

    def local_search(adj, n, target_deg, max_time=10.0):
        """
        Improve min degree using edge swaps.
        Returns (adj, min_deg).
        """
        deg = [0] * n
        for i in range(n):
            deg[i] = adj[i].bit_count()
        
        best_min_deg = min(deg)
        best_adj = adj[:]
        
        start = time.time()
        no_improve = 0
        
        while time.time() - start < max_time:
            if no_improve > 500:
                break
                
            # Find vertex with min degree
            u = min(range(n), key=lambda i: deg[i])
            current_min = deg[u]
            
            if current_min >= target_deg:
                # Try to increase further or break if stable
                if best_min_deg >= target_deg:
                     break
                # If we met target, we can stop or try to polish
                # For now, stop if we met target
                if check_valid(adj, n, target_deg):
                    return adj, best_min_deg
            
            # Try to add an edge incident to u
            # Find v such that (u,v) can be added
            added = False
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            if candidates:
                # Pick random candidate
                v = random.choice(candidates)
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                added = True
            else:
                # No edge can be added to u.
                # Try an edge swap?
                # Remove an edge (a,b) and add (c,d)?
                # Or remove (a,b) to free up space?
                # This is complex. Let's try removing an edge incident to u?
                # If we remove an edge (u,w), we might be able to add (u,v).
                
                # Find a neighbor w of u
                neighbors = []
                mask = adj[u]
                k = 0
                while mask:
                    if mask & 1:
                        neighbors.append(k)
                    mask >>= 1
                    k += 1
                
                if neighbors:
                    w = random.choice(neighbors)
                    # Remove (u,w)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    deg[u] -= 1
                    deg[w] -= 1
                    
                    # Now try to add an edge to u again
                    candidates2 = []
                    for v in range(n):
                        if u == v:
                            continue
                        if adj[u] & (1 << v):
                            continue
                        common = adj[u] & adj[v]
                        if common == 0:
                            candidates2.append(v)
                    
                    if candidates2:
                        v = random.choice(candidates2)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        deg[u] += 1
                        deg[v] += 1
                        # We effectively swapped (u,w) for (u,v)
                        # Net change in deg[u] is 0, but deg[w] decreased, deg[v] increased.
                        # This might help if v was low degree and w was high.
                        # But we specifically picked u as min-degree.
                        # So this might not help u.
                        
                        # Actually, if u is min-degree, removing an edge makes it lower.
                        # Then adding a new edge brings it back.
                        # This is a "move".
                        
                        # Check if min degree improved
                        new_min = min(deg)
                        if new_min > best_min_deg:
                            best_min_deg = new_min
                            best_adj = adj[:]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        deg[u] += 1
                        deg[w] += 1
                        no_improve += 1
                else:
                    no_improve += 1
                    continue
            
            if added:
                new_min = min(deg)
                if new_min > best_min_deg:
                    best_min_deg = new_min
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    # Keep the edge anyway? It might help later.
                    # But if min didn't improve, we might be stuck.
                    # Let's keep it to explore.
                    no_improve += 1
                    if no_improve > 200:
                         # Revert last move? Complex.
                         # Just break and restart?
                         pass

        return best_adj, best_min_deg

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        found = False
        
        # Strategy 1: Greedy Build
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 123 + 456)
            adj = greedy_build(n, min_deg)
            
            if adj is not None:
                if check_valid(adj, n, min_deg):
                    edges = get_edges(adj, n)
                    return {'n': n, 'edges': edges}
                
                # If greedy build failed min_deg, try local search to fix it
                adj_ls, min_deg_ls = local_search(adj, n, min_deg, max_time=5.0)
                if min_deg_ls >= min_deg and check_valid(adj_ls, n, min_deg):
                    edges = get_edges(adj_ls, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 2: Random Initialization + Repair
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 7919 + 13)
            
            # Initialize with random graph
            p = min(min_deg / n * 1.3, 0.7)
            adj = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Repair phase
            improved = True
            max_iter = 100
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Check for C4s
                violating_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if common.bit_count() > 1:
                            violating_pairs.append((i, j))
                
                if violating_pairs:
                    # Pick a violating pair and fix it
                    i, j = random.choice(violating_pairs)
                    common = adj[i] & adj[j]
                    common_list = []
                    mask = common
                    k = 0
                    while mask:
                        if mask & 1:
                            common_list.append(k)
                        mask >>= 1
                        k += 1
                    
                    # Remove edges to reduce common neighbors to 1
                    deg_i = adj[i].bit_count()
                    deg_j = adj[j].bit_count()
                    
                    for w in common_list[1:]:
                        if deg_i > deg_j:
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg_i -= 1
                        elif deg_j > deg_i:
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg_j -= 1
                        else:
                            if random.random() < 0.5:
                                adj[i] &= ~(1 << w)
                                adj[w] &= ~(1 << i)
                                deg_i -= 1
                            else:
                                adj[j] &= ~(1 << w)
                                adj[w] &= ~(1 << j)
                                deg_j -= 1
                else:
                    # No C4s, check min degree
                    min_deg_actual = min(adj[i].bit_count() for i in range(n))
                    if min_deg_actual >= min_deg:
                        if check_valid(adj, n, min_deg):
                            edges = get_edges(adj, n)
                            return {'n': n, 'edges': edges}
                        break
                    else:
                        # Need to add edges
                        low_verts = [i for i in range(n) if adj[i].bit_count() < min_deg]
                        if not low_verts:
                            break
                        u = random.choice(low_verts)
                        candidates = []
                        for v in range(n):
                            if v == u:
                                continue
                            if not (adj[u] & (1 << v)):
                                common = adj[u] & adj[v]
                                if common == 0:
                                    candidates.append(v)
                        
                        if candidates:
                            v = random.choice(candidates)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            pass

    # Fallback: Return a valid small graph
    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
