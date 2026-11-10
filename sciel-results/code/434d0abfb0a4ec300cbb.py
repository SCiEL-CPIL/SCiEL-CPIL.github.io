import time
import random

DEADLINE = 145.0  # Safety margin before 150s kill

def _is_c6_free_after_add(adj, u, v, n=40):
    """
    Check if adding edge (u, v) to a C6-free graph creates a C6.
    Assumes adj is a list of int bitmasks.
    A C6 containing (u, v) is u-v-x-y-z-w-u.
    We need a path v-x-y-z-w-u of length 5 with distinct vertices x,y,z,w not in {u,v}.
    """
    # Neighbors of v excluding u
    v_mask = adj[v] & ~(1 << u)
    while v_mask:
        x = (v_mask & (-v_mask)).bit_length() - 1
        v_mask &= v_mask - 1
        
        # Neighbors of x excluding v and u
        x_mask = adj[x] & ~(1 << v) & ~(1 << u)
        while x_mask:
            y = (x_mask & (-x_mask)).bit_length() - 1
            x_mask &= x_mask - 1
            
            # Neighbors of y excluding x and u
            y_mask = adj[y] & ~(1 << x) & ~(1 << u)
            while y_mask:
                z = (y_mask & (-y_mask)).bit_length() - 1
                y_mask &= y_mask - 1
                
                # Neighbors of z excluding y and u
                z_mask = adj[z] & ~(1 << y) & ~(1 << u)
                while z_mask:
                    w = (z_mask & (-z_mask)).bit_length() - 1
                    z_mask &= z_mask - 1
                    
                    # Check if w is connected to u
                    # w must be distinct from x, y, z (ensured by traversal exclusions? 
                    # x != y, y != z, z != w. 
                    # x != z? x is in N(v), z is dist 3 from v. Could x==z? 
                    # If x==z, then v-x-y-x... no, y is in N(x) and N(z)=N(x). 
                    # But we excluded x from N(y) search? No, we excluded x from y's neighbors? 
                    # y_mask = N(y) \ {x, u}. So z != x.
                    # w != z (excluded). w != y (excluded). w != x? 
                    # z_mask excludes y and u. It does not explicitly exclude x.
                    # However, if w==x, then z is in N(x) and x is in N(z). 
                    # But we need distinct vertices.
                    # Let's explicitly check distinctness.
                    
                    if w == u or w == v:
                        continue
                    if w == x or w == y or w == z:
                        continue
                        
                    if adj[u] & (1 << w):
                        return False
    return True

def _count_edges(adj, n=40):
    count = 0
    for i in range(n):
        count += bin(adj[i]).count('1')
    return count // 2

def _get_edges(adj, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def _random_greedy(n=40, seed=None, max_time=5.0):
    """Randomized greedy: try edges in random order, add if C6-free."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _is_c6_free_after_add(adj, u, v, n):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj, _count_edges(adj, n)

def _local_search(adj, n=40, max_time=10.0):
    """Local search with add and swap moves."""
    start_time = time.time()
    current_count = _count_edges(adj, n)
    best_adj = adj[:]
    best_count = current_count
    
    # Precompute non-edges and edges for efficiency in loops
    # We will regenerate these lists periodically or on the fly
    
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Try adding non-edges
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() - start_time > max_time:
                return best_adj, best_count
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _is_c6_free_after_add(adj, u, v, n):
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                improved = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if not improved:
            # 2. Try swaps: remove one edge, add one non-edge
            cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            random.shuffle(cur_edges)
            random.shuffle(non_edges)
            
            swap_improved = False
            for ru, rv in cur_edges:
                if time.time() - start_time > max_time:
                    break
                
                # Remove (ru, rv)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                current_count -= 1
                
                # Try adding non-edges
                for au, av in non_edges:
                    if time.time() - start_time > max_time:
                        break
                    if adj[au] & (1 << av):
                        continue # Already added or is the removed edge? 
                                 # If it's the removed edge, adj[au] & (1<<av) is 0 now.
                                 # We should skip if (au,av) == (ru,rv)
                    if (au == ru and av == rv) or (au == rv and av == ru):
                        continue
                        
                    adj[au] |= (1 << av)
                    adj[av] |= (1 << au)
                    
                    # Check if adding (au, av) to the graph (which now lacks (ru, rv)) creates C6
                    # Note: _is_c6_free_after_add assumes the graph was C6-free BEFORE adding.
                    # The graph with (ru, rv) removed is a subgraph of a C6-free graph, so it is C6-free.
                    if _is_c6_free_after_add(adj, au, av, n):
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_adj = adj[:]
                        swap_improved = True
                        # Keep the swap
                    else:
                        adj[au] &= ~(1 << av)
                        adj[av] &= ~(1 << au)
                
                # Restore (ru, rv) if we didn't keep a swap that effectively replaced it
                # Actually, if we kept a swap, we want to keep the new state.
                # But our loop structure restores (ru, rv) at the end.
                # We need to handle this carefully.
                # If a swap was successful, we have modified adj.
                # We should break out of the inner loop and NOT restore (ru, rv) if we want to keep the new state.
                # However, the current logic restores it.
                # Let's restructure: if a swap is accepted, we set a flag and break.
                
                # Re-evaluating swap logic for correctness
                # It's safer to just try swaps and if one works, we update best and continue searching from that new state?
                # Or just find the best swap for this removal?
                # Given time limits, let's just try to find *any* improving swap.
                
                # For now, let's just restore and break if we found an improvement, 
                # and then restart the main loop with the new best_adj.
                if swap_improved:
                    # We found an improvement. We should probably break and let the outer loop 
                    # pick up the new best_adj if we copy it back.
                    # But we are modifying 'adj' in place.
                    # Let's just break and let the outer while loop handle it.
                    # But we need to ensure 'adj' reflects the best state or we reset it.
                    # Simplest: if swap_improved, we have already updated best_adj.
                    # We should break out of the swap loop and restart the main loop.
                    break
                
                # If no swap improved, restore (ru, rv)
                if not (adj[ru] & (1 << rv)):
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    current_count += 1

            if swap_improved:
                # We found a better graph. Update current adj to best_adj and restart search
                adj = best_adj[:]
                current_count = best_count
                improved = True # Force another pass
            else:
                # No swaps found, try 2-swaps or break
                # Let's try 2-swaps briefly if time permits
                if time.time() - start_time < max_time - 5:
                    # 2-swap: remove e1, e2, add f1, f2
                    # This is complex. Let's stick to 1-swaps and adds for now.
                    # If no improvement in add or 1-swap, we are likely at a local optimum.
                    pass
    
    return best_adj, best_count

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Random Greedy to get a good starting point
    # Run multiple short greedy runs
    for i in range(5):
        remaining = DEADLINE - (time.time() - start_time) - 20
        if remaining < 2:
            break
        adj_g, count_g = _random_greedy(n, seed=42 + i, max_time=min(2.0, remaining / 5))
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
            
    # If greedy didn't find anything, start from 8xK5
    if best_adj is None or best_edges < 100:
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a+1, 5):
                    adj[base+a] |= (1 << (base+b))
                    adj[base+b] |= (1 << (base+a))
        if _count_edges(adj, n) > best_edges:
            best_edges = _count_edges(adj, n)
            best_adj = adj[:]

    # Phase 2: Local Search
    if time.time() - start_time < DEADLINE - 10:
        adj_ls, count_ls = _local_search(best_adj, n, max_time=min(10.0, DEADLINE - (time.time() - start_time) - 5))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls

    # Phase 3: More Local Search / Random Restarts if time remains
    while time.time() - start_time < DEADLINE - 5:
        # Try to perturb and search again
        adj_perturbed = best_adj[:]
        # Randomly remove a few edges
        edges_to_remove = [(i, j) for i in range(n) for j in range(i+1, n) if adj_perturbed[i] & (1 << j)]
        random.shuffle(edges_to_remove)
        for u, v in edges_to_remove[:3]:
            adj_perturbed[u] &= ~(1 << v)
            adj_perturbed[v] &= ~(1 << u)
            
        adj_ls, count_ls = _local_search(adj_perturbed, n, max_time=min(5.0, DEADLINE - (time.time() - start_time) - 2))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls

    # Final Verification
    if best_adj is None:
        # Fallback to 8xK5
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a+1, 5):
                    adj[base+a] |= (1 << (base+b))
                    adj[base+b] |= (1 << (base+a))
        best_adj = adj

    return _get_edges(best_adj, n)

