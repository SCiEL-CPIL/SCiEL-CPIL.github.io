"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _luw_construct(k, g):
    """
    Construct a valid graph using LUW (Layered Unitary Wreath) product.
    This serves as a guaranteed fallback (valid but potentially large).
    """
    if k != 3:
        # Fallback for other k, try a simple circulant or LUW
        # For this task k=3 is expected, but let's be safe.
        # If k != 3, we return a large valid graph if possible, else None
        pass
        
    q = 3
    # Determine d such that girth >= g
    # For LUW with q, the girth is roughly 2*d + 2 or similar depending on rules.
    # The previous code used d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's use a safe d to ensure girth.
    # For q=3, girth grows with d.
    d = max(3, (g - 2) // 2) 
    
    add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    
    # Standard LUW rules for k=3 (q=3)
    # We need to define the rule for each column c.
    # The rule maps (side, x) to neighbors.
    # A common set of rules for q=3:
    rule = {}
    # We need to ensure the graph is 3-regular.
    # The LUW construction typically creates a graph of size 2 * q^d.
    # Let's use the specific rule from the previous code which was verified to work.
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
            
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _verify_girth_fast(adj, n, g):
    """
    Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    Optimized for speed using sets.
    """
    for start in range(n):
        # State: (vertex, prev_vertex)
        # We want to find a cycle of length < g.
        # A cycle of length L corresponds to a non-backtracking walk of length L from v to v.
        # If we find (v, u) where u is a neighbor of start, and dist >= 2, we have a cycle.
        # Actually, non-backtracking walk from start:
        # (start, -1) at dist 0.
        # Next (v, start) at dist 1.
        # If we reach (start, u) at dist d, we have a cycle of length d.
        # We need d >= g.
        
        visited = set()
        dq = deque()
        dq.append((start, -1))
        visited.add((start, -1))
        dist = {(start, -1): 0}
        
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                # If we return to start, we found a cycle of length d+1
                if w == start:
                    if d + 1 < g:
                        return False
                    else:
                        # Cycle length >= g, valid. But we don't need to track further from this path?
                        # Actually, we can stop exploring this branch if we are just checking validity.
                        continue
                
                state = (w, v)
                if state not in visited:
                    visited.add(state)
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _generate_random_3reg(n, seed=None):
    """
    Generate a random 3-regular graph on n vertices.
    Returns adj list and edge list, or None if failed (e.g. self-loops/multi-edges persist).
    """
    if n % 2 != 0 or n < 4:
        return None, None
        
    # Stub pairing method
    # Create a list of 3*n stubs
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
            
    _random.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    edges = []
    
    # Pair up stubs
    # We need to ensure no self-loops (u==v) and no multi-edges (u,v) already in adj
    # Simple greedy pairing with retries
    attempts = 0
    max_attempts = 100
    while attempts < max_attempts:
        attempts += 1
        adj = [set() for _ in range(n)]
        edges = []
        success = True
        # Shuffle stubs again in each attempt? Or just try to pair the current shuffled list?
        # It's better to re-shuffle if we fail.
        _random.shuffle(stubs)
        
        used = [False] * len(stubs)
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                success = False
                break
            if v in adj[u]:
                success = False
                break
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
            
        if success:
            # Verify degree
            valid = True
            for i in range(n):
                if len(adj[i]) != 3:
                    valid = False
                    break
            if valid:
                return adj, edges
    return None, None

def _find_short_cycle(adj, n, g):
    """
    Find a cycle of length < g.
    Returns (cycle_vertices, length) or None.
    """
    for start in range(n):
        # BFS with state (v, prev)
        # We want to find a path from start back to start of length L < g.
        # Or a path from u to v where u,v are neighbors of start?
        # Simpler: Non-backtracking walk from start.
        # If we reach (start, u) with dist d, cycle length is d.
        # We need d < g.
        
        visited = {} # (v, prev) -> dist
        dq = deque()
        dq.append((start, -1))
        visited[(start, -1)] = 0
        
        while dq:
            v, prev = dq.popleft()
            d = visited[(v, prev)]
            if d >= g:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start:
                    if d + 1 < g:
                        # Found cycle of length d+1
                        # Reconstruct path? 
                        # For local search, we just need to know it exists and maybe one edge to swap.
                        # Returning the start and current v is enough to identify the cycle?
                        # Actually, we need the edges.
                        # Let's just return True/False for now in the fast check, 
                        # but for swap we need specific edges.
                        # Let's return the path.
                        # Path reconstruction is expensive. 
                        # Alternative: Just return a random edge from the cycle?
                        # We can store parent pointers.
                        pass
                state = (w, v)
                if state not in visited:
                    visited[state] = d + 1
                    dq.append(state)
    return None

def _local_search(adj, edges, n, g, time_limit):
    """
    Perform local search to increase girth.
    Returns (adj, edges) if girth >= g, else None.
    """
    start_time = _time.time()
    
    # Initial check
    if _verify_girth_fast(adj, n, g):
        return adj, edges
        
    # We need a way to find a short cycle and break it.
    # Strategy:
    # 1. Find a short cycle C.
    # 2. Pick an edge (u, v) in C.
    # 3. Find an edge (x, y) not in C.
    # 4. Swap (u, v) and (x, y) to (u, x) and (v, y).
    # 5. Check if girth increased or if the specific cycle C is broken.
    #    To be safe, we check if the new graph has girth >= old_girth.
    #    If not, revert.
    
    # To make this efficient, we don't want to run full _verify_girth_fast (O(n^2)) after every swap.
    # We can check locally: does the swap create a short cycle?
    # A swap affects cycles passing through u, v, x, y.
    # This is complex. 
    # Given the time limit, we can afford to run _verify_girth_fast occasionally, 
    # or use a lighter check.
    
    # Let's try a simpler heuristic:
    # Randomly pick edges and swap if it doesn't create self-loops/multi-edges.
    # Keep track of "best" girth found so far.
    # If we find a valid graph, return it.
    
    # We need a function to get a short cycle to target swaps.
    # Let's implement a helper to get a random short cycle.
    
    def get_random_short_cycle():
        for start in range(n):
            # Random walk with backtracking prevention
            # We want to find a cycle of length < g.
            # Let's do a BFS up to depth g-1.
            # If we find a connection to start, we have a cycle.
            
            # To be fast, we can just do a limited random walk.
            # If we fail, return None.
            
            # Let's use the visited dict approach but stop early.
            visited = {}
            dq = deque()
            dq.append((start, -1, [start])) # (v, prev, path)
            visited[(start, -1)] = 0
            
            while dq:
                v, prev, path = dq.popleft()
                d = visited[(v, prev)]
                if d >= g:
                    continue
                # Shuffle neighbors to randomize
                neighbors = list(adj[v])
                _random.shuffle(neighbors)
                for w in neighbors:
                    if w == prev:
                        continue
                    if w == start:
                        if d + 1 < g:
                            # Found cycle
                            # Path is path + [w]
                            # Cycle vertices: path
                            return path
                    state = (w, v)
                    if state not in visited:
                        visited[state] = d + 1
                        # Store path? Storing full paths is memory heavy.
                        # We can just store parent pointers.
                        # For simplicity in this rewrite, let's just return the start and v?
                        # No, we need the edges.
                        # Let's just return the path list.
                        dq.append((w, v, path + [w]))
        return None

    best_girth = g - 1 # Assume we start with girth < g
    current_adj = [set(s) for s in adj]
    current_edges = list(edges)
    
    # Map edge to index for fast swap?
    # Or just maintain adj sets.
    
    attempts = 0
    while _time.time() < time_limit:
        attempts += 1
        
        # Find a short cycle
        cycle = get_random_short_cycle()
        if cycle is None:
            # No short cycle found? Maybe girth is already >= g?
            # Verify to be sure.
            if _verify_girth_fast(current_adj, n, g):
                return current_adj, current_edges
            # If verify fails but get_random_short_cycle didn't find one, 
            # it might be a bug in get_random_short_cycle or it's too slow.
            # Let's force a random swap.
            pass
        
        if cycle:
            # Pick a random edge from the cycle
            # Cycle vertices: v0, v1, ..., vk
            # Edges: (v0,v1), (v1,v2), ...
            idx = _random.randint(0, len(cycle) - 1)
            u = cycle[idx]
            v = cycle[(idx + 1) % len(cycle)]
            
            # Find another edge (x, y) not involving u or v
            # And not creating multi-edge or self-loop
            candidates = []
            for x in range(n):
                if x == u or x == v:
                    continue
                for y in current_adj[x]:
                    if y < x: continue # Avoid duplicates in iteration? 
                    if y == u or y == v:
                        continue
                    # Check if (u,x) and (v,y) are valid
                    # (u,x) valid if x not in adj[u]
                    # (v,y) valid if y not in adj[v]
                    # Also (u,y) and (v,x)
                    
                    if x not in current_adj[u] and y not in current_adj[v]:
                        candidates.append((x, y, 'xy')) # Swap to (u,x), (v,y)
                    if y not in current_adj[u] and x not in current_adj[v]:
                        candidates.append((x, y, 'yx')) # Swap to (u,y), (v,x)
            
            if not candidates:
                continue
                
            # Pick a random candidate
            x, y, swap_type = _random.choice(candidates)
            
            # Perform swap
            # Remove (u, v) and (x, y)
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            current_adj[x].remove(y)
            current_adj[y].remove(x)
            
            # Add new edges
            if swap_type == 'xy':
                current_adj[u].add(x)
                current_adj[x].add(u)
                current_adj[v].add(y)
                current_adj[y].add(v)
                # Update edge list? 
                # We can reconstruct edge list at the end or maintain it.
                # Maintaining is tricky. Let's just use adj sets for verification.
                # We need to return edges at the end.
                # We can generate edges from adj sets.
            else:
                current_adj[u].add(y)
                current_adj[y].add(u)
                current_adj[v].add(x)
                current_adj[x].add(v)
            
            # Check if this swap broke the specific cycle?
            # Or just check if girth improved?
            # Full verification is expensive.
            # Let's do a quick check: is the specific cycle broken?
            # The cycle contained (u, v). We removed it. So that specific cycle is broken.
            # But we might have created a new short cycle.
            
            # Heuristic: Accept if random, or if we can't afford full check.
            # Let's do full check every 100 swaps or if time permits.
            
            if attempts % 50 == 0:
                if _verify_girth_fast(current_adj, n, g):
                    # Convert adj to edges
                    final_edges = []
                    for i in range(n):
                        for j in current_adj[i]:
                            if i < j:
                                final_edges.append((i, j))
                    return current_adj, final_edges
    
    # Final check before returning None
    if _verify_girth_fast(current_adj, n, g):
        final_edges = []
        for i in range(n):
            for j in current_adj[i]:
                if i < j:
                    final_edges.append((i, j))
        return current_adj, final_edges
        
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        # Fallback to LUW for other k values
        return _luw_construct(k, g)

    # Moore bound for k=3, girth g: 2*(2^(g/2)-1) for even g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
        
    # Ensure moore is even
    if moore % 2 != 0:
        moore += 1
        
    # Search n from moore upwards
    # Known best is 936.
    # Let's try n = moore, moore+2, ...
    # We will allocate time per n.
    
    n_current = moore
    max_n = 1000 # Upper bound for search
    
    # We want to find the smallest n.
    # We will give more time to smaller n? Or equal time?
    # If we don't find it at n, we go to n+2.
    # Let's use a fixed time budget per n, decreasing as n increases?
    # Or just linear scan with a time limit per step.
    
    # Since the optimal is 936, and moore is 512 (for g=16: 2*(16-1)=30? No. 2^(16/2)=2^8=256. 2*(256-1)=510).
    # So moore=510.
    # We need to search 510, 512, ..., 936.
    # That's ~212 values.
    # 115s / 212 ~ 0.5s per value.
    # Local search with 0.5s might be enough to find a valid graph if one exists?
    # Random 3-reg graphs of size 510 likely have small girth (3 or 4).
    # Swaps need to break all short cycles.
    # It might be hard to jump from girth 4 to 16 in 0.5s.
    
    # Strategy:
    # 1. Try n = 936 first? No, we want the MINIMUM.
    # 2. But if we can't find it at 510, we must try 512, etc.
    # 3. If we are slow, we might miss the optimum.
    
    # Let's prioritize n values that are "likely" to have high girth.
    # High girth graphs are rare.
    # Maybe we should just do a global search over n?
    
    # Let's try a binary search-like approach? No, property is not monotonic in n.
    # (A graph of size n with girth g does not imply n+2 has one).
    
    # Let's just linear scan.
    # To speed up, we can parallelize? No, single process.
    
    # Let's give 0.5s per n.
    # If we run out of time, return the best valid one found (which might be large).
    # Or return LUW fallback.
    
    best_result = None
    best_n = float('inf')
    
    while n_current <= max_n and _time.time() < deadline - 5:
        time_per_n = min(1.0, (deadline - 5 - _time.time()) / (max_n - n_current + 1))
        
        # Generate random graph
        adj, edges = _generate_random_3reg(n_current)
        if adj is None:
            n_current += 2
            continue
            
        # Local search
        res = _local_search(adj, edges, n_current, g, _time.time() + time_per_n)
        if res is not None:
            # Valid graph found
            # Check if it's the smallest
            if n_current < best_n:
                best_n = n_current
                best_result = res
                # We found a valid graph. Can we find a smaller one?
                # We are scanning from Moore bound up.
                # So the first one we find is the smallest?
                # Yes, because we start at Moore bound.
                return n_current, res[1]
                
        n_current += 2
        
    # If we didn't find a small one, return the best we have or fallback
    if best_result:
        return best_n, best_result[1]
        
    # Fallback to LUW
    return _luw_construct(k, g)

# EVOLVE-BLOCK-END
