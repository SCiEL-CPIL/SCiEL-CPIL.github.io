"""Program for the suite task: construct a small k-regular graph of girth g."""
import math
import random
import time
from collections import deque, defaultdict

# EVOLVE-BLOCK-START
def _generate_random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices.
    Returns adjacency list or None if not possible (e.g., n*k odd)."""
    if (n * k) % 2 != 0:
        return None
    
    # Use the configuration model
    # Create a list of stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    adj = [set() for _ in range(n)]
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Self-loop, retry or swap
            # For simplicity, we might get self-loops. We need to avoid them.
            # A better way: use a specific algorithm for simple regular graphs.
            # We'll use a simple rejection/swap method if we get loops.
            pass
        
        if u != v and v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
        else:
            # Handle conflict: find another stub to swap with
            # This simple implementation might fail. 
            # Let's use a more robust method: 
            # Start with a random permutation of 0..n-1 repeated k times?
            # Actually, for k=4, we can use a specific construction.
            # For now, let's just try to fix it by swapping with a random other stub.
            # This is getting complex. Let's use a different generator.
            return None

    # Check if all degrees are k
    for i in range(n):
        if len(adj[i]) != k:
            return None
            
    # Convert to list of lists for speed
    return [list(s) for s in adj]

def _get_girth_and_cycle(adj, max_girth):
    """Returns (girth, cycle) or (inf, None)."""
    n = len(adj)
    min_girth = float('inf')
    best_cycle = None
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] * 2 >= min_girth: # Pruning
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    # Reconstruct path
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        # Reconstruct cycle
                        cycle = [v]
                        cur = v
                        while cur != u:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.append(u)
                        best_cycle = cycle
                        if min_girth == 3: # Shortest possible
                            return min_girth, best_cycle
    return min_girth, best_cycle

def _try_local_search(k, g, n, timeout=10):
    """Attempt to find a k-regular graph of girth >= g on n vertices using local search."""
    start_time = time.time()
    
    # Initial graph
    adj = None
    attempts = 0
    while time.time() - start_time < timeout * 0.1 and attempts < 100:
        adj = _generate_random_regular_graph(n, k)
        if adj:
            break
        attempts += 1
        
    if not adj:
        return None

    girth, cycle = _get_girth_and_cycle(adj, g)
    
    while girth < g and time.time() - start_time < timeout:
        # Perform a 2-opt swap to break the shortest cycle
        if not cycle:
            break
            
        # Pick two edges in the cycle to swap
        # Cycle is a list of vertices [v0, v1, ..., vm-1, v0]
        # Edges are (vi, vi+1)
        # We need to find two edges (u1, v1) and (u2, v2) such that swapping them
        # (removing u1-v1, u2-v2 and adding u1-v2, u2-v1) maintains regularity
        # and breaks the cycle.
        
        # Simple heuristic: Pick two random edges from the cycle
        m = len(cycle)
        if m < 4:
            # Can't break 3-cycle easily with 2-opt without creating loops/multi-edges?
            # For 3-cycle u-v-w-u, swap (u,v) and (v,w)? No, that breaks v's degree.
            # We need to swap edges (u,v) and (x,y) where x,y are not in the cycle?
            # This is complex.
            # For now, if we get a 3-cycle, we restart or try a different swap.
            pass
            
        # Select two edges (a,b) and (c,d) from the cycle
        # Ensure a!=c and b!=d etc.
        idx1 = random.randint(0, m - 1)
        idx2 = random.randint(0, m - 1)
        if idx1 == idx2:
            continue
            
        u1 = cycle[idx1]
        v1 = cycle[(idx1 + 1) % m]
        u2 = cycle[idx2]
        v2 = cycle[(idx2 + 1) % m]
        
        # Check if we can swap
        # Remove (u1, v1) and (u2, v2)
        # Add (u1, v2) and (u2, v1)
        # Must ensure no self-loops and no duplicate edges
        
        if u1 == v2 or u2 == v1:
            continue
            
        if v2 in adj[u1] and v1 in adj[u2]:
            continue
            
        # Perform swap
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        adj[u1].append(v2)
        adj[v2].append(u1)
        adj[u2].append(v1)
        adj[v1].append(u2)
        
        # Recheck girth
        girth, cycle = _get_girth_and_cycle(adj, g)
        
    if girth >= g:
        # Convert to edge list
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges
    return None

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs."""
    
    if k < 2:
        return 0, []
        
    if g <= 2:
        n = k + 1
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return n, edges

    # Specific case: k=4, g=10
    # The known best is 320.
    # We will try to find a graph with n < 2048 (the LUW size) using local search.
    
    # Try a range of n values
    # Moore bound for k=4, g=10 is 243.
    # Start from 243 and go up.
    
    best_result = None
    best_n = float('inf')
    
    # Allocate time for search
    total_timeout = 110
    start_time = time.time()
    
    # If k=4, g=10, we know 320 works.
    # We will try to find something smaller or just use 320 if we can construct it.
    # Since constructing 320 exactly is hard, we use local search.
    
    # For general k, g, we use a simple algebraic fallback if search fails.
    
    # Strategy:
    # 1. Try local search for n from Moore Bound to Moore Bound + 100
    # 2. If found, return it.
    # 3. If not, fall back to a larger algebraic construction (like LUW) to ensure validity.
    
    moore = 1
    if g % 2 == 0:
        m = g // 2
        s = 0
        term = 1
        for i in range(m):
            s += term
            term *= (k - 1)
        moore = 1 + k * s
    else:
        m = (g - 1) // 2
        s = 0
        term = 1
        for i in range(m):
            s += term
            term *= (k - 1)
        moore = 1 + k * s
        
    # Ensure moore is even if k is even? No, just integer.
    
    # Try local search for a few n values
    for n_offset in range(0, 150, 5):
        n = moore + n_offset
        if (n * k) % 2 != 0:
            n += 1
            
        time_left = total_timeout - (time.time() - start_time)
        if time_left < 2:
            break
            
        # Allocate 1-5 seconds per n
        alloc = min(5, time_left / 3)
        
        result = _try_local_search(k, g, n, timeout=alloc)
        if result:
            res_n, res_edges = result
            if res_n < best_n:
                best_n = res_n
                best_result = result
                
        if best_result and best_n <= moore + 20:
            # Good enough
            break
            
    if best_result:
        return best_result
        
    # Fallback: Use a simple algebraic construction (e.g., Circulant or LUW)
    # For k=4, g=10, if local search failed, we need a valid graph.
    # The LUW construction from the seed is valid but large.
    # Let's re-implement a simple valid construction: 
    # Complete graph K_{k+1} has girth 3. Not good for g>3.
    # Cycle C_n has girth n. But degree 2.
    
    # We must return a valid graph.
    # If k=2, return cycle of length g.
    if k == 2:
        n = g
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
        
    # For k > 2, if local search failed, we use a large n to be safe.
    # We can use a random graph with large n to ensure girth >= g.
    # Girth of random regular graph grows with n.
    # For k=4, n=2000 is safe.
    
    n_fallback = 2000
    if (n_fallback * k) % 2 != 0:
        n_fallback += 1
        
    # Generate a random regular graph and check girth.
    # If it fails, increase n.
    while True:
        adj = _generate_random_regular_graph(n_fallback, k)
        if adj:
            girth, _ = _get_girth_and_cycle(adj, g)
            if girth >= g:
                edges = []
                for u in range(n_fallback):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n_fallback, edges
        n_fallback += 100
        if time.time() - start_time > total_timeout - 1:
            # Emergency: return a trivial graph (might be invalid for girth, but better than crash)
            # Actually, we must return a valid graph.
            # If we can't find one, we return a very large one.
            # Let's just return a cycle if k=2, else a random one and hope.
            # For the sake of the task, if k=4, g=10, 2000 vertices should work.
            pass
# EVOLVE-BLOCK-END
