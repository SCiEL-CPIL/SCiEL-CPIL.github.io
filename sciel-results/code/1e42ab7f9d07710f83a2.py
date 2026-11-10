# EVOLVE-BLOCK-START
import time
import random
import math

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Strategy:
    1. For k=3, attempt to construct a graph with n close to the Moore bound (512)
       using a random configuration model followed by a 2-switch local search.
       This is a structural attempt to beat the known 936 vertex bound.
    2. If that fails or times out, fall back to the Lazebnik-Ustimenko-Woldar (LUW)
       construction, sweeping parameters to find the smallest valid graph.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    if k != 3:
        # Fallback for other k values using LUW
        return _lnw_construct(k, g)

    # --- Phase 1: Random Search for small n (512) ---
    # The Moore bound for n(3, 16) is 512.
    # It is theoretically possible that a graph exists at this size, though extremely unlikely
    # to be found by random search in 120s. However, a local search from a random start
    # might find a local minimum with high girth.
    
    # We allocate a significant portion of time to this attempt, but keep a hard stop
    # to ensure we have time for the deterministic LUW fallback.
    search_deadline = start_time + 60
    
    # Try n=512
    # We use a specialized 2-switch search that tracks girth efficiently.
    # Note: A random 3-regular graph on 512 vertices has many short cycles.
    # Breaking them all to reach girth 16 is a huge hill to climb.
    
    # Optimization: Instead of checking full girth every step (O(N^2)), 
    # we can try to find short cycles and break them.
    
    try:
        n_target = 512
        if n_target % 2 == 0 and n_target >= 4:
            edges = _search_2switch_optimized(n_target, g, search_deadline)
            if edges is not None:
                # Verify
                adj = _edges_to_adj(edges, n_target)
                # Final verification of girth
                girth = _calculate_girth(adj, n_target)
                if girth >= g:
                    return n_target, edges
    except Exception:
        pass

    # --- Phase 2: LUW Construction ---
    # The LUW construction is deterministic and guaranteed to produce a valid graph
    # if parameters are chosen correctly.
    # For k=3, q=3.
    # n = 4 * 3^d
    # We want the smallest d such that girth >= g.
    
    # Known behavior:
    # d=6 -> n=2916, girth ~ 12-14?
    # d=7 -> n=8748, girth ~ 14-16?
    # d=8 -> n=26244, girth ~ 16+
    
    # We sweep d from a low value upwards.
    # We stop as soon as we find a valid graph.
    
    # Start d from 6.
    for d in range(6, 16):
        if time.time() > deadline - 10:
            break
        
        try:
            n, edges = _lnw_construct_params(3, d)
        except Exception:
            continue
        
        # Check girth
        adj = _edges_to_adj(edges, n)
        girth = _calculate_girth(adj, n)
        
        if girth >= g:
            return n, edges

    # Fallback to a large d that definitely works, or just return the last tried one
    # if it was valid? No, the loop breaks if invalid.
    # If we exhausted d or timed out, use a safe large d.
    d_safe = max(8, g - 2)
    try:
        return _lnw_construct_params(3, d_safe)
    except Exception:
        # Ultimate fallback: just return a trivial valid graph if possible?
        # For k=3, n must be even >= 4.
        # If everything fails, return a random 3-reg graph (might have low girth, but valid structure).
        # The evaluator scores n, but invalid graph scores 0.
        # We must return a VALID graph.
        # LUW with large d is always valid.
        return _lnw_construct_params(3, 20)


def _search_2switch_optimized(n, g, deadline):
    """
    Attempt to find a 3-regular graph on n vertices with girth >= g
    using 2-switches.
    """
    if n % 2 != 0 or n < 4:
        return None

    def gen_random_3reg(n):
        while True:
            stubs = []
            for i in range(n):
                for _ in range(3):
                    stubs.append(i)
            random.shuffle(stubs)
            
            edges = []
            adj = [set() for _ in range(n)]
            valid = True
            
            for i in range(0, 3*n, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
            if valid:
                return edges, adj
                
    edges, adj = gen_random_3reg(n)
    
    # Helper to find a short cycle < g
    # Returns a list of edges in the cycle, or None.
    def find_short_cycle(adj, n, target_girth):
        # Randomized BFS from a random start
        start = random.randint(0, n-1)
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= target_girth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        # Found a cycle
                        # We need to reconstruct the cycle to get the edges.
                        # The cycle consists of the path from u to LCA(u,v) and v to LCA(u,v) plus edge (u,v)?
                        # Actually, in BFS, if v is visited and not parent, the cycle is formed by
                        # u -> ... -> v -> ... -> u? No.
                        # The path is start -> ... -> u and start -> ... -> v.
                        # The cycle is u -> ... -> v (via LCA) -> ... -> u?
                        # Simpler: Trace back from u and v to find the first common ancestor (LCA).
                        # Then the cycle is the path u->LCA + v->LCA + edge(u,v)? No, edge(u,v) is the closing edge.
                        
                        # Let's just return the edge (u, v) as a candidate for switching.
                        # Switching (u,v) with any other edge (x,y) might break the cycle.
                        return (u, v)
        return None

    # Main loop
    while time.time() < deadline:
        short_edge = find_short_cycle(adj, n, g)
        if short_edge is None:
            # No short cycle found from this start.
            # Do a full check to be sure.
            if _calculate_girth(adj, n) >= g:
                # Success
                final_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            final_edges.append((i, j))
                return final_edges
            # If not, continue searching
            continue
        
        u, v = short_edge
        
        # Try to break this cycle by switching edge (u,v) with a random edge (x,y)
        found_swap = False
        for _ in range(5):
            x = random.randint(0, n-1)
            if x == u or x == v:
                continue
            y = random.choice(list(adj[x]))
            if y == u or y == v:
                continue
            if x == y:
                continue
            if (x == u and y == v) or (x == v and y == u):
                continue
            
            # Check if (u,x) and (v,y) are valid (not existing edges)
            if x in adj[u] or y in adj[v]:
                continue
            # Also check if (u,y) and (v,x) would be valid? 
            # We are doing (u,v)->(u,x) and (x,y)->(v,y).
            # Wait, 2-switch on edges (u,v) and (x,y) creates (u,x) and (v,y) OR (u,y) and (v,x).
            # Let's pick one.
            
            # Check for multi-edges
            if x in adj[u] or y in adj[v]:
                continue
                
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            # Check if girth improved?
            # We don't want to calculate full girth every time.
            # Just check if we still have a short cycle?
            # If we break the specific short cycle we found, that's good.
            # But we might create a new short cycle.
            # For simplicity, accept if we don't immediately find a short cycle?
            # Or just accept and hope.
            # Let's accept and move on.
            found_swap = True
            break
            
        if not found_swap:
            # Couldn't find a valid swap, try again
            pass

    return None

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth(adj, n):
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
    return min_cycle if min_cycle <= n else n + 1

def _lnw_construct_params(k, d):
    """
    LUW construction with fixed q=k and variable d.
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule generation depends on d
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

def _lnw_construct(k, g):
    # Original fallback
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
# EVOLVE-BLOCK-END
