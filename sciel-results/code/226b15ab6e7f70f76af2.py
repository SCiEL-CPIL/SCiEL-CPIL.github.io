import time
import random
from collections import deque

DEADLINE = 110.0 # Safety margin for 120s limit

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges).
    """
    start_time = time.time()
    
    # 1. Calculate Moore bound
    if g % 2 == 0:
        R = g // 2
        s = 0
        for i in range(R - 1):
            s += (k - 1) ** i
        n_min = 1 + k * s
    else:
        R = (g - 1) // 2
        s = 0
        for i in range(R):
            s += (k - 1) ** i
        n_min = 1 + k * s
    
    if n_min < k + 1:
        n_min = k + 1
        
    # Ensure n_min has correct parity for k-regular graph
    # n*k must be even
    if (n_min * k) % 2 != 0:
        n_min += 1

    best_n = None
    best_edges = None

    # 2. Search for a valid graph starting from n_min
    # We iterate n upwards.
    # We try to find a graph with n < known_best (if we had one, but we don't trust the seed's 320).
    # We will search up to a limit, or until time runs out.
    # If we find a valid graph, we return it immediately (since we are going up from min).
    # However, local search is stochastic. It might fail at 161 but succeed at 163.
    # So we should keep trying until we find one or time runs out.
    
    # Upper bound for search: 
    # The LUW construction is a safe fallback. Let's estimate its size to set a cap.
    # For k=4, g=10, LUW size is likely large.
    # Let's cap search at 500 to be safe, but prioritize lower n.
    max_search_n = 500
    
    current_n = n_min
    # Step size: 2 if k is even, 1 if k is odd?
    # Actually, n*k must be even.
    # If k is even, n can be anything.
    # If k is odd, n must be even.
    step = 1 if k % 2 == 0 else 2
    
    while current_n <= max_search_n:
        if time.time() - start_time > DEADLINE - 5.0: # Leave 5s for fallback
            break
            
        # Time slice for this n
        # We want to give more time to smaller n? Or equal?
        # Smaller n is harder to satisfy constraints?
        # Let's give a fixed slice, e.g., 2 seconds per n.
        time_slice = 2.0
        remaining = DEADLINE - 5.0 - (time.time() - start_time)
        if remaining < time_slice:
            time_slice = remaining
            
        if time_slice < 0.5:
            break
            
        success, edges_found = _local_search_attempt(k, g, current_n, time_slice)
        
        if success:
            return current_n, edges_found
            
        current_n += step

    # 3. Fallback: LUW Construction
    # If we didn't find a graph in the search range, use LUW.
    # This ensures we return a valid graph, even if large.
    try:
        n_luw, edges_luw = _luw_construct(k, g)
        if n_luw is not None and edges_luw is not None:
            return n_luw, edges_luw
    except Exception:
        pass

    # 4. Last resort: Return empty (Score 0)
    return 0, []

def _local_search_attempt(k, g, n, time_limit):
    """
    Attempts to construct a k-regular graph on n vertices with girth >= g.
    Uses randomized local search with edge swaps.
    """
    start = time.time()
    
    # Generate initial random k-regular graph
    adj = _generate_random_k_regular(n, k)
    if adj is None:
        return False, []
        
    # Convert to set for faster lookups if needed, but list is fine for small n
    # We will use a list of sets for O(1) neighbor check
    
    # Initial girth check
    girth = _find_girth(adj, g)
    if girth >= g:
        return True, _adj_to_edges(adj)
        
    # Local search loop
    # We use a simple hill-climbing / simulated annealing approach
    # Since we just want ANY valid graph, we can be aggressive.
    
    # To speed up, we can track the shortest cycle and try to break it.
    
    while time.time() - start < time_limit:
        # Find a short cycle
        cycle = _find_a_short_cycle(adj, g)
        if not cycle:
            # No short cycle found?
            # This might happen if _find_girth said girth < g but _find_a_short_cycle failed
            # Or if girth >= g (should have returned earlier)
            # Let's double check girth
            girth = _find_girth(adj, g)
            if girth >= g:
                return True, _adj_to_edges(adj)
            else:
                # Should not happen, but break to avoid infinite loop
                break
        
        # Pick a random edge in the cycle to remove
        # Actually, we do a 2-switch.
        # Remove (u, v) and (u', v'), add (u, u') and (v, v')
        # We want to break the cycle.
        # Let's pick an edge (u, v) from the cycle.
        u, v = cycle[0]
        
        # Find another edge (u', v') to swap with
        # u' and v' should not be in the cycle to avoid creating another short cycle immediately?
        # Ideally, u' and v' are far from the cycle.
        
        # Simple heuristic: Pick random u' from neighbors of u (excluding v), 
        # and random v' from neighbors of v (excluding u).
        # Then swap (u,v) and (u',v')? No.
        # Swap: Remove (u,v) and (u',v'). Add (u,u') and (v,v').
        # This requires (u,v) and (u',v') to exist.
        
        # Let's try a random 2-switch that involves u and v.
        # Pick u' in N(u) \ {v}
        # Pick v' in N(v) \ {u}
        # Ensure u' != v' and u' not in N(v) and v' not in N(u) to avoid multi-edges/loops?
        # Actually, we are removing (u,v).
        # We add (u, u') and (v, v').
        # We must remove (u', v')? No, we need to remove an edge to keep degrees constant.
        # If we just remove (u,v) and add (u,u') and (v,v'), degrees change.
        # We must remove (u',v') as well?
        # If we remove (u,v) and (u',v'), and add (u,u') and (v,v'), degrees are preserved.
        
        # So:
        # 1. Pick (u,v) from cycle.
        # 2. Pick u' in N(u) \ {v}.
        # 3. Pick v' in N(v) \ {u}.
        # 4. Check if (u', v') is an edge.
        #    If not, we can't do a standard 2-switch with these vertices directly unless we find a common neighbor or something.
        #    Actually, any two edges (u,v) and (x,y) can be swapped to (u,x) and (v,y) or (u,y) and (v,x).
        #    We need to pick a second edge (x,y) such that the new edges are valid.
        
        # Let's pick a random edge (x,y) in the graph, distinct from (u,v).
        # Try swapping (u,v) and (x,y) to (u,x) and (v,y).
        # Conditions:
        # - u != x, v != y
        # - x not in N(u), y not in N(v) (to avoid multi-edges)
        # - x != v, y != u
        # - x != y (obvious)
        
        # This is O(E) to pick a random edge and check.
        
        # To be faster, let's just try a few random swaps.
        
        improved = False
        for _ in range(10): # Try 10 random swaps
            if time.time() - start > time_limit:
                break
                
            # Pick random edge (x, y)
            # We can maintain a list of edges for this
            # For now, pick random x, then random neighbor y
            x = random.randint(0, n - 1)
            if not adj[x]:
                continue
            y = random.choice(adj[x])
            
            if x == u or x == v or y == u or y == v:
                continue
            if x == y:
                continue
                
            # Check if (x,y) is the same as (u,v)
            if (x == u and y == v) or (x == v and y == u):
                continue
                
            # Try swap: Remove (u,v) and (x,y), Add (u,x) and (v,y)
            # Check validity:
            if x in adj[u] or y in adj[v]:
                continue
            if x in adj[v] or y in adj[u]:
                continue
            if x == v or y == u:
                continue
                
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            # Check new girth
            new_girth = _find_girth(adj, g)
            
            if new_girth >= g:
                return True, _adj_to_edges(adj)
                
            # If not valid, we can accept if it improved girth, or randomly (simulated annealing)
            # For simplicity, let's only accept if it improved girth or is equal (to explore).
            # Actually, if it didn't improve, revert.
            if new_girth < girth:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
            else:
                girth = new_girth
                improved = True
                break # Accept this move and restart cycle search
        
        if not improved:
            # If no move improved girth, we might be stuck.
            # Restart with a new random graph?
            # Let's just continue trying different swaps in the next iteration.
            # Or restart.
            if random.random() < 0.1:
                adj = _generate_random_k_regular(n, k)
                if adj is None:
                    break
                girth = _find_girth(adj, g)
                if girth >= g:
                    return True, _adj_to_edges(adj)

    return False, []

def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices. Returns adjacency list (list of lists)."""
    if n * k % 2 != 0:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            continue
        edge_set.add((u, v))
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return None
            
    return adj

def _find_girth(adj, g):
    """Find the girth of the graph. Returns g if girth >= g."""
    n = len(adj)
    min_cycle = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle >= g:
                        return min_cycle
    return min_cycle

def _find_a_short_cycle(adj, g):
    """Find a cycle of length < g. Returns list of (u, v) edges."""
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return [(u, v)]
    return []

def _adj_to_edges(adj):
    """Convert adjacency list to list of edges."""
    edges = []
    n = len(adj)
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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

if __name__ == "__main__":
    n, edges = construct(4, 10)
    print(f"Constructed graph with n={n}, edges={len(edges)}")
