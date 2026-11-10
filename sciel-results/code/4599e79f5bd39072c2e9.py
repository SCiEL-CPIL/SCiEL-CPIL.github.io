# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Uses Random Configuration Model + 2-Switch Local Search.
    """
    start_time = time.time()
    deadline = start_time + 110

    if k != 3:
        return _lnw_construct(k, g)

    # Target girth is g. We need to find a 3-regular graph with girth >= g.
    # The Moore bound for n(3, 16) is 512. Best known is 936.
    # We will try sizes starting from a reasonable lower bound.
    # For girth 16, n must be even.
    
    # Estimate minimum n.
    # Moore bound: 1 + k + k(k-1) + ... for even girth.
    # For g=16, radius 7.
    # B(3, 7) = 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382? 
    # Wait, Moore bound for even girth 2r is 2 * sum_{i=0}^{r-1} (k-1)^i.
    # r = 8.
    # Sum = 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255.
    # n >= 2 * 255 = 510.
    # So n must be >= 512 (even).
    
    min_n = 512
    
    # We will try a few candidate sizes.
    # The best known is 936. Let's try around there.
    # We can start with a random graph of size n and try to increase girth.
    
    best_n = float('inf')
    best_edges = None

    # Try a range of n values. 
    # Smaller n is better. 
    # However, finding a high girth graph in small n is hard.
    # We'll try n = 512, 514, ..., up to 1000 or so.
    # But 2-switch search is stochastic.
    
    # To ensure we return something valid if search fails, we keep the LUW fallback.
    
    # Let's try n = 936 first as it is known to exist.
    # Or start smaller?
    # Let's try a sweep of n values.
    
    candidate_ns = []
    # Start from Moore bound, step by 2
    for n in range(min_n, 1500, 2):
        candidate_ns.append(n)
        
    # Shuffle to avoid getting stuck in one size if it's hard?
    # No, smaller is better. We should try smallest first.
    # But if smallest is impossible or hard, we waste time.
    # Given 120s limit, we can afford to try a few.
    
    # Strategy:
    # 1. Try n=936 (known to exist). If we find it, great.
    # 2. If not, try smaller?
    # Actually, the search is hard.
    # Let's just pick n=936 and run a robust search.
    # If it fails, try n=938, 940...
    # If all fail, return LUW.
    
    # Wait, the score is based on n.
    # If we find n=936, score is 936.
    # If we find n=1000, score is 1000.
    # If we return LUW (n~177k), score is 177k.
    
    # So we want the smallest n.
    
    # Let's try n in [512, 514, ..., 1000].
    # For each n, run search for a fixed time slice.
    
    num_trials = 10
    time_per_trial = 8.0 # 10 * 8 = 80s. Leaves 30s for LUW fallback.
    
    for n in candidate_ns[:num_trials]: # Try first 10 smallest
        if time.time() > deadline - 30:
            break
            
        edges = _search_2switch(n, g, time.time() + time_per_trial)
        
        if edges is not None:
            # Verify
            adj = _edges_to_adj(edges, n)
            girth = _calculate_girth(adj, n)
            if girth >= g:
                return n, edges
            # If verification fails, ignore (shouldn't happen if search is correct)
        
    # If search didn't find a small one, or timed out
    # Try a larger, known-good size if we have time?
    # Or just fallback to LUW.
    
    if time.time() < deadline - 10:
        # Try n=936 specifically with more time?
        # Or just return LUW.
        pass

    return _lnw_construct(3, g)

def _search_2switch(n, g, deadline):
    """
    Find a 3-regular graph on n vertices with girth >= g using 2-switches.
    """
    if n % 2 != 0 or n < 4:
        return None
        
    # Generate random 3-regular graph
    # Use configuration model
    # 1. Create list of 3n stubs.
    # 2. Randomly pair them.
    # 3. Check for multi-edges and loops. If found, retry or fix.
    
    # Better: Use a library or robust generator.
    # Standard lib only.
    
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
    
    # Initial girth check
    # We need to track short cycles.
    # For efficiency, we can just run _calculate_girth.
    # But _calculate_girth is O(N^2).
    # For N=1000, 10^6 ops. 1000 iterations = 10^9 ops. Too slow in Python.
    
    # We need a faster girth check or incremental update.
    # Alternative: Check if girth < g.
    # If yes, find a short cycle and try to break it.
    
    # Let's optimize _calculate_girth to return early if girth < g.
    
    def check_girth(adj, n, target_girth):
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
                                if min_cycle < target_girth:
                                    return min_cycle
        return min_cycle
    
    current_girth = check_girth(adj, n, g)
    
    if current_girth >= g:
        return edges
        
    # Local Search
    # While time permits:
    # 1. Find a short cycle (length < g).
    # 2. Pick a random edge in that cycle.
    # 3. Pick a random edge not in that cycle (or just any random edge).
    # 4. Perform 2-switch.
    # 5. Check if girth improved or maintained.
    
    # To find a short cycle efficiently, we can run BFS from a random node and stop when we find a cycle < g.
    
    def find_short_cycle(adj, n, target_girth):
        # BFS from random start
        start = random.randint(0, n-1)
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= target_girth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            # Reconstruct cycle
                            # Path u -> ... -> v
                            # v is ancestor of u? Or u is ancestor of v?
                            # In BFS, if v is visited, and v != parent[u],
                            # it means there's a back edge.
                            # The cycle is u-v-...-u?
                            # No, v is already visited.
                            # The cycle is formed by the path from v to LCA(v,u) and u to LCA?
                            # Simpler:
                            # We know dist[u] and dist[v].
                            # We can trace back from u and v to find LCA.
                            # But for 2-switch, we just need ANY edge in a short cycle.
                            # The edge (u,v) is part of a cycle of length cycle_len.
                            # So we can just return (u,v).
                            return (u, v)
        return None
    
    while time.time() < deadline:
        short_edge = find_short_cycle(adj, n, g)
        if short_edge is None:
            # No short cycle found from this start?
            # It might be that BFS didn't cover all, or girth is actually >= g.
            # Let's do a full check.
            if check_girth(adj, n, g) >= g:
                return edges
            # Otherwise, try again
            continue
            
        u, v = short_edge
        
        # Find another edge (x, y)
        # Random edge
        # We can pick a random vertex x, and a random neighbor y.
        # Ensure (x,y) is not (u,v) or (v,u).
        # Also ensure the swap doesn't create a loop or multi-edge.
        
        # To be efficient, pick random x, y from adj[x].
        for _ in range(10): # Try a few random swaps
            x = random.randint(0, n-1)
            if x == u or x == v:
                continue
            y = random.choice(list(adj[x]))
            if y == u or y == v:
                continue
            if x == y:
                continue
                
            # Check if (x,y) is the same edge as (u,v)
            if (x == u and y == v) or (x == v and y == u):
                continue
                
            # Perform 2-switch: (u,v) and (x,y) -> (u,x) and (v,y)
            # Check validity:
            # 1. u != x, v != y (already checked x!=u, x!=v, y!=u, y!=v? No, y can be u/v? No, we checked y!=u, y!=v)
            # 2. (u,x) not in adj
            # 3. (v,y) not in adj
            
            if x in adj[u] or y in adj[v]:
                continue
                
            # Swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            # Update edges list? 
            # We can reconstruct edges at the end.
            
            # Check girth
            # If girth improved, keep.
            # If girth worsened, revert?
            # Or just keep if it didn't create shorter cycles?
            # Simple hill climbing: Keep if girth is same or better.
            
            new_girth = check_girth(adj, n, g)
            
            if new_girth >= g:
                # Success
                # Reconstruct edges
                final_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            final_edges.append((i, j))
                return final_edges
                
            # If new_girth < g, we might still keep it if it's "closer"?
            # For simplicity, revert if it doesn't help immediately?
            # Or accept with some probability?
            # Let's revert to avoid getting stuck in bad local minima.
            # Actually, 2-switches can increase girth.
            # If we revert every time girth < g, we might never escape a bad state.
            # But if we start random, we might just need many steps.
            # Let's try: Accept if new_girth >= current_girth.
            # If new_girth < current_girth, revert.
            
            # We don't track current_girth precisely in the loop, we just check < g.
            # Let's store last_good_girth.
            
            # Revert
            adj[u].remove(x)
            adj[x].remove(u)
            adj[v].remove(y)
            adj[y].remove(v)
            
            adj[u].add(v)
            adj[v].add(u)
            adj[x].add(y)
            adj[y].add(x)
            
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
    q = k
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

def _lnw_construct(k, g):
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
# EVOLVE-BLOCK-END
