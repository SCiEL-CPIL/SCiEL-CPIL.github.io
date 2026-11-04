import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin before 150s limit
    
    # Moore bound for k=3, g=16
    # For even girth g=2m, lower bound is 2 * (k-1)^(m-1)
    # m = 8
    # LB = 2 * 2^7 = 256
    # However, known cages are often larger. 
    # The seed LUW graph for g=16 is quite large (~350k or ~4k depending on implementation details in seed).
    # Let's aim for a size that is feasible for local search.
    # n must be even.
    # Let's start with n = 2 * (k-1) * (g-2) + 2 or similar heuristic?
    # Actually, for k=3, g=16, the Hoffman-Singleton bound is 32.
    # The (3,16)-cage is known to have 936 vertices? Wait, the prompt says "smallest known ... 936".
    # But the prompt also says "best lower bound is 512".
    # Wait, 936 vertices is very large for a local search to fix if we start random.
    # However, the seed program's LUW graph size depends on the 'd' parameter.
    # The insight says: "Fixing d from 11 to 7 for g=16 reduces LUW graph size from ~354k to ~4k".
    # So the seed is likely producing a graph of ~4k vertices.
    # 4000 vertices is small enough for local search.
    
    # Let's try to find a graph with n vertices, 3-regular, girth >= 16.
    # We will try a range of n values.
    # Start with n close to the lower bound but increased to allow random generation.
    # A random 3-regular graph on n vertices has girth ~ log_2(n).
    # For girth 16, we need n such that log_2(n) ~ 16 => n ~ 65536?
    # No, girth is the length of the *shortest* cycle.
    # In a random 3-regular graph, the expected number of cycles of length l is approx (k-1)^l / (2l).
    # For k=3, expected cycles of length 16 is 2^16 / 32 = 2048.
    # So a random graph will have MANY short cycles.
    # We need to remove them.
    
    # Strategy:
    # 1. Pick n. Let's try n = 2 * (k-1) * 2^(g/2 - 1) * factor?
    # The Moore bound is 256.
    # Let's try n = 512, 1024, 2048, 4096, 8192.
    # We want to minimize n.
    # We'll iterate n from low to high.
    # For each n, generate random 3-regular graph.
    # Check girth. If girth < g, try to fix via edge swaps.
    
    def generate_random_3_regular(n):
        # Configuration model
        if n % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        seen = set()
        for i in range(0, n*3, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None # Loop
            if (u, v) in seen or (v, u) in seen:
                return None # Multi-edge
            edges.append((u, v))
            seen.add((u, v))
            seen.add((v, u))
        return edges

    def get_adjacency(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def find_shortest_cycle(adj, n, max_girth_check):
        # BFS from each node to find shortest cycle
        # Optimization: if we find a cycle of length L, we don't need to go deeper than L-1 from that start?
        # Actually, standard BFS for girth:
        # For each node s, BFS. If we find a back edge (u,v) where dist[u] + dist[v] + 1 < current_best, update.
        # We can stop BFS from s if dist[u] + 1 >= current_best_girth? No, because the other end might be close.
        # But if dist[u] >= current_best_girth, we can stop expanding u.
        
        min_girth = max_girth_check + 1
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] + 1 >= min_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    def try_local_search(edges, n, target_girth, max_iters=10000):
        adj = get_adjacency(edges, n)
        current_girth = find_shortest_cycle(adj, n, target_girth)
        if current_girth >= target_girth:
            return edges, True
        
        # We have short cycles.
        # Find a specific short cycle to break.
        # We can modify find_shortest_cycle to return the cycle itself.
        
        def find_one_short_cycle(adj, n, limit):
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    if dist[u] + 1 >= limit:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < limit:
                                # Reconstruct cycle
                                # Path s -> ... -> u and s -> ... -> v
                                path_u = []
                                curr = u
                                while curr != -1:
                                    path_u.append(curr)
                                    curr = parent[curr]
                                path_v = []
                                curr = v
                                while curr != -1:
                                    path_v.append(curr)
                                    curr = parent[curr]
                                # path_u and path_v meet at s (index 0)
                                # Cycle is path_u + reversed(path_v[1:])
                                # But we just need the edges.
                                # Let's just return the nodes in the cycle
                                cycle_nodes = path_u + path_v[1:][::-1]
                                return cycle_nodes
            return None

        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                return edges, False
            
            cycle_nodes = find_one_short_cycle(adj, n, target_girth)
            if not cycle_nodes:
                # No cycle shorter than target_girth found?
                # Maybe find_shortest_cycle was faster/accurate?
                return edges, True
            
            # We have a cycle. Pick a random edge in the cycle to swap.
            # To break the cycle, we can swap an edge (u,v) in the cycle with a non-cycle edge (u, w) or (v, w).
            # Actually, a standard move: pick two edges (u,v) and (x,y) not in the cycle (or one in, one out) and swap to (u,x) and (v,y).
            # Simpler: Pick an edge (u,v) in the cycle. Find a neighbor w of u (w not in cycle) and swap (u,v) with (u,w)? No, that changes degree.
            # Correct swap:
            # Pick edge e1=(u,v) in cycle.
            # Pick edge e2=(x,y) NOT in cycle (or just any edge such that swapping maintains 3-regularity and simplicity).
            # Swap: remove (u,v) and (x,y). Add (u,x) and (v,y) (or u,y and v,x).
            # We must ensure (u,x) and (v,y) are not loops or existing edges.
            
            cycle_set = set(cycle_nodes)
            # Get edges in cycle
            # The cycle is a path in the tree + one back edge.
            # Let's just pick a random edge from the graph that connects two nodes in the cycle? No.
            # Let's pick a random edge (u,v) from the cycle.
            # The cycle nodes are in order. Edges are (cycle_nodes[i], cycle_nodes[i+1]) and (last, first).
            cycle_edges = []
            for i in range(len(cycle_nodes)):
                u = cycle_nodes[i]
                v = cycle_nodes[(i+1) % len(cycle_nodes)]
                cycle_edges.append((u, v))
            
            # Pick a random edge from cycle_edges
            u, v = random.choice(cycle_edges)
            
            # We want to swap (u,v) with some other edge (x,y).
            # To maintain 3-regularity, we need to pick an edge (x,y) such that x and y are not u or v?
            # No, if we swap (u,v) and (x,y) to (u,x) and (v,y), degrees are preserved.
            # We need (u,x) and (v,y) to be valid (no loops, no multi-edges).
            # Let's pick x from neighbors of u (excluding v) and y from neighbors of x (excluding u)?
            # This is getting complicated.
            
            # Alternative simple move:
            # Pick two non-adjacent edges (u,v) and (w,z) in the graph.
            # If we swap to (u,w) and (v,z), we might create loops or multi-edges.
            
            # Let's use a specific heuristic:
            # Pick an edge (u,v) in the cycle.
            # Pick a neighbor w of u (w != v).
            # If (w, v) is not an edge and w != v, we can't just swap.
            
            # Let's try:
            # 1. Pick random edge (u,v) in cycle.
            # 2. Pick random edge (x,y) in graph.
            # 3. Check if swapping (u,v) and (x,y) for (u,x) and (v,y) is valid.
            #    - u != x, v != y (no loops)
            #    - (u,x) not in edges, (v,y) not in edges
            #    - x != y (always true for simple graph)
            #    - Also need to ensure we don't create a loop in the other pair?
            #    - If we swap to (u,y) and (v,x), check that too.
            
            success = False
            for _ in range(20): # Try a few random swaps
                if time.time() > deadline - 5.0:
                    return edges, False
                x, y = random.choice(edges)
                if x == u or x == v or y == u or y == v:
                    continue
                
                # Try swap 1: (u,x) and (v,y)
                if (u, x) not in seen and (v, y) not in seen and u != x and v != y:
                    # Also check if (u,x) or (v,y) are in cycle? Doesn't matter for validity, only for girth.
                    # Remove (u,v) and (x,y)
                    edges.remove((u, v))
                    edges.remove((x, y))
                    seen.discard((u, v))
                    seen.discard((v, u))
                    seen.discard((x, y))
                    seen.discard((y, x))
                    
                    # Add (u,x) and (v,y)
                    edges.append((u, x))
                    edges.append((v, y))
                    seen.add((u, x))
                    seen.add((x, u))
                    seen.add((v, y))
                    seen.add((y, v))
                    
                    # Update adjacency
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    
                    success = True
                    break
                
                # Try swap 2: (u,y) and (v,x)
                if (u, y) not in seen and (v, x) not in seen and u != y and v != x:
                    edges.remove((u, v))
                    edges.remove((x, y))
                    seen.discard((u, v))
                    seen.discard((v, u))
                    seen.discard((x, y))
                    seen.discard((y, x))
                    
                    edges.append((u, y))
                    edges.append((v, x))
                    seen.add((u, y))
                    seen.add((y, u))
                    seen.add((v, x))
                    seen.add((x, v))
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(y)
                    adj[y].append(u)
                    adj[v].append(x)
                    adj[x].append(v)
                    
                    success = True
                    break
            
            if not success:
                # If we couldn't make a valid swap, maybe try a different cycle edge?
                # Or just continue to next iteration hoping for a better random pick.
                continue

            # Re-check girth
            # Optimization: only check if girth improved?
            # We can just check the whole thing.
            current_girth = find_shortest_cycle(adj, n, target_girth)
            if current_girth >= target_girth:
                return edges, True

    # Search loop
    # Try different n values
    # Start from a reasonable lower bound.
    # Moore bound 256.
    # Let's try n = 256, 258, 260... up to some limit.
    # But random generation fails often for small n.
    # Let's jump to n=512, 1024, 2048, 4096.
    # The seed LUW was ~4k.
    # We want to beat 4k if possible, or at least match it.
    # If we find a graph with n < best_n, we keep it.
    
    best_n = None
    best_edges = None
    
    # List of candidate sizes to try
    # We want to minimize n.
    # Let's try increasing n until we find a valid graph.
    # Start with n = 2 * (k-1) * 2^(g/2 - 1) = 2 * 2 * 2^7 = 512?
    # Actually, let's just try a few sizes.
    
    candidate_ns = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144]
    
    # Shuffle to avoid getting stuck on one size if it's hard?
    # No, we want the smallest n. So iterate in order.
    
    for n in candidate_ns:
        if time.time() > deadline - 10.0:
            break
            
        # Try a few random instances for this n
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline - 10.0:
                break
                
            edges = generate_random_3_regular(n)
            if not edges:
                continue
                
            adj = get_adjacency(edges, n)
            # Quick girth check
            girth = find_shortest_cycle(adj, n, g)
            
            if girth >= g:
                # Found a valid graph!
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                break # No need to try more attempts for this n if we found one?
                      # Actually, we might find a smaller n later? No, n is fixed in this loop.
                      # We are iterating n upwards. So the first valid n is the smallest for this candidate list.
                      # We can break out of the n loop too?
                      # Yes, if we found a valid graph at n, any larger n is worse.
                break
        
        if best_n is not None:
            # We found a graph. Since we iterate n in increasing order, this is the best we can get from this list.
            # However, we might have skipped some n values (e.g. 514, 516).
            # For simplicity, we accept the first found in the candidate list.
            # To be more rigorous, we could check n, n+2, n+4...
            # But given time constraints, the candidate list approach is safer.
            pass

    # If no graph found, fallback to LUW?
    # The seed program returns LUW. We should probably return something valid.
    # If best_n is None, we must return a valid graph.
    # Let's implement a simple LUW fallback or just return the last tried graph even if invalid?
    # The evaluator checks validity. Invalid scores 0.
    # So we MUST return a valid graph.
    
    if best_n is None:
        # Fallback: Use the LUW construction from the seed, but optimized for speed/size if possible.
        # The seed's LUW for g=16 was ~4k.
        # Let's just run the seed's LUW logic here as a fallback.
        def get_luw_graph(k, g):
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

        best_n, best_edges = get_luw_graph(k, g)

    return best_n, best_edges
