import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    
    Strategy:
    1. Target a specific number of vertices N that is a multiple of 2k (here 6) 
       and lies between the Moore bound and the current best known solution.
    2. Generate a random k-regular graph of size N.
    3. Use a local search heuristic (2-Swap / Edge Switching) to increase the girth.
       - Find a short cycle (length < g).
       - Perform a random edge swap that breaks this cycle without creating 
         immediate shorter cycles or loops.
    4. Repeat until girth >= g or time limit is reached.
    5. If successful, return the graph. If not, fall back to a known valid 
       construction (e.g., LUW) to ensure a valid result is always returned.
    
    This approach explores the space of random regular graphs, which is 
    structurally different from algebraic families like LUW or Generalized 
    Petersen graphs.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    if k != 3:
        # Fallback for non-cubic, though task is specifically for k=3
        # For simplicity, we focus on k=3. If k!=3, we return a dummy or raise.
        # The task contract implies k=3.
        pass

    # The Moore bound for k=3, g=16 is 512.
    # The best known is 936.
    # Current program (LUW) likely returns ~4374 or larger.
    # We want to find a graph with N < 4374.
    # Let's try N = 936 (best known) or slightly larger to ensure existence.
    # Actually, if we can find *any* graph with N < 4374, we improve the score.
    # Let's try N = 936 first. If it fails, try N = 1000, 1100, etc.
    
    # N must be even for 3-regular graphs (sum of degrees = 3N must be even).
    # Also, N >= 512.
    
    target_sizes = [936, 948, 960, 1000, 1100, 1200, 1500, 2000, 3000, 4000]
    
    # Filter sizes: must be multiple of 2 (for 3-reg) and >= Moore bound.
    # 3-regular graphs exist for all even N >= 4.
    valid_sizes = [n for n in target_sizes if n % 2 == 0 and n >= 512]
    
    best_n = float('inf')
    best_edges = None
    best_girth_found = 0

    # Helper: Generate random 3-regular graph
    def random_3_regular(n):
        # Use configuration model
        # 3n stubs
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        seen = set()
        # Pair up stubs
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self-loop, retry logic would be complex, 
                # for large n, probability is low. 
                # If it happens, we just skip this pair and swap with another?
                # Simpler: regenerate if self-loop? 
                # For n=936, prob of self loop is 1/(3n-1) ~ 1/2800.
                # Let's just ensure no self loops in the final check.
                # Actually, configuration model can produce multi-edges too.
                # We need a simple graph.
                pass
            
            if u > v:
                u, v = v, u
            if (u, v) in seen:
                # Multi-edge, retry
                continue
            seen.add((u, v))
            edges.append((u, v))
        
        # This simple pairing might fail to produce a simple graph or 
        # might not connect all vertices properly if we just skip.
        # Better approach: Use a library or a robust random regular graph generator.
        # Since we only have stdlib, let's use a simpler "swap" based generator
        # starting from a known 3-regular graph (e.g., a cycle with chords? No, cycle is 2-reg).
        # Start with a 3-regular graph: 
        # For even n, we can start with a cycle (i, i+1) and add chords? No, that changes degree.
        
        # Let's use a known valid 3-regular graph as a seed: 
        # The "Prism" graph? No, that's specific.
        # Let's generate a random 3-regular graph using the "Wormald" style or 
        # just shuffle and fix.
        
        # Robust Random 3-Reg Generator:
        # 1. Create list of stubs.
        # 2. Shuffle.
        # 3. Pair sequentially.
        # 4. If self-loop or multi-edge, swap the partner of one stub with another random stub to fix.
        
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        
        edges = []
        adj = [[] for _ in range(n)]
        
        # We need to ensure simple graph.
        # Let's try a few times to generate a simple one.
        for _ in range(100):
            stubs_copy = stubs[:]
            random.shuffle(stubs_copy)
            edges_temp = []
            adj_temp = [[] for _ in range(n)]
            valid = True
            for i in range(0, n*3, 2):
                u = stubs_copy[i]
                v = stubs_copy[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj_temp[u]:
                    valid = False
                    break
                edges_temp.append((u, v))
                adj_temp[u].append(v)
                adj_temp[v].append(u)
            
            if valid:
                # Check degrees
                if all(len(a) == 3 for a in adj_temp):
                    return edges_temp
        # If failed 100 times (very unlikely for n>10), return None
        return None

    # Helper: Verify Girth
    def get_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_cycle = float('inf')
        
        # BFS from each node to find shortest cycle back to itself
        # This is O(N * E) which is ~ 936 * 1404 ~ 1.3M ops. Fast.
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            queue = deque()
            queue.append(start)
            dist[start] = 0
            parent[start] = -2 # Mark start
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        # But we need to be careful. 
                        # In BFS, if we see a visited node v that is not the parent of u,
                        # we have a cycle.
                        # The length is dist[u] + dist[v] + 1.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        # Optimization: if min_cycle == 3, we can break early?
                        # Girth 3 is impossible for 3-reg? No, triangles exist.
                        # But we want high girth.
                        # If min_cycle < g, we can stop early if we just need "is it < g".
                        # But we want to know the actual girth to guide search.
                        
            # Note: The above BFS finds the shortest cycle *passing through* start?
            # No, BFS from start finds the shortest cycle that includes start.
            # To find the global girth, we need to check all starts?
            # Actually, the shortest cycle in the graph will be found when we BFS from 
            # any node on that cycle. 
            # However, the standard BFS from a single node finds the shortest cycle 
            # that *contains* that node. 
            # To find the global girth, we must run BFS from all nodes? 
            # Or is there a faster way?
            # For n=936, O(N^2) is ~ 1M, which is fine.
            # But the BFS above only checks cycles containing `start`.
            # So yes, we need to run it for all starts, or use a more complex algorithm.
            # Given n is small, running BFS from all nodes is acceptable.
            # Wait, the BFS above is O(N+E) per start. Total O(N(N+E)) ~ O(N^2).
            # 936^2 ~ 870,000. Very fast.

        return min_cycle if min_cycle != float('inf') else 2*n # Fallback

    # Helper: 2-Swap
    def perform_swap(edges, adj, u, v, x, y):
        # Remove edges (u,v) and (x,y)
        # Add edges (u,x) and (v,y) or (u,y) and (v,x)
        # We need to ensure no self-loops or multi-edges.
        # This function is a bit complex to implement inline efficiently.
        # Let's use a simpler "Edge Switch" heuristic:
        # Pick a random edge (u, v).
        # Pick a random edge (x, y) such that u!=x, u!=y, v!=x, v!=y.
        # Try swapping to (u, x) and (v, y).
        # Check validity.
        
        pass # Placeholder, will implement specific swap logic in the loop

    # Main Search Loop
    for N in valid_sizes:
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a graph of size N with girth >= g
        # We do multiple restarts for this N
        max_restarts = 5
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            edges = random_3_regular(N)
            if not edges:
                continue
            
            adj = [[] for _ in range(N)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Local Search
            # We will iterate for a fixed number of steps or until girth is good
            steps = 5000
            for step in range(steps):
                if time.time() > deadline - 2.0:
                    break
                    
                # Check girth
                # Optimization: Instead of full girth calc every step, 
                # just check if girth < g.
                # We can do a BFS from a random node to see if we find a cycle < g.
                # If we do, we break that cycle.
                
                # Find a short cycle
                short_cycle = None
                # Sample a few random starts to find a short cycle quickly
                for _ in range(10):
                    start = random.randint(0, N-1)
                    dist = [-1] * N
                    parent = [-1] * N
                    queue = deque()
                    queue.append(start)
                    dist[start] = 0
                    parent[start] = -2
                    
                    while queue:
                        u = queue.popleft()
                        if dist[u] >= g: # We only care about cycles < g
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            elif v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < g:
                                    # Reconstruct cycle
                                    # Path from v to LCA, path from u to LCA
                                    # This is complex. 
                                    # Simpler: Just pick a random edge on the cycle?
                                    # We don't have the cycle path easily.
                                    # Let's just perform a random swap involving u or v?
                                    # No, that's blind.
                                    
                                    # Let's find the LCA of u and v in the BFS tree.
                                    # Actually, for local search, we can just pick 
                                    # a random edge incident to u or v and swap it 
                                    # with a random other edge to try to break the structure.
                                    
                                    # Better: Identify the cycle edges.
                                    # Since we don't have the path, let's just 
                                    # do a random valid swap in the neighborhood.
                                    short_cycle = True
                                    break
                        if short_cycle:
                            break
                    if short_cycle:
                        break
                
                if not short_cycle:
                    # No short cycle found in sample
                    # It might still exist, but we assume it's good enough for now
                    # Or we can do a full check
                    # Let's do a full check every 100 steps
                    if step % 100 == 0:
                        current_girth = get_girth(N, edges)
                        if current_girth >= g:
                            found_valid = True
                            best_n = N
                            best_edges = edges
                            break
                    else:
                        # Continue searching
                        pass
                        
                # Perform a random swap
                # Pick random edge (u, v)
                e_idx = random.randint(0, len(edges)-1)
                u, v = edges[e_idx]
                
                # Pick random edge (x, y)
                # Ensure distinct vertices
                attempts = 0
                while attempts < 10:
                    x_idx = random.randint(0, len(edges)-1)
                    x, y = edges[x_idx]
                    if u in [x, y] or v in [x, y]:
                        attempts += 1
                        continue
                    # Try swap 1: (u, x) and (v, y)
                    if x not in adj[u] and y not in adj[v] and u != x and v != y:
                        # Valid swap 1
                        # Remove (u, v) and (x, y)
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
                        edges[e_idx] = (u, x)
                        edges[x_idx] = (v, y)
                        
                        adj[u].append(x)
                        adj[x].append(u)
                        adj[v].append(y)
                        adj[y].append(v)
                        break
                    # Try swap 2: (u, y) and (v, x)
                    elif y not in adj[u] and x not in adj[v] and u != y and v != x:
                        adj[u].remove(v)
                        adj[v].remove(u)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
                        edges[e_idx] = (u, y)
                        edges[x_idx] = (v, x)
                        
                        adj[u].append(y)
                        adj[y].append(u)
                        adj[v].append(x)
                        adj[x].append(v)
                        break
                    else:
                        attempts += 1
                        continue

            if found_valid:
                break
                
        if found_valid:
            break

    # Fallback: If local search didn't find a small graph, use LUW
    if best_n == float('inf'):
        if time.time() < deadline - 10.0:
            # Use the LUW construction from the current program as a fallback
            # This ensures we return *something* valid, even if large.
            def luw_construct(k, g, d_override=None):
                if k != 3:
                    return None, None
                q = 3
                d = d_override
                if d is None:
                    d = 7
                
                add = lambda x, y: (x + y) % q
                sub = lambda x, y: (x - y) % q
                mul = lambda x, y: x * y % q
                
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
                            if side == 0:
                                y[c] = add(x[c], mul(y[a], x[b]))
                            else:
                                y[c] = sub(x[c], mul(x[a], y[b]))
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

            n_luw, edges_luw = luw_construct(3, g, 7)
            if n_luw:
                # Verify just in case
                def verify_girth(n, edges, k, g):
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    for i in range(n):
                        if len(adj[i]) != k:
                            return False
                    
                    max_depth = g // 2
                    for start in range(n):
                        visited_depth = {}
                        queue = deque()
                        for v in adj[start]:
                            queue.append((v, start, 1))
                            visited_depth[(v, start)] = 1
                        
                        while queue:
                            u, prev, depth = queue.popleft()
                            if depth == max_depth:
                                continue
                            for v in adj[u]:
                                if v == prev:
                                    continue
                                if v == start:
                                    if depth + 1 < g:
                                        return False
                                    continue
                                state = (v, u)
                                if state in visited_depth:
                                    if depth - visited_depth[state] < g:
                                        return False
                                else:
                                    visited_depth[state] = depth
                                    queue.append((v, u, depth + 1))
                    return True

                if verify_girth(n_luw, edges_luw, 3, g):
                    best_n = n_luw
                    best_edges = edges_luw

    if best_n == float('inf'):
        # Absolute last resort: return a dummy valid graph?
        # We must return something.
        # If LUW failed verification (shouldn't), we try d=8.
        def luw_construct_safe(k, g, d_override):
            q = 3
            d = d_override
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
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
                        if side == 0:
                            y[c] = add(x[c], mul(y[a], x[b]))
                        else:
                            y[c] = sub(x[c], mul(x[a], y[b]))
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

        n_luw, edges_luw = luw_construct_safe(3, g, 8)
        best_n = n_luw
        best_edges = edges_luw

    return best_n, best_edges

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
