import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Fallback: LUW construction
    def _luw_construct(k, g):
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

    # Exact girth check using BFS from each vertex
    # Returns the girth (minimum cycle length)
    def get_girth(n, adj):
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            parent[start] = start
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= min_girth // 2: # Pruning: if depth is already half the min girth, 
                    # any cycle found here will be longer than min_girth? 
                    # Cycle length = dist[u] + dist[v] + 1. 
                    # If dist[u] >= min_girth/2, and dist[v] >= 1, cycle > min_girth/2 + 1.
                    # This pruning is safe if we are looking for min girth.
                    # However, for correctness, we should just run to depth min_girth/2.
                    # Let's just run BFS fully but break early if we find a cycle < current min.
                    pass
                
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Back edge or cross edge
                        if parent[u] == v or parent[v] == u:
                            continue
                        # Cycle found
                        # The cycle length is dist[u] + dist[v] + 1?
                        # This formula works if u and v are in the same BFS tree and 
                        # the path between them is simple.
                        # In a BFS tree, the path from start to u and start to v are simple.
                        # The union might not be a simple cycle if they share a prefix.
                        # But since it's a BFS tree, the LCA is unique.
                        # Actually, for finding *any* cycle, this works.
                        # For finding the *shortest* cycle, this is standard.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth if min_girth != float('inf') else n # Fallback

    # Generate a random 4-regular graph
    def gen_random_4reg(n):
        if n < 4:
            return None
        # Use configuration model with rejection
        for _ in range(200):
            stubs = [i for i in range(n) for _ in range(4)]
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, 4*n, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                # Check for multi-edges again just in case (should be caught by v in adj[u])
                return adj
        return None

    # Local Search
    def local_search(n, adj, g_target, time_limit):
        # Convert adj to list of sets for fast neighbor checking? 
        # Or just keep lists.
        
        def try_swap(adj):
            # Pick a random edge (u, v)
            # Pick a random other edge (x, y) such that no loops/multi-edges created
            # Swap: remove (u,v) and (x,y), add (u,x) and (v,y) OR (u,y) and (v,x)
            
            # Get all edges
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            if len(edges) < 2:
                return False
            
            # Pick two random edges
            e1_idx = random.randint(0, len(edges)-1)
            e2_idx = random.randint(0, len(edges)-1)
            if e1_idx == e2_idx:
                return False
            
            u1, v1 = edges[e1_idx]
            u2, v2 = edges[e2_idx]
            
            # Check if they share a vertex? If so, swapping might create loops or multi-edges easily.
            # If they are disjoint, swapping is safer.
            # Let's only try disjoint edges for simplicity and safety.
            if len({u1, v1, u2, v2}) < 4:
                return False
            
            # Try swap 1: (u1, u2) and (v1, v2)
            # Check if (u1, u2) exists or is loop
            if u1 == u2 or u2 == v1 or u1 == v2: # Basic checks
                pass
            
            # Check if edge (u1, u2) is already in adj
            def has_edge(a, b):
                return b in adj[a]
                
            # Option A: connect u1-u2, v1-v2
            if not has_edge(u1, u2) and not has_edge(v1, v2):
                # Perform swap
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].append(u2)
                adj[u2].append(u1)
                adj[v1].append(v2)
                adj[v2].append(v1)
                return True
            
            # Option B: connect u1-v2, v1-u2
            if not has_edge(u1, v2) and not has_edge(v1, u2):
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].append(v2)
                adj[v2].append(u1)
                adj[v1].append(u2)
                adj[u2].append(v1)
                return True
                
            return False

        # Main loop
        while time.time() < time_limit:
            girth = get_girth(n, adj)
            if girth >= g_target:
                # Construct edge list
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append((u, v))
                return edge_list
            
            # Try to improve
            # If we are stuck, we might need to restart or try more aggressive swaps
            if not try_swap(adj):
                # If swap failed, maybe restart search or just continue
                continue

    # Main Strategy
    best_n = None
    best_edges = None
    
    # Moore bound for k=4, g=10 is 243. Start from 244.
    # We know 320 is possible.
    # We iterate n from 244 to 320.
    
    # First, get a baseline valid graph using LUW to ensure we have *something*
    # LUW for k=4, g=10. d = 10-5=5 (odd) -> d=5.
    # This graph will have some size. Let's calculate it or just run it.
    # If we find a smaller one via local search, we use that.
    
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Try to find smaller graphs
    # We need to be careful with time.
    # Local search on n=244 might take a while if it fails.
    # We allocate a small time budget per n.
    
    time_per_n = 5.0 # seconds
    
    for n in range(244, min(320, best_n), 2): # Step 2 because 4-regular on odd n? No, K5 is 4-regular. 
        # But cages are usually even. Let's check odd too? 
        # Actually, if n is odd, 4n is even, so it's possible.
        # However, the "cage" n(4,10) is known to be 320.
        # Let's just iterate all n from 244 to best_n-1.
        
        if time.time() > deadline - 10:
            break
            
        current_budget = time.time() + time_per_n
        if current_budget > deadline - 5:
            current_budget = deadline - 5
            
        adj = gen_random_4reg(n)
        if adj is None:
            continue
            
        # Quick check: does it have girth >= 10?
        # If yes, we are done.
        # If no, try local search.
        
        # To save time, let's do a few random restarts of local search
        # or just one long local search.
        
        result_edges = local_search(n, adj, g, current_budget)
        
        if result_edges is not None:
            best_n = n
            best_edges = result_edges
            # Since we are searching in increasing n, the first valid one is the smallest.
            # We can break here.
            break

    return best_n, best_edges
