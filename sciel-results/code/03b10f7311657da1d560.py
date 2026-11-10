import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 110
    
    def generate_random_regular_graph(N, k, max_tries=50):
        """
        Generates a random k-regular simple graph on N vertices.
        """
        if N * k % 2 != 0:
            return None
            
        for _ in range(max_tries):
            # Configuration model
            stubs = []
            for i in range(N):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj = [[] for _ in range(N)]
            valid = True
            
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]: # Check for multi-edge (simple check, O(k))
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
                
            if valid:
                # Verify degrees
                if all(len(adj[i]) == k for i in range(N)):
                    # Convert to sorted lists for consistency? Not needed.
                    return adj
                    
        return None

    def compute_girth_and_cycle(adj, N, g_target):
        """
        Computes girth. If girth < g_target, returns (girth, cycle).
        Else returns (girth, None).
        """
        min_girth = float('inf')
        best_cycle = None
        
        for start in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= min_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            # Reconstruct cycle
                            path1 = []
                            curr = u
                            while curr != -1:
                                path1.append(curr)
                                curr = parent[curr]
                            path1.reverse()
                            
                            path2 = []
                            curr = v
                            while curr != -1:
                                path2.append(curr)
                                curr = parent[curr]
                            path2.reverse()
                            
                            # Find LCA
                            # Since it's a BFS tree, the LCA is the first common node from start?
                            # Actually, the cycle is path1 + path2.
                            # The cycle is u -> ... -> LCA -> ... -> v -> u
                            # We can just store the nodes.
                            best_cycle = path1 + path2[::-1]
                            
                            if min_girth >= g_target:
                                return min_girth, None
                                
        return min_girth, best_cycle

    def try_local_search(N, k, g, seed, time_limit=10):
        random.seed(seed)
        
        adj = generate_random_regular_graph(N, k)
        if adj is None:
            return None, None
            
        current_girth, cycle = compute_girth_and_cycle(adj, N, g)
        
        if current_girth >= g:
            # Build edge list
            edges = []
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return N, edges
            
        # Local search
        # We will try to break the shortest cycle.
        
        while time.time() - start_time < time_limit:
            if current_girth >= g:
                break
                
            # Pick a random edge from the cycle
            if not cycle:
                # Recalculate if cycle is None (should not happen if girth < g)
                current_girth, cycle = compute_girth_and_cycle(adj, N, g)
                if current_girth >= g:
                    break
                if not cycle:
                    break
                    
            # Pick an edge (u, v) from the cycle
            # The cycle is a list of nodes. Edges are (c[i], c[i+1])
            # We need to be careful with the closure.
            # Let's just pick a random node from the cycle and one of its neighbors in the cycle.
            
            u_idx = random.randint(0, len(cycle) - 1)
            u = cycle[u_idx]
            # Find v in cycle that is adjacent to u in the cycle
            # The cycle list might not be ordered correctly in my reconstruction above.
            # Let's just pick a random edge in the graph that is part of a short cycle?
            # Simpler: Pick a random edge (u, v) in the graph.
            # If it's in a short cycle, great. If not, it might still help.
            
            # Let's try 2-switch on a random edge.
            # Pick u, v such that v in adj[u]
            u = random.randint(0, N-1)
            if not adj[u]: continue
            v = random.choice(adj[u])
            
            # Pick x, y such that y in adj[x]
            x = random.randint(0, N-1)
            if not adj[x]: continue
            y = random.choice(adj[x])
            
            # Ensure edges are distinct
            if u == v or x == y: continue
            if (u, v) == (x, y) or (u, v) == (y, x): continue
            
            # Try switch (u,x) and (v,y)
            # Check if u-x and v-y are valid (not self, not existing)
            if u != x and v != y:
                if x not in adj[u] and y not in adj[v]:
                    # Perform switch
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    
                    # Check girth
                    new_girth, new_cycle = compute_girth_and_cycle(adj, N, g)
                    current_girth = new_girth
                    cycle = new_cycle
                    
            # If no switch was made, loop continues
            
        # Final check
        current_girth, _ = compute_girth_and_cycle(adj, N, g)
        if current_girth >= g:
            edges = []
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return N, edges
            
        return None, None

    # Try N values
    for N in [1116, 1117, 1118]:
        if time.time() - start_time > time_limit:
            break
        for seed in range(10):
            if time.time() - start_time > time_limit:
                break
            n, edges = try_local_search(N, k, g, seed, time_limit=5)
            if n is not None:
                return n, edges
                
    # Fallback
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
