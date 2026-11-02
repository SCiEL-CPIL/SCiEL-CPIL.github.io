import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # The seed program uses L-U-W which is very large.
    # We try to construct a smaller 3-regular graph with girth >= g.
    # For k=3, g=16, Moore bound is 381. Best known is 936.
    # We try greedy construction for n in a range.
    
    if k != 3:
        # Fallback to seed logic for other k, though task specifies k=3
        # Re-implementing seed logic briefly or just returning a dummy if k!=3
        # But task says "evaluator passes this task's degree k", implying k=3.
        # If k is not 3, we might just use the seed logic or a simple cycle.
        # For safety, if k != 3, we try to use the seed's L-U-W logic if possible, 
        # but since we don't have it, we'll try a simple random 3-regular logic 
        # or just return a large cycle if k=2? 
        # Assuming k=3 for this task.
        pass

    # Target girth is g. We need no cycle < g.
    # So we need to avoid cycles of length < g.
    # Cycle length L. Path length L-1.
    # So we check paths of length < g-1.
    # Max path length to check: g-2.
    # For g=16, check paths of length up to 14.
    
    max_path_len = g - 2
    
    def creates_short_cycle(u, v, adj):
        # Check if there is a path from u to v of length < g-1
        # i.e., length <= g-2
        if u == v:
            return True
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= max_path_len:
                continue
            for neighbor in adj[node]:
                if neighbor == v:
                    # Path length is dist + 1
                    # Cycle length will be dist + 1 + 1 = dist + 2
                    # We need cycle length >= g
                    # So dist + 2 >= g => dist >= g - 2
                    # If dist < g - 2, we found a short cycle.
                    if dist + 2 < g:
                        return True
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return False

    def try_construct(n):
        # Greedy construction of n-vertex k-regular graph with girth >= g
        # We aim for k=3
        if n < 2:
            return None
        
        deg = [0] * n
        adj = [set() for _ in range(n)]
        
        # We need to add n*k/2 edges
        total_edges = (n * k) // 2
        
        # To avoid getting stuck, we prioritize vertices with low degree
        # and randomize the selection of partners
        
        attempts = 0
        max_attempts = 100000 # Hard cap for this n
        
        while total_edges > 0:
            if time.time() > deadline:
                return None
            
            # Pick u with min degree
            # Optimization: keep a list of vertices with deg < k
            # For simplicity, just scan. n is small (< 2000)
            candidates_u = [i for i in range(n) if deg[i] < k]
            if not candidates_u:
                break
                
            u = random.choice(candidates_u)
            
            # Pick v with deg < k, v != u
            candidates_v = [i for i in range(n) if deg[i] < k and i != u]
            if not candidates_v:
                # No valid partner, stuck
                return None
                
            v = random.choice(candidates_v)
            
            # Check if edge exists
            if v in adj[u]:
                continue
            
            # Check girth
            if creates_short_cycle(u, v, adj):
                continue
            
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            total_edges -= 1
            attempts = 0
        else:
            # If loop finishes without break, check if all degrees are k
            if all(d == k for d in deg):
                # Verify girth one last time (optional, but good for safety)
                # The greedy check should have ensured it
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            return None
            
        return None

    # Try different n values
    # Moore bound for k=3, g=16 is 381.
    # We try n slightly above Moore bound, up to known cage size.
    n_candidates = [400, 500, 600, 700, 800, 936, 1000, 1200, 1500, 2000]
    
    # Sort by n to try smallest first
    # But 400 might be impossible. 936 is known possible.
    # We try a mix.
    
    # First, try to find a valid graph for a few small n.
    # If we find one, it's a huge win.
    
    for n in n_candidates:
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this n
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            res = try_construct(n)
            if res:
                return res
                
    # If all failed, fallback to seed logic?
    # The seed logic is complex to replicate exactly without the code.
    # But we can implement a simple L-U-W or just return a large random graph 
    # that is likely valid? No, must be valid.
    
    # If we can't find a small one, we can try to construct a L-U-W graph 
    # with smaller parameters if possible, or just use the seed's output 
    # if we can't beat it.
    
    # However, the seed's output is ~354k. 
    # If we fail to find < 354k, we return the seed's result?
    # We don't have the seed's result.
    # We must return *something* valid.
    
    # Fallback: Construct a L-U-W graph with q=3, d=11 (girth ~ 16)
    # This is what the seed does.
    # We can just copy the seed's logic here as a fallback.
    
    def seed_fallback(k, g):
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

    return seed_fallback(k, g)
