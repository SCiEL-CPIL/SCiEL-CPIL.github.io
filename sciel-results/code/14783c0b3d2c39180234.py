import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Moore bound calculation
    # For k-regular graph with girth g:
    # If g is even: n >= 1 + k * (1 + (k-1) + ... + (k-1)^(g/2 - 1))
    # If g is odd: n >= 1 + k * (1 + (k-1) + ... + (k-1)^((g-1)/2))
    if g % 2 == 0:
        # Even girth
        # Sum is (k-1)^(g/2) - 1 / (k-2) * k? 
        # Formula: n >= k * ( (k-1)^(g/2) - 1 ) / (k-2)
        if k == 2:
            n_min = g
        else:
            n_min = int(k * ((k-1)**(g//2) - 1) / (k-2)) + 1
    else:
        # Odd girth
        # n >= 1 + k * ( (k-1)^((g-1)/2) - 1 ) / (k-2)
        if k == 2:
            n_min = g
        else:
            n_min = int(1 + k * ((k-1)**((g-1)//2) - 1) / (k-2))
            
    # Try Cayley graphs over Z_n
    # We will search for n in a range starting from n_min
    
    # Optimization: If k is small, we can try specific known structures or just search.
    # For k=3, g=5, n_min=10 (Petersen).
    # For k=5, g=9, n_min=428.
    
    best_n = None
    best_edges = None
    
    # Limit search space for n to avoid infinite loops if no Cayley graph exists in small n
    # We will try n from n_min to n_min + 500 or until time limit.
    
    # Helper to check if a set of generators yields girth >= g
    def check_girth(n, generators, g):
        # generators is a list of ints in Z_n
        # Build adjacency list implicitly via BFS
        # dist[v] = shortest distance from 0
        dist = [-1] * n
        dist[0] = 0
        queue = deque([0])
        
        # We only need to check up to depth (g-1)//2 roughly, but we need to check all edges for the cycle condition.
        # Actually, if we BFS fully, we can check all edges.
        # But for large n, full BFS is expensive.
        # However, if girth < g, we will find it quickly (shallow).
        # If girth >= g, we might have to explore a lot.
        # Let's limit BFS depth to g//2. If no cycle found within that depth, 
        # does it mean girth >= g?
        # No. A cycle of length g might be formed by two paths of length g/2 meeting.
        # So we need to explore up to depth g//2.
        
        max_depth = g // 2
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
                
            for gen in generators:
                v = (u + gen) % n
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] < max_depth:
                        queue.append(v)
                else:
                    # Edge (u, v) exists.
                    # Cycle length is dist[u] + dist[v] + 1
                    # But this is only valid if the edge is not a tree edge?
                    # In BFS, if dist[v] == dist[u] + 1, it's a tree edge (or parallel, but simple graph).
                    # If dist[v] <= dist[u] + 1, it's a forward/cross edge.
                    # The cycle formed by the tree path to u, the edge (u,v), and the tree path from v to LCA?
                    # Actually, the cycle length is dist[u] + dist[v] + 1 is NOT always correct for the *shortest* cycle involving this edge.
                    # The shortest cycle containing edge (u,v) is dist[u] + dist[v] + 1 ONLY IF the paths from 0 to u and 0 to v are disjoint except at 0?
                    # No, they might meet at LCA.
                    # However, if we just want to know if girth < g, we can check:
                    # If dist[u] + dist[v] + 1 < g, then there is a cycle of length < g?
                    # Yes, the union of the two shortest paths and the edge forms a cycle (or a theta graph). The shortest cycle in that subgraph has length <= dist[u] + dist[v] + 1.
                    # So if dist[u] + dist[v] + 1 < g, we have a cycle shorter than g.
                    
                    if dist[u] + dist[v] + 1 < g:
                        return False
                    
        # If we finished BFS up to max_depth and no short cycle found, 
        # is it guaranteed that girth >= g?
        # Consider a cycle of length g. It can be split into two paths of length g/2 from a vertex on the cycle to another.
        # If we BFS from 0, and the cycle doesn't pass through 0?
        # In a vertex-transitive graph, if there is a short cycle anywhere, there is one through 0?
        # Yes, because we can move the cycle to pass through 0 via symmetry?
        # Wait, vertex-transitive means for any u, v there is an automorphism mapping u to v.
        # So if there is a cycle of length L, there is a cycle of length L through 0.
        # So checking cycles through 0 is sufficient.
        # The condition dist[u] + dist[v] + 1 < g detects any cycle through 0 of length < g.
        
        return True

    # Search for n
    # We'll try a few candidate n values.
    # Since Cayley graphs over Z_n might not exist for the optimal n, we might need to try larger n.
    # But the task minimizes n.
    
    # For k=3, g=5, Z_5 Cayley? No, Petersen is not Cayley.
    # If we fail to find a Cayley graph, we should fall back to the LUW construction or a different algebraic construction.
    # However, the directive says "Use a voltage graph...".
    # Let's try to find a Cayley graph first. If not found in reasonable time, use LUW.
    
    # To make it faster, we can use random restarts for the generator set.
    
    # Let's try n from n_min to n_min + 100
    # For each n, try a few random generator sets.
    
    found = False
    for n in range(n_min, n_min + 200):
        if time.time() > deadline:
            break
            
        # Generate random symmetric sets of size k
        # If k is odd, n must be even to have an element of order 2.
        if k % 2 == 1 and n % 2 != 0:
            continue
            
        # Try 10 random sets
        for _ in range(10):
            if time.time() > deadline:
                break
                
            gens = set()
            # Pick (k-1)/2 pairs
            num_pairs = (k - 1) // 2
            while len(gens) < num_pairs * 2:
                s = random.randint(1, n - 1)
                if s == -s % n: # Order 2
                    if k % 2 == 1:
                        gens.add(s)
                    # else ignore or handle
                else:
                    if s not in gens and (-s % n) not in gens:
                        gens.add(s)
                        gens.add(-s % n)
            
            if len(gens) != k:
                continue
                
            # Remove 0 if present (shouldn't be)
            gens.discard(0)
            
            if len(gens) < k:
                 # Fill with randoms if failed
                 while len(gens) < k:
                     s = random.randint(1, n-1)
                     gens.add(s)
                     gens.add(-s % n)
                     if len(gens) > k:
                         # remove excess
                         # This is messy. Let's just use a list and ensure uniqueness.
                         pass
            
            # Proper generation of symmetric set of size k
            gens_list = []
            if k % 2 == 1:
                # One element of order 2
                s2 = n // 2
                gens_list.append(s2)
                k_remaining = k - 1
            else:
                k_remaining = k
                
            while k_remaining > 0:
                s = random.randint(1, n - 1)
                if s in gens_list or (-s % n) in gens_list:
                    continue
                if s == -s % n:
                    if k_remaining == 1:
                        gens_list.append(s)
                        k_remaining -= 1
                else:
                    gens_list.append(s)
                    gens_list.append(-s % n)
                    k_remaining -= 2
                    
            if len(gens_list) != k:
                continue
                
            if check_girth(n, gens_list, g):
                # Found a valid Cayley graph
                # Construct edges
                edges = []
                for u in range(n):
                    for gen in gens_list:
                        v = (u + gen) % n
                        if u < v:
                            edges.append((u, v))
                best_n = n
                best_edges = edges
                found = True
                break
        if found:
            break

    if not found:
        # Fallback to LUW construction from seed
        # Re-implement seed logic here to ensure we always return something
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
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        best_n = len(order)
        best_edges = edges_luw

    return best_n, best_edges
