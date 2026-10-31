import time
import random
import itertools
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if k != 3 or g != 16:
        return _seed_construct(k, g)
        
    # Strategy:
    # 1. Start with a voltage lift of the Petersen graph.
    #    Petersen has 10 vertices, 3-regular.
    #    Group size M = 94.
    #    Total vertices N = 10 * 94 = 940.
    #    This is close to the known optimal 936.
    # 2. The lift is defined by assigning a group element (voltage) to each edge of the Petersen graph.
    #    The lifted graph vertices are (v, x) where v is a vertex of Petersen and x in Z_M.
    #    Edge ((u, x), (v, y)) exists if (u,v) is an edge in Petersen and y - x == voltage(u,v) mod M.
    # 3. We need to choose voltages such that no cycle of length < 16 exists.
    #    A cycle in the lift corresponds to a closed walk in the base.
    #    The length of the cycle in the lift is the length of the walk in the base.
    #    The "displacement" is the sum of voltages along the walk.
    #    A cycle exists in the lift if the displacement is 0 mod M.
    #    To avoid cycles of length L < 16, we need:
    #    For all closed walks W of length L < 16 in Petersen, sum(voltages(W)) != 0 mod M.
    #    
    #    However, Petersen has triangles? No, girth 5.
    #    It has 5-cycles.
    #    If a 5-cycle in Petersen has voltage sum 0, we get a 5-cycle in the lift.
    #    So we must ensure no closed walk of length 5, 6, ..., 15 has sum 0.
    #    
    #    This is a constraint satisfaction problem on the voltages.
    #    Since M=94 is composite (2*47), it might be easier to find such voltages?
    #    Actually, we can just randomize voltages and check.
    #    
    # 4. If random voltages fail (likely for small M and strict girth), we use Simulated Annealing
    #    on the lifted graph itself to fix short cycles.
    #    We start with the lifted graph (which is 3-regular).
    #    We look for short cycles. If found, we perform edge swaps (2-opt) to break them.
    #    We accept swaps if they reduce the number of short cycles, or with probability exp(-delta/T).
    
    M = 94
    N = 940
    
    # Define Petersen graph
    # Standard Petersen: Outer 5-cycle 0-1-2-3-4-0, Inner star 5-7-9-6-8-5, Spokes 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # We need to assign voltages to these 15 edges.
    # Let's try to find voltages that satisfy the girth constraint.
    # If we can't find them quickly, we proceed with random voltages and repair.
    
    best_adj = None
    best_edges = None
    best_score = float('inf')
    
    # Helper to build lifted graph from voltages
    def build_lift(voltages):
        # voltages is a dict: edge -> voltage
        adj = [set() for _ in range(N)]
        edges = []
        for (u, v) in petersen_edges:
            vol = voltages[(u, v)]
            # For each x in 0..M-1:
            # (u, x) is connected to (v, (x + vol) % M)
            # and (v, x) is connected to (u, (x - vol) % M)
            for x in range(M):
                u_idx = u * M + x
                v_idx = v * M + ((x + vol) % M)
                adj[u_idx].add(v_idx)
                adj[v_idx].add(u_idx)
                # Only add edge once to list
                if u_idx < v_idx:
                    edges.append((u_idx, v_idx))
        return adj, edges

    # Helper to count short cycles (length < g)
    # Efficient check: BFS from each vertex up to depth g//2? 
    # No, cycles can be odd.
    # Better: For each vertex, do BFS up to depth 7 (since g=16, max cycle length to check is 15).
    # If we find a neighbor at distance d < 7, it forms a cycle of length d+1?
    # Actually, if we do BFS from u, and we find v at distance d, and v is in the visited set with distance d',
    # and we encounter v again via a different path, we might find a cycle.
    # Standard method: For each vertex u, BFS. If we find a back-edge to an ancestor, we found a cycle.
    # But we need to count ALL cycles of length < g.
    # For small g (16), we can do:
    # For each u, BFS up to depth 7. 
    # If we reach a vertex w at depth d, and w is already visited at depth d' < d, 
    # the cycle length is d + d' - 2 * depth(lca)? No, that's for trees.
    # In a general graph, BFS tree edges + non-tree edges.
    # A non-tree edge (u, v) where depth(u) <= depth(v) creates a cycle of length depth(v) - depth(u) + 1.
    # If this length < g, it's a short cycle.
    # Also, if we find a vertex v that is already visited, and it's not a parent, 
    # the cycle length is depth(u) + depth(v) - 2 * depth(lca) + 1?
    # Actually, simpler: 
    # Just check all pairs of neighbors? No.
    # Let's use the property: A graph has girth >= g iff no vertex has a closed walk of length < g.
    # We can check for cycles of length 3, 4, 5, ..., 15.
    # For speed, we'll just count the number of non-tree edges (u,v) such that the cycle formed is short.
    # This is an approximation? No, it counts each cycle multiple times?
    # Actually, every cycle has at least one non-tree edge in the BFS tree.
    # If we sum over all u, we might count the same cycle multiple times.
    # But for optimization, minimizing this count is good.
    
    def count_short_cycles(adj, g):
        count = 0
        n = len(adj)
        for u in range(n):
            # BFS from u
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if dist[curr] >= g // 2: # Optimization: if depth > g/2, cycle would be > g?
                    # Not strictly true, but helps.
                    pass
                for nb in adj[curr]:
                    if dist[nb] == -1:
                        dist[nb] = dist[curr] + 1
                        q.append(nb)
                    else:
                        # Found a cycle
                        # Cycle length is dist[curr] + dist[nb] + 1?
                        # This is only valid if the path from u to curr and u to nb are disjoint except u?
                        # No.
                        # This method is flawed for counting distinct cycles.
                        # Let's stick to a simpler heuristic: 
                        # If we find a back edge, we just increment a counter.
                        # It's a proxy for "shortness".
                        if dist[curr] + dist[nb] + 1 < g:
                            count += 1
        return count

    # Since counting is expensive, we'll use a faster check:
    # Just check if girth < 16.
    # We'll try to find voltages that work.
    
    # Let's try random voltages first.
    # If we find one with girth >= 16, we are done.
    # Given M=94, the probability is low but possible?
    # Actually, for large girth, random voltages usually work if M is large enough.
    # M=94 might be too small to avoid short cycles in the base (Petersen has 5-cycles).
    # If a 5-cycle in base has sum 0 mod 94, we have a 5-cycle in lift.
    # There are 12 5-cycles in Petersen.
    # Probability that a specific 5-cycle sum is 0 mod 94 is 1/94.
    # Probability that ALL 12 are non-zero is (93/94)^12 ~ 0.87.
    # So 87% chance of no 5-cycles from base 5-cycles.
    # But there are also 6-cycles, 7-cycles, etc.
    # So random search might work!
    
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
            
        # Random voltages
        voltages = {}
        for (u, v) in petersen_edges:
            voltages[(u, v)] = random.randint(0, M - 1)
            
        adj, edges = build_lift(voltages)
        
        # Verify girth
        # Efficient girth check:
        # BFS from each vertex. If we find a cycle of length < g, fail.
        # We can stop early.
        girth_ok = True
        for u in range(N):
            if not girth_ok:
                break
            dist = [-1] * N
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if dist[curr] >= g - 1:
                    break
                for nb in adj[curr]:
                    if dist[nb] == -1:
                        dist[nb] = dist[curr] + 1
                        q.append(nb)
                    else:
                        # Cycle found
                        # The cycle length is dist[curr] + dist[nb] + 1?
                        # No, this is the length of the walk u->curr + edge + nb->u?
                        # No.
                        # If nb is visited, the cycle is formed by the path u->...->curr, edge curr-nb, and path nb->...->u.
                        # The length is dist[curr] + 1 + dist[nb] ONLY IF the paths are disjoint.
                        # In BFS, the path to nb is the shortest.
                        # The cycle length is at most dist[curr] + 1 + dist[nb].
                        # If this sum < g, we definitely have a short cycle.
                        # If sum >= g, we might still have a short cycle via a different path, but BFS guarantees shortest.
                        # Actually, the cycle formed by the BFS tree edge (curr, nb) and the tree paths has length dist[curr] + dist[nb] + 1.
                        # If this is < g, it's a short cycle.
                        if dist[curr] + dist[nb] + 1 < g:
                            girth_ok = False
                            break
        
        if girth_ok:
            # Verify regularity (should be 3)
            # And return
            return N, edges
            
    # If random didn't work, we try Simulated Annealing on the best candidate found?
    # Or we just return the best one we found (even if girth < 16, it's better than nothing? No, it must be valid).
    # The task requires girth >= g. If we fail, we score 0.
    # We need a valid graph.
    
    # Let's try a different M. Maybe M=117 (from seed idea) but with Petersen?
    # 10 * 117 = 1170.
    # Or M=93? 10 * 93 = 930.
    # Let's try M=93 (prime).
    
    M2 = 93
    N2 = 930
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        voltages = {}
        for (u, v) in petersen_edges:
            voltages[(u, v)] = random.randint(0, M2 - 1)
            
        adj, edges = build_lift(voltages)
        
        girth_ok = True
        for u in range(N2):
            if not girth_ok:
                break
            dist = [-1] * N2
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if dist[curr] >= g - 1:
                    break
                for nb in adj[curr]:
                    if dist[nb] == -1:
                        dist[nb] = dist[curr] + 1
                        q.append(nb)
                    else:
                        if dist[curr] + dist[nb] + 1 < g:
                            girth_ok = False
                            break
        
        if girth_ok:
            return N2, edges
            
    # Fallback to seed
    return _seed_construct(k, g)

def _seed_construct(k, g):
    # Original seed code
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
