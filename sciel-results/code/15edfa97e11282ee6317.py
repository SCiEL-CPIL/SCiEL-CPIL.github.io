import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Moore bound for k=5, g=9 is 428
    # n must be even for 5-regular circulant (since 5 is odd, n must be even for the graph to be 5-regular? 
    # Actually, sum of degrees = 5n. 5n must be even => n must be even.)
    
    best_n = 1250
    best_edges = None
    
    # 1. Try to find a Circulant Graph
    # n ranges from moore_bound to best_n - 1
    # We prioritize smaller n
    
    # Fast girth check for circulant
    def check_girth_circulant(n, steps, girth_limit):
        dist = [-1] * n
        dist[0] = 0
        queue = [0]
        head = 0
        min_cycle = float('inf')
        
        while head < len(queue):
            u = queue[head]
            head += 1
            du = dist[u]
            
            # If we've gone deeper than (girth_limit-1)//2, any new cycle found will be 
            # dist[u] + dist[v] + 1.
            # Since dist[v] >= dist[u]-1, cycle >= 2*dist[u].
            # If 2*dist[u] >= girth_limit, we can stop?
            # No, we might find a shorter cycle from a different branch?
            # But BFS explores in increasing distance.
            # If du > (girth_limit-1)//2, then any cycle found involving u and a visited node v
            # has length du + dist[v] + 1.
            # The minimum dist[v] is 0 (if v=0). Length = du+1.
            # If du+1 >= girth_limit, we are safe?
            # Actually, if we find a back edge to a node with dist >= 0, 
            # the cycle length is du + dist[v] + 1.
            # The smallest possible cycle length found at depth du is du + 0 + 1 = du+1.
            # So if du + 1 >= girth_limit, any cycle found from this node or deeper 
            # will be >= girth_limit.
            # We can stop expanding this branch? 
            # But we still need to check other branches?
            # No, if the current depth du satisfies du+1 >= girth_limit, 
            # then all nodes at depth du or deeper will only form cycles >= girth_limit.
            # So we can break out of the loop if the minimum depth in the queue is too high.
            
            if du + 1 >= girth_limit:
                # All remaining nodes in queue have dist >= du
                # So any cycle will be >= girth_limit
                break

            for s in steps:
                v = (u + s) % n
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                else:
                    # Found a cycle
                    # Skip if v is the parent? 
                    # In BFS, we don't track parent explicitly, but dist[v] == du - 1 implies tree edge.
                    # However, in undirected graph, we see the parent.
                    # Cycle length = du + dist[v] + 1.
                    # If dist[v] == du - 1, cycle = 2*du.
                    # But this is the tree edge. We should not count it as a "new" cycle?
                    # Actually, the tree edge itself doesn't form a cycle.
                    # A cycle is formed when we see a visited node that is NOT the parent.
                    # But in a simple graph, the only visited node with dist du-1 is the parent.
                    # Any other visited node has dist <= du.
                    # If dist[v] < du - 1, it's a back edge to an ancestor.
                    # If dist[v] == du, it's a cross edge.
                    # If dist[v] == du + 1, it's a forward edge (already counted when v was processed).
                    
                    # To be safe, we just check if cycle_len < girth_limit.
                    cycle_len = du + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < girth_limit:
                        return False
        
        return min_cycle >= girth_limit

    # Search for Circulants
    # We will search n from 430 to 1248
    # To save time, we sample steps.
    
    # We can use a deterministic set of "good" steps first.
    # s1 = n//5, s2 = n//3
    
    for n in range(430, best_n, 2):
        if time.time() > deadline:
            break
            
        # Candidate steps
        candidates = []
        # Structured
        candidates.append((n//5, n//3))
        candidates.append((n//4, n//3))
        candidates.append((n//6, n//4))
        candidates.append((n//7, n//5))
        
        # Random
        # Limit random attempts based on remaining time
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
            
        # Estimate time per check
        # Check is O(n). n=1000 takes ~0.001s?
        # Let's try 20 random attempts per n
        for _ in range(20):
            s1 = random.randint(1, n//2 - 1)
            s2 = random.randint(1, n//2 - 1)
            if s1 != s2 and s1 + s2 != n:
                candidates.append((s1, s2))
                
        for s1, s2 in candidates:
            if time.time() > deadline:
                break
            if s1 == s2: continue
            if s1 == 0 or s2 == 0: continue
            if s1 == n//2 or s2 == n//2: continue
            if s1 + s2 == n: continue
            
            steps = [n//2, s1, -s1, s2, -s2]
            
            if check_girth_circulant(n, steps, g):
                edges = []
                for i in range(n):
                    for s in steps:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # 2. Fallback: LUW Graph
    # Re-implement LUW for k=5, g=9
    # q = 5, d = 4
    return construct_luw(5, 9)

def construct_luw(k, g):
    q = k
    d = 4 # For g=9, d=4 gives girth 10?
    # LUW(5, 4) has 1250 vertices.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
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
