import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Fallback: LUW Construction
    def luw_construct(k, g):
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

    # Primary Strategy: Search for Circulant Graphs
    # This is most effective for small k and moderate girth.
    if k == 3 and g <= 16:
        min_n = 2 * ((k - 1) ** (g // 2) - 1) // (k - 2) if k > 2 else g
        min_n = max(min_n, k + 1)
        # Moore bound for k=3, g=16 is 510.
        
        # Search range
        max_n = 1200
        # If g is small, we can search smaller n
        if g <= 8:
            max_n = 100
            
        for n in range(min_n, max_n + 1):
            if time.time() > deadline - 5:
                break
                
            # Number of trials
            # Smaller n is harder, so maybe more trials?
            # But we have limited time.
            trials = 200
            
            for _ in range(trials):
                if time.time() > deadline - 2:
                    break
                
                half_n = n // 2
                if half_n < k // 2:
                    continue
                
                # Generate random symmetric set
                # For k=3, we pick 3 distinct elements from 1..n//2
                # Ensure no element is 0
                s_set = random.sample(range(1, half_n + 1), k // 2 + (1 if k % 2 else 0))
                
                # For odd k, we need one element s such that 2s != 0 mod n?
                # No, for 3-regular, we pick 3 distinct s.
                # The neighbors of i are i+s1, i-s1, i+s2, i-s2, i+s3, i-s3.
                # This gives degree 6 if all distinct.
                # Wait, k=3 means degree 3.
                # So we pick 1 element s1, and its negative -s1.
                # And another element s2?
                # No, a 3-regular circulant graph must have an odd number of vertices?
                # No, n can be even.
                # If n is even, and we pick s = n/2, it contributes 1 neighbor.
                # If we pick s != n/2, it contributes 2 neighbors.
                # To get degree 3, we must pick one s = n/2 (if n even) and one other s.
                # Or n must be odd, and we pick 3 distinct s? No, 3 s gives degree 6.
                # Wait.
                # Degree of Circ(n, S) is |S| if n is odd? No.
                # If n is odd, -s != s. So each s adds 2 neighbors. Degree must be even.
                # So for k=3, n MUST be even, and S must contain n/2.
                
                if n % 2 != 0:
                    continue
                
                # n is even.
                # We must include n/2 in S.
                # We need k-1 = 2 more neighbors.
                # So we pick 1 more distinct s from 1..n/2-1.
                # Total S = {n/2, s1}.
                # Neighbors of i: i+n/2, i-s1, i+s1.
                # This is 3 neighbors.
                
                s1 = random.randint(1, half_n - 1)
                S = [half_n, s1]
                
                # Check if S is valid (distinct, non-zero)
                if len(set(S)) != len(S):
                    continue
                
                # Build Graph
                adj = [[] for _ in range(n)]
                for i in range(n):
                    for s in S:
                        j1 = (i + s) % n
                        j2 = (i - s) % n
                        if j1 != i: # No loops
                            if j1 not in adj[i]:
                                adj[i].append(j1)
                                adj[j1].append(i)
                        if j2 != i and j2 != j1:
                            if j2 not in adj[i]:
                                adj[i].append(j2)
                                adj[j2].append(i)
                
                # Verify Degree
                if any(len(adj[i]) != k for i in range(n)):
                    continue
                
                # Check Girth
                # Fast check: 4-cycles
                has_4_cycle = False
                for i in range(n):
                    neigh = adj[i]
                    for a in range(len(neigh)):
                        for b in range(a + 1, len(neigh)):
                            if neigh[b] in adj[neigh[a]]:
                                has_4_cycle = True
                                break
                        if has_4_cycle:
                            break
                    if has_4_cycle:
                        break
                
                if has_4_cycle:
                    continue
                
                # Strict Girth Check
                # BFS from every node is too slow.
                # We'll do BFS from 0 to find shortest cycle containing 0.
                # And assume if it's high, it's good.
                # But we need to be sure.
                # Let's do a strict check:
                # For each edge (u, v), remove it, find shortest path u->v.
                # This is O(E * (V+E)).
                # E = 3n/2 = 1.5n.
                # V+E = 2.5n.
                # Total 3.75 n^2.
                # For n=1000, 3.75M ops.
                # In Python, ~0.1s.
                # We can afford this for a few candidates.
                
                girth = n + 1
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            # Remove edge (u, v)
                            # BFS from u to v
                            dist = [-1] * n
                            dist[u] = 0
                            q = [u]
                            found = False
                            while q:
                                curr = q.pop(0)
                                if curr == v:
                                    girth = min(girth, dist[curr] + 1)
                                    found = True
                                    break
                                for w in adj[curr]:
                                    if w == v and curr == u:
                                        continue # Skip removed edge
                                    if w == u and curr == v:
                                        continue # Skip removed edge
                                    if dist[w] == -1:
                                        dist[w] = dist[curr] + 1
                                        q.append(w)
                            if found:
                                if girth < g:
                                    break
                    if girth < g:
                        break
                
                if girth >= g:
                    # Found a valid graph!
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    # Fallback
    return luw_construct(k, g)
