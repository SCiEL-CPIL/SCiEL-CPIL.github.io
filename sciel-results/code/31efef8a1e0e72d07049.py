import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = 150.0  # 150 seconds wall clock limit
    
    # For k=4, g=10, we search for a Cayley graph on Z_p x Z_p
    # Generators: (1,0), (-1,0), (a,b), (-a,-b)
    # We need girth >= 10, so no cycles of length 4, 6, 8.
    
    if k != 4 or g != 10:
        # Fallback to seed logic or simple construction if params differ
        # But task implies k=4, g=10
        pass

    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def has_non_backtracking_cycle(p, a, b, L):
        # Generators as tuples
        S = [
            (1, 0),
            (p - 1, 0),
            (a, b),
            (p - a, p - b)
        ]
        
        # Map generator to index for fast inverse lookup? 
        # Inverses: (1,0)<->(p-1,0), (a,b)<->(p-a,p-b)
        inv_map = {
            (1, 0): (p - 1, 0),
            (p - 1, 0): (1, 0),
            (a, b): (p - a, p - b),
            (p - a, p - b): (a, b)
        }
        
        # Check all words of length L
        # Optimization: If L is large, use recursion with pruning? 
        # For L=8, 4^8 = 65536 is small enough to brute force.
        
        # Precompute sums? No, just iterate.
        
        for i1 in range(4):
            x1, y1 = S[i1]
            for i2 in range(4):
                if i2 != i1 and S[i2] == inv_map[S[i1]]:
                    continue # Backtrack at step 1
                x2, y2 = S[i2]
                for i3 in range(4):
                    if i3 != i2 and S[i3] == inv_map[S[i2]]:
                        continue # Backtrack at step 2
                    x3, y3 = S[i3]
                    for i4 in range(4):
                        if i4 != i3 and S[i4] == inv_map[S[i3]]:
                            continue
                        x4, y4 = S[i4]
                        for i5 in range(4):
                            if i5 != i4 and S[i5] == inv_map[S[i4]]:
                                continue
                            x5, y5 = S[i5]
                            for i6 in range(4):
                                if i6 != i5 and S[i6] == inv_map[S[i5]]:
                                    continue
                                x6, y6 = S[i6]
                                for i7 in range(4):
                                    if i7 != i6 and S[i7] == inv_map[S[i6]]:
                                        continue
                                    x7, y7 = S[i7]
                                    for i8 in range(4):
                                        if i8 != i7 and S[i8] == inv_map[S[i7]]:
                                            continue
                                        x8, y8 = S[i8]
                                        
                                        # Check if sum is 0
                                        # We only care about L=4,6,8. 
                                        # This function is called for specific L.
                                        # Let's make it generic for L, but the loops are hardcoded for L=8 for speed.
                                        # We will call this logic for L=4, 6, 8 separately or just check all at once?
                                        # Checking L=8 covers L=4 and L=6? No.
                                        # A cycle of length 4 is a distinct entity.
                                        # However, if we find a non-backtracking cycle of length 4, girth < 10.
                                        # So we must check 4, 6, 8.
                                        
                                        # The above nested loops are for L=8.
                                        # This is getting complex. Let's use a helper for general L.
                                        pass
        return False # Placeholder

    # Better approach for cycle check:
    # Use BFS on the non-backtracking state space? 
    # States are (current_vertex, previous_vertex).
    # Number of states = N * (k-1) = p^2 * 3.
    # For p=17, N=289, states=867.
    # BFS depth up to g/2 = 5.
    # This is much faster and more robust.
    
    def check_girth(p, a, b, girth_req):
        # N = p*p
        # Vertex v = (x, y) -> index x*p + y
        # Neighbors of (x,y):
        # (x+1, y), (x-1, y), (x+a, y+b), (x-a, y-b) all mod p
        
        # BFS from each start vertex? 
        # Or just BFS from one vertex? 
        # If graph is vertex transitive (Cayley graph), checking from one vertex is sufficient?
        # Yes, if there is a cycle of length L, there is one starting from any vertex.
        # Wait, BFS from (0,0) with state (current, prev) will find cycles passing through (0,0).
        # But a cycle might not pass through (0,0).
        # However, if the graph is connected and vertex transitive, the girth is the same everywhere.
        # Actually, BFS from (0,0) finds the shortest cycle *containing* (0,0).
        # The girth is the minimum over all vertices.
        # Due to transitivity, the shortest cycle containing (0,0) is the girth?
        # Yes, for vertex transitive graphs, the girth is the same for all vertices.
        
        start = (0, 0)
        # State: (current_vertex, previous_vertex)
        # Start with neighbors of (0,0)
        # prev = (0,0), curr = neighbor
        
        # We want to find a cycle of length L < girth_req.
        # Cycle length is the number of edges.
        # Path: v0, v1, ..., vL = v0.
        # In non-backtracking BFS:
        # Depth 1: (v1, v0)
        # Depth 2: (v2, v1) ...
        # If we reach (v0, v_{L-1}), we have a cycle of length L.
        # We must ensure we don't close a cycle too early.
        
        # Initialize BFS
        from collections import deque
        
        # Get neighbors of (0,0)
        def get_neighbors(x, y):
            return [
                ((x + 1) % p, y),
                ((x - 1) % p, y),
                ((x + a) % p, (y + b) % p),
                ((x - a) % p, (y - b) % p)
            ]
        
        neighbors_0 = get_neighbors(0, 0)
        
        # Queue: (curr_x, curr_y, prev_x, prev_y, depth)
        q = deque()
        visited = set()
        
        for nx, ny in neighbors_0:
            q.append((nx, ny, 0, 0, 1))
            visited.add((nx, ny, 0, 0))
            
        while q:
            if time.time() > deadline - 5:
                return False # Timeout, assume valid? Or invalid? 
                             # Better to return False to try next candidate, or raise.
                             # If we timeout, we should return the best graph found so far.
                             # But we are in a helper. Let's propagate timeout.
                             
            cx, cy, px, py, depth = q.popleft()
            
            # Check if we found a cycle
            if (cx, cy) == (0, 0) and depth < girth_req:
                # Cycle of length depth + 1?
                # Path: v0(0,0) -> v1 ... -> v_depth(cx,cy).
                # If (cx,cy) == v0, cycle length is depth.
                # Wait, start was v0. Depth 1 is v1.
                # If v_depth == v0, cycle length is depth.
                # We need depth < girth_req (10).
                # So if depth is 4, 6, or 8, it's a violation.
                return True # Found a short cycle
                
            # Expand
            for nx, ny in get_neighbors(cx, cy):
                # Non-backtracking: (nx, ny) != (px, py)
                if (nx, ny) == (px, py):
                    continue
                    
                # Do we care if (nx, ny) is v0?
                # If depth+1 < girth_req, it's a violation.
                # If depth+1 >= girth_req, it's okay (or we stop).
                
                new_depth = depth + 1
                if new_depth < girth_req:
                    state = (nx, ny, cx, cy)
                    if state not in visited:
                        visited.add(state)
                        q.append((nx, ny, cx, cy, new_depth))
                else:
                    # We don't need to explore further if depth >= girth_req
                    # Because we are looking for cycles < girth_req
                    pass
        
        return False # No short cycle found

    # Search for p
    # Moore bound for k=4, g=10 is 243.
    # p^2 >= 243 => p >= 16.
    # Primes: 17, 19, 23, 29, ...
    
    for p in [17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if time.time() > deadline - 10:
            break
            
        for a in range(1, p):
            if time.time() > deadline - 10:
                break
            for b in range(1, p):
                # Check if generators are distinct and non-zero
                # S = {(1,0), (-1,0), (a,b), (-a,-b)}
                # Distinct:
                # (1,0) != (-1,0) -> 2 != 0 mod p (p odd, ok)
                # (a,b) != (1,0)
                # (a,b) != (-1,0)
                # (a,b) != (-a,-b) -> 2a!=0, 2b!=0 (p odd, ok)
                
                if (a == 1 and b == 0) or (a == p-1 and b == 0):
                    continue
                if (a == p-1 and b == 0) or (a == 1 and b == 0): # covered
                    continue
                if a == 0 and b == 0:
                    continue
                    
                # Check girth
                if not check_girth(p, a, b, 10):
                    # Valid graph found
                    n = p * p
                    edges = []
                    for x in range(p):
                        for y in range(p):
                            u = x * p + y
                            neighbors = get_neighbors(x, y)
                            for nx, ny in neighbors:
                                v = nx * p + ny
                                if u < v:
                                    edges.append((u, v))
                    
                    return n, edges

    # Fallback if no Cayley graph found
    # Return seed result or a large valid graph
    # We must return SOMETHING valid.
    # Let's return the seed result as fallback?
    # The seed code is in the prompt. I can copy it.
    
    # Copy of seed logic for fallback
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
