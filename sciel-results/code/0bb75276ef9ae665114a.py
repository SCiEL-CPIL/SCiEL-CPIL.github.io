import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback: LUW Construction (current best)
    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # Try to find a smaller circulant graph
    # Moore bound for k=4, g=10 is 242
    if k == 4 and g == 10:
        # Search for circulant C_N(s1, s2)
        # Girth check: BFS from 0
        def check_girth(N, s1, s2):
            # Adjacency list for 0's neighbors
            # We just need to ensure no cycle < 10 passes through 0
            # BFS from 0
            visited = {0: 0}
            queue = [(0, 0)] # node, depth
            while queue:
                node, depth = queue.pop(0)
                if depth >= 5:
                    continue
                for s in [s1, -s1, s2, -s2]:
                    neighbor = (node + s) % N
                    if neighbor == 0:
                        # Cycle found of length depth + 1
                        if depth + 1 < g:
                            return False
                    elif neighbor in visited:
                        # Cycle found of length depth + 1 + visited[neighbor]
                        if depth + 1 + visited[neighbor] < g:
                            return False
                    else:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))
            return True

        # Search N from Moore bound up
        moore_bound = 242
        # We don't want to search too many N values if it's slow
        # Randomize s1, s2 for each N?
        # Or just try a few specific Ns?
        # Let's try a random search over N in [242, 500]
        random.seed(42)
        
        best_n, best_edges = None, None
        
        # Try a few N values
        for N in range(moore_bound, 600):
            if time.time() > deadline:
                break
            
            # Try a limited number of random (s1, s2) pairs
            # Number of pairs is ~ N^2/4. We can't try all.
            # Try 100 random pairs
            for _ in range(100):
                if time.time() > deadline:
                    break
                s1 = random.randint(1, N//2 - 1)
                s2 = random.randint(1, N//2 - 1)
                if s1 == s2:
                    continue
                if s1 + s2 == N: # s1 = -s2
                    continue
                # Check validity
                if check_girth(N, s1, s2):
                    # Construct edges
                    edges = []
                    for i in range(N):
                        for s in [s1, s2]:
                            edges.append((i, (i + s) % N))
                    return N, edges
        
    # Fallback to LUW
    return luw_construct()
