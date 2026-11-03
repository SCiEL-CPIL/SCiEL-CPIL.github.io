import time
import itertools
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Target: 4-regular, girth >= 10
    # Search for circulant graphs C_n(S)
    
    # Moore bound for (4,10) is 243.
    # n must be even.
    # Try n from 244 to 8000
    best_n = 8192 # Fallback to LUW size
    best_edges = None
    
    def check_girth_numpy(n, steps, g):
        """Fast girth check using vectorized BFS on circulant graph."""
        offsets = []
        for s in steps:
            offsets.append(s)
            offsets.append(-s)
        offsets = np.array(offsets)
        
        depth = np.full(n, -1, dtype=np.int32)
        parent = np.full(n, -1, dtype=np.int64)
        depth[0] = 0
        parent[0] = -2  # root marker
        
        queue = [0]
        head = 0
        min_cycle = float('inf')
        
        max_depth = (g - 1) // 2
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if depth[u] >= max_depth:
                continue
            
            # Vectorized neighbor computation
            neighbors = (u + offsets) % n
            
            # Get depths and parents of neighbors
            neigh_depths = depth[neighbors]
            pu = parent[u]
            
            # Check each neighbor
            for i in range(len(neighbors)):
                v = int(neighbors[i])
                if neigh_depths[i] == -1:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if pu == v:
                        continue
                    cycle_len = depth[u] + neigh_depths[i] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return min_cycle
        
        return min_cycle
    
    def check_girth_fast(n, steps, g):
        """Fast girth check using BFS from vertex 0 on circulant graph."""
        offsets = []
        for s in steps:
            offsets.append(s)
            offsets.append(-s)
        
        depth = [-1] * n
        parent = [-1] * n
        depth[0] = 0
        parent[0] = -2  # root marker
        
        queue = [0]
        head = 0
        min_cycle = float('inf')
        
        max_depth = (g - 1) // 2
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if depth[u] >= max_depth:
                continue
            
            for off in offsets:
                v = (u + off) % n
                
                if depth[v] == -1:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if parent[u] == v:
                        continue
                    cycle_len = depth[u] + depth[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return min_cycle
        
        return min_cycle
    
    def try_circulant(n, steps, g):
        """Check if circulant with given n and steps has girth >= g."""
        # Use numpy version for larger n, fast version for smaller
        if n > 500:
            girth = check_girth_numpy(n, steps, g)
        else:
            girth = check_girth_fast(n, steps, g)
        if girth >= g:
            edges = []
            for u in range(n):
                for s in steps:
                    v = (u + s) % n
                    if u < v:
                        edges.append((u, v))
            return n, edges
        return None
    
    # Iterate n
    found = False
    for n in range(244, 4000, 2):
        if time.time() > deadline:
            break
        
        # Number of random samples: more for small n, fewer for large n
        if n < 400:
            num_samples = 30000
        elif n < 800:
            num_samples = 15000
        elif n < 1600:
            num_samples = 8000
        else:
            num_samples = 4000
        
        # First try some structured step sets
        # 1. Consecutive small steps
        for base in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]:
            if time.time() > deadline:
                break
            steps = [base, base*2, base*3, base*4]
            if all(0 < s < n//2 for s in steps):
                result = try_circulant(n, steps, g)
                if result:
                    return result
        
        # 2. Steps spread evenly around n/2
        for spread in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline:
                break
            steps = [int(n * spread * i) for i in range(1, 5)]
            steps = [s for s in steps if 0 < s < n//2]
            if len(steps) == 4:
                result = try_circulant(n, steps, g)
                if result:
                    return result
        
        # 3. Fibonacci-like steps
        for a in range(1, min(20, n//10)):
            if time.time() > deadline:
                break
            for b in range(a+1, min(40, n//8)):
                if time.time() > deadline:
                    break
                steps = [a, b, a+b, 2*b]
                if all(0 < s < n//2 for s in steps) and len(set(steps)) == 4:
                    result = try_circulant(n, steps, g)
                    if result:
                        return result
        
        # 4. Quadratic steps
        for base in range(1, min(15, n//20)):
            if time.time() > deadline:
                break
            steps = [base*i*i for i in range(1, 5)]
            if all(0 < s < n//2 for s in steps) and len(set(steps)) == 4:
                result = try_circulant(n, steps, g)
                if result:
                    return result
        
        # Random sampling with bias toward well-spread steps
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            
            # Generate steps that are more spread out
            # Use rejection sampling to ensure steps are reasonably separated
            while True:
                steps = random.sample(range(1, n // 2 + 1), 4)
                # Check if steps are reasonably spread (not all clustered)
                steps_sorted = sorted(steps)
                min_gap = min(steps_sorted[i+1] - steps_sorted[i] for i in range(3))
                if min_gap >= n // 20:  # At least 5% of n/2 apart
                    break
                if random.random() < 0.3:  # Sometimes accept anyway
                    break
            
            result = try_circulant(n, steps, g)
            if result:
                return result
        
        # Also try some random steps without the spread constraint
        for _ in range(num_samples // 4):
            if time.time() > deadline:
                break
            steps = random.sample(range(1, n // 2 + 1), 4)
            result = try_circulant(n, steps, g)
            if result:
                return result
    
    # If no circulant found, return LUW (seed behavior)
    # Re-implement seed logic for fallback
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
