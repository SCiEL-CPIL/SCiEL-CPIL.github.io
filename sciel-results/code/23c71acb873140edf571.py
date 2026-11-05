import random
import time
import math
import numpy as np
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    # 150s limit, leave 10s margin for overhead and return
    deadline = start_time + 140.0
    
    # Base graph: Petersen (10 vertices, 15 edges)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length 3 to g-1 from every vertex
    walks = []
    for start_v in range(n_base):
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while queue:
            curr_v, prev_e, length, path = queue.popleft()
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
    
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)
    
    # Pre-compute coefficient matrix C (K x 15)
    # Each row corresponds to a walk, entry is +1, -1, or 0
    K = len(walks)
    C = np.zeros((K, 15), dtype=np.int8)
    for i, w in enumerate(walks):
        for e_id, sign in w:
            C[i, e_id] += sign
    
    # Moore bound for n(3,16) is 512. n = 10*m. So m >= 52.
    best_n = float('inf')
    best_edges = None
    best_m = None
    best_volts = None
    
    # Search for Abelian lifts (Z_m)
    # We prioritize smaller m to minimize n.
    # We will iterate m from 52 upwards.
    # For each m, we estimate the number of tries needed.
    
    # Optimization: If K is large, small m is hard. 
    # Prob(success) ~ exp(-K/m).
    
    for m in range(52, 1000):
        if time.time() > deadline - 10.0:
            break
        
        candidate_n = 10 * m
        if candidate_n >= best_n:
            break
            
        # Estimate tries needed
        # If K/m is large, success prob is low.
        # Let's cap tries per m to avoid getting stuck on impossible small m.
        # If m is small, we try fewer times and move on? 
        # Actually, if a solution exists for m=52, we want to find it.
        # But if it doesn't, we shouldn't waste 140s on it.
        
        # Heuristic: 
        # If m < 100, try up to 5000 times (fast with NumPy)
        # If m >= 100, try up to 2000 times (likely to succeed)
        
        time_left = deadline - time.time()
        # Assume each trial takes ~10-50us with NumPy (K is small, ~100-1000)
        # Let's measure roughly: K is number of closed walks < 16 in Petersen.
        # Petersen has 10 vertices. 
        # Roughly, number of walks grows exponentially. 
        # Let's assume K ~ 1000. 
        # C @ v is 1000x15 matvec. Very fast.
        
        max_tries = 5000
        if m > 100:
            max_tries = 2000
            
        # If time is short, reduce tries
        if time_left < 10.0:
            max_tries = min(max_tries, 500)
            
        found = False
        for trial in range(max_tries):
            if time.time() > deadline - 5.0:
                break
                
            # Random voltages 1 to m-1
            # Using numpy for speed
            volts = np.random.randint(1, m, size=15)
            
            # Check: C @ volts % m != 0 for all rows
            # We want to ensure NO row sums to 0 mod m.
            # sums = C @ volts
            # if any(sums % m == 0): fail
            
            sums = C @ volts
            # Check if any sum is divisible by m
            # Note: sums can be negative? No, C has -1, 0, 1. 
            # Wait, C @ volts can be negative.
            # In Python/NumPy, negative % m is positive.
            # So (sums % m) == 0 is the correct check.
            
            # Vectorized check
            if not np.any(sums % m == 0):
                found = True
                best_m = m
                best_n = candidate_n
                best_volts = volts
                break
        
        if found:
            # Construct edges
            best_edges = []
            for i, (u, v) in enumerate(edges):
                vol = best_volts[i]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + vol) % m
                    best_edges.append((u_lift, v_lift))
            break
            
    if best_edges is None:
        # Fallback to LUW if Abelian search failed
        return _luw_fallback(k, g)
    
    return best_n, best_edges

def _luw_fallback(k, g):
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
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
