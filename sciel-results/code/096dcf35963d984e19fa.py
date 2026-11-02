import time
import sys
import math
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore bound for (k, g)
    # For even g: n >= 2k * ((k-1)^(g/2 - 1) - 1) / (k-2)
    if g % 2 == 0:
        moore_bound = 2 * k * ((k - 1) ** (g // 2 - 1) - 1) // (k - 2)
    else:
        # Moore bound for odd girth
        # n >= 1 + k * ((k-1)^((g-1)/2 - 1) * (k-1) + 1) / (k-2)
        # Simplified for k=4, g=10 (even)
        moore_bound = 1 + k * ((k - 1) ** ((g - 1) // 2 - 1) * (k - 1) + 1) // (k - 2) if k > 2 else g
    
    # For k=4, g=10, moore_bound = 320.
    
    # Strategy: Search for a circulant graph of size n >= moore_bound
    # A circulant graph on n vertices is defined by a set of offsets S.
    # For k=4, |S|=2, S={a, b}.
    # Edges: (i, i+a), (i, i-a), (i, i+b), (i, i-b) mod n.
    
    # Precompute non-backtracking walk coefficients for girth check
    # Steps: 1 (a), -1 (-a), 2 (b), -2 (-b)
    # We want to ensure no non-backtracking walk of length L < g closes (sums to 0 mod n)
    # State: (last_step, c1, c2)
    # c1: coeff of a, c2: coeff of b
    
    bad_coeffs = set()
    if g > 3:
        # Initialize length 1 states
        # (last_step, c1, c2)
        states = {
            (1, 1, 0),
            (-1, -1, 0),
            (2, 0, 1),
            (-2, 0, -1)
        }
        
        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in [1, -1, 2, -2]:
                    if step == -last_step:
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else -1 if step == -2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            # If L >= 3, a closed walk of this length implies a cycle of length <= L
            # We add all (c1, c2) to the bad set
            if L >= 3:
                for _, c1, c2 in states:
                    bad_coeffs.add((c1, c2))
    
    # Search for n starting from Moore bound
    # We try a limited range to stay within time limits
    max_n_search = moore_bound + 100
    
    for n in range(moore_bound, max_n_search + 1):
        if time.time() > deadline - 5:
            break
            
        # Iterate over pairs (a, b)
        # a in 1..n//2 - 1, b in a+1..n//2
        # Pruning: gcd(a, b, n) == 1 ensures the graph is connected (and thus a single component)
        # If not connected, it's a disjoint union, which is valid for "simple graph" but 
        # usually cage problems imply connected. The Moore bound assumes connected.
        # Also, a != b and a != -b mod n (handled by a < b <= n/2)
        
        for a in range(1, n // 2):
            if time.time() > deadline - 2:
                break
            for b in range(a + 1, n // 2 + 1):
                # Check connectivity
                if math.gcd(math.gcd(a, b), n) != 1:
                    continue
                
                # Check if any bad coefficient closes the walk
                valid = True
                for c1, c2 in bad_coeffs:
                    if (c1 * a + c2 * b) % n == 0:
                        valid = False
                        break
                
                if valid:
                    # Construct edges
                    unique_edges = set()
                    for i in range(n):
                        j1 = (i + a) % n
                        j2 = (i + b) % n
                        
                        # Add undirected edges
                        if i < j1:
                            unique_edges.add((i, j1))
                        else:
                            unique_edges.add((j1, i))
                        
                        if i < j2:
                            unique_edges.add((i, j2))
                        else:
                            unique_edges.add((j2, i))
                    
                    return n, list(unique_edges)
        
    # Fallback: LUW Construction (from seed)
    # This ensures we return a valid graph even if the circulant search fails
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
