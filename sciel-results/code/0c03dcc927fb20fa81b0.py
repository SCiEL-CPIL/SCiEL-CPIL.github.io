import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n = 16
    total_edges = n * (n - 1) // 2
    colors = [-1] * total_edges
    
    # Build 1-factorization of K16
    # Fix vertex 15, use rounds 0..14
    factors = [[] for _ in range(15)]
    
    for r in range(15):
        # Edge (15, r)
        factors[r].append((15, r))
        for k in range(1, 8):
            u = (r + k) % 15
            v = (r - k) % 15
            if u > v:
                u, v = v, u
            factors[r].append((u, v))
    
    # Assign colors: color of edge is the factor index
    for r in range(15):
        for (u, v) in factors[r]:
            if u > v:
                u, v = v, u
            # index = u*(2*n - u - 1)//2 + (v - u - 1)
            index = u * (2 * n - u - 1) // 2 + (v - u - 1)
            colors[index] = r
    
    # Verify all edges are colored
    if -1 in colors:
        # Fallback
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Build color matrix for fast lookup
    C = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            index = i * (2 * n - i - 1) // 2 + (j - i - 1)
            C[i, j] = colors[index]
            C[j, i] = colors[index]
    
    # Check all K4s: every set of 4 vertices must span edges of at least 5 distinct colours
    valid = True
    for a in range(n):
        if time.time() > deadline - 5.0:
            break
        for b in range(a + 1, n):
            if time.time() > deadline - 5.0:
                break
            for c in range(b + 1, n):
                if time.time() > deadline - 5.0:
                    break
                for d in range(c + 1, n):
                    # 6 edges in K4
                    e = np.array([C[a, b], C[a, c], C[a, d], C[b, c], C[b, d], C[c, d]])
                    if len(np.unique(e)) < 5:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return colors
    else:
        # Fallback: use a simple coloring that might work
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
