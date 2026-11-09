import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try 1-factorization of K16
    # Vertices 0..15
    # Standard construction:
    # Fix vertex 15 (infinity)
    # Others 0..14
    # Round r (0..14):
    # Match 15 with r
    # Match (r+k)%15 with (r-k)%15 for k=1..7
    
    n = 16
    colors = [-1] * (n * (n - 1) // 2)
    
    # Map (i,j) to index
    def idx(i, j):
        if i > j: i, j = j, i
        return i * n - (i * (i + 1)) // 2 + j - i - 1
        
    # Actually simpler:
    # i < j
    # index = sum_{k=0}^{i-1} (n-k-1) + (j-i-1)
    # Precompute offsets
    offsets = [0] * n
    for i in range(n):
        offsets[i] = offsets[i-1] + (n - i - 1) if i > 0 else 0
        # Wait, sum_{k=0}^{i-1} (n-k-1)
        # i=0: 0
        # i=1: 15
        # i=2: 15+14=29
    # Let's just use a dictionary or simple calculation
    # Total edges 120.
    
    # Let's build the factorization
    factors = [[] for _ in range(15)]
    
    for r in range(15):
        # Edge (15, r)
        factors[r].append((15, r))
        for k in range(1, 8):
            u = (r + k) % 15
            v = (r - k) % 15
            if u > v: u, v = v, u
            factors[r].append((u, v))
            
    # Assign colors
    # Color of edge is the factor index
    for r in range(15):
        for (u, v) in factors[r]:
            if u > v: u, v = v, u
            # Calculate index
            # i=u, j=v
            # index = u*(2*n - u - 1)//2 + (v - u - 1)
            index = u * (2 * n - u - 1) // 2 + (v - u - 1)
            colors[index] = r
            
    # Verify
    # Check every K4
    valid = True
    # To speed up, we can sample or use numpy
    import numpy as np
    # Build color matrix
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            index = i * (2 * n - i - 1) // 2 + (j - i - 1)
            C[i, j] = colors[index]
            C[j, i] = colors[index]
            
    # Check all K4
    # 16 vertices -> 1820 K4s
    # 1820 is small.
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Edges: ab, ac, ad, bc, bd, cd
                    e1 = C[a, b]
                    e2 = C[a, c]
                    e3 = C[a, d]
                    e4 = C[b, c]
                    e5 = C[b, d]
                    e6 = C[c, d]
                    distinct = len({e1, e2, e3, e4, e5, e6})
                    if distinct < 5:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
        if time.time() > deadline: break
        if time.time() > deadline: break
        if time.time() > deadline: break
        
    if valid:
        return colors
    else:
        # Fallback to seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
