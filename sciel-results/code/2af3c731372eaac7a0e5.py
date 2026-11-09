import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Try to construct the 50-vertex Chung coloring
    def try_chung_50():
        n = 50
        colors = []
        
        # Precompute modular inverses in Z_17
        inv17 = [0] * 17
        for i in range(1, 17):
            for j in range(1, 17):
                if (i * j) % 17 == 1:
                    inv17[i] = j
                    break
        
        # Vertices are (i, j) with i in 0..16, j in 0..4
        # Map vertex index to (i, j)
        # index = i * 5 + j
        
        for idx1 in range(n):
            i1 = idx1 // 5
            j1 = idx1 % 5
            for idx2 in range(idx1 + 1, n):
                i2 = idx2 // 5
                j2 = idx2 % 5
                
                u = (i1 - i2) % 17
                w = (j1 - j2) % 5
                
                if u == 0:
                    # Vertical difference
                    if w == 1 or w == 4:
                        c = 0
                    else: # w == 2 or w == 3
                        c = 1
                else:
                    # Non-vertical
                    m = (w * inv17[u]) % 5
                    if m == 1 or m == 4:
                        c = 2
                    else: # m == 2 or m == 3
                        c = 3
                
                colors.append(c)
        
        # Verify no monochromatic triangles
        # Build adjacency lists for each color
        adj = [[False] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i][j] = True
                adj[c][j][i] = True
                idx += 1
        
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[c][i][j]:
                        for k in range(j + 1, n):
                            if adj[c][i][k] and adj[c][j][k]:
                                return None # Found monochromatic triangle
        
        return {'n': n, 'colors': colors}

    result = try_chung_50()
    if result:
        return result

    # Fallback to the known valid 32-vertex solution if the above fails
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                c = log[(i % 16) ^ (j % 16)] % 3
            else:
                c = 3
            colors.append(c)
            
    return {'n': 32, 'colors': colors}
