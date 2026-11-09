import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin

    def has_mono_k222_fast(n, colors):
        """
        Fast check for monochromatic K2,2,2 using bitmasks.
        colors: list of 0/1 of length n*(n-1)/2
        """
        # Build adjacency bitmasks for each color
        adj = [[0] * n for _ in range(2)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in [0, 1]:
            for u1 in range(n):
                for u2 in range(u1 + 1, n):
                    # Common neighbors of u1 and u2 in color c
                    mask_s = adj[c][u1] & adj[c][u2]
                    # Exclude u1, u2 themselves (they shouldn't be in their own adjacency, but just in case)
                    mask_s &= ~(1 << u1)
                    mask_s &= ~(1 << u2)
                    
                    if mask_s.bit_count() < 4:
                        continue
                    
                    verts_s = [i for i in range(n) if (mask_s >> i) & 1]
                    
                    found = False
                    # Iterate over pairs v1, v2 in S
                    for i_idx in range(len(verts_s)):
                        v1 = verts_s[i_idx]
                        # Check if v1-v2 is color c
                        if not (adj[c][v1] & (1 << verts_s[i_idx+1] if i_idx+1 < len(verts_s) else -1)): 
                             # This logic is messy. Better to just check edge existence.
                             pass
                        
                        for j_idx in range(i_idx + 1, len(verts_s)):
                            v2 = verts_s[j_idx]
                            # Check if edge (v1, v2) is color c
                            if (adj[c][v1] & (1 << v2)) == 0:
                                continue
                            
                            # Find M_v: vertices in S \ {v1, v2} connected to both v1 and v2 in color c
                            mask_mv = mask_s & ~(1 << v1) & ~(1 << v2)
                            mask_mv &= adj[c][v1]
                            mask_mv &= adj[c][v2]
                            
                            if mask_mv.bit_count() < 2:
                                continue
                            
                            # Check if there is an edge in color c within M_v
                            verts_mv = [i for i in range(n) if (mask_mv >> i) & 1]
                            for k_idx in range(len(verts_mv)):
                                w1 = verts_mv[k_idx]
                                for l_idx in range(k_idx + 1, len(verts_mv)):
                                    w2 = verts_mv[l_idx]
                                    if (adj[c][w1] & (1 << w2)) != 0:
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                return True
        return False

    def generate_paley(n):
        if n % 4 != 1:
            return None
        squares = {x * x % n for x in range(1, n)}
        colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
        return colors

    # Try n=29 first (Paley)
    colors_29 = generate_paley(29)
    if colors_29:
        if not has_mono_k222_fast(29, colors_29):
            return {'n': 29, 'colors': colors_29}

    # Try n=17 (Paley)
    colors_17 = generate_paley(17)
    if colors_17:
        if not has_mono_k222_fast(17, colors_17):
            return {'n': 17, 'colors': colors_17}

    # Simulated Annealing for n=30 (limited time)
    if time.time() < start_time + 100:
        n_target = 30
        for _ in range(10): # 10 restarts
            if time.time() > start_time + 130:
                break
            colors = [random.randint(0, 1) for _ in range(n_target * (n_target - 1) // 2)]
            if not has_mono_k222_fast(n_target, colors):
                return {'n': n_target, 'colors': colors}
            for _ in range(100):
                if time.time() > start_time + 130:
                    break
                idx = random.randint(0, len(colors) - 1)
                colors[idx] ^= 1
                if not has_mono_k222_fast(n_target, colors):
                    return {'n': n_target, 'colors': colors}

    # Fallback
    n = 17
    colors = generate_paley(17)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
