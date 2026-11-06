import time

def construct():
    start_time = time.time()
    
    # 1. Generate all simple 6-cycles starting at 0 in Z17
    # Represent each cycle by the bitmask of its 6 edge differences (1..16)
    # Note: The edge (u,v) has difference d=(v-u)%17.
    # The set of differences is {d1, ..., d6}.
    
    c6_masks = set()
    # v1..v5 distinct, non-zero
    for v1 in range(1, 17):
        for v2 in range(1, 17):
            if v2 == v1: continue
            for v3 in range(1, 17):
                if v3 == v1 or v3 == v2: continue
                for v4 in range(1, 17):
                    if v4 == v1 or v4 == v2 or v4 == v3: continue
                    for v5 in range(1, 17):
                        if v5 == v1 or v5 == v2 or v5 == v3 or v5 == v4: continue
                        
                        d1 = v1
                        d2 = (v2 - v1) % 17
                        d3 = (v3 - v2) % 17
                        d4 = (v4 - v3) % 17
                        d5 = (v5 - v4) % 17
                        d6 = (17 - v5) % 17 # 0 - v5 mod 17
                        
                        # Create mask
                        mask = 0
                        for d in (d1, d2, d3, d4, d5, d6):
                            # d is 1..16
                            mask |= (1 << (d - 1))
                        c6_masks.add(mask)
                        
    # 2. For each subset S of {1..8}, compute D mask and check if it contains any c6_mask
    # D is determined by S.
    # S is subset of {1..8}.
    # D contains d and 17-d for each d in S.
    
    valid_S_masks = [False] * 256
    for S_mask in range(256):
        D_mask = 0
        for i in range(1, 9):
            if S_mask & (1 << (i-1)):
                # Add i and 17-i
                D_mask |= (1 << (i - 1))
                D_mask |= (1 << ((17 - i) - 1))
                
        # Check if any c6_mask is subset of D_mask
        contains_c6 = False
        for cm in c6_masks:
            if (cm & D_mask) == cm:
                contains_c6 = True
                break
        valid_S_masks[S_mask] = not contains_c6
        
    # 3. Search for a coloring of 1..8 into 4 colors
    # such that for each color c, the S_mask for that color is valid (i.e., valid_S_masks[S_mask] is True)
    
    for mask in range(65536):
        # Extract 4 S_masks
        S_masks = [0, 0, 0, 0]
        for i in range(1, 9):
            c = (mask >> (2*(i-1))) & 3
            S_masks[c] |= (1 << (i-1))
            
        # Check if all 4 colors have valid S_masks
        valid = True
        for c in range(4):
            if not valid_S_masks[S_masks[c]]:
                valid = False
                break
                
        if valid:
            # Construct coloring for n=17
            n = 17
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    d = (j - i) % 17
                    d_min = min(d, 17-d)
                    c = (mask >> (2*(d_min-1))) & 3
                    colors.append(c)
            return {'n': n, 'colors': colors}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
