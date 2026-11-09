import random

def construct():
    n = 512
    
    # Precompute all 4-APs and group them by their elements
    # ap_masks[x] will be a list of integers, where each integer is a bitmask
    # of the other 3 elements in a 4-AP that includes x.
    # Element i corresponds to bit i (0-indexed). So element 1 is bit 0.
    
    ap_masks = [[] for _ in range(n)]
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    # 0-indexed: a, a+d, a+2d, a+3d
    # a from 0 to n-1
    # d from 1 to (n-1-a)//3
    
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Elements are a, b, c, e
            # For each element in this AP, store the mask of the other 3
            
            # Mask for a: bits b, c, e
            mask_a = (1 << b) | (1 << c) | (1 << e)
            ap_masks[a].append(mask_a)
            
            # Mask for b: bits a, c, e
            mask_b = (1 << a) | (1 << c) | (1 << e)
            ap_masks[b].append(mask_b)
            
            # Mask for c: bits a, b, e
            mask_c = (1 << a) | (1 << b) | (1 << e)
            ap_masks[c].append(mask_c)
            
            # Mask for e: bits a, b, c
            mask_e = (1 << a) | (1 << b) | (1 << c)
            ap_masks[e].append(mask_e)
            
    best_set_int = 0
    best_size = 0
    
    # Run multiple randomized greedy attempts
    # Bit operations are fast, so we can run more iterations.
    # 512 elements. Each check involves iterating over ~100-200 masks and doing ANDs.
    # Total ops per run: 512 * 200 = ~100k bit ops.
    # 5000 runs = 500M bit ops. In Python, this might be tight but likely faster than the original.
    # Let's try 2000 runs first to be safe, or optimize the loop.
    
    for _ in range(2000):
        S = 0
        # Random permutation
        perm = list(range(n))
        random.shuffle(perm)
        
        for x in perm:
            # Check if x can be added
            can_add = True
            for mask in ap_masks[x]:
                if (S & mask) == mask:
                    can_add = False
                    break
            
            if can_add:
                S |= (1 << x)
        
        size = bin(S).count("1")
        if size > best_size:
            best_size = size
            best_set_int = S
            
    # Convert bitset to list of integers 1..n
    result = []
    for i in range(n):
        if (best_set_int >> i) & 1:
            result.append(i + 1)
            
    return result
