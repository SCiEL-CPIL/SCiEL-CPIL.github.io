import random
import time

def construct():
    N = 256
    # Precompute 3-APs
    ap3s = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3s.append((a, b, c))
            
    # Precompute 4-APs
    ap4s = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            ap4s.append((a, b, c, dd))
            
    # Precompute masks for 4-APs
    # We need to check if adding x creates a 4-AP.
    # This happens if there is a 4-AP containing x such that the other 3 elements are in A.
    # Let's precompute for each x, the list of masks of 4-APs containing x, 
    # but specifically, we need to check if (A & mask_without_x) == mask_without_x.
    # It is faster to precompute `forbidden_masks_for_add[x]` which is a list of masks 
    # representing the other 3 elements. If (A & mask) == mask, then adding x is forbidden.
    
    forbidden_masks = [[] for _ in range(N+1)]
    for a, b, c, d in ap4s:
        # x=a
        forbidden_masks[a].append((1 << (b-1)) | (1 << (c-1)) | (1 << (d-1)))
        # x=b
        forbidden_masks[b].append((1 << (a-1)) | (1 << (c-1)) | (1 << (d-1)))
        # x=c
        forbidden_masks[c].append((1 << (a-1)) | (1 << (b-1)) | (1 << (d-1)))
        # x=d
        forbidden_masks[d].append((1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)))
        
    # Precompute 3-AP pairs for each x
    # pairs_for_3ap[x] = list of (y, z) such that {x,y,z} is a 3-AP
    pairs_for_3ap = [[] for _ in range(N+1)]
    for a, b, c in ap3s:
        pairs_for_3ap[a].append((b, c))
        pairs_for_3ap[b].append((a, c))
        pairs_for_3ap[c].append((a, b))

    # Helper to count 3-APs from bitset
    def count_3aps(bitset):
        count = 0
        for a, b, c in ap3s:
            if (bitset & (1 << (a-1))) and (bitset & (1 << (b-1))) and (bitset & (1 << (c-1))):
                count += 1
        return count

    # Initialize: Random set, then repair
    initial_size = 50
    A_bits = 0
    for _ in range(initial_size):
        x = random.randint(1, N)
        A_bits |= (1 << (x-1))
        
    # Repair: Remove elements that create 4-APs
    # Iteratively remove one element from any existing 4-AP
    # To be fast, we can just check all 4-APs.
    # Since N=256, number of 4-APs is ~ (256^3)/27 ~ 60k? No.
    # d goes up to 85. a goes up to 256-3d.
    # Sum_{d=1}^{85} (256-3d) ~ 85*256 - 3*85*86/2 ~ 21760 - 10965 ~ 10k.
    # So 10k checks per repair step.
    # Repair loop might take a few iterations.
    
    changed = True
    while changed:
        changed = False
        for a, b, c, d in ap4s:
            if (A_bits & (1 << (a-1))) and (A_bits & (1 << (b-1))) and (A_bits & (1 << (c-1))) and (A_bits & (1 << (d-1))):
                A_bits &= ~(1 << (d-1))
                changed = True
                break

    current_score = count_3aps(A_bits)
    best_score = current_score
    best_bits = A_bits
    
    start_time = time.time()
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    while T > T_min and time.time() - start_time < 55:
        x = random.randint(1, N)
        bit = 1 << (x-1)
        
        if A_bits & bit:
            # Remove x
            new_bits = A_bits & ~bit
            delta = 0
            for y, z in pairs_for_3ap[x]:
                if (new_bits & (1 << (y-1))) and (new_bits & (1 << (z-1))):
                    delta -= 1
            new_score = current_score + delta
            if new_score > best_score:
                best_score = new_score
                best_bits = new_bits
            if delta >= 0 or random.random() < pow(2.718, delta / T):
                A_bits = new_bits
                current_score = new_score
        else:
            # Add x
            valid = True
            for mask in forbidden_masks[x]:
                if (A_bits & mask) == mask:
                    valid = False
                    break
            
            if valid:
                new_bits = A_bits | bit
                delta = 0
                for y, z in pairs_for_3ap[x]:
                    if (A_bits & (1 << (y-1))) and (A_bits & (1 << (z-1))):
                        delta += 1
                new_score = current_score + delta
                if new_score > best_score:
                    best_score = new_score
                    best_bits = new_bits
                if delta >= 0 or random.random() < pow(2.718, delta / T):
                    A_bits = new_bits
                    current_score = new_score
                    
        T *= cooling

    # Convert best_bits to sorted list
    result = []
    for i in range(1, N+1):
        if best_bits & (1 << (i-1)):
            result.append(i)
            
    return result
