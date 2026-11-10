import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Known best is 26. Try to find it.
    # Group Z_13 x Z_2
    n = 26
    # Elements 0..25
    # (x, e) -> x + 13*e
    # Identity: 0
    # Involution: 13
    
    # Pairs of non-identity, non-involution elements
    # Type 0: (x, 0) and (-x, 0) for x in 1..6
    # Type 1: (x, 1) and (-x, 1) for x in 1..6
    
    pairs = []
    for x in range(1, 7):
        # Type 0
        a = x
        b = (13 - x) % 13
        pairs.append((a, b))
        # Type 1
        c = x + 13
        d = (13 - x) % 13 + 13
        pairs.append((c, d))
        
    # pairs is a list of 12 tuples (u, v) where u, v are inverses
    # We need to partition these 12 pairs into 4 groups of 3 pairs each.
    # Each group forms a set S of size 6.
    
    # Pre-calculate indices
    pair_indices = list(range(12))
    
    # To speed up, we can fix the first pair in the first group to reduce symmetry
    # But 154k is small enough.
    
    best_n = 11
    best_colors = [min(j - i, n - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    def is_sidon(S):
        # S is a list of 6 elements
        diffs = {}
        for i in range(6):
            for j in range(6):
                if i == j: continue
                d = (S[i] - S[j]) % 26
                if d in diffs:
                    return False
                diffs[d] = 1
        return True

    found = False
    # Iterate over combinations of 3 pair indices for S2
    # Then 3 for S3 from remaining
    # Then 3 for S4
    # Last 3 for S5
    
    # Optimization: Use recursion or nested loops
    # Nested loops are fine for 12 choose 3
    
    # To avoid checking symmetric permutations of the 4 sets, 
    # we can enforce an ordering, e.g., min(S2) < min(S3) < min(S4) < min(S5)
    # But the sets are not naturally ordered. 
    # Just brute force all partitions.
    
    # There are 154,000 partitions.
    # Each check takes ~ 4 * 30 = 120 operations.
    # Total ~ 18 million operations.
    # In Python, this might take 5-10 seconds.
    
    for c2 in itertools.combinations(pair_indices, 3):
        if time.time() > deadline: break
        S2 = []
        for p_idx in c2:
            S2.extend(pairs[p_idx])
        if not is_sidon(S2): continue
        
        remaining1 = [i for i in pair_indices if i not in c2]
        for c3 in itertools.combinations(remaining1, 3):
            if time.time() > deadline: break
            S3 = []
            for p_idx in c3:
                S3.extend(pairs[p_idx])
            if not is_sidon(S3): continue
            
            remaining2 = [i for i in remaining1 if i not in c3]
            for c4 in itertools.combinations(remaining2, 3):
                if time.time() > deadline: break
                S4 = []
                for p_idx in c4:
                    S4.extend(pairs[p_idx])
                if not is_sidon(S4): continue
                
                S5 = []
                for p_idx in remaining2:
                    if p_idx not in c4:
                        S5.extend(pairs[p_idx])
                
                if is_sidon(S5):
                    # Found a valid partition
                    # S1 is {13}
                    sets = [
                        [13],
                        S2,
                        S3,
                        S4,
                        S5
                    ]
                    
                    # Verify
                    # Check if they partition the non-zero elements
                    all_elements = set()
                    for S in sets:
                        all_elements.update(S)
                    if len(all_elements) == 25:
                        # Construct coloring
                        colors = []
                        # Map element to color
                        elem_to_color = {}
                        for c, S in enumerate(sets):
                            for e in S:
                                elem_to_color[e] = c
                                # Since S is symmetric, -e is also in S
                                # We just need to check if d is in S
                                # But d can be negative. 
                                # (j-i) % 26 is always positive.
                                # So we just check if (j-i)%26 is in elem_to_color
                                pass
                        
                        for i in range(n):
                            for j in range(i+1, n):
                                d = (j - i) % n
                                # d is in 1..25
                                colors.append(elem_to_color[d])
                        
                        return {'n': n, 'colors': colors}
                        
    # If not found, return the seed
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
