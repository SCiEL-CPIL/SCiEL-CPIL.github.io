import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total_masks = len(all_masks)
    
    best_family_size = 0
    best_family = []
    
    # Try multiple random restarts
    # Each restart uses a different random ordering
    max_restarts = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_masks)
        
        family = []
        family_set = set()
        
        for C in all_masks:
            if time.time() > deadline:
                break
                
            # Check if C can be added to family
            # We need to ensure no A, B in family form a sunflower with C.
            # Sunflower condition: A cap B = A cap C = B cap C = I
            # Let I = A cap C. Then B must satisfy B cap C = I and A cap B = I.
            
            # Bucket sets in family by their intersection with C
            # Intersections are subsets of C, so there are at most 16 possible values
            buckets = {}
            for A in family:
                I = A & C
                if I not in buckets:
                    buckets[I] = []
                buckets[I].append(A)
            
            compatible = True
            # Check each bucket
            for I, bucket_list in buckets.items():
                if len(bucket_list) >= 2:
                    # Check pairs in this bucket
                    # A, B in bucket_list implies A&C=I and B&C=I
                    # We need to check if A&B == I
                    # Note: A&B must be a superset of I? No, A&B = I.
                    # Since I = A&C, I is subset of A.
                    # If A&B = I, then B must contain I and no elements of A\I.
                    for idx_a in range(len(bucket_list)):
                        A = bucket_list[idx_a]
                        for idx_b in range(idx_a + 1, len(bucket_list)):
                            B = bucket_list[idx_b]
                            if (A & B) == I:
                                compatible = False
                                break
                        if not compatible:
                            break
                if not compatible:
                    break
            
            if compatible:
                family.append(C)
                family_set.add(C)
        
        if len(family) > best_family_size:
            best_family_size = len(family)
            best_family = family[:]
            
            # If we reached the theoretical max or a very high number, we might stop early
            # Max possible is 250. If we hit 250, we can stop.
            if best_family_size >= 250:
                break

    # Convert bitmasks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
