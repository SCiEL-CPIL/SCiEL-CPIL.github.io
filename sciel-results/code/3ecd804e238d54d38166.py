import itertools
import math

def construct():
    # Search for a set U from 0..15 that maximizes the score
    # We try all subsets containing 0
    best_score = 0
    best_set = [0]
    
    max_val = 15
    # Iterate over all masks from 1 to 2^(max_val+1)
    # Bit i is set if i is in the set
    # Ensure 0 is in the set, so mask must have bit 0 set
    for mask in range(1, 1 << (max_val + 1)):
        if not (mask & 1):
            continue
            
        u = []
        for i in range(max_val + 1):
            if mask & (1 << i):
                u.append(i)
                
        if len(u) < 2:
            continue
            
        # Calculate score
        sums = set()
        diffs = set()
        max_u = u[-1]
        
        for i in u:
            for j in u:
                sums.add(i + j)
                diffs.add(i - j)
                
        if not sums or not diffs:
            continue
            
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            continue
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            continue
            
        score = 1 + math.log2(ratio) / denom
        
        if score > best_score:
            best_score = score
            best_set = u
            
    return best_set
