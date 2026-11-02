import math

def construct():
    best_score = 0
    best_set = [0, 1, 2, 4, 8]
    
    # Try geometric progressions with base 2
    # U = {0, 1, 2, 4, ..., 2^(k-1)}
    for k in range(3, 16):
        u = [1 << i for i in range(k)]
        # Ensure 0 is in the set. The list above starts with 1.
        # My previous analysis included 0.
        # U = {0, 1, 2, 4, ...}
        u = [0] + [1 << i for i in range(k-1)]
        
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
