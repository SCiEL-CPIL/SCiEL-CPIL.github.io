import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # We want to find a code of size 256.
    # Greedy algorithm with weight-lexicographic order.
    
    # Pre-compute candidates by weight
    # 2^18 = 262144
    # This is fast enough
    candidates_by_weight = [[] for _ in range(19)]
    for c in range(1, 1 << 18):
        candidates_by_weight[c.bit_count()].append(c)
        
    code = [0]
    
    # Iterate weights from 1 to 18
    # Weight 0 is just 0, already in code
    for w in range(1, 19):
        for c in candidates_by_weight[w]:
            if time.time() > deadline:
                return code
                
            # Check if c is at distance >= 6 from all words in code
            valid = True
            for existing in code:
                if (c ^ existing).bit_count() < 6:
                    valid = False
                    break
            
            if valid:
                code.append(c)
                
    return code
