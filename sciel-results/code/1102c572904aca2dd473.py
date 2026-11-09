def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    N = 17
    MIN_D = 6
    SIZE = 1 << N

    # Precompute weights
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    # Precompute masks for each weight up to MIN_D - 1
    masks_by_weight = []
    for w in range(MIN_D):
        masks = []
        # Generate all bitmasks of weight w in N bits
        # Using combinations of indices
        for comb in combinations(range(N), w):
            mask = 0
            for i in comb:
                mask |= (1 << i)
            masks.append(mask)
        masks_by_weight.append(np.array(masks, dtype=np.int32))

    def greedy_packing(order, time_limit):
        forbidden = np.zeros(SIZE, dtype=bool)
        selected = []
        
        for v in order:
            if time.time() > time_limit:
                break
            
            if not forbidden[v]:
                selected.append(v)
                # Mark v and all neighbors within distance < MIN_D as forbidden
                for w in range(MIN_D):
                    masks = masks_by_weight[w]
                    if len(masks) > 0:
                        indices = v ^ masks
                        forbidden[indices] = True
        
        return selected

    best_code = []
    best_size = 0
    rng = random.Random(42)
    
    all_words = list(range(SIZE))
    
    # Strategy 1: Random restarts
    while time.time() < deadline:
        order = all_words[:]
        rng.shuffle(order)
        code = greedy_packing(order, deadline - 5)
        if len(code) > best_size:
            best_code = code
            best_size = len(code)
            
    return best_code
