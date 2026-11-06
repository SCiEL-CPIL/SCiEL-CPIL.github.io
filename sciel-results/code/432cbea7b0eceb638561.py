import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    N = 512
    K = 32

    def score_bitmask(mask):
        count = 0
        # d can be at most (N-1)//2 = 255
        # If max element is 512, min is 1.
        # Max d such that a, a+d, a+2d <= 512 is 255 (1, 256, 511) or (2, 257, 512).
        # Actually, if a=1, a+2d <= 512 => 2d <= 511 => d <= 255.
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def get_mask(S):
        mask = 0
        for x in S:
            mask |= (1 << x)
        return mask

    def greedy_construct():
        S = set()
        mask = 0
        for _ in range(K):
            best_x = -1
            best_score = -1
            for x in range(1, N + 1):
                if x in S:
                    continue
                new_mask = mask | (1 << x)
                s = score_bitmask(new_mask)
                if s > best_score:
                    best_score = s
                    best_x = x
            S.add(best_x)
            mask |= (1 << best_x)
        return S

    def guided_sample_y(S, x):
        # Generate candidate y's based on x and existing S
        candidates = set()
        for s in S:
            if s == x:
                continue
            d = abs(s - x)
            # x is 1st term: x, x+d, x+2d -> y = x+d or x+2d? 
            # If we add y, we might complete an AP.
            # If y = x + d, then (x, x+d, x+2d) needs x+2d. 
            # If y = x + 2d, then (x, x+d, x+2d) needs x+d.
            # It's better to think: which y creates a new AP with x?
            # 1. (x-d, x, x+d): y = x-d or x+d
            # 2. (x, x+d, x+2d): y = x+d or x+2d
            # 3. (x-2d, x-d, x): y = x-2d or x-d
            
            # So candidates are x +/- d, x +/- 2d
            candidates.add(x + d)
            candidates.add(x - d)
            candidates.add(x + 2 * d)
            candidates.add(x - 2 * d)
        
        # Filter valid candidates
        valid_candidates = [y for y in candidates if 1 <= y <= N and y not in S]
        
        if not valid_candidates:
            # Fallback to random
            while True:
                y = random.randint(1, N)
                if y not in S:
                    return y
        else:
            return random.choice(valid_candidates)

    # 1. Greedy Initialization
    S_curr = greedy_construct()
    mask_curr = get_mask(S_curr)
    score_curr = score_bitmask(mask_curr)
    
    best_S = set(S_curr)
    best_score = score_curr
    best_mask = mask_curr

    # 2. Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    # Pre-compute list of elements for fast choice
    # We update this list when S changes
    
    while time.time() < deadline:
        # Choose x to remove
        x = random.choice(list(S_curr))
        
        # Choose y to add
        y = guided_sample_y(S_curr, x)
        
        # Apply move
        if x == y:
            continue
            
        new_mask = mask_curr
        new_mask &= ~(1 << x)
        new_mask |= (1 << y)
        
        new_score = score_bitmask(new_mask)
        
        delta = new_score - score_curr
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept move
            S_curr.remove(x)
            S_curr.add(y)
            mask_curr = new_mask
            score_curr = new_score
            
            if score_curr > best_score:
                best_score = score_curr
                best_S = set(S_curr)
                best_mask = new_mask
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Convert best_S to 1-based list (already 1-based in our logic)
    return sorted(list(best_S))
