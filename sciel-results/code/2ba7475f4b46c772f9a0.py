import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of j such that lcm(i, j) <= 256
    conflict_mask = [0] * (N + 1)
    for i in range(2, N + 1):
        mask = 0
        for j in range(2, N + 1):
            if i != j and lcm(i, j) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    def score(S):
        # S is a bitmask
        total = 0.0
        s = S
        while s:
            lsb = s & -s
            idx = lsb.bit_length() - 1
            total += 1.0 / idx
            s ^= lsb
        return total

    def can_add(S, x):
        # Check if x can be added to S
        # x is 1-indexed, so bit x is (1 << x)
        return (S & conflict_mask[x]) == 0

    def greedy(order):
        S = 0
        for x in order:
            if can_add(S, x):
                S |= (1 << x)
        return S

    # Generate different orders for greedy
    # 1. Ascending
    order1 = list(range(2, N + 1))
    # 2. Descending
    order2 = list(range(N, 1, -1))
    # 3. By 1/x (same as ascending)
    # 4. By number of conflicts (degree)
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N + 1)]
    order3 = sorted(range(2, N + 1), key=lambda x: degrees[x])
    order4 = sorted(range(2, N + 1), key=lambda x: -degrees[x])
    
    best_S = 0
    best_score = 0.0
    
    candidates = [greedy(order1), greedy(order2), greedy(order3), greedy(order4)]
    
    # Add some random restarts if time permits
    import random
    random.seed(42)
    for _ in range(10):
        order5 = list(range(2, N + 1))
        random.shuffle(order5)
        candidates.append(greedy(order5))
        
    for S in candidates:
        sc = score(S)
        if sc > best_score:
            best_score = sc
            best_S = S

    # Local Search
    S = best_S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding nodes
        # Iterate through all nodes not in S
        for x in range(2, N + 1):
            if not (S & (1 << x)):
                if can_add(S, x):
                    S |= (1 << x)
                    improved = True
                    break
        
        if improved:
            continue
            
        # 2. Try 1-for-1 swap
        # For each x not in S, check if removing some y in S allows adding x
        # and if 1/x > 1/y (i.e., x < y)
        found_swap = False
        for x in range(2, N + 1):
            if S & (1 << x):
                continue
            # x is not in S
            # Find y in S such that S - {y} + {x} is valid
            # This requires x to be compatible with S \ {y}
            # i.e., (S & conflict_mask[x]) is a subset of {y}
            # Let bad = S & conflict_mask[x]
            # If bad == 0, we could have added x (handled above, so this shouldn't happen if we checked adds first, 
            # but maybe S changed? No, we break on add. So bad != 0).
            # If bad has exactly one bit set, say y, then we can swap y with x.
            bad = S & conflict_mask[x]
            if bad and (bad & (bad - 1)) == 0:
                # bad is a power of 2
                y = bad.bit_length() - 1
                if x < y: # 1/x > 1/y, so score increases
                    S &= ~(1 << y)
                    S |= (1 << x)
                    improved = True
                    found_swap = True
                    break
            elif bad == 0:
                # This case is handled by the add step, but just in case
                # If x < min(S), we might want to swap? No, if bad==0 we can just add.
                # The add step would have caught this.
                pass

        if found_swap:
            continue
            
        # 3. Try 2-for-1 swap (remove 2, add 1) - unlikely to improve sum of reciprocals if we remove 2 and add 1, 
        # unless the 2 removed are very large and the 1 added is small?
        # 1/x + 1/y vs 1/z. If z is small, 1/z is large.
        # But we need z to be compatible with S \ {x, y}.
        
        # 4. Try 1-for-2 swap (remove 1, add 2)
        # 1/y vs 1/x1 + 1/x2.
        # This can increase score if x1, x2 are small enough.
        
        # Let's try a simple 1-for-2:
        # For each y in S, remove y.
        # Then try to add two nodes x1, x2 from S^c that are compatible with S\{y} and each other.
        # This is expensive.
        
        # Instead, let's do a multi-swap search:
        # Try all pairs of additions? No.
        
        # Let's stick to 1-for-1 and maybe try to improve the initial candidates more.
        
        # If no improvement, break.
        
    return [i for i in range(2, N + 1) if (S & (1 << i))]
