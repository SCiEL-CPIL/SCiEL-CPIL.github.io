import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for evaluator overhead
    
    # Precompute conflicts for 4-AP
    # conflict_mask[x] is a bitmask where bit y is set if x and y are in a 4-AP
    # Note: A 4-AP is a, a+d, a+2d, a+3d.
    # If we have a set S, we cannot add x if any other 3 elements in S form a 4-AP with x.
    # So for each x, we precompute the set of all y such that x and y are part of some 4-AP.
    # Wait, that's not quite right.
    # If we add x to S, we check if there exists a 4-AP {x, a, b, c} subset of S U {x}.
    # This is equivalent to: are there 3 elements a, b, c in S such that {x, a, b, c} is a 4-AP?
    # Precomputing "conflict pairs" is not enough, because the constraint involves 3 other elements.
    # However, we can precompute for each x, the set of "potential partners".
    # Actually, the standard trick for 4-AP is:
    # If x is added, we check if S contains any 3 elements that complete a 4-AP with x.
    # There are O(N^2) such triples for a fixed x? No, O(N).
    # For each d, x can be at pos 1, 2, 3, 4.
    # Pos 1: x, x+d, x+2d, x+3d. Need x+d, x+2d, x+3d in S.
    # Pos 2: x-d, x, x+d, x+2d. Need x-d, x+d, x+2d in S.
    # Pos 3: x-2d, x-d, x, x+d. Need x-2d, x-d, x+d in S.
    # Pos 4: x-3d, x-2d, x-d, x. Need x-3d, x-2d, x-d in S.
    
    # So for each x, we have O(N) checks, each requiring 3 set lookups.
    # With bitmask, set lookup is O(1) bitwise AND.
    # So can_add is O(N) bitwise ops.
    # N=512, so 512 bitwise ops. Very fast.
    
    # Let's use a Python integer as a bitmask. Python ints are arbitrary precision, 
    # so we can use a single int for the whole set.
    # bit i represents element i+1? Or element i?
    # Let's use bit i for element i+1 (1-indexed).
    # So element 1 is bit 0, element 512 is bit 511.
    
    def to_bitmask(S):
        mask = 0
        for x in S:
            mask |= (1 << (x - 1))
        return mask
        
    def from_bitmask(mask):
        S = []
        for i in range(N):
            if mask & (1 << i):
                S.append(i + 1)
        return S
        
    def can_add(mask, x):
        # x is 1-indexed
        # Check if adding x creates a 4-AP
        # Iterate d
        max_d = N // 3
        # Case 1: x is first term
        # Need x+d, x+2d, x+3d in mask
        for d in range(1, max_d + 1):
            # Case 1
            if x + 3*d <= N:
                b1 = 1 << (x + d - 1)
                b2 = 1 << (x + 2*d - 1)
                b3 = 1 << (x + 3*d - 1)
                if (mask & b1) and (mask & b2) and (mask & b3):
                    return False
            # Case 2: x is second term
            if x - d >= 1 and x + 2*d <= N:
                b1 = 1 << (x - d - 1)
                b2 = 1 << (x + d - 1)
                b3 = 1 << (x + 2*d - 1)
                if (mask & b1) and (mask & b2) and (mask & b3):
                    return False
            # Case 3: x is third term
            if x - 2*d >= 1 and x + d <= N:
                b1 = 1 << (x - 2*d - 1)
                b2 = 1 << (x - d - 1)
                b3 = 1 << (x + d - 1)
                if (mask & b1) and (mask & b2) and (mask & b3):
                    return False
            # Case 4: x is fourth term
            if x - 3*d >= 1:
                b1 = 1 << (x - 3*d - 1)
                b2 = 1 << (x - 2*d - 1)
                b3 = 1 << (x - d - 1)
                if (mask & b1) and (mask & b2) and (mask & b3):
                    return False
        return True

    # Initial solution: Greedy
    # Randomize order for diversity? Or use natural?
    # Let's use a few different greedy seeds and pick the best.
    
    def greedy(order):
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << (x - 1))
        return mask

    # Try a few random permutations for initial greedy
    best_mask = 0
    best_size = 0
    
    random.seed(42)
    
    # 1. Natural order
    m = greedy(range(1, N + 1))
    s = bin(m).count('1')
    if s > best_size:
        best_size = s
        best_mask = m
        
    # 2. Reverse order
    m = greedy(range(N, 0, -1))
    s = bin(m).count('1')
    if s > best_size:
        best_size = s
        best_mask = m
        
    # 3. Random orders
    for _ in range(20):
        order = list(range(1, N + 1))
        random.shuffle(order)
        m = greedy(order)
        s = bin(m).count('1')
        if s > best_size:
            best_size = s
            best_mask = m

    current_mask = best_mask
    current_size = best_size
    
    # Simulated Annealing
    # State: mask
    # Move: Remove one element, Add one element (or just add if possible, but SA usually keeps size constant or varies)
    # Objective: Maximize size.
    # If we remove 1 and add 1, size is constant. We need to allow size to change.
    # Better: 
    # Move 1: Add a random element from [1..N] not in S. If valid, accept. (Increases size)
    # Move 2: Remove a random element from S. (Decreases size)
    # Move 3: Swap: Remove one, Add one.
    
    # To escape local optima where we can't add anything, we need to remove something.
    # SA accepts worse solutions (smaller size) with probability exp(-delta/T).
    # Here delta = -1 (size decreases by 1).
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # We need to be careful with time.
    # can_add is O(N) ~ 500 ops.
    # We can do ~10^5 moves in 5s? 10^5 * 500 = 5*10^7 ops. Python is slow, maybe 10^6-10^7 ops/sec.
    # So maybe 10^4-10^5 moves.
    
    # Optimization: Precompute valid additions? No, mask changes.
    
    # Let's implement SA.
    
    elements = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Choose move type
        # 80% Swap, 10% Add, 10% Remove
        r = random.random()
        
        new_mask = current_mask
        new_size = current_size
        delta = 0
        
        if r < 0.1:
            # Try to Add
            # Pick random x not in S
            # Efficient way: pick random index, check if in S. If in S, retry or pick another.
            # Better: maintain a list of available elements? Or just sample.
            # Sampling is fine.
            x = random.choice(elements)
            if not (current_mask & (1 << (x - 1))):
                if can_add(current_mask, x):
                    new_mask = current_mask | (1 << (x - 1))
                    new_size = current_size + 1
                    delta = 1
        elif r < 0.2:
            # Try to Remove
            # Pick random x in S
            # Find a set bit in current_mask
            # random bit?
            # Get list of set bits? Expensive.
            # Pick random index, check if in S.
            x = random.choice(elements)
            if current_mask & (1 << (x - 1)):
                new_mask = current_mask & ~(1 << (x - 1))
                new_size = current_size - 1
                delta = -1
        else:
            # Swap
            # Remove x, Add y
            # Pick x in S
            x = random.choice(elements)
            if not (current_mask & (1 << (x - 1))):
                # Retry or skip? Skip is safer for time.
                continue
                
            # Pick y not in S
            y = random.choice(elements)
            if current_mask & (1 << (y - 1)):
                continue
                
            # Try swap
            # Remove x
            temp_mask = current_mask & ~(1 << (x - 1))
            # Add y
            if can_add(temp_mask, y):
                new_mask = temp_mask | (1 << (y - 1))
                new_size = current_size
                delta = 0
            else:
                continue
                
        # Acceptance
        if delta >= 0:
            # Always accept improvement or equal
            current_mask = new_mask
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
        else:
            # delta = -1
            # Accept with prob exp(1/T)
            if random.random() < np.exp(1.0 / T):
                current_mask = new_mask
                current_size = new_size
                
        T *= cooling

    # Final polish: Greedy add on the best found?
    # Try to add any remaining elements in a few random orders
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Greedy on top of best_mask? 
        # No, greedy builds from scratch.
        # Let's just do a local search add on best_mask.
        m = best_mask
        s = best_size
        for x in order:
            if not (m & (1 << (x - 1))):
                if can_add(m, x):
                    m |= (1 << (x - 1))
                    s += 1
        if s > best_size:
            best_size = s
            best_mask = m

    return from_bitmask(best_mask)

# EVOLVE-BLOCK-END
