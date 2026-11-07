import time
import random
import math

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 85.0 # Safety margin
    
    # Bitmask representation of the set S.
    # Bit i is 1 if (i+1) is in S.
    # We use indices 0..127 for values 1..128.
    
    def bit_pos(val):
        return val - 1
        
    def bit_mask(val):
        return 1 << bit_pos(val)
        
    def mask_to_list(mask):
        return [i + 1 for i in range(n) if (mask >> i) & 1]
        
    # Check if adding x to S is valid.
    # S is a bitmask. x is a value 1..128.
    # If x is already in S, return False (or True if we handle removal separately).
    # We assume x not in S.
    def can_add(S_mask, x):
        if (S_mask >> bit_pos(x)) & 1:
            return False
            
        # N_x = { u in S | u + x in S }
        # We need to check if there exist a, b in N_x such that:
        # 1. a + b in S
        # 2. a + b == x
        # Note: If a+b == x, then a+b is in S_new (since x is added).
        # If a+b in S, then a+b is in S_old.
        
        # Construct N_x as a bitmask
        N_x_mask = 0
        # Iterate through all u in S
        # Optimization: We can just check pairs in N_x directly.
        # But first, let's identify N_x.
        # u is in N_x if (S_mask has bit u) and (S_mask has bit u+x).
        # u ranges 1..128. u+x <= 128 => u <= 128-x.
        
        # Instead of building N_x mask, we can just iterate potential a in S.
        # For each a in S, if a+x in S, then a is a candidate.
        # We need to check if there is another b in S (b != a) such that b+x in S AND (a+b in S OR a+b == x).
        
        # Let's collect candidates in a list for simplicity, as N is small.
        candidates = []
        for u in range(1, n + 1):
            if (S_mask >> bit_pos(u)) & 1:
                # Check if u+x is in S
                if u + x <= n and (S_mask >> bit_pos(u + x)) & 1:
                    candidates.append(u)
                    
        # Now check pairs in candidates
        # If there are a, b in candidates such that a+b in S or a+b == x, then adding x creates a triangle (a, b, x).
        # Note: a and b must be distinct.
        L = len(candidates)
        for i in range(L):
            a = candidates[i]
            for j in range(i + 1, L):
                b = candidates[j]
                s = a + b
                if s == x:
                    return False
                if s <= n and (S_mask >> bit_pos(s)) & 1:
                    return False
        return True

    # Check if removing x from S keeps it valid.
    # Removing elements never creates a new invalid triple.
    # So removal is always valid.
    def can_remove(S_mask, x):
        if not ((S_mask >> bit_pos(x)) & 1):
            return False
        return True

    # Initialize with a greedy valid set
    # Start with empty, add largest to smallest?
    # Or start with top half.
    S_mask = 0
    # Greedy: add elements from 128 down to 1 if possible.
    # This tends to create sets with large numbers, which have fewer sums in range.
    for x in range(n, 0, -1):
        if can_add(S_mask, x):
            S_mask |= bit_mask(x)
            
    best_mask = S_mask
    best_size = bin(S_mask).count('1')
    
    # Simulated Annealing
    # T starts high, decays.
    T = 1.0
    cooling = 0.999
    
    current_mask = S_mask
    current_size = best_size
    
    # Precompute bit masks for all values for speed?
    # bit_mask(x) is fast enough.
    
    # List of all values 1..128
    all_vals = list(range(1, n + 1))
    
    while time.time() < deadline:
        # Choose a move
        # 70% Add, 30% Remove
        if random.random() < 0.7:
            # Pick random x not in S
            # To speed up, pick from a random subset or just random int and check
            # Since N is small, just pick random int.
            x = random.choice(all_vals)
            if (current_mask >> bit_pos(x)) & 1:
                continue # Already in set, skip
            if can_add(current_mask, x):
                new_mask = current_mask | bit_mask(x)
                new_size = current_size + 1
                # Always accept if size increases?
                # In SA, we accept with prob 1 if better.
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # Invalid move, ignore
                pass
        else:
            # Pick random x in S to remove
            if current_size == 0:
                continue
            # Pick a random bit from current_mask
            # Get list of set bits? Or pick random index and check?
            # To ensure uniformity among set bits, we can pick random index 0..n-1
            # and if not set, retry.
            while True:
                idx = random.randint(0, n - 1)
                if (current_mask >> idx) & 1:
                    x = idx + 1
                    break
            new_mask = current_mask & ~bit_mask(x)
            new_size = current_size - 1
            # Accept removal with probability based on T?
            # Delta = -1. Prob = exp(-Delta/T) = exp(1/T).
            # This allows escaping local maxima.
            if random.random() < math.exp(1.0 / T):
                current_mask = new_mask
                current_size = new_size
        T *= cooling
        
    # Final Greedy Addition on best_mask
    # Try to add any remaining element to best_mask
    final_mask = best_mask
    for x in range(1, n + 1):
        if not ((final_mask >> bit_pos(x)) & 1):
            if can_add(final_mask, x):
                final_mask |= bit_mask(x)
                
    # Convert to list of ints
    result = []
    for i in range(n):
        if (final_mask >> i) & 1:
            result.append(i + 1)
            
    return result

