"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s kill)
    N = 128
    
    # Precompute conflict masks for each x
    # conflict_masks[x] is a list of integers. Each integer is a bitmask of 5 bits: a, b, a+b, a+x, b+x
    # If (S & mask) == mask, then adding x is invalid.
    conflict_masks = [[] for _ in range(N)]
    for x in range(N):
        masks = []
        # a, b must be distinct from x and each other
        # a+b, a+x, b+x must be < 128
        for a in range(N):
            if a == x: continue
            # a+x < 128 => a < 128-x
            if a + x >= N: continue
            for b in range(a + 1, N):
                if b == x: continue
                if b + x >= N: continue
                s_ab = a + b
                if s_ab >= N: continue
                # Check distinctness of sums? 
                # "The six displayed values need not be mutually distinct."
                # But a, b, x are distinct.
                # Sums a+b, a+x, b+x might coincide with a,b,x or each other.
                # The condition is just that they are in S.
                # Mask bits: a, b, s_ab, a+x, b+x
                mask = (1 << a) | (1 << b) | (1 << s_ab) | (1 << (a + x)) | (1 << (b + x))
                masks.append(mask)
        conflict_masks[x] = masks

    def can_add(S, x):
        if (S >> x) & 1:
            return False # Already in S
        for mask in conflict_masks[x]:
            if (S & mask) == mask:
                return False
        return True

    # Initialize S as empty
    S = 0
    best_S = 0
    best_size = 0
    
    # Greedy Construction
    # Order: Random or Sorted? Random helps avoid local traps.
    # Let's try multiple random orders.
    
    def run_greedy_and_local_search(order):
        S_local = 0
        # Greedy
        for x in order:
            if can_add(S_local, x):
                S_local |= (1 << x)
        
        size = bin(S_local).count('1')
        
        # Local Search: Hill Climbing
        # Try to add any missing element
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in order:
                if not ((S_local >> x) & 1):
                    if can_add(S_local, x):
                        S_local |= (1 << x)
                        size += 1
                        changed = True
        
        # Local Search: Swaps
        # Try to swap x in S with y not in S if it allows adding more later?
        # Or just accept valid swaps to escape local minima.
        # Since we only care about size, a swap keeps size same.
        # We can accept a swap if it doesn't reduce size (it doesn't) and is valid.
        # This helps find better configurations for further additions.
        for _ in range(100): # Limit iterations
            if time.time() > deadline: break
            swapped = False
            # Pick random x in S, y not in S
            in_S = [i for i in range(N) if (S_local >> i) & 1]
            not_in_S = [i for i in range(N) if not ((S_local >> i) & 1)]
            if not in_S or not not_in_S:
                break
            x = random.choice(in_S)
            y = random.choice(not_in_S)
            
            # Try swap
            new_S = S_local
            new_S &= ~(1 << x)
            new_S |= (1 << y)
            
            # Check validity of new_S
            # We need to check if new_S is valid.
            # Since we only changed x->y, we only need to check triples involving y.
            # Triples not involving y were valid in S_local (assuming S_local was valid).
            # Wait, S_local might not be valid if we started with invalid? No, we started valid.
            # So we just check if y creates a forbidden triple with other elements in new_S.
            valid = True
            for mask in conflict_masks[y]:
                if (new_S & mask) == mask:
                    valid = False
                    break
            
            if valid:
                S_local = new_S
                swapped = True
                # Try to add more after swap
                for z in order:
                    if not ((S_local >> z) & 1):
                        if can_add(S_local, z):
                            S_local |= (1 << z)
                            size += 1
        
        return S_local, size

    # Run multiple restarts
    best_S = 0
    best_size = 0
    
    while time.time() < deadline:
        order = list(range(N))
        random.shuffle(order)
        S_candidate, size_candidate = run_greedy_and_local_search(order)
        
        if size_candidate > best_size:
            best_size = size_candidate
            best_S = S_candidate
            
        # If we reached a high size, we might stop early if time is tight?
        # No, keep going.

    # Convert best_S to list of ints
    result = []
    for i in range(N):
        if (best_S >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
