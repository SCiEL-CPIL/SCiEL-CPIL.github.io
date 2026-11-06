import time
import random
import sys

N = 512

# Precompute 4-APs
# For each x, store a list of masks representing the other 3 elements in 4-APs containing x.
# A 4-AP is (a, a+d, a+2d, a+3d).
# If x is a: others are a+d, a+2d, a+3d.
# If x is a+d: others are a, a+2d, a+3d.
# If x is a+2d: others are a, a+d, a+3d.
# If x is a+3d: others are a, a+d, a+2d.

def build_ap_masks():
    # masks[x] will be a list of integers (bitmasks)
    # Bit i corresponds to element i+1.
    # So element 1 is bit 0, element 512 is bit 511.
    masks = [[] for _ in range(N + 1)]
    
    # Iterate over all possible 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            a1 = a
            a2 = a + d
            a3 = a + 2 * d
            a4 = a + 3 * d
            if a4 > N:
                break
            
            # Create bitmask for the set {a1, a2, a3, a4}
            # We need masks for each element in the AP, representing the *other three*.
            
            # For a1: others are a2, a3, a4
            m1 = (1 << (a2 - 1)) | (1 << (a3 - 1)) | (1 << (a4 - 1))
            masks[a1].append(m1)
            
            # For a2: others are a1, a3, a4
            m2 = (1 << (a1 - 1)) | (1 << (a3 - 1)) | (1 << (a4 - 1))
            masks[a2].append(m2)
            
            # For a3: others are a1, a2, a4
            m3 = (1 << (a1 - 1)) | (1 << (a2 - 1)) | (1 << (a4 - 1))
            masks[a3].append(m3)
            
            # For a4: others are a1, a2, a3
            m4 = (1 << (a1 - 1)) | (1 << (a2 - 1)) | (1 << (a3 - 1))
            masks[a4].append(m4)
            
    return masks

AP_MASKS = build_ap_masks()

def is_valid_add(mask, x):
    """Check if adding x to the set represented by mask is valid (no 4-AP)."""
    # If x is already in mask, it's not an add, but let's assume x not in mask.
    # We need to check if any 4-AP containing x is completed by mask.
    # A 4-AP containing x is completed if the other 3 elements are in mask.
    for m in AP_MASKS[x]:
        if (mask & m) == m:
            return False
    return True

def greedy_construct(order, deadline):
    mask = 0
    for x in order:
        if time.time() > deadline:
            break
        if is_valid_add(mask, x):
            mask |= (1 << (x - 1))
    return mask

def local_search(mask, deadline):
    """
    Local search to maximize size.
    Moves:
    1. Add 1 element.
    2. Remove 1, Add 2.
    3. Remove 2, Add 3.
    """
    current_mask = mask
    current_size = bin(current_mask).count("1")
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add 1 element
        # Iterate through elements not in mask
        # To speed up, we can shuffle candidates or just iterate 1..N
        # Since N=512, iterating 1..N is fast.
        for x in range(1, N + 1):
            if (current_mask >> (x - 1)) & 1:
                continue
            if is_valid_add(current_mask, x):
                current_mask |= (1 << (x - 1))
                current_size += 1
                improved = True
                # Restart phase 1 as state changed
                break
        
        if improved:
            continue
            
        # Phase 2: Try to remove 1, add 2
        # This increases size by 1.
        # We need to find x in S, and y, z not in S, such that S - {x} + {y, z} is valid.
        # Note: S - {x} is always valid. We need S - {x} + {y} valid, then + {z} valid.
        # Or just check final set.
        
        # Optimization: Only try this if we have time.
        # Number of x is ~size. Number of pairs (y,z) is ~N^2.
        # This might be slow. Let's limit the search space or randomize.
        
        # Let's try a random subset of x and y,z to avoid O(N^3)
        # Or, since N=512, maybe we can be smart.
        
        # Actually, let's just try a few random swaps per iteration.
        # If we do systematic search, it might take too long.
        
        # Let's try systematic for x, but random for y,z?
        # Or random for x and systematic for y,z?
        
        # Given the deadline, let's do:
        # Pick a random x in S.
        # Try to find y, z not in S such that adding them to S-{x} is valid.
        
        # To make it faster, let's collect candidates for y.
        # A candidate y is "good" if S-{x}+{y} is valid.
        # Then from those, find z.
        
        # Let's just try random swaps to keep it fast.
        # 100 random attempts per loop?
        
        # Actually, if Phase 1 failed, we are at a local max for additions.
        # We need to break the deadlock.
        
        # Let's try:
        # For a few random x in S:
        #   Let S' = S - {x}
        #   Find all y not in S such that S' + {y} is valid.
        #   For each such y, let S'' = S' + {y}
        #   Find all z not in S'' such that S'' + {z} is valid.
        #   If found, accept S'' + {z}.
        
        # This is O(N * K_y * K_z).
        
        # Let's limit the number of x we try to, say, 10 random ones.
        
        tried_x = set()
        for _ in range(10):
            if time.time() > deadline: break
            # Pick random x in S
            # Extract indices
            bits = []
            temp = current_mask
            i = 0
            while temp:
                if temp & 1:
                    bits.append(i + 1)
                temp >>= 1
                i += 1
            if not bits: break
            x = random.choice(bits)
            if x in tried_x: continue
            tried_x.add(x)
            
            mask_without_x = current_mask & ~(1 << (x - 1))
            
            # Find valid y's
            valid_ys = []
            for y in range(1, N + 1):
                if (mask_without_x >> (y - 1)) & 1:
                    continue
                if is_valid_add(mask_without_x, y):
                    valid_ys.append(y)
            
            if len(valid_ys) < 2:
                continue
                
            # Try to find z for some y
            # Randomize valid_ys to avoid bias
            random.shuffle(valid_ys)
            
            found = False
            for y in valid_ys[:20]: # Limit checks
                mask_with_y = mask_without_x | (1 << (y - 1))
                for z in range(1, N + 1):
                    if z == y: continue
                    if (mask_with_y >> (z - 1)) & 1:
                        continue
                    if is_valid_add(mask_with_y, z):
                        # Accept
                        current_mask = mask_with_y | (1 << (z - 1))
                        current_size += 1
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if improved:
            continue
            
        # Phase 3: Try to remove 2, add 3
        # Similar logic, but more expensive.
        # If time permits.
        
        # If no improvement in Phase 2, try Phase 3.
        # Or just restart with a different random seed/greedy order?
        # Let's try a simple "kick": remove a random element, then run Phase 1.
        # This might allow new additions.
        
        # But we must maintain size? No, we want to maximize size.
        # Removing an element decreases size.
        # So we should only do this if it leads to a net gain.
        
        # Let's stick to Phase 2 for now. If it fails, we might be stuck.
        # We can break and try a new greedy start.
        
        # To avoid infinite loop if stuck:
        # If we did 100 iterations without improvement, break.
        # We need a counter.
        
        # For simplicity in this rewrite, let's just break if Phase 2 fails after some attempts.
        # The outer loop in construct() will handle restarts.
        break

    return current_mask, current_size

def mask_to_set(mask):
    return [i + 1 for i in range(N) if (mask >> i) & 1]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_mask = 0
    best_size = 0
    
    # 1. Initial Greedy with different orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(1, N + 1, 2)),
        list(range(2, N + 1, 2)),
    ]
    
    # Add a few random orders
    for _ in range(5):
        random.seed(int(time.time() * 1000) + _)
        r_order = list(range(1, N + 1))
        random.shuffle(r_order)
        orders.append(r_order)
        
    for order in orders:
        if time.time() > deadline - 10:
            break
        m = greedy_construct(order, deadline - 5)
        s = bin(m).count("1")
        if s > best_size:
            best_size = s
            best_mask = m
            
    # 2. Local Search on best candidate
    if time.time() < deadline - 5:
        improved_mask, improved_size = local_search(best_mask, deadline - 2)
        if improved_size > best_size:
            best_size = improved_size
            best_mask = improved_mask
            
    # 3. Try a few more random greedy starts + local search if time permits
    attempts = 0
    while time.time() < deadline - 2:
        random.seed(attempts + 100)
        r_order = list(range(1, N + 1))
        random.shuffle(r_order)
        m = greedy_construct(r_order, deadline - 5)
        s = bin(m).count("1")
        
        # Only do local search if it's promising or if we have lots of time
        if s > best_size or (s >= best_size - 2 and time.time() < deadline - 10):
            improved_mask, improved_size = local_search(m, deadline - 2)
            if improved_size > best_size:
                best_size = improved_size
                best_mask = improved_mask
        attempts += 1
        
    return mask_to_set(best_mask)

# EVOLVE-BLOCK-END
