# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct a large 4-term-progression-free subset of [1, 512].
    Strategy:
      1. Use bitset representation for fast set operations.
      2. Precompute conflict masks for each element.
      3. Initialize with multiple randomized greedy passes.
      4. Improve via local search: try additions, and if stuck, explore 
         same-size valid configurations via random 1-swaps to find 
         a state where an addition is possible.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    n = 512
    # Elements are 1..512, so we use bits 1..512 in an integer.
    # Bit 0 is unused.

    def time_up():
        return time.time() - start_time > deadline

    # Precompute conflict masks
    # conflict_masks[x] is a list of integers.
    # Each integer is a bitmask of the other 3 elements in a 4-AP containing x.
    # If S & mask == mask, then x conflicts with S.
    conflict_masks = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            a1 = a
            a2 = a + d
            a3 = a + 2 * d
            a4 = a + 3 * d
            
            # For each element in the AP, the other 3 form a conflict mask
            # For a1: mask has a2, a3, a4
            mask_a1 = (1 << a2) | (1 << a3) | (1 << a4)
            conflict_masks[a1].append(mask_a1)
            
            # For a2: mask has a1, a3, a4
            mask_a2 = (1 << a1) | (1 << a3) | (1 << a4)
            conflict_masks[a2].append(mask_a2)
            
            # For a3: mask has a1, a2, a4
            mask_a3 = (1 << a1) | (1 << a2) | (1 << a4)
            conflict_masks[a3].append(mask_a3)
            
            # For a4: mask has a1, a2, a3
            mask_a4 = (1 << a1) | (1 << a2) | (1 << a3)
            conflict_masks[a4].append(mask_a4)

    def is_valid_add(S, x):
        """Check if x can be added to S without creating a 4-AP."""
        # x is not in S (assumed)
        for mask in conflict_masks[x]:
            if (S & mask) == mask:
                return False
        return True

    def bit_count(x):
        return bin(x).count('1')

    def random_greedy():
        """Perform one random greedy construction."""
        S = 0
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        for x in elements:
            if is_valid_add(S, x):
                S |= (1 << x)
        return S

    # Step 1: Initialization with multiple random greedy restarts
    best_S = 0
    best_size = 0
    num_restarts = 200
    
    for i in range(num_restarts):
        if time_up():
            break
        S = random_greedy()
        size = bit_count(S)
        if size > best_size:
            best_size = size
            best_S = S

    # Step 2: Local Search
    # We try to increase the size. If we can't add any element, we do swaps.
    S = best_S
    current_size = best_size
    elements_list = list(range(1, n + 1))
    
    # Cache of elements currently in S for faster random selection?
    # We can just pick random x and check if it's in S.
    
    # To speed up "pick random element not in S", we can maintain a list, 
    # but updating it on every swap is O(N). 
    # Given N=512, O(N) is very fast.
    
    # We will use a simple loop.
    # To avoid getting stuck in an infinite loop of swaps that don't help,
    # we limit the number of swaps between successful additions.
    
    max_swaps_per_add = 5000
    swaps_since_add = 0
    no_improvement_count = 0
    
    while not time_up():
        # Try to add a random element
        added = False
        # Try a few random elements for addition
        # Instead of 1, try a batch to increase chance
        for _ in range(10):
            if time_up():
                break
            x = random.choice(elements_list)
            if not (S & (1 << x)):  # x not in S
                if is_valid_add(S, x):
                    S |= (1 << x)
                    current_size += 1
                    added = True
                    swaps_since_add = 0
                    no_improvement_count = 0
                    break
        
        if added:
            continue
        
        # If no element could be added, try to improve the configuration via swaps
        # We do a random 1-swap: remove u, add v.
        # This keeps size constant but changes the structure.
        
        # Pick u in S
        # Efficient way: pick random index, check if bit is set.
        u = None
        for _ in range(10): # Try a few times to find a u in S
            if time_up():
                break
            candidate = random.choice(elements_list)
            if S & (1 << candidate):
                u = candidate
                break
        
        if u is None:
            # S is empty? Unlikely.
            break
            
        # Pick v not in S
        v = None
        for _ in range(10):
            if time_up():
                break
            candidate = random.choice(elements_list)
            if not (S & (1 << candidate)):
                v = candidate
                break
                
        if v is None:
            # S is full?
            break

        # Perform swap: S' = S \ {u} U {v}
        # Check if v is valid in S \ {u}
        S_temp = (S & ~(1 << u)) | (1 << v)
        
        # We need to check if v conflicts with S \ {u}.
        # This is equivalent to checking if v conflicts with S, 
        # EXCEPT for conflicts that were "resolved" by removing u.
        # is_valid_add(S_temp, v) checks if v conflicts with S_temp.
        # But S_temp contains v. So we should check if v conflicts with (S_temp \ {v}).
        # Which is S \ {u}.
        
        # So we check: is v valid in S \ {u}?
        # We can reuse the logic of is_valid_add but on S \ {u}.
        
        valid = True
        S_minus_u = S & ~(1 << u)
        for mask in conflict_masks[v]:
            if (S_minus_u & mask) == mask:
                valid = False
                break
        
        if valid:
            S = S_temp
            swaps_since_add += 1
            if swaps_since_add > max_swaps_per_add:
                # If we've done many swaps without adding, 
                # maybe this local optimum is deep. 
                # We could restart local search or just continue.
                # Let's just continue, but maybe increase max_swaps?
                # For now, keep going.
                pass
        else:
            # Swap not possible, try next iteration
            pass
            
        # If we've done a lot of swaps without improvement, 
        # we might be stuck. 
        # We can try to restart the local search from a different random greedy solution?
        # Or just keep trying.
        
        # To prevent infinite loop on a dead-end local optimum, 
        # we can track if we've seen this S before? (Too expensive to hash big ints frequently)
        # Instead, we rely on the time limit.
        
    return sorted([i for i in range(1, n + 1) if S & (1 << i)])

# EVOLVE-BLOCK-END
