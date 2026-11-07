import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin
    
    N = 256
    numbers = list(range(2, N + 1))
    nums_list = list(numbers)
    
    # Precompute divisors for products
    max_prod = 256 * 255
    divs = [set() for _ in range(max_prod + 1)]
    for a in range(2, N + 1):
        for multiple in range(a, max_prod + 1, a):
            divs[multiple].add(a)
    
    # Build conflict structure: for each pair (b,c) with b!=c, which a's divide b*c?
    # conflict_mask[x] is a bitmask where bit i is set if x conflicts with numbers[i] in some triple
    # Actually, let's use a different approach: for each x, precompute which (b,c) pairs cause x to conflict
    # i.e., x in divs[b*c]
    
    # For can_add(S, x): check if there exist distinct b,c in S such that x in divs[b*c]
    # Precompute: for each x, a set of pairs (b,c) where b!=c and x divides b*c
    # But that's too large. Instead, for each x, precompute a list of b such that there exists c with x|b*c
    # Even better: for each x, precompute a bitmask over numbers where bit b is set if there exists c != b with x | b*c
    
    # Let's use a simpler but faster approach:
    # For each x in [2,256], precompute a set of "conflict partners" 
    # Actually, the condition is: x cannot be added if there exist b,c in S, b!=c, with x | b*c
    # This is equivalent to: the set {b in S : exists c in S, c!=b, x|b*c} has size >= 2
    # Wait no, it's: there exist DISTINCT b,c in S such that x | b*c
    # So for a given x and set S, we need to check if any pair (b,c) from S (b!=c) has x | b*c
    
    # Precompute for each x: a dictionary mapping b to the set of c values such that x | b*c
    # Then for can_add: for each b in S, check if any c in S (c != b) is in conflict_c[x][b]
    
    # Alternative: precompute a 256x256 boolean matrix conflict[x][b] = set of c such that x|b*c
    # But memory... 256*256 = 65536 entries, each a set. That's fine.
    
    # Let me use a different representation:
    # For each x, create a list of bitmasks. For each b, bitmask_x_b has bit c set if x | b*c
    # Then can_add(S, x): for each b in S, if (S_mask & bitmask_x[b]) has at least one bit set that's not b itself... 
    # Wait, we need c != b. So: (S_mask & bitmask_x[b]) & ~(1 << b) != 0
    
    # Let's index numbers 0..255 corresponding to values 2..256
    # num_to_idx: value -> index (0-based)
    num_to_idx = {v: v - 2 for v in range(2, 257)}
    idx_to_num = list(range(2, 257))
    
    # For each x (value), precompute bitmask_x[b_idx] = bitmask of c_idx such that x | b*c
    # x ranges over 2..256, b ranges over 2..256, c ranges over 2..256
    # That's 255 * 255 * 255 checks = ~16.6M. Might be slow in Python.
    
    # Better: for each x, for each b, the condition x | b*c means c is a multiple of (x * b) / gcd(x,b) / something...
    # x | b*c iff x/gcd(x,b) | c
    # So for given x and b, c must be a multiple of x/gcd(x,b)
    
    import math
    
    # Precompute: for each x in 2..256, for each b in 2..256:
    #   g = gcd(x, b)
    #   req = x // g
    #   c must be a multiple of req
    #   So c_idx such that idx_to_num[c_idx] is a multiple of req
    #   bitmask_x[b_idx] = bitmask of all c_idx where idx_to_num[c_idx] % req == 0
    
    # Let's precompute multiples bitmasks
    # mult_mask[d] = bitmask of indices i where idx_to_num[i] is a multiple of d
    max_val = 256
    mult_mask = [0] * (max_val + 1)
    for d in range(1, max_val + 1):
        mask = 0
        for m in range(d, max_val + 1, d):
            if m >= 2 and m <= 256:
                idx = m - 2
                mask |= (1 << idx)
        mult_mask[d] = mask
    
    # Now for each x, precompute bitmask_x as a list of 255 bitmasks (one per b)
    # bitmask_x[b_idx] = mult_mask[x // gcd(x, b)]
    bitmask_x = {}
    for x in range(2, 257):
        bm_list = []
        for b in range(2, 257):
            g = math.gcd(x, b)
            req = x // g
            bm_list.append(mult_mask[req])
        bitmask_x[x] = bm_list
    
    def can_add_fast(S_mask, x):
        """
        Check if x can be added given current set represented as S_mask (bitmask).
        x is the value (2..256).
        Returns True if x can be added (no conflict).
        """
        bm_list = bitmask_x[x]
        x_idx = x - 2
        for b in range(2, 257):
            b_idx = b - 2
            if not (S_mask & (1 << b_idx)):
                continue
            # Check if there exists c in S, c != b, such that x | b*c
            # c must be in mult_mask[req] where req = x//gcd(x,b)
            # We need c in S and c != b
            candidates = S_mask & bm_list[b_idx]
            # Remove b itself if it's in candidates
            candidates &= ~(1 << b_idx)
            if candidates != 0:
                return False
        return True
    
    def is_valid_mask(S_mask):
        """Check if the set represented by S_mask is valid."""
        # For each a in S, check that no pair b,c in S (b!=c, both != a) has a | b*c
        # But actually the condition is: for all DISTINCT a,b,c in S, a does not divide b*c
        # So a, b, c are all distinct
        # For each a in S: for each pair (b,c) in S with b!=c, b!=a, c!=a: check a !| b*c
        # Equivalently: for each a in S, let S_without_a = S \ {a}
        # Check that no pair (b,c) from S_without_a with b!=c has a | b*c
        # This is the same as can_add_fast(S_without_a_mask, a) but with the twist that
        # we're checking a against S_without_a, not adding a to S_without_a
        # Actually it's the same check!
        
        for a in range(2, 257):
            a_idx = a - 2
            if not (S_mask & (1 << a_idx)):
                continue
            S_without_a = S_mask & ~(1 << a_idx)
            if not can_add_fast(S_without_a, a):
                return False
        return True
    
    def mask_to_list(S_mask):
        return [idx_to_num[i] for i in range(255) if S_mask & (1 << i)]
    
    def list_to_mask(lst):
        mask = 0
        for v in lst:
            mask |= (1 << (v - 2))
        return mask
    
    # 1. Greedy Initialization with multiple orders
    best_mask = 0
    best_size = 0
    
    orders = [
        list(range(256, 1, -1)),  # Descending
        list(range(2, 257)),      # Ascending
        [x for x in range(256, 1, -1) if x % 2 != 0] + [x for x in range(256, 1, -1) if x % 2 == 0],
        [x for x in range(2, 257) if x % 2 == 0] + [x for x in range(2, 257) if x % 2 != 0],
    ]
    
    for order in orders:
        S_mask = 0
        S_size = 0
        for x in order:
            if can_add_fast(S_mask, x):
                S_mask |= (1 << (x - 2))
                S_size += 1
        if S_size > best_size:
            best_size = S_size
            best_mask = S_mask
    
    # 2. Local Search with bitmask operations
    current_mask = best_mask
    current_size = best_size
    
    random.seed(42)
    
    no_improve_count = 0
    max_no_improve = 500
    
    all_indices = list(range(255))
    
    while time.time() < DEADLINE:
        if no_improve_count > max_no_improve:
            # Kick: remove 2-3 random elements, try to add
            if current_size >= 3:
                num_remove = random.randint(2, 3)
                current_indices = [i for i in range(255) if current_mask & (1 << i)]
                to_remove = random.sample(current_indices, num_remove)
                temp_mask = current_mask
                for i in to_remove:
                    temp_mask &= ~(1 << i)
                
                # Try to add random elements
                candidates = [i for i in range(255) if not (temp_mask & (1 << i))]
                random.shuffle(candidates)
                
                added = 0
                for i in candidates:
                    if added >= num_remove:
                        break
                    x = idx_to_num[i]
                    if can_add_fast(temp_mask, x):
                        temp_mask |= (1 << i)
                        added += 1
                
                temp_size = bin(temp_mask).count('1')
                if temp_size > current_size:
                    current_mask = temp_mask
                    current_size = temp_size
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            continue
        
        # 1-Swap Move
        current_indices = [i for i in range(255) if current_mask & (1 << i)]
        absent_indices = [i for i in range(255) if not (current_mask & (1 << i))]
        
        if not current_indices or not absent_indices:
            break
        
        x_idx = random.choice(current_indices)
        y_idx = random.choice(absent_indices)
        
        new_mask = (current_mask & ~(1 << x_idx)) | (1 << y_idx)
        
        # Check validity of new_mask
        if is_valid_mask(new_mask):
            # Try to greedily add more elements
            improved = False
            # Try adding in descending order first, then ascending
            for order in [range(254, -1, -1), range(255)]:
                for i in order:
                    if not (new_mask & (1 << i)):
                        x = idx_to_num[i]
                        if can_add_fast(new_mask, x):
                            new_mask |= (1 << i)
                            improved = True
            
            new_size = bin(new_mask).count('1')
            if new_size > current_size:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                no_improve_count = 0
            elif improved:
                current_mask = new_mask
                no_improve_count = 0
            else:
                # Accept with some probability for exploration
                if random.random() < 0.3:
                    current_mask = new_mask
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    # Final polish: try to add any remaining elements in various orders
    final_mask = best_mask
    for order in [range(254, -1, -1), range(255), random.sample(range(255), 255)]:
        for i in order:
            if not (final_mask & (1 << i)):
                x = idx_to_num[i]
                if can_add_fast(final_mask, x):
                    final_mask |= (1 << i)
    
    final_size = bin(final_mask).count('1')
    if final_size > best_size:
        best_mask = final_mask
        best_size = final_size
    
    # Convert to list
    result = mask_to_list(best_mask)
    result.sort()
    
    # Final validation
    if not is_valid_mask(best_mask):
        # Fallback
        S_mask = 0
        for x in range(256, 1, -1):
            if can_add_fast(S_mask, x):
                S_mask |= (1 << (x - 2))
        result = mask_to_list(S_mask)
        result.sort()
    
    return result
