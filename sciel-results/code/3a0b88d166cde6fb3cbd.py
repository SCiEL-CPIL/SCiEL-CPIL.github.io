import time
import random

def construct():
    n = 512
    deadline = time.time() + 55.0  # Safety margin for 90s limit
    
    # Precompute conflict masks
    # conflicts[i] is a list of bitmasks. If (S & mask) == mask, element i (0-indexed) cannot be added.
    # Bit i in the mask corresponds to element i+1 (1-indexed).
    conflicts = [[] for _ in range(n)]
    
    for x in range(1, n + 1):
        i = x - 1
        # Case 1: x is 1st term: x, x+d, x+2d, x+3d
        # Need x+d, x+2d, x+3d in S
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            y1 = x + d
            y2 = x + 2 * d
            y3 = x + 3 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 2: x is 2nd term: x-d, x, x+d, x+2d
        # Need x-d, x+d, x+2d in S
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            y1 = x - d
            y2 = x + d
            y3 = x + 2 * d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 3: x is 3rd term: x-2d, x-d, x, x+d
        # Need x-2d, x-d, x+d in S
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            y1 = x - 2 * d
            y2 = x - d
            y3 = x + d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)
            
        # Case 4: x is 4th term: x-3d, x-2d, x-d, x
        # Need x-3d, x-2d, x-d in S
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            y1 = x - 3 * d
            y2 = x - 2 * d
            y3 = x - d
            mask = (1 << (y1 - 1)) | (1 << (y2 - 1)) | (1 << (y3 - 1))
            conflicts[i].append(mask)

    def can_add(x_idx, S):
        """Check if element at index x_idx (0-based) can be added to bitset S."""
        for mask in conflicts[x_idx]:
            if (S & mask) == mask:
                return False
        return True

    def bit_count(S):
        """Count bits in S."""
        return bin(S).count('1')

    def bitset_to_list(S):
        """Convert bitset to list of 1-based indices."""
        result = []
        for i in range(n):
            if S & (1 << i):
                result.append(i + 1)
        return result

    def greedy_fill(S, order=None):
        """Fill S with all possible elements greedily. Returns new S."""
        if order is None:
            order = list(range(n))
        for i in order:
            if not (S & (1 << i)):
                if can_add(i, S):
                    S |= (1 << i)
        return S

    def random_greedy():
        """Perform a greedy fill with a random order."""
        S = 0
        order = list(range(n))
        random.shuffle(order)
        S = greedy_fill(S, order)
        return S

    # Base-3 no-zero-digit construction
    def base3_no_zero(x):
        if x == 0:
            return True
        while x > 0:
            if x % 3 == 0:
                return False
            x //= 3
        return True

    # Initialization: Try several diverse starting points
    random.seed(42)
    
    # 1. Base-3 set
    S_init = 0
    for x in range(1, n + 1):
        if base3_no_zero(x):
            S_init |= (1 << (x - 1))
    S_best = greedy_fill(S_init)
    best_size = bit_count(S_best)
    
    # 2. Multiple random greedy starts
    num_init_tries = 20
    for _ in range(num_init_tries):
        if time.time() > deadline - 40: # Reserve 40s for search
            break
        S_curr = random_greedy()
        curr_size = bit_count(S_curr)
        if curr_size > best_size:
            best_size = curr_size
            S_best = S_curr

    # Local Search: Iterated Local Search with Restarts
    S_current = S_best
    best_list = bitset_to_list(S_best)
    
    # Stagnation counter
    stagnation_count = 0
    max_stagnation = 50 # Restart if no improvement for 50 kicks
    
    while time.time() < deadline - 5:
        # 1. Greedy fill current set
        S_filled = greedy_fill(S_current)
        filled_size = bit_count(S_filled)
        
        if filled_size > best_size:
            best_size = filled_size
            S_best = S_filled
            best_list = bitset_to_list(S_best)
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        S_current = S_filled
        
        # 2. Check for restart
        if stagnation_count >= max_stagnation:
            # Restart from a new random greedy fill
            S_current = random_greedy()
            stagnation_count = 0
            continue
            
        # 3. Kick: Remove k random elements
        # Dynamic kick size: larger if stuck longer
        k = random.randint(2, 4 + stagnation_count // 10)
        if k > filled_size:
            k = filled_size // 2
            
        if k > 0:
            set_bits = []
            temp = S_current
            for i in range(n):
                if temp & (1 << i):
                    set_bits.append(i)
            
            if len(set_bits) > k:
                to_remove = random.sample(set_bits, k)
                for x in to_remove:
                    S_current &= ~(1 << x)
    
    # Final greedy pass on best
    S_final = greedy_fill(S_best)
    if bit_count(S_final) > best_size:
        best_size = bit_count(S_final)
        best_list = bitset_to_list(S_final)

    return sorted(best_list)
