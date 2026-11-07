import math
import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 256
    numbers = list(range(2, N + 1))
    
    # Precompute mults[k]: bitmask of numbers in 2..256 that are multiples of k
    mults = [0] * (N + 1)
    for k in range(2, N + 1):
        mask = 0
        for m in range(k, N + 1, k):
            mask |= (1 << m)
        mults[k] = mask

    def can_add(x, M):
        """
        Check if x can be added to set M (represented as bitmask).
        M must not contain x.
        """
        m = M
        while m:
            lsb = m & -m
            b = lsb.bit_length() - 1
            m ^= lsb
            
            # Check 1: x | b*c for some c in M, c != b
            # x | b*c  <=>  (x/gcd(x,b)) | c
            g = math.gcd(x, b)
            x_prime = x // g
            # We need c in M, c != b, such that c is multiple of x_prime
            if (M & mults[x_prime] & ~(1 << b)) != 0:
                return False
            
            # Check 2: b | x*c for some c in M, c != b
            # b | x*c  <=>  (b/gcd(b,x)) | c
            b_prime = b // g
            if (M & mults[b_prime] & ~(1 << b)) != 0:
                return False
                
        return True

    def is_valid(M):
        """Check if set M is valid."""
        m = M
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            # Check if a can be added to M \ {a}
            if not can_add(a, M ^ (1 << a)):
                return False
        return True

    # Greedy initialization
    def greedy_build(order):
        M = 0
        for x in order:
            if (M >> x) & 1:
                continue
            if can_add(x, M):
                M |= (1 << x)
        return M

    best_M = 0
    best_size = 0
    
    # Multiple restarts with different orders
    orders = [
        sorted(numbers, reverse=True),
        sorted(numbers),
    ]
    
    # Add some random orders
    for _ in range(20):
        rnd_order = numbers[:]
        random.shuffle(rnd_order)
        orders.append(rnd_order)

    for order in orders:
        if time.time() > DEADLINE - 5.0:
            break
        M = greedy_build(order)
        size = bin(M).count("1")
        if size > best_size:
            best_size = size
            best_M = M

    # Local Search on best_M
    current_M = best_M
    current_size = best_size
    
    improved = True
    while improved and time.time() < DEADLINE - 2.0:
        improved = False
        
        # Phase 1: Try to add any element
        for x in numbers:
            if (current_M >> x) & 1:
                continue
            if can_add(x, current_M):
                current_M |= (1 << x)
                current_size += 1
                improved = True
                break
        
        if time.time() > DEADLINE - 2.0:
            break
            
        # Phase 2: If no element could be added, try 1-swap
        if not improved:
            swaps_tried = 0
            max_swaps = 5000
            
            in_M = []
            not_in_M = []
            for i in range(2, N + 1):
                if (current_M >> i) & 1:
                    in_M.append(i)
                else:
                    not_in_M.append(i)
            
            random.shuffle(in_M)
            random.shuffle(not_in_M)
            
            for u in in_M:
                if swaps_tried >= max_swaps:
                    break
                for v in not_in_M:
                    if swaps_tried >= max_swaps:
                        break
                    swaps_tried += 1
                    
                    new_M = (current_M ^ (1 << u)) | (1 << v)
                    if is_valid(new_M):
                        current_M = new_M
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                 # Try a few random restarts from scratch if we have time
                 if time.time() < DEADLINE - 10.0:
                     for _ in range(10):
                         rnd_order = numbers[:]
                         random.shuffle(rnd_order)
                         M = greedy_build(rnd_order)
                         size = bin(M).count("1")
                         if size > current_size:
                             current_M = M
                             current_size = size
                             improved = True
                             break
                     if not improved:
                         break

    result = []
    for i in range(2, N + 1):
        if (current_M >> i) & 1:
            result.append(i)
            
    return result
