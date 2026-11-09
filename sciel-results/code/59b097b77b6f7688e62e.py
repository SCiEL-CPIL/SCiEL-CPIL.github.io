import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    all_elements = list(range(1, n + 1))
    
    def popcount(mask):
        return bin(mask).count('1')
    
    def is_valid_addition_fast(mask, x):
        """
        Fast check: can we add x to the set represented by mask?
        Returns True if valid.
        """
        bit_x = 1 << (x - 1)
        if mask & bit_x:
            return True
        
        # Case 1: x is a vertex c. Need a,b in S with a+x in S, b+x in S, and a+b in S (or a+b==x)
        C_a_mask = mask & (mask >> x)
        
        # Get list of candidates from C_a_mask
        candidates = []
        temp = C_a_mask
        while temp:
            low_bit = temp & (-temp)
            idx = low_bit.bit_length() - 1
            a = idx + 1
            if a != x:
                candidates.append(a)
            temp ^= low_bit
        
        k = len(candidates)
        if k >= 2:
            for i in range(k):
                a = candidates[i]
                for j in range(i+1, k):
                    b = candidates[j]
                    s = a + b
                    if s == x:
                        return False
                    if s <= n and (mask >> (s-1)) & 1:
                        return False
        
        # Case 2: x is a sum. x = a+b for some a,b in S.
        for a in range(1, x//2 + 1):
            b = x - a
            if b <= a:
                continue
            if (mask >> (a-1)) & 1 and (mask >> (b-1)) & 1:
                C_a = mask & (mask >> a)
                C_b = mask & (mask >> b)
                potential_c = C_a & C_b
                potential_c &= ~(1 << (a-1))
                potential_c &= ~(1 << (b-1))
                if potential_c != 0:
                    return False
        
        return True
    
    def greedy_fill(mask):
        """Try to greedily add elements in order."""
        for x in range(1, n+1):
            if not (mask >> (x-1)) & 1:
                if is_valid_addition_fast(mask, x):
                    mask |= (1 << (x-1))
        return mask
    
    def local_search(initial_mask, time_limit):
        """Run local search with add/remove moves."""
        current = initial_mask
        best = initial_mask
        best_size = popcount(best)
        current_size = popcount(current)
        
        end_time = time.time() + time_limit
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < end_time and no_improve < max_no_improve:
            improved = False
            no_improve += 1
            
            # Try adding a random element
            for _ in range(5):
                x = random.randint(1, n)
                bit_x = 1 << (x-1)
                if not (current >> (x-1)) & 1:
                    if is_valid_addition_fast(current, x):
                        current |= bit_x
                        current_size += 1
                        improved = True
                        if current_size > best_size:
                            best = current
                            best_size = current_size
                            no_improve = 0
                        break
            
            if improved:
                continue
            
            # Try removing a random element to enable future additions
            removed_any = False
            for _ in range(3):
                idx = random.randint(0, n-1)
                if (current >> idx) & 1:
                    x = idx + 1
                    current &= ~(1 << idx)
                    current_size -= 1
                    removed_any = True
                    break
            
            if not removed_any:
                break
        
        return best, best_size
    
    # Strategy: multiple restarts with different initial sets
    best_mask = 0
    best_size = 0
    
    # Try multiple initial strategies
    initial_strategies = []
    
    # Upper half
    m1 = 0
    for i in range(65, 129):
        m1 |= (1 << (i-1))
    initial_strategies.append(m1)
    
    # Upper third
    m2 = 0
    for i in range(85, 129):
        m2 |= (1 << (i-1))
    initial_strategies.append(m2)
    
    # Random with higher density
    for _ in range(3):
        mr = 0
        for i in range(1, n+1):
            if random.random() < 0.5:
                mr |= (1 << (i-1))
        initial_strategies.append(mr)
    
    # Process each strategy
    for init in initial_strategies:
        if time.time() > deadline - 5:
            break
        
        # Greedy fill
        g_mask = greedy_fill(init)
        g_size = popcount(g_mask)
        if g_size > best_size:
            best_mask = g_mask
            best_size = g_size
        
        # Local search
        if deadline - start_time - time.time() > 3:
            ls_mask, ls_size = local_search(g_mask, 2)
            if ls_size > best_size:
                best_mask = ls_mask
                best_size = ls_size
    
    # Final polish: try to add any remaining valid elements
    if time.time() < deadline - 1:
        for x in range(1, n+1):
            if not (best_mask >> (x-1)) & 1:
                if is_valid_addition_fast(best_mask, x):
                    best_mask |= (1 << (x-1))
                    best_size += 1
    
    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return result
