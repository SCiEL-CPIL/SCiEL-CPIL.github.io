import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 128
    
    # Precompute: for each x in 1..n, which pairs (a,b) with a<b in 1..n have a+b == x
    # and which pairs have a+x <= n (i.e., x is a "shift" that keeps sums in range)
    
    # We'll use bitmask representation: bit i represents element (i+1)
    # So element e is represented by bit (e-1)
    
    # Precompute for each possible x (1..n):
    # - pairs_sum[x] = list of (a, b) with a < b, a+b == x, both in 1..n
    # - shifts[x] = list of a in 1..n such that a+x <= n (so a and a+x could both be in S)
    
    pairs_sum = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            s = a + b
            if s <= n:
                pairs_sum[s].append((a, b))
    
    # Precompute shift lists: for each x, which a values satisfy a + x <= n
    shifts = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, n - x + 1):
            shifts[x].append(a)
    
    def is_valid_addition(mask, x):
        """
        Check if adding element x to the set represented by mask is valid.
        mask is a Python int where bit (e-1) is set iff e is in the set.
        x is an integer in 1..n, not currently in the set.
        """
        x_bit = 1 << (x - 1)
        
        # Case 1: x is one of {a, b, c} (say x = c)
        # Need: distinct a, b in S such that:
        #   (a+b == x OR a+b in S) AND (a+x in S) AND (b+x in S)
        # 
        # Let A = {a in S : a + x <= n AND (a+x) in S}
        # We need two distinct elements a, b in A such that a+b == x OR a+b in S.
        
        # Build list of a in S where a+x is also in S
        candidates = []
        for a in shifts[x]:
            a_bit = 1 << (a - 1)
            ax_bit = 1 << (a + x - 1)
            if (mask & a_bit) and (mask & ax_bit):
                candidates.append(a)
        
        if len(candidates) >= 2:
            for i in range(len(candidates)):
                a = candidates[i]
                for j in range(i + 1, len(candidates)):
                    b = candidates[j]
                    s = a + b
                    if s == x:
                        return False
                    if s <= n:
                        s_bit = 1 << (s - 1)
                        if mask & s_bit:
                            return False
        
        # Case 2: x is one of the sums {a+b, a+c, b+c}
        # Subcase 2a: x = a+b. Need c in S, c distinct from a,b, such that a+c in S and b+c in S.
        # For each pair (a,b) with a+b == x:
        #   Need c in S, c != a, c != b, such that (a+c) in S and (b+c) in S.
        
        for (a, b) in pairs_sum[x]:
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if not (mask & a_bit) or not (mask & b_bit):
                continue
            
            # For each c in S, c != a, c != b:
            # Check if (a+c) in S and (b+c) in S
            for c in shifts[max(a, b)]:  # c must satisfy c + max(a,b) <= n
                if c == a or c == b:
                    continue
                c_bit = 1 << (c - 1)
                if not (mask & c_bit):
                    continue
                ac_bit = 1 << (a + c - 1)
                bc_bit = 1 << (b + c - 1)
                if (mask & ac_bit) and (mask & bc_bit):
                    return False
        
        return True
    
    # Initialize with upper half: elements 65..128
    # Bits 64..127 set
    current_mask = 0
    for e in range(65, 129):
        current_mask |= (1 << (e - 1))
    
    best_mask = current_mask
    best_size = 64  # len of upper half
    
    # Track all elements 1..n
    all_elements = list(range(1, n + 1))
    
    attempts = 0
    max_attempts = 2000000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Find missing elements
        missing = []
        for e in all_elements:
            if not (current_mask & (1 << (e - 1))):
                missing.append(e)
        
        if missing:
            random.shuffle(missing)
            added = False
            for x in missing:
                if is_valid_addition(current_mask, x):
                    current_mask |= (1 << (x - 1))
                    new_size = bin(current_mask).count('1')
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = current_mask
                    added = True
                    break
            
            if added:
                continue
        
        # If no element could be added, remove a random element
        if current_mask:
            # Find a random element to remove
            present = []
            for e in all_elements:
                if current_mask & (1 << (e - 1)):
                    present.append(e)
            
            if present:
                to_remove = random.choice(present)
                current_mask &= ~(1 << (to_remove - 1))
        
        # Periodic restart if stuck for too long
        if attempts % 50000 == 0:
            # Try a different initial configuration
            if attempts % 100000 == 0:
                # Restart with random subset of size ~60
                current_mask = 0
                random_sample = random.sample(all_elements, 60)
                for e in random_sample:
                    current_mask |= (1 << (e - 1))
    
    # Convert best_mask to list of ints
    result = []
    for e in range(1, n + 1):
        if best_mask & (1 << (e - 1)):
            result.append(e)
    
    return result
