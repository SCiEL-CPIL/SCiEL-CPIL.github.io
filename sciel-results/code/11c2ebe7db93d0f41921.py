"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    # Bitmask representation of the set S
    # Bit i represents integer i+1
    # S = {x | (mask >> (x-1)) & 1 == 1}
    
    def get_size(mask):
        return bin(mask).count("1")
    
    def is_valid_addition(x, S_mask):
        """
        Check if adding x to S_mask keeps it valid.
        x is 1-indexed.
        """
        # Case 1: x is one of the elements {a, b, c}.
        # Let a = x. Need b, c in S, b != c.
        # Sums: x+b, x+c, b+c must be in S U {x}.
        # x+b > x, so x+b must be in S.
        # x+c > x, so x+c must be in S.
        # b+c can be in S or equal to x.
        
        # Find B = { y in S | x+y in S }
        # y ranges from 1 to 128-x
        B_bits = 0
        # Iterate through set bits in S_mask
        # To speed up, we can just iterate 1..128-x
        # But S is sparse enough? No, S is dense in upper half.
        # Let's iterate through possible y
        for y in range(1, n - x + 1):
            # Check if y in S
            if (S_mask >> (y - 1)) & 1:
                # Check if x+y in S
                if (S_mask >> (x + y - 1)) & 1:
                    B_bits |= (1 << (y - 1))
        
        # If |B| >= 2, check pairs
        if B_bits:
            # Get list of y in B
            Bs = []
            temp = B_bits
            while temp:
                lsb = temp & -temp
                idx = (lsb.bit_length() - 1) + 1
                Bs.append(idx)
                temp ^= lsb
            
            if len(Bs) >= 2:
                for i in range(len(Bs)):
                    for j in range(i + 1, len(Bs)):
                        b = Bs[i]
                        c = Bs[j]
                        s3 = b + c
                        # s3 must be in S or s3 == x
                        if s3 == x:
                            return False
                        elif s3 <= n:
                            if (S_mask >> (s3 - 1)) & 1:
                                return False
                        # else s3 > n, so not in S U {x} (since x < s3 usually? No, x is fixed)
                        # If s3 > n, it's not in S. If s3 != x, it's not in S U {x}.
                        # So condition holds: must be in S. If not, no conflict from this pair.
        
        # Case 2: x is one of the sums.
        # a+b = x. a, b in S, a != b.
        # Need c in S, c != a, c != b.
        # a+c in S U {x}, b+c in S U {x}.
        # a+c = x => c = x-a = b. But c != b. So a+c != x. Thus a+c in S.
        # Similarly b+c in S.
        
        # Find pairs (a, b) in S such that a+b=x
        # a < b < x
        for a in range(1, x // 2 + 1):
            b = x - a
            if a == b:
                continue
            # Check if a, b in S
            if ((S_mask >> (a - 1)) & 1) and ((S_mask >> (b - 1)) & 1):
                # We have a valid pair (a, b) summing to x
                # Check if there exists c in S \ {a, b} such that a+c in S and b+c in S
                # c ranges 1..n
                # Optimization: 
                # N_a = { c | a+c in S }
                # N_b = { c | b+c in S }
                # We need intersection of N_a and N_b, excluding a, b, and intersected with S.
                
                # Let's just iterate c
                # c must be in S
                # c != a, c != b
                # a+c in S, b+c in S
                found_c = False
                # Iterate through set bits in S_mask
                temp = S_mask
                while temp:
                    lsb = temp & -temp
                    c = (lsb.bit_length() - 1) + 1
                    temp ^= lsb
                    
                    if c == a or c == b:
                        continue
                    
                    # Check a+c in S
                    if a + c <= n and ((S_mask >> (a + c - 1)) & 1):
                        # Check b+c in S
                        if b + c <= n and ((S_mask >> (b + c - 1)) & 1):
                            found_c = True
                            break
                
                if found_c:
                    return False
        
        return True

    # Initial Set: Upper half
    S_mask = 0
    for i in range(65, n + 1):
        S_mask |= (1 << (i - 1))
    
    best_size = get_size(S_mask)
    best_mask = S_mask
    
    # Greedy Phase
    # Try adding 1 to 64
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S_mask):
            S_mask |= (1 << (x - 1))
            
    if get_size(S_mask) > best_size:
        best_size = get_size(S_mask)
        best_mask = S_mask
        
    # Local Search Phase
    # Try to add any remaining element
    # If stuck, try swaps?
    # For now, let's do multiple random greedy restarts if time permits
    # But the current greedy is deterministic.
    # Let's try a simple local search:
    # 1. Try to add any x not in S.
    # 2. If none, try to remove one y in S and see if we can add two x, z?
    #    This is complex.
    #    Alternative: Randomized Greedy.
    
    # Let's implement a few randomized greedy runs
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Run several random orders
        candidates = list(range(1, 65))
        for _ in range(10): # Adjust based on time
            if time.time() > deadline - 2:
                break
            random.shuffle(candidates)
            curr_mask = 0
            for i in range(65, n + 1):
                curr_mask |= (1 << (i - 1))
            
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                if is_valid_addition(x, curr_mask):
                    curr_mask |= (1 << (x - 1))
            
            if get_size(curr_mask) > best_size:
                best_size = get_size(curr_mask)
                best_mask = curr_mask
                
    # Convert best_mask to list of ints
    result = []
    for i in range(1, n + 1):
        if (best_mask >> (i - 1)) & 1:
            result.append(i)
            
    return result
# EVOLVE-BLOCK-END
