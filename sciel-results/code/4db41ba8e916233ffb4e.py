import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    N = 128
    # Universe is 1..128, map to 0..127 for bit operations
    # Bit i represents integer (i+1)
    
    def get_bitset(s):
        """Convert set of ints (1..128) to 128-bit int where bit (x-1) is set."""
        b = 0
        for x in s:
            b |= (1 << (x - 1))
        return b
    
    def bits_to_set(b):
        """Convert 128-bit int to set of ints (1..128)."""
        s = set()
        for i in range(N):
            if b & (1 << i):
                s.add(i + 1)
        return s
    
    # Precompute masks for each x in 1..128:
    # mask_pair_sum[x] has bit (a-1) set iff a+x is in 1..128 AND a+x is a valid sum target
    # Actually, we need to check if adding x creates a triple (a,b,c) where pair sums are in S.
    # A triple {a,b,c} is bad if a+b, a+c, b+c are all in S.
    # When we add x, the only new triples are those containing x: {x, a, b}.
    # These are bad if (x+a in S), (x+b in S), and (a+b in S).
    # So for a fixed x, the "conflict" pairs (a,b) are those where:
    #   a,b in S, a != b, a != x, b != x
    #   (x+a in S), (x+b in S), (a+b in S)
    
    # Let's define for each x, a bitmask M_x such that:
    #   M_x has bit (a-1) set iff (x+a is in 1..128)
    #   Actually, we need to intersect S with (S - x) and (S - x) and check pair sums.
    
    # Faster approach:
    # For a given set S (as bitset B) and candidate x (not in S):
    #   Let A = {a in S : x+a in S}  -> this is S & (S >> x) ... wait, bit shifts.
    #   If B is the bitset, then (B >> x) has bit i set iff bit (i+x) is set in B.
    #   So S ∩ (S - x) corresponds to bits where both bit i and bit (i+x) are set.
    #   That is: A_bits = B & (B >> x) ... but we need to be careful with indices.
    #   Let's say x is the value. We want a such that a in S and a+x in S.
    #   Bit for a is (a-1). Bit for a+x is (a+x-1).
    #   So if we shift B right by x, bit (a-1) in (B >> x) corresponds to bit (a-1+x) = bit (a+x-1) in B.
    #   So yes, A_bits = B & (B >> x).
    #   Now, for any pair (a, b) in A (a < b), we need to check if a+b is in S.
    #   This is still O(|A|^2). But |A| is typically small.
    
    # Even faster: Precompute for each x, and for each possible pair (a,b) in 1..128,
    # whether a+b is in 1..128. But that's static.
    
    # Let's use a different formulation:
    # A triple {a,b,c} is bad if all pair sums are in S.
    # Count the number of bad triples in S. If 0, valid.
    # When adding x, the new bad triples are those containing x.
    # New bad triples = number of pairs (a,b) in S such that x+a in S, x+b in S, a+b in S.
    
    # Let's precompute for each x, a list of "witness" pairs.
    # Actually, let's just compute the conflict count incrementally.
    
    # For each x in 1..128, precompute:
    #   partners[x] = set of a in 1..128 such that x+a <= 128
    #   sum_ok[a][b] = (a+b <= 128)
    
    # Let's try a direct but optimized local search.
    
    def count_conflicts(B, x):
        """
        Count how many pairs (a,b) in S (represented by B) would form a bad triple with x.
        i.e., a,b in S, a != b, a != x, b != x, and (x+a in S), (x+b in S), (a+b in S).
        Note: x is NOT in S when we call this for an "add" move.
        """
        if x in bits_to_set(B):  # Should not happen for add, but safe
            return 0
        # A = {a in S : x+a in S}
        # B has bits for S. B >> x has bit i set iff bit (i+x) set in B.
        # So A_bits = B & (B >> x)
        A_bits = B & (B >> x)
        # Now, for each a in A (a != x, a+x != x -> a != 0 which is true),
        # and b in A (b != a), check if a+b in S.
        # Also need a != x and b != x, but x not in B, so a,b not x.
        # Wait, a is in S, so a != x. Same for b.
        # So just iterate over a in A, b in A, a < b, and check if a+b in S.
        
        # Extract a's from A_bits
        # A_bits has bit (a-1) set.
        count = 0
        # To speed up, iterate over set bits of A_bits
        # A_bits is at most 128 bits.
        a_list = []
        temp = A_bits
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            a = idx + 1
            a_list.append(a)
            temp ^= lsb
        
        n = len(a_list)
        for i in range(n):
            a = a_list[i]
            # For each b > a in a_list
            for j in range(i + 1, n):
                b = a_list[j]
                s = a + b
                if s <= N:
                    # Check if s in S -> bit (s-1) in B
                    if B & (1 << (s - 1)):
                        count += 1
        return count
    
    # Initial set: upper half 65..128
    # This set has no bad triples because min pair sum is 65+66=131 > 128.
    init_set = set(range(65, 129))
    init_B = get_bitset(init_set)
    
    best_B = init_B
    best_size = len(init_set)
    best_set = init_set
    
    def is_valid_B(B):
        """Check if bitset B represents a valid set."""
        s = bits_to_set(B)
        sl = sorted(s)
        n = len(sl)
        for i in range(n):
            a = sl[i]
            for j in range(i + 1, n):
                b = sl[j]
                if a + b > N:
                    continue
                if not (B & (1 << (a + b - 1))):
                    continue
                for k in range(j + 1, n):
                    c = sl[k]
                    if a + c <= N and b + c <= N:
                        if (B & (1 << (a + c - 1))) and (B & (1 << (b + c - 1))):
                            return False
        return True
    
    # Simulated Annealing
    current_B = init_B
    current_size = len(init_set)
    
    # Precompute all possible x
    all_x = list(range(1, N + 1))
    
    T = 50.0  # Initial temperature
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        # Move types: Add, Remove, Swap
        r = random.random()
        
        if r < 0.4:
            # Add
            candidates = [x for x in all_x if not (current_B & (1 << (x - 1)))]
            if not candidates:
                continue
            x = random.choice(candidates)
            new_B = current_B | (1 << (x - 1))
            # Conflict count for adding x
            # We need to count pairs (a,b) in current_S such that x+a in S, x+b in S, a+b in S.
            # But wait, count_conflicts assumes x not in B. Here x is not in current_B.
            # However, the new bad triples are those containing x.
            # Are there other new bad triples? No, only those containing x.
            # So delta_conflicts = count_conflicts(current_B, x)
            # But count_conflicts checks a,b in S (current), and x+a in S (current), etc.
            # This is correct.
            c = count_conflicts(current_B, x)
            # Valid if c == 0
            if c == 0:
                # Always accept valid move that increases size
                current_B = new_B
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_B = current_B
                    best_set = bits_to_set(current_B)
            else:
                # Invalid, reject
                pass
                
        elif r < 0.7:
            # Remove
            s = bits_to_set(current_B)
            if not s:
                continue
            x = random.choice(list(s))
            new_B = current_B & ~(1 << (x - 1))
            # Removing x might fix conflicts, but we only accept if it doesn't decrease size?
            # No, in SA we can accept decreases if it helps escape local minima.
            # But we want to maximize size. Let's only accept removes if they lead to a better state later.
            # For simplicity, let's only accept removes if the set remains valid?
            # Actually, if current set is valid, removing x keeps it valid.
            # If current set is invalid (which shouldn't happen if we start valid and only add valid),
            # we might want to remove. But we start valid and only add if valid, so current is always valid.
            # So remove always leads to a valid set of smaller size.
            # We should only accept remove if it's part of a swap or if we are stuck.
            # Let's skip pure removes for now, or accept with low probability if stuck.
            # For now, let's implement Swap instead.
            pass
            
        else:
            # Swap: Remove x, Add y
            s = bits_to_set(current_B)
            if not s:
                continue
            x = random.choice(list(s))
            # Pick y not in S
            candidates = [z for z in all_x if not (current_B & (1 << (z - 1)))]
            if not candidates:
                continue
            y = random.choice(candidates)
            
            # New set: S' = (S \ {x}) ∪ {y}
            # We need to check if S' is valid.
            # This is expensive to check from scratch.
            # Approximation: 
            # 1. Check if adding y to (S \ {x}) is valid.
            #    This requires that no bad triple in S' exists.
            #    Bad triples in S' are either:
            #    a) Old bad triples in S that don't involve x. (S is valid, so 0)
            #    b) New bad triples involving y.
            #    c) Old bad triples involving x that are now broken. (S is valid, so 0)
            #    So we just need to check if adding y to (S \ {x}) creates any bad triple.
            #    Let S_temp = S \ {x}.
            #    Check count_conflicts(S_temp_bitset, y) == 0.
            
            S_temp_B = current_B & ~(1 << (x - 1))
            c = count_conflicts(S_temp_B, y)
            if c == 0:
                # Valid swap
                current_B = S_temp_B | (1 << (y - 1))
                # Size remains same
                # Check if this leads to a better state? 
                # Since size is same, we accept to explore.
                # But we should only keep it if it's part of a path to larger size.
                # For SA, we accept.
                pass
            else:
                # Reject
                pass
                
        # Update best
        # (Already done on add)
        
        # Cooling
        T *= cooling
        
    # Final validation
    if not is_valid_B(best_B):
        # Fallback
        best_set = init_set
        best_size = len(init_set)
    
    return sorted(best_set)
