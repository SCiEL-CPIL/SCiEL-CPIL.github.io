import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 85.0

    def make_set(lst):
        s = 0
        for x in lst:
            s |= (1 << (x - 1))
        return s

    def is_valid_full(s):
        elems = [i + 1 for i in range(n) if (s >> i) & 1]
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    continue
                if not ((s >> (ab - 1)) & 1):
                    continue
                for k in range(j + 1, L):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        continue
                    if ((s >> (ac - 1)) & 1) and ((s >> (bc - 1)) & 1):
                        return False
        return True

    def can_add(s, x):
        """
        Check if adding x to set s (bitmask) is valid.
        Forbidden if there exist distinct a, b in s such that:
        1. a + b in s
        2. a + x in s
        3. b + x in s
        Note: x is not in s yet.
        """
        if (s >> (x - 1)) & 1:
            return True  # Already in set, trivially valid to "add" (no change)
        
        # M contains bits for a such that a + x is in s
        # Bit i of M is 1 if (s >> (i + x - 1)) & 1
        # So M = s >> (x - 1)
        M = s >> (x - 1)
        
        # We need to find distinct a, b in M such that a + b is in s.
        # a and b are 1-indexed values in the original set context.
        # In M, bit index i corresponds to value i+1.
        # Let a_val = i+1, b_val = j+1.
        # We need a_val + b_val to be in s.
        
        # Iterate over set bits in M
        # Optimization: If M has few bits, this is fast.
        # Max bits in M is n.
        
        # To avoid checking a,b and b,a, we can just check all pairs.
        # With n=128, worst case M has 128 bits. 128*128 = 16k ops.
        # But usually M is sparse.
        
        # Get list of indices in M
        # Instead of building a list, iterate
        m_temp = M
        # We need to check pairs (i, j) where i < j
        # i, j are 0-indexed in M, corresponding to values i+1, j+1
        
        # If M has 0 or 1 bits, no pair exists.
        if M == 0 or (M & (M - 1)) == 0:
            return True
            
        # Iterate i
        i = 0
        while M:
            if M & 1:
                a_val = i + 1
                # Check if there is a j > i such that bit j is set in M and a_val + (j+1) is in s
                # j+1 is b_val.
                # We need (s >> (a_val + b_val - 1)) & 1
                # b_val = j + 1
                # sum = a_val + j + 1
                # Check bits j in M where j > i
                
                m_rest = M >> (i + 1) # Bits for j > i
                j_offset = i + 1
                while m_rest:
                    if m_rest & 1:
                        j_val = j_offset + 1
                        sum_val = a_val + j_val
                        if sum_val <= n:
                            if (s >> (sum_val - 1)) & 1:
                                return False
                    m_rest >>= 1
                    j_offset += 1
            M >>= 1
            i += 1
            
        return True

    # Start with seed: 65-128
    best_s = make_set(range(65, 129))
    best_size = 64

    # Multiple greedy passes with different random orderings
    for trial in range(20):
        if time.time() > deadline - 25:
            break
        trial_s = best_s
        candidates = list(range(1, 65))
        random.seed(trial * 1000 + 42)
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline - 25:
                break
            if can_add(trial_s, x):
                trial_s |= (1 << (x - 1))

        # Try adding any remaining element
        for x in range(1, n + 1):
            if time.time() > deadline - 25:
                break
            if (trial_s >> (x - 1)) & 1:
                continue
            if can_add(trial_s, x):
                trial_s |= (1 << (x - 1))

        sz = bin(trial_s).count('1')
        if sz > best_size:
            best_s = trial_s
            best_size = sz

    # Local search on best solution
    for iteration in range(20000):
        if time.time() > deadline - 3:
            break

        in_set = [i + 1 for i in range(n) if (best_s >> i) & 1]
        out_set = [i + 1 for i in range(n) if not ((best_s >> i) & 1)]

        if not out_set:
            break

        improved = False

        # Try single adds
        random.shuffle(out_set)
        for x in out_set:
            if can_add(best_s, x):
                best_s |= (1 << (x - 1))
                best_size += 1
                improved = True
                break
        if improved:
            continue

        # Try remove-one-add-two
        if len(in_set) >= 1 and len(out_set) >= 2:
            random.shuffle(in_set)
            random.shuffle(out_set)
            for rem in in_set[:5]:
                if time.time() > deadline - 3:
                    break
                s2 = best_s & ~(1 << (rem - 1))
                added = 0
                for add in out_set:
                    if add == rem:
                        continue
                    if can_add(s2, add):
                        s2 |= (1 << (add - 1))
                        added += 1
                    if added >= 2:
                        break
                if added >= 2:
                    best_s = s2
                    best_size = bin(best_s).count('1')
                    improved = True
                    break

        if not improved and iteration > 5000:
            break

    # Final verification
    if not is_valid_full(best_s):
        best_s = make_set(range(65, 129))
        best_size = 64

    result = [i + 1 for i in range(n) if (best_s >> i) & 1]
    return result
