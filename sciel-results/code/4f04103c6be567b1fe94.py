import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit

    # Bitmask representation of S
    # Bit i is 1 if i is in S.
    # Max value is 128, so we need bits 0..128.
    
    def can_add(S_mask, x):
        """
        Check if adding x to S_mask results in a valid set.
        S_mask is an integer representing the set.
        x is an integer 1..128.
        """
        # Case 1: x is a member of the triple (a=x)
        # Need distinct b, c in S such that:
        # (x+b in S) and (x+c in S) and (b+c in S or b+c == x)
        
        # V = { y in S | x+y in S }
        # y in S means bit y is set in S_mask.
        # x+y in S means bit (x+y) is set in S_mask.
        # So y must be such that S_mask has bit y AND S_mask has bit (x+y).
        
        # Efficiently find V:
        # S_mask & (S_mask >> x) gives bits y such that bit y is in S and bit y+x is in S.
        # Wait, if S_mask has bit y and bit y+x, then (S_mask >> x) has bit y.
        # So V_mask = S_mask & (S_mask >> x)
        
        V_mask = S_mask & (S_mask >> x)
        
        # If V has fewer than 2 bits, no pair b,c exists.
        if V_mask.bit_count() < 2:
            # Case 1 passes
            pass
        else:
            # Check if there exist distinct b, c in V such that b+c in S or b+c == x.
            # We can iterate through bits in V_mask.
            
            # Let's extract bits
            v_list = []
            temp = V_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                v_list.append(idx)
                temp -= lsb
                
            n_v = len(v_list)
            if n_v >= 2:
                for i in range(n_v):
                    b = v_list[i]
                    for j in range(i + 1, n_v):
                        c = v_list[j]
                        s = b + c
                        # Check if s in S or s == x
                        if s == x or ((S_mask >> s) & 1):
                            return False
        
        # Case 2: x is a sum of two members (a+b=x)
        # Need distinct a, b, c in S such that:
        # a+b = x
        # a+c in S
        # b+c in S
        
        # Iterate a in S. b = x - a.
        # If b in S and a != b:
        #   Check if there exists c in S, c != a, c != b, such that (a+c in S) and (b+c in S).
        
        for a in range(1, x // 2 + 1):
            if ((S_mask >> a) & 1) == 0:
                continue
            b = x - a
            if b == a:
                continue
            if ((S_mask >> b) & 1) == 0:
                continue
            
            # Compute U_mask
            # N_a = S_mask & (S_mask >> a)
            # N_b = S_mask & (S_mask >> b)
            # U = N_a & N_b
            # Remove a and b from U
            U_mask = (S_mask & (S_mask >> a)) & (S_mask & (S_mask >> b))
            U_mask &= ~(1 << a)
            U_mask &= ~(1 << b)
            
            if U_mask != 0:
                return False
        
        return True

    def greedy_construct(order, S_mask):
        """Greedy construction given a fixed order of candidates."""
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask, max_passes=50):
        """Improve S via additions and systematic swaps."""
        # Convert mask to set for easier manipulation of "not in S"
        S_set = set()
        temp = S_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S_set.add(idx)
            temp -= lsb
            
        not_in_S = [x for x in range(1, 129) if x not in S_set]
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            
            improved = False
            random.shuffle(not_in_S)
            
            for x in not_in_S[:]:
                if time.time() > deadline:
                    break
                
                # Try to add x directly
                if can_add(S_mask, x):
                    S_mask |= (1 << x)
                    S_set.add(x)
                    not_in_S.remove(x)
                    improved = True
                    continue
                
                # Try swaps: remove y from S, add x
                # We want to find y in S such that can_add(S_mask without y, x) is True.
                # S_mask without y is S_mask ^ (1 << y)
                
                S_list = list(S_set)
                random.shuffle(S_list)
                
                found_swap = False
                # Try up to 30 random swaps for this x to increase chance of escape
                for y in S_list[:30]: 
                    if time.time() > deadline:
                        break
                    S_temp_mask = S_mask ^ (1 << y)
                    if can_add(S_temp_mask, x):
                        S_mask = S_temp_mask | (1 << x)
                        S_set.remove(y)
                        S_set.add(x)
                        not_in_S.append(y)
                        if x in not_in_S:
                            not_in_S.remove(x)
                        found_swap = True
                        improved = True
                        break
                
                if found_swap:
                    # Re-check additions after swap might be beneficial, but let's just restart pass
                    break
            
            if not improved:
                break
        
        return S_mask

    # Phase 1: Multiple greedy constructions with diverse orderings
    random.seed(42)
    
    best_mask = 0
    
    # Ascending
    S_mask = greedy_construct(list(range(1, 129)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    # Descending
    S_mask = greedy_construct(list(range(128, 0, -1)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask
    
    # Several random orders
    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask
    
    # Phase 2: Local search from best set
    if time.time() < deadline:
        best_mask = local_search(best_mask, max_passes=30)
    
    # Phase 3: More random restarts with local search
    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        S_mask = local_search(S_mask, max_passes=20)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask
    
    # Final: try to add any remaining elements
    if time.time() < deadline:
        for x in range(1, 129):
            if (best_mask >> x) & 1:
                continue
            if can_add(best_mask, x):
                best_mask |= (1 << x)
    
    # One more local search pass on the final best set
    if time.time() < deadline and best_mask.bit_count() < 128:
        best_mask = local_search(best_mask, max_passes=10)
    
    # Convert mask to list of ints
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(int(idx))
        temp -= lsb
    
    return result
