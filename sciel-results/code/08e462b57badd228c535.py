"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    mask = (1 << 201) - 1  # Bits 0 to 200
    
    def compute_reachable(elements):
        """Compute reachable sums bitset for a list of elements."""
        r = 1
        for x in elements:
            r |= (r << x)
            r &= mask
        return r
    
    # Greedy descending with bitset
    selected = []
    reachable = 1  # Bit 0 set
    
    for x in range(100, 0, -1):
        if (reachable >> (200 - x)) & 1:
            pass
        else:
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= mask
            
    S = set(selected)
    U = set(range(1, 101)) - S
    
    def try_swap_1_for_2(S, U):
        """Try to remove 1 from S and add 2 from U. Returns True if successful."""
        for s in list(S):
            S_minus = list(S)
            S_minus.remove(s)
            R_temp = compute_reachable(S_minus)
            
            for u1 in U:
                R1 = R_temp | (R_temp << u1)
                R1 &= mask
                if (R1 >> 200) & 1:
                    continue
                
                for u2 in U:
                    if u1 == u2:
                        continue
                    R2 = R1 | (R1 << u2)
                    R2 &= mask
                    if not ((R2 >> 200) & 1):
                        S.remove(s)
                        S.add(u1)
                        S.add(u2)
                        U.remove(u1)
                        U.remove(u2)
                        return True
        return False
    
    def try_swap_2_for_3(S, U):
        """Try to remove 2 from S and add 3 from U. Returns True if successful."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                s1, s2 = S_list[i], S_list[j]
                S_minus = list(S)
                S_minus.remove(s1)
                S_minus.remove(s2)
                R_temp = compute_reachable(S_minus)
                
                for u1 in U:
                    R1 = R_temp | (R_temp << u1)
                    R1 &= mask
                    if (R1 >> 200) & 1:
                        continue
                    
                    for u2 in U:
                        if u1 == u2:
                            continue
                        R2 = R1 | (R1 << u2)
                        R2 &= mask
                        if (R2 >> 200) & 1:
                            continue
                        
                        for u3 in U:
                            if u3 == u1 or u3 == u2:
                                continue
                            R3 = R2 | (R2 << u3)
                            R3 &= mask
                            if not ((R3 >> 200) & 1):
                                S.remove(s1)
                                S.remove(s2)
                                S.add(u1)
                                S.add(u2)
                                S.add(u3)
                                U.remove(u1)
                                U.remove(u2)
                                U.remove(u3)
                                return True
        return False
    
    # Multiple passes of local search
    for _ in range(5):
        improved = False
        # Try 1-for-2 swaps
        while try_swap_1_for_2(S, U):
            improved = True
        # Try 2-for-3 swaps
        while try_swap_2_for_3(S, U):
            improved = True
        if not improved:
            break
    
    return list(S)
# EVOLVE-BLOCK-END
