"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 90.0  # Safety margin
    
    # Start with the safe set [91, 180]
    S = set(range(91, 181))
    
    # Try to add numbers from 90 down to 1
    # We iterate descending because larger 'a' have larger 'b,c'
    # and are less likely to conflict with the dense S_safe range?
    # Actually, smaller 'a' have smaller 'b,c' which might be in S_safe?
    # No, b,c > a.
    # Let's just try all.
    
    for a in range(90, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if a can be added
        # Condition: No b, c in S with b != c such that 1/a = 1/b + 1/c
        # (b-a)(c-a) = a^2
        conflict = False
        a_sq = a * a
        
        # Iterate through divisors of a^2
        # b-a = d, c-a = a^2/d
        # b = a + d, c = a + a^2/d
        # We need b, c in S and b != c
        
        d = 1
        while d * d <= a_sq:
            if a_sq % d == 0:
                d1 = d
                d2 = a_sq // d
                
                # Pair 1
                b1 = a + d1
                c1 = a + d2
                if b1 != c1:
                    if b1 in S and c1 in S:
                        conflict = True
                        break
                # Note: if b1 == c1, it's not a valid distinct triple, 
                # but we still check the other pair if d1 != d2
                
                # Pair 2 (if d1 != d2)
                if d1 != d2:
                    b2 = a + d2
                    c2 = a + d1
                    if b2 != c2:
                        if b2 in S and c2 in S:
                            conflict = True
                            break
            d += 1
            
        if not conflict:
            S.add(a)
            
    return list(S)
# EVOLVE-BLOCK-END
