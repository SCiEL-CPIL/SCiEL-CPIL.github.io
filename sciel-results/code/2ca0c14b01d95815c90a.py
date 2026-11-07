"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # safety margin
    n = 128
    
    # Start with the upper half: {65, ..., 128}
    S = set(range(65, n + 1))
    
    # For fast checking, maintain a bitmask or set
    # Check if adding x creates a forbidden triple
    
    def is_valid_addition(x, S):
        """Check if adding x to S keeps S valid (no forbidden triple)."""
        # A forbidden triple is {a,b,c} distinct in S∪{x} with a+b,a+c,b+c all in S∪{x}
        # Since x is new, any new forbidden triple must include x.
        # Case 1: x is one of a,b,c.
        #   Let x=a. Then need b,c in S (distinct from x), and x+b, x+c, b+c all in S∪{x}.
        #   x+b and x+c: if they equal x, then b=0 or c=0, not possible. So x+b, x+c must be in S.
        #   b+c must be in S∪{x}. If b+c == x, that's ok (it's in S∪{x}). Otherwise must be in S.
        # Case 2: x is one of the sums.
        #   E.g., a+b = x. Then a,b,c in S, a+b=x, a+c in S∪{x}, b+c in S∪{x}.
        #   a+c and b+c: if they equal x, then c = x-a or c = x-b. 
        #   This gets complicated. Let me just check all triples involving x.
        
        # Actually, let's check all triples {a,b,c} that include x as a member
        # and all triples where x is one of the three sums.
        
        # Triples where x is a member:
        # {x, b, c} with b,c in S, b≠c, b≠x, c≠x
        # Need: x+b, x+c, b+c all in S∪{x}
        for b in S:
            if b == x:
                continue
            for c in S:
                if c == x or c == b:
                    continue
                s1 = x + b
                s2 = x + c
                s3 = b + c
                if s1 in S or s1 == x:
                    if s2 in S or s2 == x:
                        if s3 in S or s3 == x:
                            return False
        # Triples where x is a sum:
        # a+b = x, a,c in S, b,c in S, a≠b≠c distinct
        # a+c in S∪{x}, b+c in S∪{x}
        # Also a+c = x or b+c = x or both
        for a in S:
            if a == x:
                continue
            b = x - a
            if b <= 0 or b == a or b not in S:
                continue
            # Now a+b = x. Need c in S, c distinct from a,b.
            # a+c in S∪{x}, b+c in S∪{x}
            for c in S:
                if c == a or c == b or c == x:
                    continue
                s1 = a + c
                s2 = b + c
                if (s1 in S or s1 == x) and (s2 in S or s2 == x):
                    return False
        # Also: a+c = x and b+c = x can't both happen unless a=b.
        # a+b = x, a+c = x => b=c, not distinct.
        # So we've covered all cases.
        return True
    
    # Greedily add small numbers
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S):
            S.add(x)
    
    return sorted(S)
# EVOLVE-BLOCK-END
