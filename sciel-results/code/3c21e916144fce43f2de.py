from fractions import Fraction

def construct():
    S = []
    Sums = {Fraction(0, 1)}
    
    # Order: 24 down to 1
    for x in range(24, 0, -1):
        rx = Fraction(1, x)
        
        # Check if rx is in Sums (subset of S sums to rx)
        if rx in Sums:
            continue
            
        # Check if adding x makes any existing a invalid
        # 1/a = rx + sum(T) for some T subset of S
        # sum(T) = 1/a - rx
        # We need 1/a - rx > 0 and in Sums
        valid_add = True
        for a in S:
            ra = Fraction(1, a)
            target = ra - rx
            if target > 0:
                if target in Sums:
                    valid_add = False
                    break
            # if target <= 0, no subset of positive reciprocals can sum to it (except empty which is 0, but target!=0)
            # if target == 0, ra == rx => a == x, impossible since x not in S
            
        if valid_add:
            S.append(x)
            # Update Sums
            # New sums are old sums and old sums + rx
            # We need to be careful not to modify Sums while iterating
            new_sums = set(Sums)
            for s in Sums:
                new_sums.add(s + rx)
            Sums = new_sums
            
    return S
