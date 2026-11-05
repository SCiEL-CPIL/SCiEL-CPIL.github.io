import time

def construct():
    # Start with the valid set [81, 180]
    S = set(range(81, 181))
    
    # We want to add numbers from 80 down to 1
    # If adding a number creates a triple, we remove one of the other elements in the triple
    # to maintain validity.
    
    # Helper to check if adding x creates a triple
    def can_add(x, S):
        # Check if there exist b, c in S such that 1/x = 1/b + 1/c
        # Equivalent to (b-x)(c-x) = x^2
        # b, c must be in S and distinct from x (they are > x)
        x2 = x * x
        # Iterate through potential b in S
        # b must be > x
        for b in S:
            if b <= x:
                continue
            # Check if (b-x) divides x^2
            if x2 % (b - x) != 0:
                continue
            c_minus_x = x2 // (b - x)
            c = x + c_minus_x
            # c must be in S, distinct from b, and distinct from x
            if c in S and c != b and c != x:
                return False, b, c
        return True, None, None

    # Greedy addition
    for x in range(80, 0, -1):
        if x in S:
            continue
        ok, b, c = can_add(x, S)
        if ok:
            S.add(x)
        else:
            # Conflict with b, c. Remove the largest one to minimize impact?
            # Or remove c?
            # Removing the largest element seems to preserve the "density" of the upper range
            # which might be more compatible with other additions.
            # Actually, removing the element that causes the most future conflicts is ideal,
            # but hard to predict.
            # Let's try removing the largest of b, c.
            if b > c:
                S.remove(b)
            else:
                S.remove(c)
            S.add(x)
            
    return sorted(S)
