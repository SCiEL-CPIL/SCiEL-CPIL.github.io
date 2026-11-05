def construct():
    S = set()
    # Iterate from largest to smallest
    for a in range(180, 0, -1):
        valid = True
        # Check for conflicts with existing elements in S
        # 1/a = 1/b + 1/c  => c = ab / (b-a)
        # b, c must be in S, distinct from each other and from a.
        # Since we add in decreasing order, all b in S are > a.
        # We need to check if there exists b in S such that c is integer, in S, and c != b.
        for b in S:
            # b > a is guaranteed
            denom = b - a
            num = a * b
            if num % denom == 0:
                c = num // denom
                if c in S and c != b:
                    valid = False
                    break
        if valid:
            S.add(a)
    return list(S)
