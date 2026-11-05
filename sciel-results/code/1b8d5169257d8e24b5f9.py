def construct():
    n = 12
    A = []
    S = {0}
    for _ in range(n):
        c = 1
        while True:
            if c in A:
                c += 1
                continue
            # Check if adding c preserves distinct subset sums
            # New sums are S and S+c
            # We need S and S+c to be disjoint
            # Equivalent to: no s in S such that s+c is in S
            # Or: (S + c) & S is empty
            valid = True
            for s in S:
                if s + c in S:
                    valid = False
                    break
            if valid:
                A.append(c)
                # Update S
                new_S = set(S)
                for s in S:
                    new_S.add(s + c)
                S = new_S
                break
            c += 1
    return A
