def construct():
    n = 256
    candidates = list(range(2, n + 1))
    
    def is_compatible(a, S):
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                b, c = S[i], S[j]
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
        return True
    
    # Try multiple orderings and take the best
    import random
    best = []
    
    orderings = [
        list(range(2, n + 1)),
        list(range(n, 1, -1)),
        sorted(range(2, n + 1), key=lambda x: -x.bit_count()),  # more bits first
        sorted(range(2, n + 1), key=lambda x: x.bit_count()),
    ]
    
    for order in orderings:
        S = []
        for a in order:
            if is_compatible(a, S):
                S.append(a)
        if len(S) > len(best):
            best = S
    
    # Verify final result
    S = best
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            for k in range(j + 1, len(S)):
                a, b, c = S[i], S[j], S[k]
                if (b * c) % a == 0:
                    raise ValueError(f"Violation: {a} | {b}*{c}")
    
    return list(S)
