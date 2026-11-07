    # Phase 2: Fast Removal
    S2 = list(range(256, 1, -1))
    
    def find_violation(S):
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                for k in range(j + 1, n):
                    c = S[k]
                    if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                        return (a, b, c)
        return None

    for _ in range(200): # Limit steps
        if time.time() > deadline:
            break
        viol = find_violation(S2)
        if viol is None:
            break
        # Remove the largest element from the violating triple?
        # Or the one that seems "most central"?
        # Removing the largest is often good because it's less likely to divide others.
        # But if a|bc, a is the divisor.
        # Let's remove the element that is the divisor in the violation if possible.
        a, b, c = viol
        # Check which one divides the product of the other two
        if (b * c) % a == 0:
            S2.remove(a)
        elif (a * c) % b == 0:
            S2.remove(b)
        else:
            S2.remove(c)
