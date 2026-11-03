def decompose(n, m, p):
    # If n, m, p all even and >= 2: Strassen
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0 and n >= 2 and m >= 2 and p >= 2:
        # Strassen
        ...
    else:
        # Can't Strassen directly. Split off a 1 from the largest odd dimension, or schoolbook.
        # To keep it simple and correct: if any dimension is 1, schoolbook.
        # Otherwise, split the odd dimension into even + 1.
        # Find which dimension is odd.
        if n % 2 == 1 and n >= 3:
            # split n = (n-1) + 1
            U1, V1, W1 = decompose(n-1, m, p)
            U2, V2, W2 = decompose(1, m, p)
            return concat(U1,U2), concat(V1,V2), concat(W1,W2)
        elif m % 2 == 1 and m >= 3:
            U1, V1, W1 = decompose(n, m-1, p)
            U2, V2, W2 = decompose(n, 1, p)
            return concat(U1,U2), concat(V1,V2), concat(W1,W2)
        elif p % 2 == 1 and p >= 3:
            U1, V1, W1 = decompose(n, m, p-1)
            U2, V2, W2 = decompose(n, m, 1)
            return concat(U1,U2), concat(V1,V2), concat(W1,W2)
        else:
            # All dimensions are 1 or 2 (and at least one is 1, or all 2 but that's even)
            # Actually if we get here, at least one dim is odd and < 3, i.e., == 1, or all are 2 (even, handled above)
            # So at least one dim is 1. Schoolbook.
            return schoolbook(n, m, p)
