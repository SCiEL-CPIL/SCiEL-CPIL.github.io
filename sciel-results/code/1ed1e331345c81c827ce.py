def construct():
    p = 1009
    n = 1008

    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)

    # A[x] = min(x, inv[x]), B[x] = max(x, inv[x])
    # We need M(s) = min { B[x] | A[x] >= s }
    # Initialize min_B_for_A with infinity
    min_B_for_A = [float('inf')] * (n + 2)
    
    for x in range(1, n + 1):
        a = min(x, inv[x])
        b = max(x, inv[x])
        if b < min_B_for_A[a]:
            min_B_for_A[a] = b

    # Compute suffix minimum of min_B_for_A
    # M[s] = min(min_B_for_A[s], M[s+1])
    M = [float('inf')] * (n + 2)
    current_min = float('inf')
    for s in range(n, 0, -1):
        if min_B_for_A[s] < current_min:
            current_min = min_B_for_A[s]
        M[s] = current_min

    best_start = 1
    best_length = 0

    for s in range(1, n + 1):
        # Max valid end e is M[s] - 1
        max_e = M[s] - 1
        if max_e > n:
            max_e = n
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    return [best_start, best_length]
