DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block i=0
    for j in range(m):
        for c in range(3):
            U[j][c] = 1
            if j == c:
                V[j * p + 0][c] = 1 # Wait, V index is j*p+k. 
                # In my derivation V_{j, 6+k, c} = delta_{j,c}.
                # So V[j*p + k, c] = 1 if j==c else 0.
                # But it must be 1 for ALL k?
                # Yes, V'' was 1 for all k.
                # So V[j*p + k, c] = 1 for all k if j==c.
                pass
    # Let's refine the loop
    
    for c in range(3):
        # U for i=0
        for j in range(m):
            U[j][c] = 1
        # V for i=0
        for j in range(m):
            if j == c:
                for k in range(p):
                    V[j * p + k][c] = 1
        # W for i=0
        for k in range(p):
            W[k * n + 0][c] = 1
            
    for c in range(3, 6):
        c_prime = c - 3
        # U for i=1
        for j in range(m):
            U[n*m/2 + j][c] = 1 # n*m/2 = 3
            # U index: i*m + j. i=1, m=3. 3+j.
        # V for i=1
        for j in range(m):
            if j == c_prime:
                for k in range(p):
                    V[j * p + k][c] = 1
        # W for i=1
        for k in range(p):
            W[k * n + 1][c] = 1

    return {"U": U, "V": V, "W": W}
