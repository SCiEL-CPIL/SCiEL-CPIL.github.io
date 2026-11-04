import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0  # 10s is plenty for this construction
    
    n, m, p = DIMENSIONS
    
    # We use the block decomposition:
    # Split m=3 into m1=2 and m2=1.
    # A = [A1, A2], B = [B1; B2]
    # C = A1 B1 + A2 B2
    # A1 is 2x2, B1 is 2x10.
    # A2 is 2x1, B2 is 1x10.
    
    # Part 1: A2 B2 (2x1 * 1x10)
    # This is 20 scalar multiplications.
    # U, V, W indices for this part.
    # i in 0..1 (n=2), j in 2 (m index 2, 0-indexed), k in 0..9 (p=10)
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    
    # Part 2: A1 B1 (2x2 * 2x10)
    # Split B1 into 5 blocks of 2x2.
    # B1 = [B1_0, B1_1, B1_2, B1_3, B1_4]
    # C1 = A1 B1 = [A1 B1_0, ...]
    # Each A1 B1_l is a 2x2x2 multiplication.
    # Use Strassen for each.
    
    # Total rank = 20 + 5 * 7 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: A2 B2 ---
    # j=2 (0-indexed) corresponds to the 3rd column of A, 3rd row of B.
    # i in 0..1, k in 0..9
    for i in range(n):
        for k in range(p):
            # U index: i*m + 2
            U[i * m + 2][col] = 1
            # V index: 2*p + k
            V[2 * p + k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    # --- Part 2: A1 B1 ---
    # j in 0..1 (first two columns of A, first two rows of B)
    # k in 0..9
    # Split k into 5 blocks of size 2.
    # Strassen for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B21 - B22)
    # M4 = A22 (B12 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the global U, V, W.
    # For a block starting at k0:
    # k indices are k0, k0+1.
    # j indices are 0, 1.
    # i indices are 0, 1.
    
    for block_idx in range(5):
        k0 = block_idx * 2
        # k0, k0+1 are the column indices in B1, and row indices in C1 (part of W)
        
        # M1
        # U: (A11 + A22) -> U[0*m+0]=1, U[1*m+1]=1
        # V: (B11 + B22) -> V[0*p+k0]=1, V[1*p+(k0+1)]=1
        # W: (C11 + C22 - C21 + C12? No, C11 and C22 contributions)
        # M1 contributes to C11 and C22.
        # C11 is W[k0*n + 0], C22 is W[(k0+1)*n + 1].
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + (k0 + 1)][col] = 1
        W[k0 * n + 0][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1
        if time.time() > deadline: return {"U": U, "V": V, "W": W}

        # M2
        # U: (A21 + A22) -> U[1*m+0]=1, U[1*m+1]=1
        # V: B11 -> V[0*p+k0]=1
        # W: C21 -> W[k0*n + 1]
        U[1 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        if time.time() > deadline: return {"U": U, "V": V, "W": W}

        # M3
        # U: A11 -> U[0*m+0]=1
        # V: (B21 - B22) -> V[1*p+k0]=1, V[1*p+(k0+1)]=-1
        # W: C12 -> W[(k0+1)*n + 0]
        U[0 * m + 0][col] = 1
        V[1 * p + k0][col] = 1
        V[1 * p + (k0 + 1)][col] = -1
        W[(k0 + 1) * n + 0][col] = 1
        col += 1
        if time.time() > deadline: return {"U": U, "V": V, "W": W}

        # M4
        # U: A22 -> U[1*m+1]=1
        # V: (B12 - B11) -> V[0*p+(k0+1)]=1, V[0*p+k0]=-1
        # W: C21 and C11?
        # C21 = M2 + M4. So W[k0*n+1] += 1
        # C11 = M1 + M4 - M5 + M7. So W[k0*n+0] += 1
        U[1 * m + 1][col] = 1
        V[0 * p + (k0 + 1)][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        if time.time() > deadline: return {"U": U, "V": V, "W": W}

        # M5
        # U: (A11 + A12) -> U[0*m+0]=1, U[0*m+1]=1
        # V: B22 -> V[1*p+(k0+1)]=1
        # W: C12 and C11?
        # C12 = M3 + M5. So W[(k0+1)*n+0] += 1
        # C11 = ... - M5 ... So W[k0*n+0] -= 1
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[1 * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + 0][col] = 1
        W[k0 * n + 0][col] = -1
        col += 1
        if time.time() > deadline: return {"U": U, "V": V, "W": W}

        # M6
        # U: (A21 - A11) -> U[1*m+0]=1, U[0*m+0]=-1
        # V: (B11 + B12) -> V[0*p+k0]=1, V[0*p+(k0+1)]=1
        # W: C22 -> W[(k0+1)*n+1]
        U[1 * m + 0][col] = 1
        U[0 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + 1][col] = 1
        col += 1
        if time.time() > deadline: return {"U": U, "V": V, "W": W}

        # M7
        # U: (A12 - A22) -> U[0*m+1]=1, U[1*m+1]=-1
        # V: (B21 + B22) -> V[1*p+k0]=1, V[1*p+(k0+1)]=1
        # W: C11 -> W[k0*n+0]
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + (k0 + 1)][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        if time.time() > deadline: return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
