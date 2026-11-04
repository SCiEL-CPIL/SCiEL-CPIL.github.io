"""Matrix multiplication 2x4x5 with rank-36 Strassen-based construction."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]   # 8 x rank
    V = [[0.0] * rank for _ in range(m * p)]   # 20 x rank
    W = [[0.0] * rank for _ in range(p * n)]   # 10 x rank

    col = 0

    def add_strassen_224(A_i, A_j1, A_j2, B_row0, B_row1, B_cols):
        """
        For 2x2x4 subproblem:
        A = [[a, b], [c, d]] where a=A[i0,j1], b=A[i0,j2], c=A[i1,j1], d=A[i1,j2]
        B is 2x4 with rows r0,r1 and 4 columns.
        C = A @ B, C is 2x4.
        
        U indices: U[A_i*4 + A_j1] for a, U[A_i*4 + A_j2] for b,
                   U[(A_i+1)*4 + A_j1] for c, U[(A_i+1)*4 + A_j2] for d
        V indices: V[r0*p + col] and V[r1*p + col]
        W indices: W[k*2 + A_i] for C row 0, W[k*2 + (A_i+1)] for C row 1
        """
        nonlocal col
        r0, r1 = B_row0, B_row1
        c0, c1, c2, c3 = B_cols
        i0 = A_i
        i1 = A_i + 1
        j1 = A_j1
        j2 = A_j2
        
        # U indices
        ua = i0 * 4 + j1  # a = A[i0, j1]
        ub = i0 * 4 + j2  # b = A[i0, j2]
        uc = i1 * 4 + j1  # c = A[i1, j1]
        ud = i1 * 4 + j2  # d = A[i1, j2]
        
        # W indices for C row i0 and i1, columns c0,c1,c2,c3
        # W[k*n + i] = W[k*2 + i]
        
        def emit(U_vals, V_vals, W_vals):
            nonlocal col
            for idx, val in U_vals:
                U[idx][col] += val
            for idx, val in V_vals:
                V[idx][col] += val
            for idx, val in W_vals:
                W[idx][col] += val
            col += 1
        
        # B1 = [[B(r0,c0), B(r0,c1)], [B(r1,c0), B(r1,c1)]]
        # e=B(r0,c0), f=B(r0,c1), g=B(r1,c0), h=B(r1,c1)
        
        # M1 = (a+d)(e+h)
        emit([(ua,1),(ud,1)], [(r0*p+c0,1),(r1*p+c1,1)], [(c0*2+i0,1),(c1*2+i1,1)])
        # M2 = (c+d)e
        emit([(uc,1),(ud,1)], [(r0*p+c0,1)], [(c0*2+i1,1)])
        # M3 = a(f-h)
        emit([(ua,1)], [(r0*p+c1,1),(r1*p+c1,-1)], [(c1*2+i0,1),(c1*2+i1,1)])
        # M4 = d(g-e)
        emit([(ud,1)], [(r1*p+c0,1),(r0*p+c0,-1)], [(c0*2+i0,1),(c0*2+i1,1)])
        # M5 = (a+b)h
        emit([(ua,1),(ub,1)], [(r1*p+c1,1)], [(c0*2+i0,-1),(c1*2+i0,1)])
        # M6 = (c-a)(e+f)
        emit([(uc,1),(ua,-1)], [(r0*p+c0,1),(r0*p+c1,1)], [(c1*2+i1,1)])
        # M7 = (b-d)(g+h)
        emit([(ub,1),(ud,-1)], [(r1*p+c0,1),(r1*p+c1,1)], [(c0*2+i0,1)])
        
        # B2 = [[B(r0,c2), B(r0,c3)], [B(r1,c2), B(r1,c3)]]
        # e2=B(r0,c2), f2=B(r0,c3), g2=B(r1,c2), h2=B(r1,c3)
        
        # M1_2 = (a+d)(e2+h2)
        emit([(ua,1),(ud,1)], [(r0*p+c2,1),(r1*p+c3,1)], [(c2*2+i0,1),(c3*2+i1,1)])
        # M2_2 = (c+d)e2
        emit([(uc,1),(ud,1)], [(r0*p+c2,1)], [(c2*2+i1,1)])
        # M3_2 = a(f2-h2)
        emit([(ua,1)], [(r0*p+c3,1),(r1*p+c3,-1)], [(c3*2+i0,1),(c3*2+i1,1)])
        # M4_2 = d(g2-e2)
        emit([(ud,1)], [(r1*p+c2,1),(r0*p+c2,-1)], [(c2*2+i0,1),(c2*2+i1,1)])
        # M5_2 = (a+b)h2
        emit([(ua,1),(ub,1)], [(r1*p+c3,1)], [(c2*2+i0,-1),(c3*2+i0,1)])
        # M6_2 = (c-a)(e2+f2)
        emit([(uc,1),(ua,-1)], [(r0*p+c2,1),(r0*p+c3,1)], [(c3*2+i1,1)])
        # M7_2 = (b-d)(g2+h2)
        emit([(ub,1),(ud,-1)], [(r1*p+c2,1),(r1*p+c3,1)], [(c2*2+i0,1)])
    
    # Part 1: A[:,0:2] @ B[0:2,:] -> 2x2x4, rank 14
    # A: i=0,1; j=0,1. B: rows 0,1; cols 0,1,2,3. C: rows 0,1; cols 0,1,2,3
    add_strassen_224(A_i=0, A_j1=0, A_j2=1, B_row0=0, B_row1=1, B_cols=[0,1,2,3])
    
    # Part 2: A[:,2:4] @ B[2:4,:] -> 2x2x4, rank 14
    # A: i=0,1; j=2,3. B: rows 2,3; cols 0,1,2,3. C: rows 0,1; cols 0,1,2,3
    add_strassen_224(A_i=0, A_j1=2, A_j2=3, B_row0=2, B_row1=3, B_cols=[0,1,2,3])
    
    # Part 3: A @ B[:,4] -> 2x4x1, rank 8
    # C[i,4] = sum_j A[i,j]*B[j,4]
    for i in range(2):
        for j in range(4):
            U[i*4 + j][col] = 1.0
            V[j*5 + 4][col] = 1.0
            W[4*2 + i][col] = 1.0  # W[8+i]
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _verify():
    """Verify the decomposition is correct."""
    import random
    result = construct()
    U, V, W = result['U'], result['V'], result['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for trial in range(10):
        A = [[random.random() for _ in range(m)] for _ in range(n)]
        B = [[random.random() for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[i][j]*B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_decomp = [[0.0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                C_decomp[i][k] = s
        
        for i in range(n):
            for k in range(p):
                if abs(C_decomp[i][k] - C_true[i][k]) > 1e-9:
                    print(f"MISMATCH at ({i},{k}): decomp={C_decomp[i][k]}, true={C_true[i][k]}")
                    return False
    return True

if _verify():
    print("Verification passed")
else:
    print("Verification FAILED")
# EVOLVE-BLOCK-END
