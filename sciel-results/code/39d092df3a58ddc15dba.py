"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Partition middle dim: 3 = 2 + 1
    # Term 1: A1(2x2) @ B1(2x10), partition output cols into 5 groups of 2, use Strassen (rank 7 each)
    # Term 2: A2(2x1) @ B2(1x10), rank 1
    # Total rank: 5*7 + 1 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0] * rank for _ in range(m * p)]  # 30 x 36
    W = [[0] * rank for _ in range(p * n)]  # 20 x 36
    
    col = 0
    # Strassen coefficients for 2x2x2
    # U_s: coefficients for A (local indices 0=A00, 1=A01, 2=A10, 3=A11)
    strassen_U = [
        {0: 1, 3: 1},       # M1: A00+A11
        {2: 1, 3: 1},       # M2: A10+A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00+A01
        {2: 1, 0: -1},      # M6: A10-A00
        {1: 1, 3: -1},      # M7: A01-A11
    ]
    # V_s: coefficients for B (local indices 0=B00, 1=B01, 2=B10, 3=B11)
    strassen_V = [
        {0: 1, 3: 1},       # M1: B00+B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01-B11
        {2: 1, 0: -1},      # M4: B10-B00
        {3: 1},             # M5: B11
        {0: 1, 2: 1},       # M6: B00+B10
        {1: 1, 3: 1},       # M7: B01+B11
    ]
    # W_s: coefficients for C (local indices 0=C00, 1=C01, 2=C10, 3=C11)
    strassen_W = [
        {0: 1, 3: 1},       # M1: C00+=1, C11+=1
        {2: 1, 3: -1},      # M2: C10+=1, C11-=1
        {1: 1, 3: 1},       # M3: C01+=1, C11+=1
        {0: 1, 2: 1},       # M4: C00+=1, C10+=1
        {0: -1, 1: 1},      # M5: C00-=1, C01+=1
        {3: 1},             # M6: C11+=1
        {0: 1},             # M7: C00+=1
    ]
    
    for g in range(5):
        for s in range(7):
            # U: local A indices (i,j) with i in {0,1}, j in {0,1} -> global i*m+j
            for local_idx, coeff in strassen_U[s].items():
                li = local_idx // 2  # local i
                lj = local_idx % 2   # local j
                global_A_idx = li * m + lj  # li*3 + lj
                U[global_A_idx][col] += coeff
            
            # V: local B indices (j,k) with j in {0,1}, k in {0,1} -> global j*p+(2g+k)
            for local_idx, coeff in strassen_V[s].items():
                lj = local_idx // 2   # local j
                lk = local_idx % 2    # local k
                global_B_idx = lj * p + (2 * g + lk)  # lj*10 + 2g+lk
                V[global_B_idx][col] += coeff
            
            # W: local C indices (i,k) with i in {0,1}, k in {0,1} -> global (2g+k)*n+i
            for local_idx, coeff in strassen_W[s].items():
                li = local_idx // 2   # local i
                lk = local_idx % 2    # local k
                global_C_idx = (2 * g + lk) * n + li  # (2g+lk)*2 + li
                W[global_C_idx][col] += coeff
            
            col += 1
    
    # Term 2: A2(2x1) @ B2(1x10), rank 1
    # C2[i,k] = A[i*3+2] * B[20+k]
    # U[i*3+2, col] = 1, V[20+k, col] = 1, W[k*2+i, col] = 1
    # But this needs to work for all i,k. Rank 1 means one column.
    # U[i*3+2, col] should be 1 for i=0,1 -> U[2, col]=1, U[5, col]=1
    # V[20+k, col] should be 1 for k=0..9 -> V[20..29, col]=1
    # W[k*2+i, col] should be 1 for all k,i -> W[0..19, col]=1
    
    # Wait, that gives C[k*2+i] = sum_j U[i*3+j,c]*V[j*10+k,c]*W[k*2+i,c]
    # For j=2: U[i*3+2,c]*V[2*10+k,c]*W[k*2+i,c] = 1*1*1 = 1 for all i,k
    # But we need C[k*2+i] = A[i*3+2]*B[20+k], not 1.
    
    # The tensor value T[i,j,k] = 1 for all (i,j,k). The decomposition is:
    # T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # So for Term 2, I need: for i in {0,1}, j=2, k in {0..9}:
    # U[i*3+2,c]*V[2*10+k,c]*W[k*2+i,c] = 1
    # With one column: U[2,c]=1, U[5,c]=1, V[20+k,c]=1 for all k, W[k*2+i,c]=1 for all k,i
    # That gives 1*1*1=1. Correct!
    
    U[2][col] = 1
    U[5][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
