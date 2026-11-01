import time

def construct():
    n, m, p = 2, 2, 11
    start_time = time.time()
    deadline = start_time + 110
    
    # We want to compute C = A * B where A is 2x2 and B is 2x11.
    # C is 2x11.
    # We can group the 11 columns of B into 5 pairs and 1 single column.
    # For each pair of columns (B[:, 2t], B[:, 2t+1]), we can use Strassen's algorithm 
    # to multiply a 2x2 matrix A by a 2x2 matrix M_t (which consists of these two columns).
    # This takes 7 multiplications per pair.
    # For the last single column, we use 4 standard multiplications.
    # Total rank = 5 * 7 + 4 = 35 + 4 = 39.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39
    
    # Indices for U: A[i,j] -> index i*m + j = i*2 + j
    # Indices for V: B[j,k] -> index j*p + k = j*11 + k
    # Indices for W: C[i,k] -> index k*n + i = k*2 + i (Trace convention W[k*n+i, r])
    
    col = 0
    for t in range(5):
        # Columns 2*t and 2*t+1
        c1 = 2 * t
        c2 = 2 * t + 1
        
        # Strassen's 7 products for M_t = [B[:, c1], B[:, c2]]
        # M11 = B[0, c1], M12 = B[0, c2]
        # M21 = B[1, c1], M22 = B[1, c2]
        
        # P1 = (A00 + A11) * (M11 + M22)
        U[0][col] = 1; U[3][col] = 1
        V[0*11 + c1][col] = 1; V[1*11 + c2][col] = 1
        # C00 += P1, C11 += P1
        W[c1*2 + 0][col] = 1; W[c2*2 + 1][col] = 1
        col += 1
        
        # P2 = (A10 + A11) * M11
        U[2][col] = 1; U[3][col] = 1
        V[0*11 + c1][col] = 1
        # C10 += P2, C11 += P2
        W[c1*2 + 1][col] = 1; W[c2*2 + 1][col] = 1
        col += 1
        
        # P3 = A00 * (M12 - M11)
        U[0][col] = 1
        V[0*11 + c2][col] = 1; V[0*11 + c1][col] = -1
        # C01 += P3
        W[c2*2 + 0][col] = 1
        col += 1
        
        # P4 = A11 * (M21 - M22)
        U[3][col] = 1
        V[1*11 + c1][col] = 1; V[1*11 + c2][col] = -1
        # C11 += P4
        W[c2*2 + 1][col] = 1
        col += 1
        
        # P5 = (A00 + A01) * M22
        U[0][col] = 1; U[1][col] = 1
        V[1*11 + c2][col] = 1
        # C01 += P5, C11 += P5
        W[c2*2 + 0][col] = 1; W[c2*2 + 1][col] = 1
        col += 1
        
        # P6 = (A10 - A00) * (M11 + M12)
        U[2][col] = 1; U[0][col] = -1
        V[0*11 + c1][col] = 1; V[0*11 + c2][col] = 1
        # C10 += P6
        W[c1*2 + 1][col] = 1
        col += 1
        
        # P7 = (A01 - A11) * (M21 + M22)
        U[1][col] = 1; U[3][col] = -1
        V[1*11 + c1][col] = 1; V[1*11 + c2][col] = 1
        # C01 += P7
        W[c2*2 + 0][col] = 1
        col += 1
        
    # Last column c = 10
    c = 10
    # Standard 4 multiplications
    # C00 += A00 * B0c
    U[0][col] = 1
    V[0*11 + c][col] = 1
    W[c*2 + 0][col] = 1
    col += 1
    
    # C01 += A01 * B1c
    U[1][col] = 1
    V[1*11 + c][col] = 1
    W[c*2 + 0][col] = 1
    col += 1
    
    # C10 += A10 * B0c
    U[2][col] = 1
    V[0*11 + c][col] = 1
    W[c*2 + 1][col] = 1
    col += 1
    
    # C11 += A11 * B1c
    U[3][col] = 1
    V[1*11 + c][col] = 1
    W[c*2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
