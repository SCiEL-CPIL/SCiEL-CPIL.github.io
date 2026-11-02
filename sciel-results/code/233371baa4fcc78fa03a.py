"""Matrix multiplication M_{2,2,15} using Strassen's algorithm on 2x2 column blocks."""

def construct():
    n, m, p = 2, 2, 15
    # 7 blocks of 2 columns + 1 block of 1 column
    # Strassen for 2x2x2: rank 7
    # Schoolbook for 2x2x1: rank 4
    rank = 7 * 7 + 4  # 53
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank  
    W = [[0] * rank for _ in range(p * n)]  # 30 x rank
    
    idx = 0
    
    # Strassen's 7 products for M_{2,2,2}
    # A = [[A00, A01], [A10, A11]], B = [[B0j, B0k], [B1j, B1k]]
    # C = A*B, where C[i,j] = sum A[i,l]*B[l,j]
    # 
    # M1 = (A00+A11)*(B0j+B1k) -> contributes to C0j, C1k
    # M2 = (A10+A11)*B0j       -> contributes to C1j, -C1k  
    # M3 = A00*(B1k-B0k)       -> contributes to C0k, C1k
    # M4 = A11*(B1j-B0j)       -> contributes to C0j, C1j
    # M5 = (A00+A01)*B1k       -> contributes to -C0j, C0k
    # M6 = (A10-A00)*(B0j+B0k) -> contributes to C1k
    # M7 = (A01-A11)*(B1j+B1k) -> contributes to C0j
    
    def add_strassen_block(j, k):
        nonlocal idx
        # M1
        U[0][idx] += 1; U[3][idx] += 1
        V[j][idx] += 1; V[15+k][idx] += 1
        W[2*j][idx] += 1; W[2*k+1][idx] += 1
        idx += 1
        # M2
        U[2][idx] += 1; U[3][idx] += 1
        V[j][idx] += 1
        W[2*j+1][idx] += 1; W[2*k+1][idx] += -1
        idx += 1
        # M3
        U[0][idx] += 1
        V[k][idx] += 1; V[15+k][idx] += -1
        W[2*k][idx] += 1; W[2*k+1][idx] += 1
        idx += 1
        # M4
        U[3][idx] += 1
        V[15+j][idx] += 1; V[j][idx] += -1
        W[2*j][idx] += 1; W[2*j+1][idx] += 1
        idx += 1
        # M5
        U[0][idx] += 1; U[1][idx] += 1
        V[15+k][idx] += 1
        W[2*j][idx] += -1; W[2*k][idx] += 1
        idx += 1
        # M6
        U[2][idx] += 1; U[0][idx] += -1
        V[j][idx] += 1; V[k][idx] += 1
        W[2*k+1][idx] += 1
        idx += 1
        # M7
        U[1][idx] += 1; U[3][idx] += -1
        V[15+j][idx] += 1; V[15+k][idx] += 1
        W[2*j][idx] += 1
        idx += 1
    
    # Process 7 blocks of 2 columns
    for b in range(7):
        j = 2 * b
        k = 2 * b + 1
        add_strassen_block(j, k)
    
    # Last single column: schoolbook
    j = 14
    # C0j = A00*B0j + A01*B1j
    U[0][idx] += 1; V[j][idx] += 1; W[2*j][idx] += 1
    idx += 1
    U[1][idx] += 1; V[15+j][idx] += 1; W[2*j][idx] += 1
    idx += 1
    # C1j = A10*B0j + A11*B1j
    U[2][idx] += 1; V[j][idx] += 1; W[2*j+1][idx] += 1
    idx += 1
    U[3][idx] += 1; V[15+j][idx] += 1; W[2*j+1][idx] += 1
    idx += 1
    
    return {"U": U, "V": V, "W": W}
