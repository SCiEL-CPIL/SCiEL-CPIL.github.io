"""Rewritten program using Strassen-based recursive decomposition for rank minimization."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication.
    This is derived from the Kronecker product of two 2x2 Strassen decompositions.
    
    Strassen 2x2 decomposition has rank 7.
    The 4x4 decomposition has rank 7 * 7 = 49.
    
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    
    Indices:
    U[i*m + j, c]
    V[j*p + k, c]
    W[k*n + i, c]
    """
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 2x2 Strassen factors (rank 7)
    # U2x2[i*2+j, c], V2x2[j*2+k, c], W2x2[k*2+i, c]
    # The 7 products in Strassen's algorithm:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # U2x2 rows: (0,0), (0,1), (1,0), (1,1)
    # V2x2 rows: (0,0), (0,1), (1,0), (1,1)
    # W2x2 rows: (0,0), (0,1), (1,0), (1,1)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Set U2, V2, W2 based on the 7 terms
    # Term 0: M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 1: M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1
    
    # Term 2: M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = -1
    
    # Term 3: M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[0][3] = -1
    
    # Term 4: M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[1][4] = 1; W2[3][4] = 1
    
    # Term 5: M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 6: M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1
    
    # Now compute 4x4 matrices via Kronecker product
    # Index mapping for 4x4:
    # i = 2*i1 + i2, j = 2*j1 + j2
    # U4x4[i*4+j, C] where C = c1*7 + c2
    # U4x4[ (2*i1+i2)*4 + (2*j1+j2), c1*7+c2 ] = U2[2*i1+j1, c1] * U2[2*i2+j2, c2]
    
    # Let's iterate over all combinations
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    u4_idx = i*4 + j
                    
                    u2_11 = 2*i1 + j1
                    u2_22 = 2*i2 + j2
                    
                    for c1 in range(7):
                        val1 = U2[u2_11][c1]
                        if val1 == 0: continue
                        for c2 in range(7):
                            val2 = U2[u2_22][c2]
                            if val2 == 0: continue
                            C = c1*7 + c2
                            U[u4_idx][C] = val1 * val2

    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    v4_idx = j*4 + k
                    
                    v2_11 = 2*j1 + k1
                    v2_22 = 2*j2 + k2
                    
                    for c1 in range(7):
                        val1 = V2[v2_11][c1]
                        if val1 == 0: continue
                        for c2 in range(7):
                            val2 = V2[v2_22][c2]
                            if val2 == 0: continue
                            C = c1*7 + c2
                            V[v4_idx][C] = val1 * val2

    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    w4_idx = k*4 + i
                    
                    w2_11 = 2*k1 + i1
                    w2_22 = 2*k2 + i2
                    
                    for c1 in range(7):
                        val1 = W2[w2_11][c1]
                        if val1 == 0: continue
                        for c2 in range(7):
                            val2 = W2[w2_22][c2]
                            if val2 == 0: continue
                            C = c1*7 + c2
                            W[w4_idx][C] = val1 * val2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

# Verification helper (not part of submission, just for logic check)
def verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] 
    equals the coefficient of A_ij * B_jk in C_ik.
    For standard matrix mult, coefficient is 1 if indices match in chain, 0 otherwise?
    Actually, the tensor T_{ijk} = 1 if we are multiplying A (n x m) and B (m x p) to get C (n x p).
    Wait, the convention is W[k*n+i, r].
    The tensor is T[i, j, k] = 1 for all i, j, k? No.
    The matrix multiplication tensor T_{i,j,k} is 1 if we sum over j?
    Let's check the definition.
    T_{i,j,k} corresponds to the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    So T_{i,j,k} = 1 for all valid i,j,k.
    Wait, no. The tensor is defined such that C_{ik} = sum_{j} A_{ij} B_{jk}.
    So the tensor entries are T_{i,j,k} = 1.
    We need to check if sum_{c} U_{i,m+j,c} V_{j,p+k,c} W_{k,n+i,c} == 1 for all i,j,k.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

# Local test
if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    valid = verify_decomposition(res['U'], res['V'], res['W'], n, m, p)
    print(f"Rank: {len(res['U'][0])}")
    print(f"Valid: {valid}")
