"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return rank-7 decomposition of M_{2,2,2} (Strassen)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = -1
    
    # Term 1: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # Term 2: A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1
    
    # Term 3: A11(B00-B01)
    U[3][3] = 1
    V[0][3] = 1; V[1][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # Term 4: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = -1
    
    # Term 6: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Return schoolbook decomposition of M_{n,m,p}."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """
    Kronecker product of two decompositions.
    M_{n1,m1,p1} rank r1, M_{n2,m2,p2} rank r2
    -> M_{n1*n2, m1*m2, p1*p2} rank r1*r2
    """
    r1 = len(W1[0]) if W1 else 0
    r2 = len(W2[0]) if W2 else 0
    rank = r1 * r2
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            idx_u = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
                            val1 = U1[i1*m1+j1][c1]
                            val2 = U2[i2*m2+j2][c2]
                            if val1 != 0 and val2 != 0:
                                U[idx_u][c] += val1 * val2
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            idx_v = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
                            val1 = V1[j1*p1+k1][c1]
                            val2 = V2[j2*p2+k2][c2]
                            if val1 != 0 and val2 != 0:
                                V[idx_v][c] += val1 * val2
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            idx_w = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
                            val1 = W1[k1*n1+i1][c1]
                            val2 = W2[k2*n2+i2][c2]
                            if val1 != 0 and val2 != 0:
                                W[idx_w][c] += val1 * val2
    
    return U, V, W

def _sum_decomp(U1, V1, W1, U2, V2, W2, n, m, p):
    """
    Direct sum of two decompositions for the same dimensions.
    M_{n,m,p} rank r1 + M_{n,m,p} rank r2 -> M_{n,m,p} rank r1+r2
    """
    r1 = len(W1[0])
    r2 = len(W2[0])
    rank = r1 + r2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(r1):
            U[i][c] = U1[i][c]
        for c in range(r2):
            U[i][r1+c] = U2[i][c]
            
    for j in range(m*p):
        for c in range(r1):
            V[j][c] = V1[j][c]
        for c in range(r2):
            V[j][r1+c] = V2[j][c]
            
    for k in range(p*n):
        for c in range(r1):
            W[k][c] = W1[k][c]
        for c in range(r2):
            W[k][r1+c] = W2[k][c]
            
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > tol:
                    return False, (i, j, k, s)
    return True, None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook baseline
    U_sb, V_sb, W_sb = _schoolbook_nm_p(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p  # 84
    
    # Construct M_{3,4,7} using block decomposition
    # n=3 (2+1), m=4 (2+2), p=7 (4+3)
    
    # Blocks of A (3x4):
    # A00 (2x2), A01 (2x2)
    # A10 (1x2), A11 (1x2)
    
    # Blocks of B (4x7):
    # B00 (2x4), B01 (2x3)
    # B10 (2x4), B11 (2x3)
    
    # C00 (2x4) = A00*B00 + A01*B10
    # C01 (2x3) = A00*B01 + A01*B11
    # C10 (1x4) = A10*B00 + A11*B10
    # C11 (1x3) = A10*B01 + A11*B11
    
    # We need decompositions for:
    # M_{2,2,4}: Rank 14 (2 * M_{2,2,2})
    # M_{2,2,3}: Rank 12 (Schoolbook)
    # M_{1,2,4}: Rank 8 (Schoolbook)
    # M_{1,2,3}: Rank 6 (Schoolbook)
    
    # Get base decompositions
    U_222, V_222, W_222 = _strassen_222()
    
    # M_{2,2,4} = M_{2,2,2} (cols 0-1) + M_{2,2,2} (cols 2-3)
    # We can use sum_decomp if we shift indices, but it's easier to just use schoolbook for simplicity 
    # if rank doesn't improve, but here Strassen helps.
    # M_{2,2,4} rank 14.
    # Let's build M_{2,2,4} by summing two M_{2,2,2} decompositions.
    # However, sum_decomp assumes same dimensions.
    # M_{2,2,4} is not M_{2,2,2} + M_{2,2,2} in the same space.
    # It's a block diagonal in the output.
    # C00 (2x2) and C01 (2x2) parts of the 2x4 output.
    
    # Let's create a helper to place a decomposition into a larger space.
    def _embed_decomp(U, V, W, n, m, p, i_off, j_off, k_off, n_sub, m_sub, p_sub):
        """Embed a decomposition of M_{n_sub, m_sub, p_sub} into M_{n, m, p}."""
        r = len(W[0])
        U_out = [[0]*r for _ in range(n*m)]
        V_out = [[0]*r for _ in range(m*p)]
        W_out = [[0]*r for _ in range(p*n)]
        
        for i in range(n_sub):
            for j in range(m_sub):
                idx_u = (i_off + i)*m + (j_off + j)
                for c in range(r):
                    U_out[idx_u][c] = U[i*m_sub + j][c]
                    
        for j in range(m_sub):
            for k in range(p_sub):
                idx_v = (j_off + j)*p + (k_off + k)
                for c in range(r):
                    V_out[idx_v][c] = V[j*p_sub + k][c]
                    
        for k in range(p_sub):
            for i in range(n_sub):
                idx_w = (k_off + k)*n + (i_off + i)
                for c in range(r):
                    W_out[idx_w][c] = W[k*n_sub + i][c]
                    
        return U_out, V_out, W_out

    # Initialize global U, V, W for M_{3,4,7}
    # We will accumulate terms.
    # Total rank will be sum of ranks of all components.
    
    components = []
    
    # 1. A00 * B00 (2x2 * 2x4) -> 2x4
    # This is M_{2,2,4}.
    # We can split B00 (2x4) into B00a (2x2) and B00b (2x2).
    # A00 * B00a: M_{2,2,2} rank 7.
    # A00 * B00b: M_{2,2,2} rank 7.
    # Total 14.
    
    # Embed A00*B00a (i=0, j=0, k=0)
    U1, V1, W1 = _embed_decomp(U_222, V_222, W_222, 3, 4, 7, 0, 0, 0, 2, 2, 2)
    components.append((U1, V1, W1))
    
    # Embed A00*B00b (i=0, j=0, k=2)
    U2, V2, W2 = _embed_decomp(U_222, V_222, W_222, 3, 4, 7, 0, 0, 2, 2, 2, 2)
    components.append((U2, V2, W2))
    
    # 2. A01 * B10 (2x2 * 2x4) -> 2x4
    # A01 is at i=0, j=2. B10 is at j=2, k=0..3.
    # Split B10 into B10a (k=0,1) and B10b (k=2,3).
    
    # A01 * B10a (i=0, j=2, k=0)
    U3, V3, W3 = _embed_decomp(U_222, V_222, W_222, 3, 4, 7, 0, 2, 0, 2, 2, 2)
    components.append((U3, V3, W3))
    
    # A01 * B10b (i=0, j=2, k=2)
    U4, V4, W4 = _embed_decomp(U_222, V_222, W_222, 3, 4, 7, 0, 2, 2, 2, 2, 2)
    components.append((U4, V4, W4))
    
    # 3. A00 * B01 (2x2 * 2x3) -> 2x3
    # M_{2,2,3} rank 12 (schoolbook)
    U_223, V_223, W_223 = _schoolbook_nm_p(2, 2, 3)
    # Embed at i=0, j=0, k=4
    U5, V5, W5 = _embed_decomp(U_223, V_223, W_223, 3, 4, 7, 0, 0, 4, 2, 2, 3)
    components.append((U5, V5, W5))
    
    # 4. A01 * B11 (2x2 * 2x3) -> 2x3
    # Embed at i=0, j=2, k=4
    U6, V6, W6 = _embed_decomp(U_223, V_223, W_223, 3, 4, 7, 0, 2, 4, 2, 2, 3)
    components.append((U6, V6, W6))
    
    # 5. A10 * B00 (1x2 * 2x4) -> 1x4
    # M_{1,2,4} rank 8 (schoolbook)
    U_124, V_124, W_124 = _schoolbook_nm_p(1, 2, 4)
    # Embed at i=2, j=0, k=0
    U7, V7, W7 = _embed_decomp(U_124, V_124, W_124, 3, 4, 7, 2, 0, 0, 1, 2, 4)
    components.append((U7, V7, W7))
    
    # 6. A11 * B10 (1x2 * 2x4) -> 1x4
    # Embed at i=2, j=2, k=0
    U8, V8, W8 = _embed_decomp(U_124, V_124, W_124, 3, 4, 7, 2, 2, 0, 1, 2, 4)
    components.append((U8, V8, W8))
    
    # 7. A10 * B01 (1x2 * 2x3) -> 1x3
    # M_{1,2,3} rank 6 (schoolbook)
    U_123, V_123, W_123 = _schoolbook_nm_p(1, 2, 3)
    # Embed at i=2, j=0, k=4
    U9, V9, W9 = _embed_decomp(U_123, V_123, W_123, 3, 4, 7, 2, 0, 4, 1, 2, 3)
    components.append((U9, V9, W9))
    
    # 8. A11 * B11 (1x2 * 2x3) -> 1x3
    # Embed at i=2, j=2, k=4
    U10, V10, W10 = _embed_decomp(U_123, V_123, W_123, 3, 4, 7, 2, 2, 4, 1, 2, 3)
    components.append((U10, V10, W10))
    
    # Sum all components
    total_rank = sum(len(U[0]) for U, V, W in components)
    
    U_final = [[0]*total_rank for _ in range(n*m)]
    V_final = [[0]*total_rank for _ in range(m*p)]
    W_final = [[0]*total_rank for _ in range(p*n)]
    
    offset = 0
    for U_c, V_c, W_c in components:
        r_c = len(U_c[0])
        for i in range(n*m):
            for c in range(r_c):
                U_final[i][offset+c] = U_c[i][c]
        for j in range(m*p):
            for c in range(r_c):
                V_final[j][offset+c] = V_c[j][c]
        for k in range(p*n):
            for c in range(r_c):
                W_final[k][offset+c] = W_c[k][c]
        offset += r_c
        
    # Verify
    valid, err = _verify_decomp(U_final, V_final, W_final, n, m, p)
    if valid:
        best = {"U": U_final, "V": V_final, "W": W_final}
        best_rank = total_rank
        
    return best
# EVOLVE-BLOCK-END
