"""Optimized tensor decomposition for 2x2 * 2x16 matrix multiplication."""
import time
import random

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x16 matrix.
    
    The matrix multiplication tensor T is defined as:
    T[i, j, k, l] = 1 if and only if the output element (i,l) depends on A[i,j] * B[j,k]
    
    For C = A * B where A is 2x2, B is 2x16:
    C[i, l] = sum_j A[i,j] * B[j,l]
    
    The tensor has dimensions n x m x p = 2 x 2 x 16
    
    We seek rank-r decomposition:
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    such that T[i,j,k,l] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Note the W index convention: W[k*n + i, c] where output index is (k, i) with k in [0,p), i in [0,n)
    So output element C[i, k] = sum_j A[i,j] * B[j,k]
    
    The schoolbook rank is n*m*p = 2*2*16 = 64.
    
    For rectangular matrix multiplication, the minimum rank is known to be:
    - For 2x2 * 2x2: rank 7 (Strassen)
    - For 2x2 * 2xp: can be done in less than 4p
    
    Let's use the fact that we can reuse Strassen-like tricks.
    
    For 2x2 * 2x16, we can group the 16 columns into 8 groups of 2 columns each.
    For each group of 2 columns, we multiply 2x2 * 2x2 which can be done in rank 7.
    Total rank: 8 * 7 = 56.
    
    But we can do better by sharing across groups.
    
    Actually, let's think more carefully. The tensor is:
    T[i, j, k, l] where i in [0,2), j in [0,2), k in [0,16), l in [0,2)
    T[i,j,k,l] = delta_{j,k//8?} ... no, let me re-read the convention.
    
    The output factor follows W[k*n+i, r] where k is the column index of B (0..p-1=0..15) and i is the row index of A/output (0..n-1=0..1).
    
    So the tensor T has indices (i, j, k) where:
    - i: row of A, 0..n-1
    - j: column of A / row of B, 0..m-1  
    - k: column of B, 0..p-1
    
    T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k]
    
    This is just the standard matrix multiplication tensor. For each (i,j,k), there's exactly one nonzero entry.
    
    The rank of this tensor for 2x2*2x16:
    
    Known results:
    - 2x2*2x2 has rank 7
    - 2x2*2xp has rank at most 7*p/2 for even p (using Strassen on pairs of columns)
    
    For p=16: 7*8 = 56
    
    But we might do even better. Let's think about it differently.
    
    Actually, for 2x2 * 2xp, we can view this as computing 16 outputs, each being a 2x2 matrix-vector product.
    Each 2x2 * 2x1 (vector) product can be done in rank 3 (since 2x2*2x1 is just 4 scalar mults but can be done in 3: (a+b)(c+d)-ac-bd gives ad).
    
    Wait, 2x2 * 2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1], C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    This is 4 scalar multiplications, but can be done in 3:
    m1 = A[0,0]*B[0]
    m2 = A[0,1]*B[1]
    m3 = (A[0,0]+A[0,1])*(B[0]+B[1])
    Then C[0] = m1 + m2, C[1] = ... no that doesn't work for different rows.
    
    Actually for 2x2*2x1, the minimum rank is 4 (it's just 4 independent bilinear forms).
    
    Let me reconsider. The tensor for 2x2*2x16 has 2*2*16 = 64 nonzero entries.
    
    Using Strassen on pairs of columns: group columns 0-1, 2-3, ..., 14-15.
    For each pair, we compute 2x2 * 2x2 in rank 7.
    Total: 8 * 7 = 56.
    
    Can we do better? The asymptotic rank for 2x2*2xp is known to be approximately 7p/2 for large p, but for small p there might be better constructions.
    
    Let me just implement the Strassen-based construction with rank 56.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # Group p columns into pairs: p/2 = 8 groups
    num_groups = p // 2  # 8
    
    # For each group, use Strassen 2x2*2x2 decomposition (rank 7)
    # Strassen: M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B10-B11), M4=A11(B01-B00)
    #           M5=(A00+A01)B11, M6=(A10-A00)(B00+B10), M7=(A01-A11)(B01+B11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    
    # Strassen rank-7 decomposition for 2x2*2x2:
    # U factors (4 entries per factor, indexed by i*2+j):
    # V factors (4 entries per factor, indexed by j*2+k):
    # W factors (4 entries per factor, indexed by k*2+i):
    
    # Let me define the Strassen decomposition explicitly.
    # For 2x2*2x2, the 7 rank-1 tensors are:
    
    # Using standard Strassen:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # U[i*2+j] for the 7 factors:
    strassen_U = [
        [1, 0, 0, 1],  # m1: a00+a11
        [0, 0, 1, 1],  # m2: a10+a11
        [1, 0, 0, 0],  # m3: a00
        [0, 0, 0, 1],  # m4: a11
        [1, 1, 0, 0],  # m5: a00+a01
        [0, -1, 1, 0], # m6: a10-a00
        [0, 1, 0, -1], # m7: a01-a11
    ]
    
    # V[j*2+k] for the 7 factors:
    strassen_V = [
        [1, 1, 0, 0],  # m1: b00+b11
        [1, 0, 0, 0],  # m2: b00
        [0, 0, 1, -1], # m3: b10-b11
        [0, 1, -1, 0], # m4: b01-b00
        [0, 0, 0, 1],  # m5: b11
        [1, 0, 1, 0],  # m6: b00+b10
        [0, 1, 0, 1],  # m7: b01+b11
    ]
    
    # W[k*2+i] for the 7 factors:
    # Output c[i,k], W index is k*2+i
    # c00 = m1 + m4 - m5 + m7 -> W[0*2+0]=W[0]
    # c01 = m3 + m5 -> W[1*2+0]=W[2]
    # c10 = m2 + m4 -> W[0*2+1]=W[1]
    # c11 = m1 - m2 + m3 + m6 -> W[1*2+1]=W[3]
    
    strassen_W = [
        [1, 0, 0, 0],   # m1: c00
        [0, 0, 0, 0],   # m2: c10
        [0, 1, 0, 0],   # m3: c01
        [0, 0, 1, 0],   # m4: c10
        [0, 0, 0, 0],   # m5: c01
        [0, 0, 0, 0],   # m6: c11
        [0, 0, 0, 0],   # m7: c00
    ]
    
    # Let me redo W more carefully:
    # W[k*2+i, c] where c is the factor index
    # For each of the 7 factors, W has 4 entries (k=0,1; i=0,1)
    
    # Factor 0 (m1): contributes to c00 (+1) and c11 (+1)
    # Factor 1 (m2): contributes to c10 (+1) and c11 (-1)
    # Factor 2 (m3): contributes to c01 (+1) and c11 (+1)
    # Factor 3 (m4): contributes to c00 (+1) and c10 (+1)
    # Factor 4 (m5): contributes to c00 (-1) and c01 (+1)
    # Factor 5 (m6): contributes to c11 (+1)
    # Factor 6 (m7): contributes to c00 (+1)
    
    # W[k*2+i, factor]:
    # k=0,i=0 (c00): m1:+1, m4:+1, m5:-1, m7:+1
    # k=0,i=1 (c10): m2:+1, m4:+1
    # k=1,i=0 (c01): m3:+1, m5:+1
    # k=1,i=1 (c11): m1:+1, m2:-1, m3:+1, m6:+1
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # k=0,i=0 (c00)
        [0, 1, 0, 1, 0, 0, 0],   # k=0,i=1 (c10)
        [0, 0, 1, 0, 1, 0, 0],   # k=1,i=0 (c01)
        [1, -1, 1, 0, 0, 1, 0],  # k=1,i=1 (c11)
    ]
    
    # Now build the full decomposition for 2x2*2x16
    rank = num_groups * 7  # 8 * 7 = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    for g in range(num_groups):
        base_col = g * 7
        # U: same for all groups (A is shared)
        for i in range(n):
            for j in range(m):
                for c in range(7):
                    U[i * m + j][base_col + c] = strassen_U[c][i * m + j]
        
        # V: for group g, columns 2g and 2g+1 of B
        for j in range(m):
            for k in range(2):
                for c in range(7):
                    V[j * p + (2 * g + k)][base_col + c] = strassen_V[c][j * 2 + k]
        
        # W: for group g, columns 2g and 2g+1 of output
        for k in range(2):
            for i in range(n):
                for c in range(7):
                    W[(2 * g + k) * n + i][base_col + c] = strassen_W[k * 2 + i][c]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, rank):
        # Check T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for c in range(rank):
                        actual += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
