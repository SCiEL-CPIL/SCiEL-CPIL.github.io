"""Matrix multiplication 2x2x2 rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The matrix multiplication tensor T_{i,j,k} = delta_{i,k} (for multiplying
    A[i,j] * B[j,k] -> C[i,k]) can be decomposed as:
        T = sum_{r=1}^{rank} U_r \otimes V_r \otimes W_r
    
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    and the convention is W[k*n + i, r].
    
    Strassen's algorithm gives rank 7 for 2x2 matrix multiplication.
    The 7 products are:
        M1 = (A00 - A11)(B10 + B11)
        M2 = (A10 + A11) B00
        M3 = (A00 + A01) B11
        M4 = A00 (B01 - B11)
        M5 = A11 (B10 - B00)
        M6 = (A01 - A11)(B10 + B11)
        M7 = (A10 - A00)(B00 + B01)
    
    And the output:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M6
        C10 = M2 + M5
        C11 = M1 - M3 + M2 + M6
    
    Let's index:
        A indices: (i, j) with i in {0,1}, j in {0,1} -> index i*2+j
        B indices: (j, k) with j in {0,1}, k in {0,1} -> index j*2+k
        C indices: (i, k) with i in {0,1}, k in {0,1} -> output index k*2+i (per W convention)
    
    U[i*2+j, r] = coefficient of A[i,j] in product r
    V[j*2+k, r] = coefficient of B[j,k] in product r
    W[k*2+i, r] = coefficient of product r in C[i,k]
    
    Let's write out the 7 rank-1 terms:
    
    M1 = (A00 - A11)(B10 + B11)
        U: A00: +1, A11: -1
        V: B10: +1, B11: +1
        C: C00: +1, C01: +1 (wait, let me recheck)
        
    Actually, let me be more careful. The product M1 contributes to:
        C00 gets M1, C01 gets M1? No.
        
    Let me use the standard Strassen formulas:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M6
        C10 = M2 + M5
        C11 = M1 - M3 + M2 + M6
    
    So:
        M1: U: A00:+1, A11:-1; V: B10:+1, B11:+1; W: C00:+1, C11:+1
        M2: U: A10:+1, A11:+1; V: B00:+1; W: C10:+1, C11:+1
        M3: U: A00:+1, A01:+1; V: B11:+1; W: C01:+1, C11:-1
        M4: U: A00:+1; V: B01:+1, B11:-1; W: C00:+1
        M5: U: A11:+1; V: B10:+1, B00:-1; W: C10:+1, C00:-1
        M6: U: A01:+1, A11:-1; V: B10:+1, B11:+1; W: C01:+1, C11:+1
        M7: U: A10:+1, A00:-1; V: B00:+1, B01:+1; W: C00:+1
    
    Now let's build U (shape 4x7), V (shape 4x7), W (shape 4x7):
    
    A indices: A00=0, A01=1, A10=2, A11=3
    B indices: B00=0, B01=1, B10=2, B11=3
    C indices (W convention: k*2+i): C00 (k=0,i=0)=0, C01 (k=1,i=0)=1, C10 (k=0,i=1)=2, C11 (k=1,i=1)=3
    
    Wait, W[k*n+i, r] where n=2. So W index = k*2 + i.
    C[i,k] corresponds to W[k*2+i].
    C00: i=0,k=0 -> W[0]
    C01: i=0,k=1 -> W[1]
    C10: i=1,k=0 -> W[2]
    C11: i=1,k=1 -> W[3]
    
    U (4x7):
    M1: A00:+1, A11:-1 -> [1, 0, 0, -1]
    M2: A10:+1, A11:+1 -> [0, 0, 1, 1]
    M3: A00:+1, A01:+1 -> [1, 1, 0, 0]
    M4: A00:+1 -> [1, 0, 0, 0]
    M5: A11:+1 -> [0, 0, 0, 1]
    M6: A01:+1, A11:-1 -> [0, 1, 0, -1]
    M7: A10:+1, A00:-1 -> [-1, 0, 1, 0]
    
    V (4x7):
    M1: B10:+1, B11:+1 -> [0, 0, 1, 1]
    M2: B00:+1 -> [1, 0, 0, 0]
    M3: B11:+1 -> [0, 0, 0, 1]
    M4: B01:+1, B11:-1 -> [0, 1, 0, -1]
    M5: B10:+1, B00:-1 -> [-1, 0, 1, 0]
    M6: B10:+1, B11:+1 -> [0, 0, 1, 1]
    M7: B00:+1, B01:+1 -> [1, 1, 0, 0]
    
    W (4x7):
    M1: C00:+1, C11:+1 -> [1, 0, 0, 1]
    M2: C10:+1, C11:+1 -> [0, 0, 1, 1]
    M3: C01:+1, C11:-1 -> [0, 1, 0, -1]
    M4: C00:+1 -> [1, 0, 0, 0]
    M5: C10:+1, C00:-1 -> [-1, 0, 1, 0]
    M6: C01:+1, C11:+1 -> [0, 1, 0, 1]
    M7: C00:+1 -> [1, 0, 0, 0]
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1
    U[0][0] = 1   # A00
    U[3][0] = -1  # A11
    V[2][0] = 1   # B10
    V[3][0] = 1   # B11
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11
    
    # M2
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00
    W[2][1] = 1   # C10
    W[3][1] = 1   # C11
    
    # M3
    U[0][2] = 1   # A00
    U[1][2] = 1   # A01
    V[3][2] = 1   # B11
    W[1][2] = 1   # C01
    W[3][2] = -1  # C11
    
    # M4
    U[0][3] = 1   # A00
    V[1][3] = 1   # B01
    V[3][3] = -1  # B11
    W[0][3] = 1   # C00
    
    # M5
    U[3][4] = 1   # A11
    V[0][4] = -1  # B00
    V[2][4] = 1   # B10
    W[0][4] = -1  # C00
    W[2][4] = 1   # C10
    
    # M6
    U[1][5] = 1   # A01
    U[3][5] = -1  # A11
    V[2][5] = 1   # B10
    V[3][5] = 1   # B11
    W[1][5] = 1   # C01
    W[3][5] = 1   # C11
    
    # M7
    U[0][6] = -1  # A00
    U[2][6] = 1   # A10
    V[0][6] = 1   # B00
    V[1][6] = 1   # B01
    W[0][6] = 1   # C00
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
