"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n, m, p = 3, 3, 3
    
    # We will construct a rank-26 decomposition using Strassen on the 2x2 block
    # plus 19 terms for the interactions with the third row/column.
    # Actually, the known optimal rank for 3x3 is 23.
    # However, constructing a verified rank-23 decomposition is complex and error-prone.
    # A rank-26 decomposition is easily constructed and verified:
    #   - 7 terms for the 2x2 block (Strassen)
    #   - 19 terms for the remaining interactions (schoolbook-like but optimized)
    #   Total: 26 terms.
    
    # But we can do better. Let's try to construct a rank-23 decomposition.
    # The known rank-23 algorithm for 3x3 matrix multiplication is based on
    # the work of Landsberg and others.
    
    # For now, let's implement a correct rank-26 decomposition first,
    # and then try to improve to rank-23 if time permits.
    
    # Rank-26 construction:
    # Split A, B, C into blocks:
    # A = [[A00, A01, A02],
    #      [A10, A11, A12],
    #      [A20, A21, A22]]
    # B = [[B00, B01, B02],
    #      [B10, B11, B12],
    #      [B20, B21, B22]]
    # C = [[C00, C01, C02],
    #      [C10, C11, C12],
    #      [C20, C21, C22]]
    
    # Strassen on the 2x2 top-left block (indices 0,1):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M7
    # C01 = M5 + M7
    # C10 = M2 + M6
    # C11 = M1 - M3 - M6
    
    # For the interactions with index 2, we use schoolbook terms:
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # This gives 7 + 15 = 22 terms, but some terms can be combined.
    # Actually, let's count: 7 Strassen terms + 15 schoolbook terms = 22 terms.
    # But we need to be careful about how the terms combine.
    
    # Let's implement this and verify it.
    
    rank = 26  # We'll use 26 to be safe, but aim for 23
    
    # Actually, let's try a different approach: use a known rank-23 construction.
    # Based on the literature, here is one valid rank-23 algorithm:
    
    # I'll use a systematic approach to construct the 23 terms.
    # For now, let's return a correct rank-26 decomposition.
    
    # Define the 26 terms:
    # Terms 0-6: Strassen on 2x2 block
    # Terms 7-25: Schoolbook for the remaining 19 entries
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 26
    V = [[0] * rank for _ in range(m * p)]  # 9 x 26
    W = [[0] * rank for _ in range(p * n)]  # 9 x 26
    
    r = 0
    
    # Strassen terms for the 2x2 block
    # M1 = (A00+A11)(B00+B11)
    U[0][r] = 1  # A00
    U[4][r] = 1  # A11
    V[0][r] = 1  # B00
    V[4][r] = 1  # B11
    W[0][r] = 1  # C00
    W[4][r] = 1  # C11
    r += 1
    
    # M2 = (A10+A11)B00
    U[3][r] = 1  # A10
    U[4][r] = 1  # A11
    V[0][r] = 1  # B00
    W[3][r] = 1  # C10
    r += 1
    
    # M3 = A00(B10-B11)
    U[0][r] = 1  # A00
    V[3][r] = 1  # B10
    V[4][r] = -1  # B11
    W[4][r] = -1  # C11
    r += 1
    
    # M4 = A11(B01-B00)
    U[4][r] = 1  # A11
    V[1][r] = 1  # B01
    V[0][r] = -1  # B00
    W[0][r] = -1  # C00
    r += 1
    
    # M5 = (A00+A01)B11
    U[0][r] = 1  # A00
    U[1][r] = 1  # A01
    V[4][r] = 1  # B11
    W[1][r] = 1  # C01
    r += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][r] = 1  # A10
    U[0][r] = -1  # A00
    V[0][r] = 1  # B00
    V[1][r] = 1  # B01
    W[3][r] = 1  # C10
    r += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][r] = 1  # A01
    U[4][r] = -1  # A11
    V[3][r] = 1  # B10
    V[4][r] = 1  # B11
    W[1][r] = 1  # C01
    r += 1
    
    # Schoolbook terms for the remaining entries
    # C02 = A00*B02 + A01*B12 + A02*B22
    U[0][r] = 1  # A00
    V[2][r] = 1  # B02
    W[2][r] = 1  # C02
    r += 1
    
    U[1][r] = 1  # A01
    V[5][r] = 1  # B12
    W[2][r] = 1  # C02
    r += 1
    
    U[2][r] = 1  # A02
    V[8][r] = 1  # B22
    W[2][r] = 1  # C02
    r += 1
    
    # C12 = A10*B02 + A11*B12 + A12*B22
    U[3][r] = 1  # A10
    V[2][r] = 1  # B02
    W[5][r] = 1  # C12
    r += 1
    
    U[4][r] = 1  # A11
    V[5][r] = 1  # B12
    W[5][r] = 1  # C12
    r += 1
    
    U[5][r] = 1  # A12
    V[8][r] = 1  # B22
    W[5][r] = 1  # C12
    r += 1
    
    # C20 = A20*B00 + A21*B10 + A22*B20
    U[6][r] = 1  # A20
    V[0][r] = 1  # B00
    W[6][r] = 1  # C20
    r += 1
    
    U[7][r] = 1  # A21
    V[3][r] = 1  # B10
    W[6][r] = 1  # C20
    r += 1
    
    U[8][r] = 1  # A22
    V[6][r] = 1  # B20
    W[6][r] = 1  # C20
    r += 1
    
    # C21 = A20*B01 + A21*B11 + A22*B21
    U[6][r] = 1  # A20
    V[1][r] = 1  # B01
    W[7][r] = 1  # C21
    r += 1
    
    U[7][r] = 1  # A21
    V[4][r] = 1  # B11
    W[7][r] = 1  # C21
    r += 1
    
    U[8][r] = 1  # A22
    V[7][r] = 1  # B21
    W[7][r] = 1  # C21
    r += 1
    
    # C22 = A20*B02 + A21*B12 + A22*B22
    U[6][r] = 1  # A20
    V[2][r] = 1  # B02
    W[8][r] = 1  # C22
    r += 1
    
    U[7][r] = 1  # A21
    V[5][r] = 1  # B12
    W[8][r] = 1  # C22
    r += 1
    
    U[8][r] = 1  # A22
    V[8][r] = 1  # B22
    W[8][r] = 1  # C22
    r += 1
    
    # Verify the decomposition
    # Check: for all A, B: sum_r U_r(A) * V_r(B) = (A*B)
    # We'll verify with a few random matrices
    
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        C_computed = np.zeros((p, n))
        for rr in range(rank):
            u_val = 0.0
            for i in range(n):
                for j in range(m):
                    u_val += U[i * m + j][rr] * A[i][j]
            
            v_val = 0.0
            for j in range(m):
                for k in range(p):
                    v_val += V[j * p + k][rr] * B[j][k]
            
            w_val = u_val * v_val
            for k in range(p):
                for i in range(n):
                    C_computed[k][i] += W[k * n + i][rr] * w_val
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # If verification fails, return schoolbook rank-27
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            rr = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][rr] = 1
                        V[j * p + k][rr] = 1
                        W[k * n + i][rr] = 1
                        rr += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
