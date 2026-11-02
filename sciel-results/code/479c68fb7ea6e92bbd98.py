import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    start_time = time.time()
    deadline = start_time + 140  # Leave some margin before 150s limit

    # We construct a rank-55 decomposition using:
    # 5 blocks of 2x3x2, each decomposed as:
    #   - Strassen 2x2x2 (rank 7) for the top-left 2x2 sub-block of B
    #   - Schoolbook 2x1x2 (rank 4) for the bottom row of B (column 2)
    # Total per block: 11, total: 55

    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    #
    # In tensor decomposition form for M_{2,2,2}:
    # U is 4x7 (indices i*2+j for i,j in {0,1}), V is 4x7 (indices j*2+k), W is 4x7 (indices k*2+i)
    #
    # For the 2x1x2 part (bottom row of A times column 2 of B):
    # C00 += A02 * B20
    # C01 += A02 * B21
    # C10 += A12 * B20
    # C11 += A12 * B21
    # This is 4 rank-1 terms.

    # Let's build the 2x3x2 block decomposition explicitly.
    # A is 2x3: indices (i,j) for i in {0,1}, j in {0,1,2}
    # B is 3x2: indices (j,k) for j in {0,1,2}, k in {0,1}
    # C is 2x2: indices (i,k) for i in {0,1}, k in {0,1}

    # U: (2*3, R) = (6, R), indexed by i*3 + j
    # V: (3*2, R) = (6, R), indexed by j*2 + k
    # W: (2*2, R) = (4, R), indexed by k*2 + i

    # We'll use R=11 for each 2x3x2 block.
    # Terms 0-6: Strassen for the 2x2 part (j in {0,1})
    # Terms 7-10: Schoolbook for the 2x1x2 part (j=2)

    def make_block_decomp():
        R = 11
        U = np.zeros((6, R), dtype=int)
        V = np.zeros((6, R), dtype=int)
        W = np.zeros((4, R), dtype=int)
        
        # Strassen terms for the 2x2 sub-block (j in {0,1})
        # A indices: i*3+j for i in {0,1}, j in {0,1}
        # B indices: j*2+k for j in {0,1}, k in {0,1}
        # C indices: k*2+i for k in {0,1}, i in {0,1}
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        U[0*3+0, 0] = 1  # A00
        U[1*3+1, 0] = 1  # A11
        V[0*2+0, 0] = 1  # B00
        V[1*2+1, 0] = 1  # B11
        W[0*2+0, 0] = 1  # C00
        W[1*2+1, 0] = 1  # C11
        
        # M2 = (A10 + A11) B00
        U[1*3+0, 1] = 1  # A10
        U[1*3+1, 1] = 1  # A11
        V[0*2+0, 1] = 1  # B00
        W[0*2+1, 1] = 1  # C10
        
        # M3 = A00 (B01 - B11)
        U[0*3+0, 2] = 1  # A00
        V[0*2+1, 2] = 1  # B01
        V[1*2+1, 2] = -1 # B11
        W[0*2+1, 2] = 1  # C01
        
        # M4 = A11 (B10 - B00)
        U[1*3+1, 3] = 1  # A11
        V[1*2+0, 3] = 1  # B10
        V[0*2+0, 3] = -1 # B00
        W[0*2+0, 3] = -1 # C00 (because C00 = M1 - M4 + ...)
        W[1*2+1, 3] = -1 # C11 (because C11 = M1 - M3 + M2 - M5)
        
        # Wait, let me re-derive the Strassen coefficients carefully.
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        
        # Let me redo this more carefully.
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B01 - B11)
        # M4 = A11 (B10 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Reset and do it properly.
        U = np.zeros((6, R), dtype=int)
        V = np.zeros((6, R), dtype=int)
        W = np.zeros((4, R), dtype=int)
        
        # Term 0: M1 = (A00 + A11)(B00 + B11)
        U[0*3+0, 0] = 1
        U[1*3+1, 0] = 1
        V[0*2+0, 0] = 1
        V[1*2+1, 0] = 1
        # W coefficients: C00 += 1, C11 += 1
        W[0*2+0, 0] = 1   # C00
        W[1*2+1, 0] = 1   # C11
        
        # Term 1: M2 = (A10 + A11) B00
        U[1*3+0, 1] = 1
        U[1*3+1, 1] = 1
        V[0*2+0, 1] = 1
        # W coefficients: C10 += 1
        W[0*2+1, 1] = 1   # C10
        
        # Term 2: M3 = A00 (B01 - B11)
        U[0*3+0, 2] = 1
        V[0*2+1, 2] = 1
        V[1*2+1, 2] = -1
        # W coefficients: C01 += 1
        W[0*2+1, 2] = 1   # C01
        
        # Term 3: M4 = A11 (B10 - B00)
        U[1*3+1, 3] = 1
        V[1*2+0, 3] = 1
        V[0*2+0, 3] = -1
        # W coefficients: C00 += -1, C10 += 1
        W[0*2+0, 3] = -1  # C00
        W[0*2+1, 3] = 1   # C10
        
        # Term 4: M5 = (A00 + A01) B11
        U[0*3+0, 4] = 1
        U[0*3+1, 4] = 1
        V[1*2+1, 4] = 1
        # W coefficients: C01 += 1, C11 += -1
        W[0*2+1, 4] = 1   # C01
        W[1*2+1, 4] = -1  # C11
        
        # Term 5: M6 = (A10 - A00)(B00 + B01)
        U[1*3+0, 5] = 1
        U[0*3+0, 5] = -1
        V[0*2+0, 5] = 1
        V[0*2+1, 5] = 1
        # W coefficients: C00 += 1
        W[0*2+0, 5] = 1   # C00
        
        # Term 6: M7 = (A01 - A11)(B10 + B11)
        U[0*3+1, 6] = 1
        U[1*3+1, 6] = -1
        V[1*2+0, 6] = 1
        V[1*2+1, 6] = 1
        # W coefficients: C00 += 1, C11 += 1
        W[0*2+0, 6] = 1   # C00
        W[1*2+1, 6] = 1   # C11
        
        # Now verify: C00 should be M1 - M4 + M6 + M7
        # C00 coefficients: 1 - 1 + 1 + 1 = 2? That's wrong.
        # Let me re-check the Strassen formulas.
        # 
        # Actually, the standard Strassen is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        #
        # Let me expand C00:
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B10 - A11B00
        # M6 = A10B00 + A10B01 - A00B00 - A00B01
        # M7 = A01B10 + A01B11 - A11B10 - A11B11
        #
        # C00 = M1 - M4 + M6 + M7
        #     = (A00B00 + A00B11 + A11B00 + A11B11) - (A11B10 - A11B00) + (A10B00 + A10B01 - A00B00 - A00B01) + (A01B10 + A01B11 - A11B10 - A11B11)
        #     = A00B00 + A00B11 + A11B00 + A11B11 - A11B10 + A11B00 + A10B00 + A10B01 - A00B00 - A00B01 + A01B10 + A01B11 - A11B10 - A11B11
        #     = A00B11 + 2*A11B00 - A11B10 + A10B00 + A10B01 - A00B01 + A01B10 + A01B11
        #
        # This doesn't look right. C00 should be A00B00 + A01B10.
        # I think I have the wrong Strassen formula. Let me look it up.
        #
        # The correct Strassen formulas are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 - M4 + M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 - M5
        #
        # Let me try a different approach. I'll just use a known valid rank-7 decomposition.
        # Actually, let me just use schoolbook for the 2x2 part too, to be safe.
        # Schoolbook for 2x2x2 has rank 4.
        # Total per block: 4 + 4 = 8. Total: 5*8 = 40.
        # That's even better!
        
        # Let's use schoolbook for everything.
        # For the 2x3x2 block, schoolbook rank is 2*3*2 = 12.
        # But we can do better by splitting.
        
        # Actually, let me just implement a clean schoolbook for the full 2x3x10.
        # Schoolbook rank is 60. That's the baseline.
        # 
        # To improve, I'll use Strassen for the 2x2 parts.
        # Let me use a verified rank-7 decomposition for 2x2x2.
        # 
        # I'll use this known rank-7 decomposition:
        # The 7 terms are:
        # t0: A00*A11 + A10*A01 (no, that's not right)
        
        # Let me just use a simple approach: schoolbook for 2x3x2 (rank 12), total 60.
        # Or use the 55-rank approach with careful Strassen.
        
        # I'll implement Strassen correctly this time.
        # From Wikipedia, Strassen's algorithm:
        # m1 = (a11 + a22)(b11 + b22)
        # m2 = (a21 + a22)b11
        # m3 = a11(b12 - b22)
        # m4 = a22(b21 - b11)
        # m5 = (a11 + a12)b22
        # m6 = (a21 - a11)(b11 + b12)
        # m7 = (a12 - a22)(b21 + b22)
        #
        # c11 = m1 - m4 + m6 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m3 + m2 - m5
        
        # Note: using 1-based indexing in the formula, but 0-based in code.
        # a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
        # b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
        # c11=C[0,0], c12=C[0,1], c21=C[1,0], c22=C[1,1]
        
        # Let me re-implement with 0-based indexing:
        # m1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # m2 = (A[1,0]+A[1,1])B[0,0]
        # m3 = A[0,0](B[0,1]-B[1,1])
        # m4 = A[1,1](B[1,0]-B[0,0])
        # m5 = (A[0,0]+A[0,1])B[1,1]
        # m6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        # m7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        #
        # C[0,0] = m1 - m4 + m6 + m7
        # C[0,1] = m3 + m5
        # C[1,0] = m2 + m4
        # C[1,1] = m1 - m3 + m2 - m5
        
        # Now in tensor form:
        # For each term mr, we have U_r (4x1), V_r (4x1), W_r (4x1)
        # where U_r[i*2+j] is the coefficient of A[i,j], etc.
        
        U = np.zeros((6, 11), dtype=int)
        V = np.zeros((6, 11), dtype=int)
        W = np.zeros((4, 11), dtype=int)
        
        # Term 0: m1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        U[0*3+0, 0] = 1  # A[0,0]
        U[1*3+1, 0] = 1  # A[1,1]
        V[0*2+0, 0] = 1  # B[0,0]
        V[1*2+1, 0] = 1  # B[1,1]
        W[0*2+0, 0] = 1  # C[0,0]
        W[1*2+1, 0] = 1  # C[1,1]
        
        # Term 1: m2 = (A[1,0]+A[1,1])B[0,0]
        U[1*3+0, 1] = 1  # A[1,0]
        U[1*3+1, 1] = 1  # A[1,1]
        V[0*2+0, 1] = 1  # B[0,0]
        W[1*2+0, 1] = 1  # C[1,0]
        
        # Term 2: m3 = A[0,0](B[0,1]-B[1,1])
        U[0*3+0, 2] = 1  # A[0,0]
        V[0*2+1, 2] = 1  # B[0,1]
        V[1*2+1, 2] = -1 # B[1,1]
        W[0*2+1, 2] = 1  # C[0,1]
        
        # Term 3: m4 = A[1,1](B[1,0]-B[0,0])
        U[1*3+1, 3] = 1  # A[1,1]
        V[1*2+0, 3] = 1  # B[1,0]
        V[0*2+0, 3] = -1 # B[0,0]
        W[0*2+0, 3] = -1 # C[0,0]
        W[1*2+0, 3] = 1  # C[1,0]
        
        # Term 4: m5 = (A[0,0]+A[0,1])B[1,1]
        U[0*3+0, 4] = 1  # A[0,0]
        U[0*3+1, 4] = 1  # A[0,1]
        V[1*2+1, 4] = 1  # B[1,1]
        W[0*2+1, 4] = 1  # C[0,1]
        W[1*2+1, 4] = -1 # C[1,1]
        
        # Term 5: m6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        U[1*3+0, 5] = 1  # A[1,0]
        U[0*3+0, 5] = -1 # A[0,0]
        V[0*2+0, 5] = 1  # B[0,0]
        V[0*2+1, 5] = 1  # B[0,1]
        W[0*2+0, 5] = 1  # C[0,0]
        
        # Term 6: m7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        U[0*3+1, 6] = 1  # A[0,1]
        U[1*3+1, 6] = -1 # A[1,1]
        V[1*2+0, 6] = 1  # B[1,0]
        V[1*2+1, 6] = 1  # B[1,1]
        W[0*2+0, 6] = 1  # C[0,0]
        W[1*2+1, 6] = 1  # C[1,1]
        
        # Terms 7-10: Schoolbook for j=2 (the third column of A, third row of B)
        # C[0,0] += A[0,2]*B[2,0]
        U[0*3+2, 7] = 1  # A[0,2]
        V[2*2+0, 7] = 1  # B[2,0]
        W[0*2+0, 7] = 1  # C[0,0]
        
        # C[0,1] += A[0,2]*B[2,1]
        U[0*3+2, 8] = 1  # A[0,2]
        V[2*2+1, 8] = 1  # B[2,1]
        W[0*2+1, 8] = 1  # C[0,1]
        
        # C[1,0] += A[1,2]*B[2,0]
        U[1*3+2, 9] = 1  # A[1,2]
        V[2*2+0, 9] = 1  # B[2,0]
        W[1*2+0, 9] = 1  # C[1,0]
        
        # C[1,1] += A[1,2]*B[2,1]
        U[1*3+2, 10] = 1  # A[1,2]
        V[2*2+1, 10] = 1  # B[2,1]
        W[1*2+1, 10] = 1  # C[1,1]
        
        return U, V, W
    
    U_block, V_block, W_block = make_block_decomp()
    R_block = 11
    total_rank = 5 * R_block  # 55
    
    # Construct the full 2x3x10 decomposition
    U_full = np.zeros((n*m, total_rank), dtype=int)
    V_full = np.zeros((m*p, total_rank), dtype=int)
    W_full = np.zeros((p*n, total_rank), dtype=int)
    
    for t in range(5):
        # U is the same for all blocks
        for i in range(n):
            for j in range(m):
                for r in range(R_block):
                    U_full[i*m + j, t*R_block + r] = U_block[i*m + j, r]
        
        # V: block t uses columns 2t and 2t+1 of B
        for j in range(m):
            for k_local in range(2):
                k = 2*t + k_local
                for r in range(R_block):
                    V_full[j*p + k, t*R_block + r] = V_block[j*2 + k_local, r]
        
        # W: block t uses columns 2t and 2t+1 of C
        for k_local in range(2):
            k = 2*t + k_local
            for i in range(n):
                for r in range(R_block):
                    W_full[k*n + i, t*R_block + r] = W_block[k_local*2 + i, r]
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(total_rank):
                    s += U_full[i*m + j, r] * V_full[j*p + k, r] * W_full[k*n + i, r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
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
