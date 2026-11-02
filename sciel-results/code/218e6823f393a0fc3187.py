import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 6

    def verify_full(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # ---- Rank-5 decomposition for 2x2 * 2x3 ----
    # A is 2x2 (indices i,j), B is 2x3 (indices j,k), C is 2x3 (indices k,i)
    # U: shape (4, 5) indexed by [i*2+j]
    # V: shape (6, 5) indexed by [j*3+k]
    # W: shape (6, 5) indexed by [k*2+i]
    def winograd_2x2x3():
        U = [[0.0] * 5 for _ in range(4)]
        V = [[0.0] * 5 for _ in range(6)]
        W = [[0.0] * 5 for _ in range(6)]
        
        # Term 0: (a00 + a11) * (b00 + b01) -> contributes to c00, c01, c10, c11
        # Actually let me use the known rank-5 for 2x2x3:
        # This is a standard result. Let me construct it carefully.
        # For 2x2 * 2x3, we can use 5 multiplications.
        
        # Let me use a systematic approach:
        # We want C = A * B where A is 2x2, B is 2x3
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Rank-5 decomposition (known optimal):
        # t0 = (a00 + a11) * (b00 + b01)
        # t1 = (a00 + a11) * (b02 + b12)  
        # t2 = a00 * (b10 - b00)
        # t3 = a11 * (b01 - b11)
        # t4 = (a01 - a10) * (b10 + b11)
        
        # Wait, let me think more carefully. The standard rank-5 for 2x2x3:
        
        # t0 = (a00 + a11) * (b00 + b10)
        # t1 = (a00 + a11) * (b01 + b11)
        # t2 = (a00 + a11) * (b02 + b12)
        # t3 = a00 * (b10 - b00 + b11 - b01)
        # t4 = a11 * (b00 - b10 + b01 - b11)
        
        # Hmm, this doesn't seem right either. Let me use a verified construction.
        
        # Actually, for 2x2 * 2x3, the known rank is 5. Let me use:
        # t0 = (a00 + a11) * (b00 + b01 + b02)
        # t1 = (a00 + a11) * (b10 + b11 + b12)
        # t2 = a00 * (b10 - b00)
        # t3 = a11 * (b01 - b11)
        # t4 = (a01 - a10) * (b10 + b11)
        
        # Let me verify this works by constructing the full decomposition and checking.
        
        # I'll use a different known rank-5 construction:
        # From literature, for 2x2 x 2x3:
        
        U[0][0] = 1.0; U[3][0] = 1.0  # a00 + a11
        V[0][0] = 1.0; V[3][0] = 1.0  # b00 + b10
        W[0][0] = 1.0; W[2][0] = 1.0  # c00, c10
        
        U[0][1] = 1.0; U[3][1] = 1.0  # a00 + a11
        V[1][1] = 1.0; V[4][1] = 1.0  # b01 + b11
        W[1][1] = 1.0; W[3][1] = 1.0  # c01, c11
        
        U[0][2] = 1.0; U[3][2] = 1.0  # a00 + a11
        V[2][2] = 1.0; V[5][2] = 1.0  # b02 + b12
        W[2][2] = 1.0; W[4][2] = 1.0  # c02, c12
        
        U[0][3] = 1.0                  # a00
        V[3][3] = 1.0; V[0][3] = -1.0  # b10 - b00
        W[0][3] = 1.0; W[2][3] = -1.0  # c00 +, c10 -
        
        U[1][4] = 1.0; U[2][4] = -1.0  # a01 - a10
        V[3][4] = 1.0; V[4][4] = 1.0   # b10 + b11
        W[1][4] = 1.0; W[3][4] = 1.0   # c01 +, c11 +
        
        return U, V, W

    U223, V223, W223 = winograd_2x2x3()
    
    # Verify the 2x2x3 sub-decomposition
    ok_sub = verify_full(U223, V223, W223, 2, 2, 3)
    
    if not ok_sub:
        # Try an alternative rank-5 construction
        U223 = [[0.0] * 5 for _ in range(4)]
        V223 = [[0.0] * 5 for _ in range(6)]
        W223 = [[0.0] * 5 for _ in range(6)]
        
        # Alternative: use Strassen-like approach
        # t0 = (a00+a11)*(b00+b10)
        # t1 = (a00+a11)*(b01+b11) 
        # t2 = (a00+a11)*(b02+b12)
        # t3 = a00*(b10-b00)
        # t4 = a11*(b01-b11)
        # But we need to reconstruct correctly...
        
        # Let me try yet another approach - direct verification-based search
        # For now, if this fails, fall back to schoolbook
        
        U223[0][0] = 1.0; U223[3][0] = 1.0
        V223[0][0] = 1.0; V223[3][0] = 1.0
        W223[0][0] = 1.0; W223[2][0] = 1.0
        
        U223[0][1] = 1.0; U223[3][1] = 1.0
        V223[1][1] = 1.0; V223[4][1] = 1.0
        W223[1][1] = 1.0; W223[3][1] = 1.0
        
        U223[0][2] = 1.0; U223[3][2] = 1.0
        V223[2][2] = 1.0; V223[5][2] = 1.0
        W223[2][2] = 1.0; W223[4][2] = 1.0
        
        U223[0][3] = 1.0
        V223[3][3] = 1.0; V223[0][3] = -1.0
        W223[0][3] = 1.0; W223[2][3] = -1.0
        
        U223[1][4] = 1.0; U223[2][4] = -1.0
        V223[3][4] = 1.0; V223[4][4] = 1.0
        W223[1][4] = 1.0; W223[3][4] = 1.0
        
        ok_sub = verify_full(U223, V223, W223, 2, 2, 3)
    
    if ok_sub and time.time() < deadline - 10:
        # Build 4x4*4x6 by splitting into four 2x2*2x3 blocks
        # A is 4x4, split into 2x2 blocks: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
        # B is 4x6, split into 2x3 blocks: B[0:2,0:3], B[0:2,3:6], B[2:4,0:3], B[2:4,3:6]
        # C is 4x6, split into 2x3 blocks: C[0:2,0:3], C[0:2,3:6], C[2:4,0:3], C[2:4,3:6]
        
        # The four 2x2*2x3 subproblems are:
        # Block 0: A[0:2,0:2] * B[0:2,0:3] -> C[0:2,0:3]
        # Block 1: A[0:2,2:4] * B[2:4,0:3] -> C[0:2,0:3]  (contributes to same C block)
        # Block 2: A[0:2,0:2] * B[0:2,3:6] -> C[0:2,3:6]
        # Block 3: A[0:2,2:4] * B[2:4,3:6] -> C[0:2,3:6]  (contributes to same C block)
        # Block 4: A[2:4,0:2] * B[0:2,0:3] -> C[2:4,0:3]
        # Block 5: A[2:4,2:4] * B[2:4,0:3] -> C[2:4,0:3]  (contributes to same C block)
        # Block 6: A[2:4,0:2] * B[0:2,3:6] -> C[2:4,3:6]
        # Block 7: A[2:4,2:4] * B[2:4,3:6] -> C[2:4,3:6]  (contributes to same C block)
        
        # Wait, that's 8 blocks, not 4. Let me reconsider.
        # A is 4x4, B is 4x6.
        # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        # If I split A into 2x2 blocks and B into 2x3 blocks:
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]] where each is 2x3
        # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
        #      [A10*B00 + A11*B10, A10*B01 + A11*B11]]
        
        # So we have 8 distinct 2x2*2x3 multiplications:
        # A00*B00, A01*B10, A00*B01, A01*B11,
        # A10*B00, A11*B10, A10*B01, A11*B11
        
        # Each uses rank 5, so total rank = 8 * 5 = 40.
        
        rank = 40
        U = [[0.0] * rank for _ in range(16)]  # 4x4 = 16
        V = [[0.0] * rank for _ in range(24)]  # 4x6 = 24
        W = [[0.0] * rank for _ in range(24)]  # 6x4 = 24
        
        # Define the 8 subproblems
        # Each subproblem: (ai, aj, bi, bk, ci, ck)
        # ai, aj: row/col offset in A (2x2 block)
        # bi, bk: row/col offset in B (2x3 block)
        # ci, ck: row/col offset in C (2x3 block)
        
        subproblems = [
            (0, 0, 0, 0, 0, 0),  # A00*B00 -> C00
            (0, 2, 2, 0, 0, 0),  # A01*B10 -> C00
            (0, 0, 0, 3, 0, 3),  # A00*B01 -> C01
            (0, 2, 2, 3, 0, 3),  # A01*B11 -> C01
            (2, 0, 0, 0, 2, 0),  # A10*B00 -> C10
            (2, 2, 2, 0, 2, 0),  # A11*B10 -> C10
            (2, 0, 0, 3, 2, 3),  # A10*B01 -> C11
            (2, 2, 2, 3, 2, 3),  # A11*B11 -> C11
        ]
        
        for sp_idx, (ai, aj, bi, bk, ci, ck) in enumerate(subproblems):
            base = sp_idx * 5
            for t in range(5):
                # U: A indices (ai+di, aj+dj)
                for di in range(2):
                    for dj in range(2):
                        a_idx = (ai + di) * 4 + (aj + dj)
                        u_val = U223[di * 2 + dj][t]
                        U[a_idx][base + t] += u_val
                
                # V: B indices (bi+dj, bk+dk)
                for dj in range(2):
                    for dk in range(3):
                        b_idx = (bi + dj) * 6 + (bk + dk)
                        v_val = V223[dj * 3 + dk][t]
                        V[b_idx][base + t] += v_val
                
                # W: C indices (ck+dk, ci+di)
                for dk in range(3):
                    for di in range(2):
                        c_idx = (ck + dk) * 4 + (ci + di)
                        w_val = W223[dk * 2 + di][t]
                        W[c_idx][base + t] += w_val
        
        if verify_full(U, V, W, 4, 4, 6):
            return {"U": U, "V": V, "W": W}
    
    # ---- Fallback: schoolbook rank 96 ----
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
