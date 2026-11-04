"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Schoolbook decomposition (Rank 24) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1
                V_school[j * p + k, idx] = 1
                W_school[k * n + i, idx] = 1
                idx += 1
    
    def verify(U, V, W):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(U.shape[1]):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    # Try rank-22 decomposition using Strassen on 2x2 blocks
    # A is 2x3, B is 3x4
    # Split: A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    #        B = [B1; B2] where B1 is 2x4, B2 is 1x4
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 * 2x4 = two 2x2 multiplications (rank 7 each = 14)
    # A2*B2: 2x1 * 1x4 (rank 8)
    # Total: 22
    
    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))
    
    def add_u(i, j, c, val):
        U22[i * m + j, c] += val
    def add_v(j, k, c, val):
        V22[j * p + k, c] += val
    def add_w(k, i, c, val):
        W22[k * n + i, c] += val
    
    # Strassen for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b10+b11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    def add_strassen_block(ai, aj, bi, bj, ki, kj, offset):
        """
        Add Strassen decomposition for A[ai:ai+2, aj:aj+2] * B[bi:bi+2, kj:kj+2]
        ai, aj: top-left of A block (rows ai, ai+1; cols aj, aj+1)
        bi, bj: top-left of B block (rows bi, bi+1; cols kj, kj+1) -- note: bj is col offset
        ki, kj: output C block starts at row ki (k-index), col offset in C is implicit
        Actually: C[i, k] where i in {ai, ai+1}, k in {kj, kj+1}
        """
        # A indices: A[ai, aj], A[ai, aj+1], A[ai+1, aj], A[ai+1, aj+1]
        a00 = (ai, aj)
        a01 = (ai, aj+1)
        a10 = (ai+1, aj)
        a11 = (ai+1, aj+1)
        
        # B indices: B[bi, kj], B[bi, kj+1], B[bi+1, kj], B[bi+1, kj+1]
        b00 = (bi, kj)
        b01 = (bi, kj+1)
        b10 = (bi+1, kj)
        b11 = (bi+1, kj+1)
        
        # C indices: C[ai, kj], C[ai, kj+1], C[ai+1, kj], C[ai+1, kj+1]
        # W index: k*n + i
        c00 = (kj, ai)      # C[ai, kj]
        c01 = (kj+1, ai)    # C[ai, kj+1]
        c10 = (kj, ai+1)    # C[ai+1, kj]
        c11 = (kj+1, ai+1)  # C[ai+1, kj+1]
        
        # m1 = (a00+a11)(b00+b11)
        add_u(*a00, offset + 0, 1)
        add_u(*a11, offset + 0, 1)
        add_v(*b00, offset + 0, 1)
        add_v(*b11, offset + 0, 1)
        add_w(*c00, offset + 0, 1)
        add_w(*c11, offset + 0, 1)
        
        # m2 = (a10+a11)b00
        add_u(*a10, offset + 1, 1)
        add_u(*a11, offset + 1, 1)
        add_v(*b00, offset + 1, 1)
        add_w(*c10, offset + 1, 1)
        add_w(*c11, offset + 1, -1)
        
        # m3 = a00(b01-b11)
        add_u(*a00, offset + 2, 1)
        add_v(*b01, offset + 2, 1)
        add_v(*b11, offset + 2, -1)
        add_w(*c01, offset + 2, 1)
        add_w(*c11, offset + 2, 1)
        
        # m4 = a11(b10-b00)
        add_u(*a11, offset + 3, 1)
        add_v(*b10, offset + 3, 1)
        add_v(*b00, offset + 3, -1)
        add_w(*c00, offset + 3, 1)
        add_w(*c10, offset + 3, 1)
        
        # m5 = (a00+a01)b11
        add_u(*a00, offset + 4, 1)
        add_u(*a01, offset + 4, 1)
        add_v(*b11, offset + 4, 1)
        add_w(*c00, offset + 4, -1)
        add_w(*c01, offset + 4, 1)
        
        # m6 = (a10-a00)(b00+b10)
        add_u(*a10, offset + 5, 1)
        add_u(*a00, offset + 5, -1)
        add_v(*b00, offset + 5, 1)
        add_v(*b10, offset + 5, 1)
        add_w(*c11, offset + 5, 1)
        
        # m7 = (a01-a11)(b10+b11)
        add_u(*a01, offset + 6, 1)
        add_u(*a11, offset + 6, -1)
        add_v(*b10, offset + 6, 1)
        add_v(*b11, offset + 6, 1)
        add_w(*c00, offset + 6, 1)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 0, 0, 0, 2, 7)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # This is 2x1 * 1x4, rank 8
    offset = 14
    for i in range(2):
        for k in range(4):
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
    
    # Verify
    if verify(U22, V22, W22):
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
    
    # Try other rank-22 variants or fall back
    # Check time
    if time.time() - t0 > 120:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
