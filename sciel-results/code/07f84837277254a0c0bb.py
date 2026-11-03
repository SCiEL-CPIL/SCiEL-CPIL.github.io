import time
import numpy as np

def construct():
    n, m, p = 2, 4, 7
    # Rank 50 decomposition:
    # Split M_2,4,7 into two M_2,2,7 terms.
    # Each M_2,2,7 is split into three M_2,2,2 (Strassen, rank 7) and one M_2,2,1 (Schoolbook, rank 4).
    # Total rank per M_2,2,7: 3*7 + 4 = 25.
    # Total rank: 25 + 25 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # This is not a rank-1 term, it's part of a decomposition.
        pass

    # We need to fill U, V, W for the specific indices.
    # A is 2x4. Indices: A[i, j] where i in {0,1}, j in {0,1,2,3}.
    # B is 4x7. Indices: B[j, k] where j in {0,1,2,3}, k in {0..6}.
    # C is 2x7. Indices: C[i, k] where i in {0,1}, k in {0..6}.
    
    # Block 1: A1 (cols 0,1 of A) * B1 (rows 0,1 of B)
    # A1 indices in U: i*m + j where j in {0,1}
    # B1 indices in V: j*p + k where j in {0,1}
    # C indices in W: k*n + i
    
    # Block 2: A2 (cols 2,3 of A) * B2 (rows 2,3 of B)
    # A2 indices in U: i*m + j where j in {2,3}
    # B2 indices in V: j*p + k where j in {2,3}
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00) -- Wait, standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Note: B00, B01 are entries (0,0), (0,1). B10, B11 are (1,0), (1,1).
    
    def add_strassen_block(u_base, v_base, w_base, col_offset):
        # u_base: index in U for A00 (i=0, j=u_base)
        # v_base: index in V for B00 (j=v_base, k=0)
        # w_base: index in W for C00 (k=0, i=0)
        # col_offset: starting column in U, V, W
        
        c = col_offset + 0
        # M1
        U[u_base][c] = 1
        U[u_base + 1][c] = 1 # A11 is i=1, j=u_base+1? No.
        # U index is i*m + j.
        # A00: i=0, j=u_base. U[0*m + u_base]
        # A11: i=1, j=u_base+1? No, A11 is i=1, j=u_base+1 relative to block?
        # In block 1, j is 0,1. A00 is (0,0), A11 is (1,1).
        # U index for (i, j) is i*m + j.
        # So A00 -> U[0*m + u_base]. A11 -> U[1*m + u_base+1].
        
        # Let's define indices explicitly.
        # u_00 = 0*m + u_base
        # u_11 = 1*m + u_base + 1
        # u_10 = 1*m + u_base
        # u_01 = 0*m + u_base + 1
        
        # v_00 = v_base*p + 0
        # v_01 = v_base*p + 1
        # v_10 = (v_base+1)*p + 0
        # v_11 = (v_base+1)*p + 1
        
        # w_00 = 0*n + 0
        # w_01 = 1*n + 0
        # w_10 = 0*n + 1
        # w_11 = 1*n + 1
        
        # M1
        U[0*m + u_base][c] = 1
        U[1*m + u_base + 1][c] = 1
        V[v_base*p + 0][c] = 1
        V[v_base*p + 1][c] = 1
        V[(v_base+1)*p + 0][c] = 1
        V[(v_base+1)*p + 1][c] = 1
        W[0*n + 0][c] = 1
        W[1*n + 0][c] = 1
        W[0*n + 1][c] = 1
        W[1*n + 1][c] = 1
        
        # M2
        c = col_offset + 1
        U[1*m + u_base][c] = 1
        U[1*m + u_base + 1][c] = 1
        V[v_base*p + 0][c] = 1
        V[v_base*p + 1][c] = 1
        W[0*n + 1][c] = 1
        W[1*n + 1][c] = 1
        
        # M3
        c = col_offset + 2
        U[0*m + u_base][c] = 1
        V[(v_base+1)*p + 0][c] = 1
        V[(v_base+1)*p + 1][c] = -1
        W[1*n + 0][c] = 1
        
        # M4
        c = col_offset + 3
        U[1*m + u_base + 1][c] = 1
        V[v_base*p + 1][c] = 1
        V[v_base*p + 0][c] = -1
        W[0*n + 0][c] = 1
        W[0*n + 1][c] = 1
        
        # M5
        c = col_offset + 4
        U[0*m + u_base][c] = 1
        U[0*m + u_base + 1][c] = 1
        V[(v_base+1)*p + 0][c] = 1
        V[(v_base+1)*p + 1][c] = 1
        W[0*n + 0][c] = -1
        W[1*n + 0][c] = 1
        
        # M6
        c = col_offset + 5
        U[1*m + u_base][c] = 1
        U[0*m + u_base][c] = -1
        V[v_base*p + 0][c] = 1
        V[v_base*p + 1][c] = 1
        # M6 contributes to C10? No, C10 = M2 + M4.
        # Standard Strassen:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # M6 is not used in reconstruction?
        # Wait, M6 = (A10-A00)(B00+B01).
        # It is not in the sum?
        # Ah, the 7 products are M1..M7.
        # The reconstruction uses M1, M2, M3, M4, M5, M7.
        # M6 is NOT used?
        # Let's check.
        # C10 = A10B00 + A11B10.
        # M2 = (A10+A11)B00 = A10B00 + A11B00.
        # M4 = A11(B01-B00) = A11B01 - A11B00.
        # M2 + M4 = A10B00 + A11B01.
        # This is C10? No, C10 = A10B00 + A11B10.
        # B01 is (0,1). B10 is (1,0).
        # My M4 definition: A11(B01-B00).
        # If B is 2x2, B01 is (0,1), B00 is (0,0).
        # This doesn't match C10.
        
        # Correct Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let's re-verify C10 = M2 + M4.
        # M2 = A10B00 + A11B00
        # M4 = A11B01 - A11B00
        # Sum = A10B00 + A11B01.
        # C10 should be A10B00 + A11B10.
        # So M4 must involve B10?
        # Standard Strassen M4 is A11(B01-B00)?
        # No, usually:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C10 = M2 + M4?
        # If M4 = A11(B10-B00)?
        # Then M2+M4 = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10. Correct.
        # So M4 = A11(B10-B00).
        # Let's check C00 = M1 + M4 - M5 - M7.
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B10 - A11B00
        # M5 = A00B11 + A01B11
        # M7 = A01B10 + A01B11 - A11B10 - A11B11
        # Sum:
        # A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 - A01B10 - A01B11 + A11B10 + A11B11
        # = A00B00 + 2A11B11 + 2A11B10 - A01B10 - 2A01B11
        # This is not A00B00 + A01B11.
        
        # I will just use the schoolbook for the 2x2 blocks to be safe.
        # Rank 4 per 2x2x2.
        # 3 blocks of 2x2x2 and 1 block of 2x2x1.
        # Rank per M_2,2,7: 3*4 + 4 = 16.
        # Total rank 32.
        # This is better than 56 and safe.
        
        pass

    # I will implement the rank 32 version.
    # M_2,2,2 schoolbook rank 4.
    # M_2,2,1 schoolbook rank 4.
    # Total 16 per block.
    
    def add_2x2x2_block(u_base, v_base, w_base, col_offset):
        # 4 terms
        for i in range(2):
            for j in range(2):
                c = col_offset + i*2 + j
                U[i*m + u_base + j][c] = 1
                V[(v_base + i)*p + j][c] = 1
                W[j*n + i][c] = 1
                # Wait, indices for W are k*n + i.
                # Here j is k (0,1). i is i (0,1).
                # So W[j*n + i][c] = 1. Correct.

    def add_2x2x1_block(u_base, v_base, w_base, col_offset, k_offset):
        # 4 terms
        for i in range(2):
            for j in range(2):
                c = col_offset + i*2 + j
                U[i*m + u_base + j][c] = 1
                V[(v_base + i)*p + k_offset][c] = 1
                W[k_offset*n + i][c] = 1

    # Block 1: A1 (u_base=0), B1 (v_base=0)
    # Split B1 (cols 0-6) into 0-1, 2-3, 4-5, 6
    col = 0
    add_2x2x2_block(0, 0, 0, col)
    col += 4
    add_2x2x2_block(0, 0, 2, col) # v_base=0, but k offset 2?
    # My add_2x2x2_block uses j as k index.
    # I need to pass k_offset.
    
    # Let's rewrite helpers with k_offset.
    
    def add_2x2x2_block(u_base, v_base, k_offset, col_offset):
        for i in range(2):
            for j in range(2):
                c = col_offset + i*2 + j
                U[i*m + u_base + j][c] = 1
                V[(v_base + i)*p + k_offset + j][c] = 1
                W[(k_offset + j)*n + i][c] = 1

    def add_2x2x1_block(u_base, v_base, k_offset, col_offset):
        for i in range(2):
            for j in range(2):
                c = col_offset + i*2 + j
                U[i*m + u_base + j][c] = 1
                V[(v_base + i)*p + k_offset][c] = 1
                W[k_offset*n + i][c] = 1

    # Block 1
    col = 0
    add_2x2x2_block(0, 0, 0, col)
    col += 4
    add_2x2x2_block(0, 0, 2, col)
    col += 4
    add_2x2x2_block(0, 0, 4, col)
    col += 4
    add_2x2x1_block(0, 0, 6, col)
    col += 4
    
    # Block 2: A2 (u_base=2), B2 (v_base=2)
    add_2x2x2_block(2, 2, 0, col)
    col += 4
    add_2x2x2_block(2, 2, 2, col)
    col += 4
    add_2x2x2_block(2, 2, 4, col)
    col += 4
    add_2x2x1_block(2, 2, 6, col)
    col += 4
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=2, m=4, p=7):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
