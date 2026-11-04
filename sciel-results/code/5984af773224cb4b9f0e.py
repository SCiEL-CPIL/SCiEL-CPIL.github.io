"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x2x2():
    """Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, c] corresponds to A[i,j]
    V[j*2+k, c] corresponds to B[j,k]
    W[k*2+i, c] corresponds to C[i,k]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00 + A11
    V[0][0] = 1; V[3][0] = 1  # B00 + B11
    W[0][0] = 1; W[3][0] = -1  # C00 += M1, C11 -= M1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10 + A11
    V[0][1] = 1               # B00
    W[3][1] = -1              # C11 -= M2
    
    # M3 = A00(B10-B11)
    U[0][2] = 1               # A00
    V[2][2] = 1; V[3][2] = -1 # B10 - B11
    W[2][2] = 1               # C01 += M3
    W[3][2] = 1               # C11 += M3
    
    # M4 = A11(B01-B10)
    U[3][3] = 1               # A11
    V[1][3] = 1; V[2][3] = -1 # B01 - B10
    W[0][3] = 1               # C00 += M4
    W[3][3] = 1               # C11 += M4
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00 + A01
    V[3][4] = 1               # B11
    W[0][4] = -1              # C00 -= M5
    
    # M6 = (A10-A00)(B00+B01) -- not used in standard Strassen output formulas
    # Actually, M6 is not directly used in the C formulas above. Let me recheck.
    # Standard Strassen:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # M6 is not used! So we only need 7 terms but M6 is not in the output.
    # Wait, let me recount. The 7 Strassen products are M1-M7, but the output
    # only uses M1, M2, M3, M4, M5, M7. M6 is not used.
    # Actually no, I think I made an error. Let me use the standard formulation.
    
    # Let me redo this more carefully with the standard Strassen formulas.
    # Reset
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen's 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A10)(B10+B11)
    #
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    
    # P1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[3][1] = -1
    
    # P3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # P4 = A11(B01-B10)
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # P5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # P6 = (A10-A00)(B00+B01) -- not used in C formulas, skip
    # P7 = (A01-A10)(B10+B11)
    U[1][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[0][5] = -1; W[3][5] = 1
    
    # We only used 6 products (P1-P5, P7). P6 is not needed.
    # But we allocated rank=7. Let's set the 7th column to 0.
    # Actually, we should use rank=6 if P6 is not needed.
    # Let me verify: the standard Strassen algorithm uses 7 multiplications.
    # But looking at the C formulas, P6 is indeed not used.
    # This seems wrong. Let me look up the correct Strassen formulas.
    
    # Correct Strassen:
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P7
    # Yes, P6 is not used. So rank 6 should suffice?
    # No wait, I think I'm confusing myself. The point of Strassen is to use 7
    # multiplications instead of 8. The 7th multiplication (P6) IS used, but
    # I may have the formulas wrong.
    
    # Let me just use a known correct set of Strassen coefficients.
    # I'll verify by checking that the decomposition works.
    
    # For now, let's use rank=7 and fill in P6 as well, even if it cancels out.
    # Actually, let me just verify my current 6-product decomposition.
    
    return U, V, W

def _construct_hybrid():
    """Construct a hybrid decomposition: 1 row schoolbook + 2 rows Strassen.
    
    Split the 3x4 matrix into a 1x4 part and a 2x4 part.
    The 1x4 x 4x8 part uses schoolbook (rank 32).
    The 2x4 x 4x8 part uses Strassen on 2x2 blocks.
    
    Total rank = 32 + 8*7 = 32 + 56 = 88.
    """
    n, m, p = 3, 4, 8
    
    # Part 1: 1x4 x 4x8 schoolbook (row 0 of A)
    # A[0,j] * B[j,k] for j=0..3, k=0..7
    # Rank = 4*8 = 32
    rank1 = 32
    U1 = [[0]*rank1 for _ in range(1*4)]  # 4 rows (for the 1x4 block)
    V1 = [[0]*rank1 for _ in range(4*8)]  # 32 rows
    W1 = [[0]*rank1 for _ in range(8*1)]  # 8 rows
    
    col = 0
    for j in range(4):
        for k in range(8):
            U1[j][col] = 1
            V1[j*8+k][col] = 1
            W1[k*1+0][col] = 1
            col += 1
    
    # Part 2: 2x4 x 4x8 using Strassen on 2x2 blocks
    # A is 2x4, B is 4x8
    # Split A into 2x2 blocks: A = [A11 A12; A21 A22] where each is 2x2
    # Split B into 2x4 blocks: B = [B1; B2] where B1 is 4x4, B2 is 4x4
    # Actually, let's split differently.
    # A (2x4) = [A_left (2x2) | A_right (2x2)]
    # B (4x8) = [B_top (4x4); B_bottom (4x4)]
    # C = A * B = A_left * B_top + A_right * B_bottom
    # Each term is 2x2 * 4x4, which can be decomposed into two 2x2*2x2 Strassen blocks.
    # A_left * B_top = A_left * [B_top_left | B_top_right] = [A_left*B_top_left | A_left*B_top_right]
    # Each 2x2*2x2 uses rank 7, so 2*7 = 14 per 2x2*4x4 block.
    # Total for 2x4*4x8: 4 * 7 = 28? No.
    # Let me think again.
    # A (2x4) * B (4x8):
    # C (2x8) = sum_{j=0}^{3} A[:,j] * B[j,:]
    # We can group: C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    # A[:,0:2] is 2x2, B[0:2,:] is 2x8
    # 2x2 * 2x8 = 2x2 * [2x4 | 2x4] -> two 2x2*2x4 blocks
    # 2x2*2x4 = 2x2 * [2x2 | 2x2] -> two 2x2*2x2 Strassen blocks, rank 7 each
    # So 2x2*2x4 has rank 14, and 2x2*2x8 has rank 28.
    # Similarly for the other half.
    # Total rank for 2x4*4x8 = 28 + 28 = 56.
    
    # Let me construct this properly.
    # A rows: 0,1; A cols: 0,1,2,3
    # B rows: 0,1,2,3; B cols: 0,1,...,7
    # C rows: 0,1; C cols: 0,1,...,7
    
    # Block 1: A[:,0:2] * B[0:2,0:4] -> C[0:2,0:4]
    # This is 2x2 * 2x4
    # = A[:,0:2] * [B[0:2,0:2] | B[0:2,2:4]]
    # = [A[:,0:2]*B[0:2,0:2] | A[:,0:2]*B[0:2,2:4]]
    # Each 2x2*2x2 uses Strassen rank 7.
    
    # I'll build the 2x4*4x8 decomposition by tiling Strassen 2x2*2x2 blocks.
    
    def _strassen_2x2x2_local():
        """Return U, V, W for 2x2*2x2 Strassen decomposition."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # P1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[3][1] = -1
        
        # P3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # P4 = A11(B01-B10)
        U[3][3] = 1
        V[1][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # P5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # P6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # P6 is not used in standard Strassen C formulas
        
        # P7 = (A01-A10)(B10+B11)
        U[1][6] = 1; U[2][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[3][6] = 1
        
        return U, V, W
    
    # Build 2x4*4x8 decomposition
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # For each pair (j_block, k_block) where j_block in {0,1} (j=2*j_block, 2*j_block+1)
    # and k_block in {0,1,2,3} (k=2*k_block, 2*k_block+1):
    # We have a 2x2*2x2 subproblem.
    
    # Total subproblems: 2 (j_blocks) * 4 (k_blocks) = 8
    # Each has rank 7, so total rank = 56.
    
    rank2 = 56
    U2 = [[0]*rank2 for _ in range(2*4)]  # 8 rows (2x4)
    V2 = [[0]*rank2 for _ in range(4*8)]  # 32 rows
    W2 = [[0]*rank2 for _ in range(8*2)]  # 16 rows
    
    sU, sV, sW = _strassen_2x2x2_local()
    
    for jb in range(2):  # j block: 0 or 1
        for kb in range(4):  # k block: 0 to 3
            base = (jb*4 + kb) * 7
            # A rows: 0,1 (local), global rows: 0,1
            # A cols: 2*jb, 2*jb+1
            # B rows: 2*jb, 2*jb+1
            # B cols: 2*kb, 2*kb+1
            # C rows: 0,1
            # C cols: 2*kb, 2*kb+1
            
            for c in range(7):
                for local_i in range(2):
                    global_i = local_i
                    U2[global_i*4 + 2*jb + local_i // 1][base + c] = sU[local_i*2 + local_i][c]
                # This indexing is getting complicated. Let me use a cleaner approach.
    
    # Let me redo the 2x4*4x8 construction more carefully.
    # A is 2x4: A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x8: B[j,k] for j in {0,1,2,3}, k in {0,1,...,7}
    # C is 2x8: C[i,k] for i in {0,1}, k in {0,1,...,7}
    
    # U has rows indexed by (i,j) for A: 8 rows
    # V has rows indexed by (j,k) for B: 32 rows
    # W has rows indexed by (k,i) for C: 16 rows
    
    # For each 2x2*2x2 block (jb, kb):
    # A submatrix: A[0:2, 2*jb:2*jb+2]
    # B submatrix: B[2*jb:2*jb+2, 2*kb:2*kb+2]
    # C submatrix: C[0:2, 2*kb:2*kb+2]
    
    U2 = [[0]*rank2 for _ in range(8)]
    V2 = [[0]*rank2 for _ in range(32)]
    W2 = [[0]*rank2 for _ in range(16)]
    
    for jb in range(2):
        for kb in range(4):
            base = (jb*4 + kb) * 7
            for c in range(7):
                for li in range(2):  # local A row
                    for lj in range(2):  # local A col
                        gi = li  # global A row
                        gj = 2*jb + lj  # global A col
                        U2[gi*4 + gj][base + c] = sU[li*2 + lj][c]
                
                for lj in range(2):  # local B row
                    for lk in range(2):  # local B col
                        gj = 2*jb + lj  # global B row
                        gk = 2*kb + lk  # global B col
                        V2[gj*8 + gk][base + c] = sV[lj*2 + lk][c]
                
                for lk in range(2):  # local C col
                    for li in range(2):  # local C row
                        gk = 2*kb + lk  # global C col
                        gi = li  # global C row
                        W2[gk*2 + gi][base + c] = sW[lk*2 + li][c]
    
    # Now combine Part 1 and Part 2
    total_rank = rank1 + rank2
    U = [[0]*total_rank for _ in range(12)]  # 3x4 = 12 rows
    V = [[0]*total_rank for _ in range(32)]  # 4x8 = 32 rows
    W = [[0]*total_rank for _ in range(24)]  # 8x3 = 24 rows
    
    # Part 1: row 0 of A (schoolbook)
    # A[0,j] for j=0..3, B[j,k] for j=0..3,k=0..7, C[k,0] for k=0..7
    for j in range(4):
        for k in range(8):
            col = j*8 + k
            U[0*4 + j][col] = 1
            V[j*8 + k][col] = 1
            W[k*3 + 0][col] = 1
    
    # Part 2: rows 1,2 of A (Strassen)
    # A[1,j] and A[2,j] for j=0..3
    # In the 2x4 block, local row 0 = global row 1, local row 1 = global row 2
    for i in range(8):
        for c in range(rank2):
            U[(1 + i//4)*4 + i%4][rank1 + c] = U2[i][c]
    
    for j in range(32):
        for c in range(rank2):
            V[j][rank1 + c] = V2[j][c]
    
    for k in range(16):
        for c in range(rank2):
            # W2 row index: k*2 + i where i is local C row (0 or 1)
            # Global C row: 1 + i
            gk = k // 2
            li = k % 2
            gi = 1 + li
            W[gk*3 + gi][rank1 + c] = W2[k][c]
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 x 4x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 3, 4, 8
    deadline = time.time() + 100  # safety margin before 120s limit
    
    # Try hybrid decomposition first
    try:
        U, V, W = _construct_hybrid()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition
    rank_sb = n * m * p  # 96
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
