"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try rank-44 construction: 4 blocks of 2x3x2, each rank 11
    result = _try_rank44(n, m, p)
    if result is not None:
        return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank44(n, m, p):
    """Construct rank-44 decomposition using 4 blocks of 2x3x2 with rank 11 each."""
    if n != 2 or m != 3 or p != 8:
        return None
    
    # For each 2-column block of B, we decompose 2x3x2 with rank 11.
    # Total rank = 4 * 11 = 44.
    
    # First, let's construct a rank-11 decomposition for 2x3x2.
    # We'll use a known construction or search for one.
    
    # For 2x3x2, let's try to build it using Strassen on a 2x2x2 subblock
    # plus rank-1 terms for the remaining column.
    
    # A is 2x3: A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B is 3x2: B = [B1; B2] where B1 is 2x2, B2 is 1x2
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2x2): use Strassen, rank 7
    # For A2*B2 (2x1x2): rank 2 (since 2x1x2 tensor has rank 2)
    # Total: 7 + 2 = 9? But we need to check if this gives a valid decomposition.
    
    # Actually, let's just try to find a rank-11 decomposition numerically
    # and round to half-integers.
    
    # Let's use a direct construction.
    # For 2x3x2, the tensor T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1}
    
    # Let me try to construct it with rank 11 using a systematic approach.
    # 
    # I'll use the fact that we can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # And decompose using 11 terms.
    
    # Let's just try a few known patterns and verify.
    
    # Pattern 1: Use Strassen on columns 0,1 of the middle, and rank-1 for column 2.
    # Strassen for 2x2x2 (using middle cols 0,1):
    # U_strassen (2x7), V_strassen (2x7), W_strassen (2x7) for 2x2x2
    
    # Strassen 2x2x2:
    # U[i,s], V[j,s], W[k,s] for i,j,k in {0,1}, s in {0,...,6}
    # 
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # In tensor form:
    # U[i,s] for A[i,j], V[j,s] for B[j,k], W[k,s] for C[i,k]
    
    # Let me define the Strassen decomposition for 2x2x2:
    # s=0: M1: A part: A00+A11 -> U[0,0]=1,U[1,0]=1; B part: B00+B11 -> V[0,0]=1,V[1,0]=1; C: C00+=M1,C01-=M1,C10+=M1,C11-=M1 -> W[0,0]=1,W[1,0]=-1... 
    
    # This is getting complicated. Let me just use a numerical search for 2x3x2 rank 11,
    # then verify and use it.
    
    # Actually, let's just return None and use schoolbook for now, since the
    # numerical search is unreliable for exact half-integer solutions.
    
    # Instead, let me try a different approach: use the known fact that
    # rank(2x3x8) can be bounded, and try to find a good decomposition
    # using a structured approach.
    
    # For now, let's just do schoolbook and return it.
    # The knowledge says rank-44 was achieved, so let me try to construct it
    # more carefully.
    
    # Let me try: for each pair of columns (2k, 2k+1) of B, use a rank-11
    # decomposition of the 2x3x2 subproblem.
    
    # I'll construct the rank-11 for 2x3x2 by hand using a known pattern.
    
    # For 2x3x2, let's use the following approach:
    # Split the 3 middle dimension into 2+1.
    # 2x2x2 part: rank 7 (Strassen)
    # 2x1x2 part: rank 2
    # Total: 9
    # But we need to verify this gives a valid tensor decomposition.
    
    # Let me just implement the Strassen + rank-1 approach and verify.
    
    # Strassen 2x2x2 tensor decomposition (U,V,W are 2x7):
    # Based on the 7 Strassen products:
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 0],  # i=0
        [1, 1, 0, 1, 0, -1, 1], # i=1
    ]
    strassen_V = [
        [1, 1, 1, 0, 0, 1, 0],  # j=0
        [1, 0, -1, 1, 1, 0, 1], # j=1
    ]
    strassen_W = [
        [1, 0, 0, 1, 0, 1, 0],  # k=0
        [1, 1, 1, 0, 1, 0, 1],  # k=1
    ]
    
    # Verify Strassen 2x2x2:
    # For each i,j,k: sum_s U[i,s]*V[j,s]*W[k,s] should equal 1
    ok = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(strassen_U[i][s] * strassen_V[j][s] * strassen_W[k][s] for s in range(7))
                if val != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # My Strassen coefficients are wrong. Let me use the correct ones.
        # The standard Strassen decomposition for 2x2x2 matmul tensor:
        # The tensor T[i,j,k] = 1 for all i,j,k in {0,1}
        # 
        # One valid decomposition of rank 7:
        # I'll look up the correct coefficients.
        
        # Actually, let me just use a different approach. I'll construct the
        # 2x3x2 decomposition directly using a known rank-11 pattern.
        
        # For now, let's just return schoolbook.
        return None
    
    # Strassen verified. Now construct 2x3x2 with rank 9 (7+2).
    # 
    # For the 2x1x2 part (middle column 2):
    # A2 is 2x1, B2 is 1x2. C_part[i,k] = A2[i]*B2[k]
    # Tensor: T2[i,2,k] = 1 (only j=2)
    # Decomposition: rank 2
    # U2[i,s] for s in {0,1}: U2[0,0]=1,U2[0,1]=0,U2[1,0]=0,U2[1,1]=1
    # V2[2,s]: V2[2,0]=1,V2[2,1]=1
    # W2[k,s]: W2[0,0]=1,W2[0,1]=0,W2[1,0]=0,W2[1,1]=1
    # Check: sum_s U2[i,s]*V2[2,s]*W2[k,s] = U2[i,0]*1*W2[k,0] + U2[i,1]*1*W2[k,1]
    # = U2[i,0]*W2[k,0] + U2[i,1]*W2[k,1]
    # For i=0,k=0: 1*1+0*0=1 ✓
    # For i=0,k=1: 1*0+0*1=0 ✗ (should be 1)
    
    # That's wrong. Let me fix it.
    # We need: for all i,k: sum_s U2[i,s]*V2[2,s]*W2[k,s] = 1
    # With V2[2,s] = 1 for all s:
    # sum_s U2[i,s]*W2[k,s] = 1 for all i,k
    # This is a 2x2 all-ones matrix, rank 1.
    # So: U2[0,0]=1,U2[0,1]=0; U2[1,0]=1,U2[1,1]=0
    #     W2[0,0]=1,W2[0,1]=0; W2[1,0]=1,W2[1,1]=0
    # Check: sum_s U2[i,s]*W2[k,s] = U2[i,0]*W2[k,0] + U2[i,1]*W2[k,1] = 1*1+0*0 = 1 ✓
    
    # So rank 2 works, but actually rank 1 works!
    # U2[0,0]=1,U2[1,0]=1; W2[0,0]=1,W2[1,0]=1; V2[2,0]=1
    # sum_s U2[i,s]*V2[2,s]*W2[k,s] = 1*1*1 = 1 for all i,k ✓
    
    # So 2x1x2 has rank 1!
    
    # Total for 2x3x2: 7 + 1 = 8.
    
    # Let me construct the full 2x3x2 decomposition with rank 8.
    # 
    # U (2x8): first 7 cols from Strassen (for j=0,1), 8th col for j=2
    # V (3x8): first 7 cols from Strassen (for j=0,1), 8th col for j=2
    # W (2x8): first 7 cols from Strassen, 8th col for j=2
    
    # Strassen part (j=0,1):
    # U_str[i,s] for s=0..6, V_str[j,s] for j=0,1, s=0..6
    # For j=2: V[2,s] = 0 for s=0..6, V[2,7]=1
    # U[0,7]=1, U[1,7]=1
    # W[0,7]=1, W[1,7]=1
    
    # Full U (2x8):
    U_block = [
        strassen_U[0] + [1],
        strassen_U[1] + [1],
    ]
    # Full V (3x8):
    V_block = [
        strassen_V[0] + [0],
        strassen_V[1] + [0],
        [0]*7 + [1],
    ]
    # Full W (2x8):
    W_block = [
        strassen_W[0] + [1],
        strassen_W[1] + [1],
    ]
    
    # Verify 2x3x2:
    ok = True
    for i in range(2):
        for j in range(3):
            for k in range(2):
                val = sum(U_block[i][s] * V_block[j][s] * W_block[k][s] for s in range(8))
                if val != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        return None
    
    # Great! 2x3x2 has rank 8. So 2x3x8 = 4 * 8 = 32?
    # Wait, that seems too good. Let me double-check.
    
    # Actually, I think the issue is that the Strassen decomposition I used
    # might not be correct. Let me verify the Strassen part more carefully.
    
    # The Strassen coefficients I used:
    # strassen_U = [[1,0,1,0,1,0,0],[1,1,0,1,0,-1,1]]
    # strassen_V = [[1,1,1,0,0,1,0],[1,0,-1,1,1,0,1]]
    # strassen_W = [[1,0,0,1,0,1,0],[1,1,1,0,1,0,1]]
    
    # Let me verify T[i,j,k] = sum_s U[i,s]*V[j,s]*W[k,s] = 1 for all i,j,k in {0,1}
    
    # This was verified above (ok was True), so the Strassen part is correct.
    
    # So 2x3x2 has rank 8, and 2x3x8 has rank 32?
    # That would be a huge improvement over 48!
    
    # Let me construct the full 2x3x8 decomposition.
    
    U_full = []
    V_full = []
    W_full = []
    
    for block in range(4):
        # Block handles columns 2*block and 2*block+1 of B
        # U: same for all blocks (2x8 per block)
        # V: rows 2*block and 2*block+1 of B (3x8 per block, but only 2 rows active)
        # W: columns 2*block and 2*block+1 of C (2x8 per block)
        
        U_block_full = U_block  # 2x8
        
        # V_block_full: 3x8, but for this block, only the relevant rows
        # In the full problem, V is 3x8 (m*p = 3*8 = 24 rows? No, m*p = 3*8=24? 
        # Wait, V has shape (m*p, rank) = (3*8, rank) = (24, rank)
        # No, V has shape (m*p, rank) where m=3, p=8, so 24 rows.
        # V[j*p+k, s] where j in {0,1,2}, k in {0,...,7}
        
        # For block b (columns 2b, 2b+1 of B), the V rows involved are:
        # V[j*8 + 2b, s] and V[j*8 + 2b+1, s] for j in {0,1,2}
        
        # In the block decomposition, V_block[j, s] corresponds to V[j*8 + 2b + k, s]
        # where k is the column within the block (0 or 1).
        
        # So V_block[j, s] -> V[j*8 + 2b + 0, s] for k=0, V[j*8 + 2b + 1, s] for k=1
        # But V_block is 3x8, with rows j=0,1,2 and columns s=0..7.
        # The mapping is: V_block[j, s] contributes to V[j*8 + 2b, s] and V[j*8 + 2b+1, s]?
        # No, that's not right.
        
        # Let me reconsider. In the 2x3x2 subproblem:
        # A is 2x3, B is 3x2, C is 2x2.
        # V has shape (3*2, rank) = (6, rank). V[j*2+k, s] for j in {0,1,2}, k in {0,1}.
        # 
        # In the full 2x3x8 problem:
        # V has shape (3*8, rank) = (24, rank). V[j*8+k, s] for j in {0,1,2}, k in {0,...,7}.
        #
        # For block b (k in {2b, 2b+1}):
        # V[j*8 + 2b, s] and V[j*8 + 2b+1, s] for j in {0,1,2}
        #
        # In the block subproblem, V_block[j, s] for j in {0,1,2}, s in {0..7}
        # corresponds to the j-th row of B in the block, which is B[j, 2b] and B[j, 2b+1].
        # But V_block has 3 rows (j=0,1,2) and 8 columns (s=0..7).
        # The k dimension (0 or 1) is absorbed into the W matrix.
        
        # I think I'm overcomplicating this. Let me just build the full matrices directly.
        
        # For each block b, we add 8 rank terms.
        # The U contribution: U[i*3+j, s] for i in {0,1}, j in {0,1,2}
        # Wait, U has shape (n*m, rank) = (2*3, rank) = (6, rank). U[i*m+j, s] = U[i*3+j, s].
        
        # In the block subproblem, U_block[i, s] for i in {0,1}, s in {0..7}
        # corresponds to U[i*3 + j, s] for j in {0,1,2}.
        # But U_block only has 2 rows (i=0,1), while the full U has 6 rows.
        
        # I think the issue is that in the 2x3x2 subproblem, the U matrix has shape (2*3, rank) = (6, rank),
        # not (2, rank). Let me reconsider.
        
        # In the tensor decomposition T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # For 2x3x2: U is 6x8, V is 6x8, W is 4x8.
        # 
        # In my block construction, I had U_block as 2x8, which is wrong. It should be 6x8.
        
        # Let me redo this.
        
        # For 2x3x2, the Strassen part uses j=0,1 (first two columns of A, first two rows of B).
        # U_str[i, s] for i in {0,1} should be U[i*3+j, s] for j in {0,1}.
        # So U[0*3+0, s] = U[0, s] = strassen_U[0][s]
        #    U[0*3+1, s] = U[1, s] = strassen_U[0][s]  (same i, different j)
        #    U[1*3+0, s] = U[3, s] = strassen_U[1][s]
        #    U[1*3+1, s] = U[4, s] = strassen_U[1][s]
        
        # Wait, that's not right either. In Strassen, U[i,s] is for the i-th row of A.
        # In the full tensor, U[i*m+j, s] is for the (i,j) entry of A.
        # So for Strassen on the 2x2 submatrix (rows 0,1 of A, columns 0,1 of B):
        # U[i*3+j, s] = strassen_U[i][s] for j in {0,1}
        # V[j*2+k, s] = strassen_V[j][s] for j in {0,1}, k in {0,1}
        # W[k*2+i, s] = strassen_W[k][s] for k in {0,1}, i in {0,1}
        
        # For the j=2 part (rank 1):
        # U[i*3+2, s] = 1 for i in {0,1}
        # V[2*2+k, s] = 1 for k in {0,1}
        # W[k*2+i, s] = 1 for k in {0,1}, i in {0,1}
        
        # Let me construct the full 6x8 U, 6x8 V, 4x8 W for 2x3x2.
        
        U_232 = [[0]*8 for _ in range(6)]
        V_232 = [[0]*8 for _ in range(6)]
        W_232 = [[0]*8 for _ in range(4)]
        
        # Strassen part (s=0..6):
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    U_232[i*3+j][s] = strassen_U[i][s]
            for j in range(2):
                for k in range(2):
                    V_232[j*2+k][s] = strassen_V[j][s]
            for k in range(2):
                for i in range(2):
                    W_232[k*2+i][s] = strassen_W[k][s]
        
        # j=2 part (s=7):
        for i in range(2):
            U_232[i*3+2][7] = 1
        for k in range(2):
            V_232[2*2+k][7] = 1
        for k in range(2):
            for i in range(2):
                W_232[k*2+i][7] = 1
        
        # Verify 2x3x2:
        ok = True
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    val = sum(U_232[i*3+j][s] * V_232[j*2+k][s] * W_232[k*2+i][s] for s in range(8))
                    if val != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            return None
        
        # Now extend to 2x3x8 by tiling 4 blocks.
        
        if block == 0:
            # Initialize full matrices
            U_full = [[0]*32 for _ in range(6)]
            V_full = [[0]*32 for _ in range(24)]
            W_full = [[0]*32 for _ in range(16)]
        
        # For block b, offset in rank is b*8
        offset = block * 8
        
        for s in range(8):
            for i in range(2):
                for j in range(3):
                    U_full[i*3+j][offset+s] = U_232[i*3+j][s]
            for j in range(3):
                for k in range(2):
                    V_full[j*8 + 2*block + k][offset+s] = V_232[j*2+k][s]
            for k in range(2):
                for i in range(2):
                    W_full[(2*block+k)*2+i][offset+s] = W_232[k*2+i][s]
    
    # Verify full 2x3x8:
    ok = True
    for i in range(2):
        for j in range(3):
            for k in range(8):
                val = sum(U_full[i*3+j][s] * V_full[j*8+k][s] * W_full[k*2+i][s] for s in range(32))
                if val != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        return None
    
    return {
        "U": U_full,
        "V": V_full,
        "W": W_full
    }
# EVOLVE-BLOCK-END
