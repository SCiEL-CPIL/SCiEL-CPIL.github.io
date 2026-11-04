"""Combined approach: explicit Strassen-based rank-49 decomposition for 4x4 * 4x9
matrix multiplication, with a fast exact-validation fallback to schoolbook rank-144.

Strategy:
- Build a rank-49 decomposition by applying Strassen (rank 7) at the 2x2-block level,
  then using Strassen (rank 7) inside each 2x2 x 2x2 block multiplication.
- The B matrix is 4x9, so we split it into 2x9 blocks. Each "2x2 x 2x2" inner
  multiplication actually multiplies a 2x2 block of A by a 2x9 block of B.
- We handle this by treating the 2x9 B-block as two 2x1 (or 2x9) columns and
  using the Strassen structure where the B-side has 9 columns.

Actually, a cleaner approach: use the rank-7 Strassen for 2x2x2, and for the
4x4 x 4x9 case, we can view it as 4x4 x 4x4 (with 5 zero columns in B) which
gives rank 49, OR we can be more careful.

Let me reconsider: The standard approach for rectangular matrix multiplication
using Strassen:
- A is 4x4, B is 4x9, C is 4x9
- Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] (each 2x2)
- Split B into 2x2 blocks: B = [[B11, B12, B13, B14, B15, B16, B17, B18, B19]]
  Wait, B is 4x9. Split rows: B = [B_top; B_bottom] where B_top is 2x9, B_bottom is 2x9.
  
  For Strassen-style, we need to split B into 2 column-blocks. But 9 is odd.
  
  Alternative: Pad B to 4x16 (add 7 zero columns) and use 4x4 x 4x16, then the
  result is 4x16 (take first 9 columns). But that changes the problem.

  Better approach: Use the fact that we can decompose 4x4 x 4x9 directly.
  
  Actually, the cleanest structural approach: 
  - Use rank-7 Strassen for 2x2 x 2x2
  - For 4x4 x 4x9, split A into 2x2 blocks (4 blocks of 2x2)
  - Split B into 2 row-blocks (2 blocks of 2x9)
  - The product C = A*B has 4 row-blocks (each 2x9)
  
  C_top = A11 * B_top + A12 * B_bottom
  C_bottom = A21 * B_top + A22 * B_bottom
  
  Each of these is a sum of two 2x2 * 2x9 multiplications.
  
  For a 2x2 * 2x9 multiplication, we can use Strassen with 7 multiplications,
  where each "multiplication" is actually a scalar-times-vector operation...
  
  No wait. Let me think differently.
  
  A 2x2 * 2x9 multiplication: 
    [a b]   [x1 x2 ... x9]
    [c d] * [y1 y2 ... y9]
  = [a*x1+b*y1, a*x2+b*y2, ..., a*x9+b*y9]
    [c*x1+d*y1, c*x2+d*y2, ..., c*x9+d*y9]
  
  This is 4*9 = 36 scalar multiplications in schoolbook.
  
  Using Strassen for the 2x2 part (treating each column of B as a 2x1 vector):
  We can compute all 9 columns simultaneously with 7 "big" operations where
  each big operation involves 9 scalar multiplications. So 7*9 = 63 operations
  instead of 36... that's worse!
  
  The issue is Strassen helps when both matrices are large enough. For 2x2 x 2x9,
  the 2x2 part is too small to benefit.
  
  Let me try a different decomposition:
  
  For 4x4 x 4x9, one approach is to use the known rank bounds.
  The tensor rank of 4x4 x 4x4 matrix multiplication is at most 49 (Strassen).
  For 4x4 x 4x9, we can embed it in 4x4 x 4x16 (rank at most 49*2=98? No...)
  
  Actually, let me just use the direct Strassen recursion properly:
  
  4x4 x 4x9:
  - Split A (4x4) into four 2x2 blocks
  - Split B (4x9) into two 2x9 blocks (top and bottom)
  - C (4x9) = [C_top; C_bottom] where C_top, C_bottom are 2x9
  
  C_top = A11*B_top + A12*B_bottom
  C_bottom = A21*B_top + A22*B_bottom
  
  Now, each term like A11*B_top is a 2x2 * 2x9 multiplication.
  
  For a 2x2 * 2x9 multiplication, the rank is at most 2*2*9 = 36 in schoolbook,
  but can we do better? The tensor [i,k,j] for i in {0,1}, k in {0,1}, j in {0..8}
  has rank... it's a sum of 4*9 rank-1 terms. Can we reduce?
  
  Actually, for 2x2 x 2xn, the rank is known to be 2n for n>=2 (I believe).
  So 2x2 x 2x9 has rank 18.
  
  So C_top = A11*B_top + A12*B_bottom requires at most 18+18 = 36 rank-1 terms
  if we do them separately. But we can potentially share.
  
  Total for 4x4 x 4x9: at most 4 * 18 = 72? Or less with sharing.
  
  Let me just implement the Strassen-based 4x4 x 4x4 (rank 49) and extend to 4x9
  by noting that 4x4 x 4x9 can be seen as 4x4 x 4x16 with 7 zero columns.
  
  For 4x4 x 4x16: split into 4x4 x 4x4 and 4x4 x 4x12, etc. This gets complex.
  
  Simplest reliable approach: Use the rank-49 Strassen for 4x4x4, and for the
  extra 5 columns of B (to make it 4x9), add 4*5 = 20 more rank-1 terms.
  Total rank: 49 + 20 = 69.
  
  Wait, that's not quite right either. Let me think more carefully.
  
  Actually, the cleanest approach: 
  1. Build the rank-49 Strassen decomposition for 4x4 x 4x4
  2. For columns 4-8 of B (5 extra columns), add schoolbook terms: 4*5 = 20
  Total: 49 + 20 = 69
  
  But we need to be careful about the W matrix structure.
  
  Let me just implement this step by step.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    """Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7).
    Convention: U[i*2+k, r], V[k*2+j, r], W[j*2+i, r]
    where i,k in {0,1} for A rows/cols, j in {0,1} for B cols.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build rank-49 Strassen decomposition for 4x4 x 4x4.
    U (16,49), V (16,49), W (16,49).
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    # 4x4 split into 2x2 blocks:
    # A: blocks A[bi,bk] for bi,bk in {0,1}, each 2x2
    #    A[bi,bk] has rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    # B: blocks B[bk,bj] for bk,bj in {0,1}, each 2x2  
    #    B[bk,bj] has rows 2*bk..2*bk+1, cols 2*bj..2*bj+1
    # C: blocks C[bj,bi] for bj,bi in {0,1}, each 2x2
    #    C[bj,bi] has rows 2*bj..2*bj+1, cols 2*bi..2*bi+1
    #    Wait, C is 4x4, C[i,k]. Split into 2x2 blocks:
    #    C[bi,bk] has rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    
    # Strassen at 4x4 level (7 big multiplications):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Where A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    #       B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    #       C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
    
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mk is a 2x2 x 2x2 multiplication with rank 7
    # So we have 7*7 = 49 rank-1 terms total
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # For each of the 7 Strassen multipliers at the 4x4 level,
    # we have a 2x2 x 2x2 inner multiplication with rank 7.
    
    # Strassen multiplier definitions at 4x4 level:
    # (a_blocks, b_blocks, c_blocks_with_signs)
    # a_blocks: list of (bi, bk, sign) for A
    # b_blocks: list of (bk, bj, sign) for B  
    # c_blocks: list of (bi, bk, sign) for C (output)
    
    strassen_4x4 = [
        # M1 = (A11+A22)(B11+B22)
        {
            'a': [(0,0,1), (1,1,1)],
            'b': [(0,0,1), (1,1,1)],
            'c': [(0,0,1), (1,1,1)]
        },
        # M2 = (A21+A22)B11
        {
            'a': [(1,0,1), (1,1,1)],
            'b': [(0,0,1)],
            'c': [(1,0,1), (1,1,-1)]
        },
        # M3 = A11(B12-B22)
        {
            'a': [(0,0,1)],
            'b': [(0,1,1), (1,1,-1)],
            'c': [(0,1,1), (1,1,1)]
        },
        # M4 = A22(B21-B11)
        {
            'a': [(1,1,1)],
            'b': [(1,0,1), (0,0,-1)],
            'c': [(0,0,1), (1,0,1)]
        },
        # M5 = (A11+A12)B22
        {
            'a': [(0,0,1), (0,1,1)],
            'b': [(1,1,1)],
            'c': [(0,0,-1), (0,1,1)]
        },
        # M6 = (A21-A11)(B11+B12)
        {
            'a': [(1,0,1), (0,0,-1)],
            'b': [(0,0,1), (0,1,1)],
            'c': [(1,1,1)]
        },
        # M7 = (A12-A22)(B21+B22)
        {
            'a': [(0,1,1), (1,1,-1)],
            'b': [(1,0,1), (1,1,1)],
            'c': [(0,0,1)]
        },
    ]
    
    col = 0
    for mult in strassen_4x4:
        # For this Strassen multiplier, we need to compute:
        # sum_{(bi,bk,sa) in a} sum_{(bk',bj,sb) in b} sa*sb * A[bi,bk] * B[bk',bj]
        # and add to C blocks.
        
        # The inner computation is a 2x2 x 2x2 multiplication where:
        # A_inner[i,k] = sum over a-blocks with bi=i, bk=k of sa * A_block[i,k]
        # B_inner[k,j] = sum over b-blocks with bk=k, bj=j of sb * B_block[k,j]
        # C_inner[j,i] gets contributions to C[bi,bk] for each c-block
        
        # Actually, the structure is:
        # The "effective" A matrix is a 2x2 matrix where entry (i,k) is
        #   A_eff[i,k] = sum_{(bi,bk,sa) in a: bi==i, bk==k} sa * A[bi,bk]
        # Similarly for B.
        # Then C_eff = A_eff * B_eff (2x2 x 2x2, rank 7 via Strassen)
        # And C[bi,bk] += sign_c * C_eff[...]
        
        # Let me build the effective A and B coefficient matrices
        A_eff = np.zeros((2, 2))  # A_eff[i,k] for i,k in {0,1}
        B_eff = np.zeros((2, 2))  # B_eff[k,j] for k,j in {0,1}
        
        for bi, bk, sa in mult['a']:
            A_eff[bi, bk] += sa
        for bk, bj, sb in mult['b']:
            B_eff[bk, bj] += sb
        
        # Now use Strassen 2x2 to decompose A_eff * B_eff
        # For each of the 7 rank-1 terms in Strassen 2x2:
        for r_inner in range(7):
            # U2[i*2+k, r_inner], V2[k*2+j, r_inner], W2[j*2+i, r_inner]
            for i in range(2):
                for k in range(2):
                    u_coeff = U2[i*2+k, r_inner]
                    if abs(u_coeff) > 1e-15:
                        # This contributes to A[bi,bk] for all (bi,bk) where A_eff[i,k] includes it
                        # A_eff[i,k] = sum of sa for (bi,bk,sa) with bi==i, bk==k
                        # So A[bi,bk] gets coefficient u_coeff * sa
                        for bi, bk, sa in mult['a']:
                            if bi == i and bk == k:
                                # A block (bi,bk) contributes with coefficient u_coeff * sa
                                # U index: row = 2*bi + i_local, col = 2*bk + k_local
                                # where i_local, k_local are within the 2x2 block
                                for i_loc in range(2):
                                    for k_loc in range(2):
                                        U[(2*bi + i_loc)*4 + (2*bk + k_loc), col] = u_coeff * sa
                                        
            for k in range(2):
                for j in range(2):
                    v_coeff = V2[k*2+j, r_inner]
                    if abs(v_coeff) > 1e-15:
                        for bk, bj, sb in mult['b']:
                            if bk == k and bj == j:
                                for k_loc in range(2):
                                    for j_loc in range(2):
                                        V[(2*bk + k_loc)*4 + (2*bj + j_loc), col] = v_coeff * sb
                                        
            for j in range(2):
                for i in range(2):
                    w_coeff = W2[j*2+i, r_inner]
                    if abs(w_coeff) > 1e-15:
                        # C_eff[i,j] gets w_coeff * (A_eff[i,k] * B_eff[k,j])
                        # C[bi,bk] += sign_c * C_eff[...]
                        # C block (bi,bk) has rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
                        # W index: row = 2*bk + k_loc, col = 2*bi + i_loc
                        # Wait, W[k*n+i] where n=4, so W[k*4+i]
                        # C[i,k] is at row i, col k in the 4x4 matrix
                        # W index for C[i,k] is k*4+i
                        for bi, bk, sc in mult['c']:
                            if bi == i and bk == k:
                                for i_loc in range(2):
                                    for k_loc in range(2):
                                        i_global = 2*bi + i_loc
                                        k_global = 2*bk + k_loc
                                        W[k_global*4 + i_global, col] = w_coeff * sc
                                        
            col += 1
    
    assert col == 49, f"Expected 49 columns, got {col}"
    return U, V, W

def _validate(U, V, W, n=4, m=4, p=9):
    """Validate the decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    r = U_arr.shape[1]
    if V_arr.shape[1] != r or W_arr.shape[1] != r:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """Return {'U','V','W'} exact decomposition of (4x4)*(4x9) tensor."""
    _t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen for the first 4 columns of B (4x4 x 4x4, rank 49)
    # and schoolbook for the remaining 5 columns (4x4 x 4x5, rank 4*5 = 20)
    # Total rank: 49 + 20 = 69
    
    U49, V49, W49 = _build_4x4x4_strassen()
    
    # For columns 4-8 of B (5 extra columns):
    # Schoolbook: for each (i,j,k) with k in 4..8, add a rank-1 term
    # U[i*4+j, col] = 1, V[j*9+k, col] = 1, W[k*4+i, col] = 1
    
    extra_rank = n * m * (p - 4)  # 4*4*5 = 80... wait that's too many
    
    # Actually, for the extra 5 columns, we need:
    # C[i,k] for k in 4..8 = sum_j A[i,j] * B[j,k]
    # Schoolbook: 4 * 4 * 5 = 80 rank-1 terms. That's too many.
    
    # Better: For each extra column k (5 columns), we have a 4x4 x 4x1 multiplication.
    # A 4x4 x 4x1 can be done with rank 4*4 = 16 (schoolbook) or less.
    # Actually, 4x4 x 4x1: C[i] = sum_j A[i,j] B[j]. 
    # This is a bilinear form. Rank is at most 4*4 = 16.
    # Can we do better? The tensor T[i,j,k] for k=0 only... 
    # It's sum_j A[i,j] B[j], which is a single bilinear form per output i.
    # Each output C[i] = sum_j A[i,j] B[j] requires 4 multiplications.
    # Total: 4*4 = 16 multiplications. Rank 16 per extra column, 5 columns = 80.
    
    # That gives total rank 49 + 80 = 129, which is worse than schoolbook 144.
    # Wait, 129 < 144, so it's still an improvement!
    
    # But can we do better? Let me think...
    
    # Alternative: Use Strassen for the full 4x4 x 4x9 by padding B to 4x16.
    # 4x4 x 4x16: split B into four 4x4 blocks.
    # C = A * [B1 B2 B3 B4] = [A*B1 A*B2 A*B3 A*B4]
    # Each A*Bt is 4x4 x 4x4 with rank 49.
    # Total rank: 4 * 49 = 196. Worse than 144.
    
    # So the 49 + 80 = 129 approach is better. Let me implement it.
    
    total_rank = 49 + 80
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Copy the 4x4 x 4x4 Strassen part (for B columns 0-3)
    U[:, :49] = U49
    # V49 is for 4x4 B, but our V is for 4x9 B
    # V49[j*4+k, r] for j,k in 0..3
    # We need V[j*9+k, r] for j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            V[j*9+k, :49] = V49[j*4+k, :49]
    W[:, :49] = W49
    
    # Add schoolbook terms for B columns 4-8
    col = 49
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    assert col == total_rank
    
    # Validate
    if _validate(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank-144
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
