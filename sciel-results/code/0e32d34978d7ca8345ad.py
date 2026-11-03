"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy:
- Use Strassen (rank 7) for 2x2*2x2 blocks.
- Use schoolbook (rank 4) for 2x2*2x1 blocks.
- Tile 4x4*4x6 into four 2x3 output blocks, each computed as
  sum of two (2x2)*(2x3) products. Each (2x2)*(2x3) product is
  decomposed as Strassen 2x2*2x2 (rank 7) + schoolbook 2x2*2x1 (rank 4) = rank 11.
- Total rank: 4 blocks * 2 products per block * 11 = 88.

Wait, let me recount. Each output block Ci,j (2x3) = A_block * B_block1 + A_block2 * B_block2.
Each A_block * B_block (2x2 * 2x3) = 11 terms.
So each Ci,j = 2 * 11 = 22 terms.
4 blocks * 22 = 88.

Hmm, that's higher than the current 44. Let me re-examine.

Actually looking at the current code more carefully:
- product_2x2_2x3 is called 8 times (for each of the 8 A_block*B_block products)
- Each call adds 11 terms
- Total = 8 * 11 = 88

But the current program claims score 1.0909 which is 96/88 ≈ 1.0909. So rank is 88.

The strategy suggests: "Apply a direct 4x4x4 Strassen decomposition (rank 28) to the first four columns of B, and combine it with a rank-16 schoolbook decomposition for the remaining 2x4x2 block, yielding a total rank of 44."

Let me think about this:
- 4x4 * 4x4: Strassen recursively gives rank 7*7 = 49? No, that's not right either.
  Actually, Strassen for 2x2*2x2 is rank 7. For 4x4*4x4, we tile into four 2x2*2x2 blocks.
  C11 = A11*B11 + A12*B21
  C12 = A11*B12 + A12*B22
  C21 = A21*B11 + A22*B21
  C22 = A21*B12 + A22*B22
  
  Each Ci,j = sum of two 2x2*2x2 products = 2*7 = 14 terms.
  4 blocks * 14 = 56 terms.
  
  But wait, there's a better way. The 4x4*4x4 multiplication can be done with rank 49 by Kronecker product of two Strassen, or with rank 56 by the naive tiling. But actually, there might be a direct rank-28 decomposition? No, that seems too low.

  Actually, for 4x4*4x4, the best known is rank 49 (Kronecker of Strassen) or possibly lower. The schoolbook is 64.

  Hmm, the strategy says "rank 28" for 4x4x4 Strassen. That doesn't sound right for standard Strassen. Let me reconsider.

  Wait, maybe they mean: for the 4x4*4x4 part, use Strassen recursively. 4x4*4x4 = 4 blocks of 2x2*2x2. Each block needs 2*7=14 terms. Total 4*14=56. That's not 28.

  Or maybe: treat 4x4*4x4 as a single Strassen-like operation. Actually, I think the "rank 28" might be incorrect or referring to something else.

  Let me try a different interpretation: 
  - 4x4 * 4x6 = (4x4 * 4x4) + (4x4 * 4x2)
  - For 4x4 * 4x4: use tiling with Strassen. 4 output blocks of 2x2, each = sum of 2 Strassen products = 2*7=14. Total 4*14=56.
  - For 4x4 * 4x2: schoolbook would be 4*4*2=32. Or tile: 4 output blocks of 2x1, each = sum of 2 schoolbook 2x2*2x1 = 2*4=8. Total 4*8=32.
  - Total: 56+32 = 88. Same as before.

  Alternatively:
  - For 4x4 * 4x2: split into 2x2*2x1 blocks. 4 output blocks (2x1), each = sum of 2 (2x2*2x1) schoolbook = 2*4 = 8. Total 4*8 = 32.
  
  Hmm, I'm getting 88 either way with this tiling approach.

  Let me try yet another approach: 
  - 4x4 * 4x6: Split B into B_left (4x4) and B_right (4x2).
  - C_left (4x4) = A * B_left: Use Kronecker Strassen. 4x4*4x4 has rank 49 via Kronecker of two rank-7 Strassen.
  - C_right (4x2) = A * B_right: Schoolbook rank 4*4*2 = 32.
  - Total: 49 + 32 = 81.

  Or:
  - C_right (4x2): Tile into 4 blocks of 2x1. Each 2x1 block = sum of 2 (2x2*2x1) = 2*4 = 8. Total 4*8 = 32.
  
  So 49+32 = 81. That's better than 88!

  Can we do even better for 4x4*4x2?
  - 4x4 * 4x2: Split A into 2x2 blocks (4 blocks), B_right into 2x2 and 2x... wait, B_right is 4x2.
  - Split B_right into B1 (2x2) and B2 (2x2) vertically? No, B_right is 4x2, so split rows: B_top (2x2), B_bot (2x2).
  - C_right (4x2) = A * B_right where A is 4x4, B_right is 4x2.
  - C_top (2x2) = A_top * B_top + A_bot... wait, no.
  
  Let me be more careful. A is 4x4, B_right is 4x2.
  A = [[A11, A12], [A21, A22]] (each 2x2)
  B_right = [[B1], [B2]] where B1 is 2x2, B2 is 2x2.
  
  C_right (4x2) = A * B_right
  C_top (2x2) = A11*B1 + A12*B2
  C_bot (2x2) = A21*B1 + A22*B2
  
  Each Ci (2x2) = sum of two 2x2*2x2 products.
  Using Strassen: each 2x2*2x2 = 7 terms. So each Ci = 2*7 = 14 terms.
  Total for C_right: 2*14 = 28 terms.
  
  Total: 49 + 28 = 77. Even better!

  Can we do better for 4x4*4x4? The Kronecker Strassen gives 49. Is there a lower rank?
  The best known for 4x4*4x4 is actually 49 (or possibly lower with more advanced techniques, but 49 is the standard).
  
  So total rank = 49 + 28 = 77.

  Let me verify: 96/77 ≈ 1.2468. That would be a significant improvement over 1.0909.

  Let me implement this:
  1. 4x4*4x4 via Kronecker Strassen (rank 49)
  2. 4x4*4x2 via tiling with Strassen (rank 28)
  Total: 77.

  For the Kronecker Strassen 4x4*4x4:
  The 4x4*4x4 tensor is the Kronecker product of two 2x2*2x2 tensors.
  If the 2x2*2x2 tensor has rank-7 decomposition (U1, V1, W1), then the 4x4*4x4 tensor has rank-49 decomposition where:
  U = U1 ⊗ U1 (in some appropriate sense)
  
  Actually, let me think about this more carefully. The matrix multiplication tensor for n×m×p is:
  T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the indices correspond to A[i,j], B[j,k], C[i,k].
  
  For 4x4*4x4, we can write it as a sum of 16 terms of 2x2*2x2 tensors (one for each 2x2 block combination). But that gives 16*7 = 112, which is worse.
  
  The Kronecker product approach: if we have a rank-r decomposition of the 2x2*2x2 tensor, then the 4x4*4x4 tensor (which is the "square" of the 2x2*2x2 tensor in a sense) has a rank-r^2 decomposition.
  
  Specifically, if T_2 = sum_{s=1}^7 u_s ⊗ v_s ⊗ w_s is the Strassen decomposition of 2x2*2x2, then:
  T_4 = sum_{s,t=1}^7 (u_s ⊗ u_t) ⊗ (v_s ⊗ v_t) ⊗ (w_s ⊗ w_t)
  
  where the tensor products are in the appropriate index spaces.
  
  For A (4x4), the index is (i1,i2,j1,j2) -> flat index. If we split i = i1*2+i2, j = j1*2+j2, then:
  A[i,j] = A[i1,i2,j1,j2]
  
  The Kronecker product u_s ⊗ u_t means: for the 4x4 A matrix, entry (i,j) where i=i1*2+i2, j=j1*2+j2:
  (u_s ⊗ u_t)[i,j] = u_s[i1,j1] * u_t[i2,j2]
  
  Similarly for B and C.
  
  So I need to:
  1. Define the Strassen 2x2*2x2 decomposition as 7 triples (u_s, v_s, w_s) where each is a 4-element vector (for the 2x2 matrix entries).
  2. For each pair (s,t), create a 16-element vector for A, 16-element for B, 16-element for C.
  3. Do the same for the 4x4*4x2 part.

  Let me code this up.

  For the 4x4*4x2 part:
  A is 4x4, B is 4x2, C is 4x2.
  Split: A = [[A11,A12],[A21,A22]] (2x2 blocks), B = [[B1],[B2]] (2x2 blocks vertically).
  C_top (2x2) = A11*B1 + A12*B2
  C_bot (2x2) = A21*B1 + A22*B2
  
  For A11*B1: Strassen 2x2*2x2, 7 terms.
  For A12*B2: Strassen 2x2*2x2, 7 terms.
  For A21*B1: Strassen 2x2*2x2, 7 terms.
  For A22*B2: Strassen 2x2*2x2, 7 terms.
  Total: 28 terms.

  Now let me implement the Kronecker Strassen for 4x4*4x4.
"""

import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6  # A is n x m, B is m x p, C is n x p
    
    U_rows = n * m  # 16
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def zeros(length):
        return [0] * length
    
    # Strassen 2x2*2x2 decomposition: 7 terms
    # Each term is (u, v, w) where u, v, w are 4-element vectors
    # A indices: 0=a11, 1=a12, 2=a21, 3=a22
    # B indices: 0=b11, 1=b12, 2=b21, 3=b22
    # C indices: 0=c11, 1=c12, 2=c21, 3=c22
    # W convention: W[k*n+i] where k is col, i is row. For 2x2: W[0]=c11, W[1]=c12, W[2]=c21, W[3]=c22
    
    strassen_2x2 = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1=(a11+a22)(b11+b22) -> c11+=, c22+=
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2=(a21+a22)b11 -> c21+=, c22-=
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3=a11(b12-b22) -> c12+=, c22+=
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # M4=a22(b21-b11) -> c11+=, c21+=
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5=(a11+a12)b22 -> c11-=, c12+=
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6=(a21-a11)(b11+b12) -> c22+=
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7=(a12-a22)(b21+b22) -> c11+=
    ]
    
    # Verify Strassen correctness
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Check: for entry (i,j,k) of the 2x2*2x2 tensor, the sum should be 1 if A[i,j]*B[j,k] appears in C[i,k]
    # i=row of A/C, j=col of A/row of B, k=col of B/C
    # A index: i*2+j, B index: j*2+k, C index: k*2+i (since W[k*n+i] with n=2)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_idx = i*2+j
                b_idx = j*2+k
                c_idx = k*2+i  # W index
                total = 0
                for s in range(7):
                    u, v, w = strassen_2x2[s]
                    total += u[a_idx] * v[b_idx] * w[c_idx]
                assert total == 1, f"Strassen verification failed at ({i},{j},{k}): got {total}"
    
    # Part 1: 4x4 * 4x4 via Kronecker Strassen (rank 49)
    # A is 4x4, B_left is 4x4 (first 4 columns of B), C_left is 4x4 (first 4 columns of C)
    # A index: i*4+j where i=row, j=col. Split: i=i1*2+i2, j=j1*2+j2 -> flat = (i1*2+i2)*4 + (j1*2+j2)
    # B index: j*6+k where j=row, k=col. For B_left, k is 0..3. Split: j=j1*2+j2, k=k1*2+k2 -> flat = (j1*2+j2)*6 + (k1*2+k2)
    # C index: k*4+i where k=col, i=row (W convention: W[k*n+i]). Split: k=k1*2+k2, i=i1*2+i2 -> flat = (k1*2+k2)*4 + (i1*2+i2)
    
    def a4_idx(i1, i2, j1, j2):
        return (i1*2+i2)*4 + (j1*2+j2)
    
    def b4_idx(j1, j2, k1, k2):
        return (j1*2+j2)*6 + (k1*2+k2)
    
    def c4_idx(k1, k2, i1, i2):
        return (k1*2+k2)*4 + (i1*2+i2)
    
    # For each pair (s, t) in Strassen terms:
    for s in range(7):
        for t in range(7):
            us, vs, ws = strassen_2x2[s]
            ut, vt, wt = strassen_2x2[t]
            
            u = zeros(U_rows)
            v = zeros(V_rows)
            w = zeros(W_rows)
            
            # u[a4_idx(i1,i2,j1,j2)] = us[i1*2+j1] * ut[i2*2+j2]
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u[a4_idx(i1, i2, j1, j2)] = us[i1*2+j1] * ut[i2*2+j2]
            
            # v[b4_idx(j1,j2,k1,k2)] = vs[j1*2+k1] * vt[j2*2+k2]
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v[b4_idx(j1, j2, k1, k2)] = vs[j1*2+k1] * vt[j2*2+k2]
            
            # w[c4_idx(k1,k2,i1,i2)] = ws[k1*2+i1] * wt[k2*2+i2]
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w[c4_idx(k1, k2, i1, i2)] = ws[k1*2+i1] * wt[k2*2+i2]
            
            add_term(u, v, w)
    
    # Part 2: 4x4 * 4x2 via tiling with Strassen (rank 28)
    # B_right is 4x2 (columns 4,5 of B), C_right is 4x2 (columns 4,5 of C)
    # A = [[A11, A12], [A21, A22]] (2x2 blocks)
    # B_right = [[B1], [B2]] where B1 is 2x2 (rows 0-1, cols 4-5), B2 is 2x2 (rows 2-3, cols 4-5)
    # C_top (2x2) = A11*B1 + A12*B2
    # C_bot (2x2) = A21*B1 + A22*B2
    
    def a2_idx(block_r, block_c, i, j):
        """A index for 2x2 block at (block_r, block_c), local (i,j)."""
        return (block_r*2+i)*4 + (block_c*2+j)
    
    def b2_idx(block_r, i, j):
        """B index for 2x2 block at row block_r, local (i,j). B is 4x6, so col offset is 4."""
        return (block_r*2+i)*6 + (4+j)
    
    def c2_idx(block_r, block_c, i, j):
        """C index for 2x2 block at (block_r, block_c), local (i,j). C is 4x6, W index = k*4+i."""
        # block_c corresponds to column block: 0 -> cols 0-3, 1 -> cols 4-5
        # For C_right, block_c=1, so cols 4-5
        col = block_c*2 + j
        row = block_r*2 + i
        return col*4 + row
    
    def add_strassen_block(a_br, a_bc, b_br, c_cr, c_cc):
        """Add Strassen 2x2*2x2 for A block at (a_br,a_bc) * B block at row a_br, cols (c_cc*2+0, c_cc*2+1) -> C block at (c_cr, c_cc)."""
        for s in range(7):
            u_s, v_s, w_s = strassen_2x2[s]
            u = zeros(U_rows)
            v = zeros(V_rows)
            w = zeros(W_rows)
            
            for i in range(2):
                for j in range(2):
                    u[a2_idx(a_br, a_bc, i, j)] = u_s[i*2+j]
            
            for i in range(2):
                for j in range(2):
                    v[b2_idx(b_br, i, j)] = v_s[i*2+j]
            
            for i in range(2):
                for j in range(2):
                    w[c2_idx(c_cr, c_cc, i, j)] = w_s[j*2+i]  # W index = k*n+i, so k=j, i=i
            
            add_term(u, v, w)
    
    # C_top = A11*B1 + A12*B2
    # A11 is at (0,0), B1 is at row block 0, C_top is at (0,1)
    add_strassen_block(0, 0, 0, 0, 1)  # A11 * B1 -> C_top
    add_strassen_block(0, 1, 1, 0, 1)  # A12 * B2 -> C_top
    
    # C_bot = A21*B1 + A22*B2
    # A21 is at (1,0), B1 is at row block 0, C_bot is at (1,1)
    add_strassen_block(1, 0, 0, 1, 1)  # A21 * B1 -> C_bot
    add_strassen_block(1, 1, 1, 1, 1)  # A22 * B2 -> C_bot
    
    rank = len(U_cols)
    
    # Convert column lists to row-major matrices
    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(V_rows)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}
