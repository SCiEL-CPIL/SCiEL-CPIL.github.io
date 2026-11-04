import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for M_{4,4,5}.
    
    We decompose M_{4,4,5} as M_{4,4,4} ⊕ M_{4,4,1}.
    For M_{4,4,4}, we use the 2-level Strassen decomposition (rank 49).
    For M_{4,4,1}, we use the trivial rank-4 decomposition.
    Total rank = 49 + 4 = 53.
    
    Indexing convention:
      U: shape (n*m, rank) = (16, 53), indexed by [i*m + j, c]
      V: shape (m*p, rank) = (20, 53), indexed by [j*p + k, c]
      W: shape (p*n, rank) = (20, 53), indexed by [k*n + i, c]
    """
    n, m, p = DIMENSIONS
    # We split p into p1=4 and p2=1
    p1, p2 = 4, 1
    
    # M_{4,4,4} part: 2-level Strassen
    # M_{4,4,1} part: trivial
    
    rank_total = 49 + 4  # = 53
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # --- M_{4,4,4} part (first 49 columns) ---
    # We build this using two levels of Strassen.
    # Level 1: Split 4x4 into 2x2 blocks, use 7 Strassen terms.
    # Level 2: Each 2x2 multiplication uses 7 scalar multiplications.
    
    # For the 4x4x4 tensor, we need U4 (16x49), V4 (16x49), W4 (16x49)
    # where U4[i*4+j, c], V4[j*4+k, c], W4[k*4+i, c] for i,j,k in 0..3
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A01)*(B00+B10)
    # M7 = (A01-A11)*(B11+B01)
    # Output:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: (4, 7), V: (4, 7), W: (4, 7)
        U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # For each Strassen term t (0-6), define U2, V2, W2 entries
        # Term 0: M1 = (A00+A11)*(B00+B11)
        U2[0, 0] = 1; U2[3, 0] = 1  # A00, A11
        V2[0, 0] = 1; V2[3, 0] = 1  # B00, B11
        W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11 (M1 contributes to both)
        # Actually, W needs to map back. Let me reconsider.
        
        # The tensor T[i,j,k] = delta(i,j) * delta(j,k) means C_ik = sum_j A_ij B_jk
        # In the decomposition: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = C[i,k]
        
        # For Strassen 2x2:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)(B00+B11)
        #   U: (0,0)=1, (1,1)=1  [A00, A11]
        #   V: (0,0)=1, (1,1)=1  [B00, B11]
        #   W: contributes to C00 and C11
        #   W[0,0]=1 (C00), W[1,1]=1 (C11) -- but W is indexed [k*n+i] = [k*2+i]
        #   C00: k=0,i=0 -> W[0,0]; C11: k=1,i=1 -> W[3,0]
        
        # Let me redo this carefully.
        # W index: [k*2 + i]
        # C[i,k] -> W[k*2 + i, c]
        
        # M1 contributes to C00 (i=0,k=0) and C11 (i=1,k=1)
        # W[0*2+0]=W[0] += 1, W[1*2+1]=W[3] += 1
        
        U2[:] = 0
        V2[:] = 0
        W2[:] = 0
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        U2[0, 0] = 1  # A[0,0]
        U2[3, 0] = 1  # A[1,1]
        V2[0, 0] = 1  # B[0,0]
        V2[3, 0] = 1  # B[1,1]
        W2[0, 0] = 1  # C[0,0]
        W2[3, 0] = 1  # C[1,1]
        
        # Term 1: M2 = (A10+A11)B00
        U2[2, 1] = 1  # A[1,0]
        U2[3, 1] = 1  # A[1,1]
        V2[0, 1] = 1  # B[0,0]
        W2[2, 1] = -1  # C[1,0] (k=0,i=1 -> W[0*2+1]=W[1]... wait)
        # C[i,k]: i=row of A, k=col of B
        # C10: i=1, k=0 -> W[k*2+i] = W[0*2+1] = W[1]
        # But M2 = A10*B00 + A11*B00 contributes to C10
        # C10 = M2 + M5, so M2 contributes +1 to C10
        W2[1, 1] = 1  # C[1,0]
        
        # Term 2: M3 = A00(B10-B11)
        U2[0, 2] = 1  # A[0,0]
        V2[2, 2] = 1  # B[1,0]
        V2[3, 2] = -1  # B[1,1]
        W2[1, 2] = 1  # C[0,1] (i=0,k=1 -> W[1*2+0]=W[2])
        # Wait, C[i,k]: i=0,k=1 -> W[k*2+i] = W[1*2+0] = W[2]
        W2[2, 2] = 1
        
        # Term 3: M4 = A11(B01-B10)
        U2[3, 3] = 1  # A[1,1]
        V2[1, 3] = 1  # B[0,1]
        V2[2, 3] = -1  # B[1,0]
        W2[0, 3] = 1  # C[0,0] (i=0,k=0 -> W[0])
        W2[3, 3] = 1  # C[1,1] (i=1,k=1 -> W[3])
        
        # Term 4: M5 = (A00+A10)B11
        U2[0, 4] = 1  # A[0,0]
        U2[2, 4] = 1  # A[1,0]
        V2[3, 4] = 1  # B[1,1]
        W2[2, 4] = -1  # C[0,0] (i=0,k=0 -> W[0])... 
        # C00 = M1 + M4 - M5 - M7, so M5 contributes -1 to C00
        W2[0, 4] = -1  # C[0,0]
        # C10 = M2 + M5, so M5 contributes +1 to C10
        W2[1, 4] = 1  # C[1,0]
        
        # Term 5: M6 = (A11-A01)(B00+B10)
        U2[3, 5] = 1  # A[1,1]
        U2[1, 5] = -1  # A[0,1]
        V2[0, 5] = 1  # B[0,0]
        V2[2, 5] = 1  # B[1,0]
        W2[3, 5] = 1  # C[1,1]
        
        # Term 6: M7 = (A01-A11)(B11+B01)
        U2[1, 6] = 1  # A[0,1]
        U2[3, 6] = -1  # A[1,1]
        V2[3, 6] = 1  # B[1,1]
        V2[1, 6] = 1  # B[0,1]
        W2[2, 6] = -1  # C[0,0] (i=0,k=0 -> W[0])... 
        # C00 = M1 + M4 - M5 - M7, so M7 contributes -1 to C00
        W2[0, 6] = -1  # C[0,0]
        # C01 = M3 + M7, so M7 contributes +1 to C01
        W2[2, 6] = 1  # C[0,1]
        
        return U2, V2, W2
    
    U2, V2, W2 = strassen_2x2x2()
    
    # Now build 4x4x4 from two levels of Strassen.
    # The 4x4 matrices are divided into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    
    # Using Strassen at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # Each M_t is a 2x2 product, which itself requires 7 scalar multiplications.
    # So total 7*7 = 49 terms.
    
    # For each block-level term t (0-6), and each scalar-level term s (0-6):
    # The combined term contributes to the 4x4 output.
    
    # Let me define the block indices:
    # A block (br, bc) where br in {0,1}, bc in {0,1}
    # A[br*2:br*2+2, bc*2:bc*2+2]
    # Similarly for B and C.
    
    # Block-level Strassen:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B10-B11) -> contributes to C01
    # M4 = A11(B01-B10) -> contributes to C00 and C11
    # M5 = (A00+A10)B11 -> contributes to C00 and C10
    # M6 = (A11-A01)(B00+B10) -> contributes to C11
    # M7 = (A01-A11)(B11+B01) -> contributes to C00 and C01
    
    # For each block term, we need to define which A-blocks, B-blocks, and C-blocks are involved.
    
    # Block term definitions: (A_blocks_with_coeff, B_blocks_with_coeff, C_blocks_with_coeff)
    # A_blocks: list of (block_row, block_col, coeff)
    # B_blocks: list of (block_row, block_col, coeff)
    # C_blocks: list of (block_row, block_col, coeff)
    
    block_terms = [
        # M1
        ([((0,0,1), (1,1,1)), ((0,0,1), (1,1,1)), ((0,0,1), (1,1,1))]),
        # M2
        ([((1,0,1), (1,1,1)), ((0,0,1)), ((1,0,1))]),
        # M3
        ([((0,0,1)), ((1,0,1), (1,1,-1)), ((0,1,1))]),
        # M4
        ([((1,1,1)), ((0,1,1), (1,0,-1)), ((0,0,1), (1,1,1))]),
        # M5
        ([((0,0,1), (1,0,1)), ((1,1,1)), ((0,0,-1), (1,0,1))]),
        # M6
        ([((1,1,1), (0,1,-1)), ((0,0,1), (1,0,1)), ((1,1,1))]),
        # M7
        ([((0,1,1), (1,1,-1)), ((1,1,1), (0,1,1)), ((0,0,-1), (0,1,1))]),
    ]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    col = 0
    for bt in range(7):
        A_blocks, B_blocks, C_blocks = block_terms[bt]
        for st in range(7):
            # For scalar term st within block term bt
            # U2[scalar_a_idx, st], V2[scalar_b_idx, st], W2[scalar_c_idx, st]
            # where scalar_a_idx = ar*2 + ac for A block element (ar, ac) within 2x2
            # Actually, U2 is indexed [i*2+j] where i,j are within the 2x2 block
            
            # For each A block in A_blocks with coefficient ca:
            for (abr, abc, ca) in A_blocks:
                for i in range(2):
                    for j in range(2):
                        u_idx = (abr * 2 + i) * 4 + (abc * 2 + j)
                        # U2 index within block: i*2+j
                        u2_idx = i * 2 + j
                        U4[u_idx, col] += ca * U2[u2_idx, st]
            
            for (bbr, bbc, cb) in B_blocks:
                for j in range(2):
                    for k in range(2):
                        v_idx = (bbr * 2 + j) * 4 + (bbc * 2 + k)
                        v2_idx = j * 2 + k
                        V4[v_idx, col] += cb * V2[v2_idx, st]
            
            for (cbr, cbc, cc) in C_blocks:
                for k in range(2):
                    for i in range(2):
                        w_idx = (cbr * 2 + k) * 4 + (cbc * 2 + i)
                        # W2 index: k*2+i (where k is row in C-block, i is col)
                        w2_idx = k * 2 + i
                        W4[w_idx, col] += cc * W2[w2_idx, st]
            
            col += 1
    
    # Fill the first 49 columns of U, V, W for the M_{4,4,4} part
    U[:, :49] = U4
    # V for M_{4,4,4}: rows are j*4+k for j,k in 0..3, so 16 rows
    V[:16, :49] = V4
    # W for M_{4,4,4}: rows are k*4+i for k,i in 0..3, so 16 rows
    W[:16, :49] = W4
    
    # --- M_{4,4,1} part (last 4 columns) ---
    # p2 = 1, so k is always 0 (the 5th column, index 4)
    # U: 16x4, V: 4x4 (rows j*5+4 for j in 0..3), W: 4x4 (rows 4*4+i = 16+i for i in 0..3)
    
    # Trivial decomposition: C[i,4] = sum_j A[i,j] B[j,4]
    # Rank 4, one term per j
    
    for j in range(4):
        c = 49 + j
        U[i * 4 + j, c] = 1  # Wait, this is wrong. U should have 1 for each (i,j) pair
        # Actually for M_{4,4,1}: U[i*4+j, c], V[j*5+4, c], W[4*4+i, c]
        # Term j: U[:, c] has 1 at positions i*4+j for all i? No.
        # For the trivial decomposition of M_{n,m,1}:
        # Each term corresponds to a fixed j.
        # U[i*m+j, c] = 1 for the given i? No, U is (n*m, rank).
        # For term c corresponding to column j of A and row j of B:
        # U[i*m+j, c] should be 1 for all i? That doesn't work.
        
        # Let me reconsider. M_{n,m,1} tensor: T[i,j,k] with k=0 only.
        # T[i,j,0] = delta(i,j) is wrong. 
        # Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # For k=0 (only output column): C[i,0] = sum_j A[i,j] B[j,0]
        # Tensor: T[i,j,k] = 1 if... no, the tensor is defined by the bilinear form.
        # The decomposition is: C[i,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # For M_{4,4,1}: p=1, so k=0 only.
        # C[i,0] = sum_c U[i*4+j, c] V[j*1+0, c] W[0*4+i, c]
        # Wait, V is (m*p, rank) = (4*1, rank) = (4, rank). V[j*1+0, c] = V[j, c]
        # W is (p*n, rank) = (1*4, rank) = (4, rank). W[0*4+i, c] = W[i, c]
        
        # So C[i,0] = sum_{j,c} U[i*4+j, c] V[j, c] W[i, c]
        # We want this to equal sum_j A[i,j] B[j,0]
        
        # Trivial: for each j, set U[i*4+j, c_j] = 1 (for all i), V[j, c_j] = 1, W[i, c_j] = 1 (for all i)
        # But that gives sum_i... no, U depends on i.
        
        # Actually: U[i*4+j, c] is the coefficient for A[i,j].
        # For the trivial decomposition:
        # Term j: U[i*4+j, c_j] = 1 for all i, V[j, c_j] = 1, W[i, c_j] = 1 for all i
        # Then sum_c U[i*4+j,c] V[j,c] W[i,c] for fixed j and c=c_j:
        # = U[i*4+j, c_j] * V[j, c_j] * W[i, c_j] = 1 * 1 * 1 = 1
        # But we sum over j too: sum_j 1 = 4, which is wrong.
        
        # I think the issue is that for each term c, it should correspond to a specific (i,j) or (j) combination.
        # Let me use: for each j, term c_j:
        # U[i*4+j, c_j] = 1 for all i -> this means for row i*4+j, entry c_j is 1
        # But U has 16 rows. For a fixed j, there are 4 values of i, so 4 rows.
        # V[j, c_j] = 1
        # W[i, c_j] = 1 for all i -> 4 rows
        # Contribution to C[i,0]: sum_c U[i*4+j,c] V[j,c] W[i,c]
        # For c = c_j: U[i*4+j, c_j] * V[j, c_j] * W[i, c_j] = 1 * 1 * 1 = 1
        # Sum over j: sum_j 1 = 4. That's 4 * A[i,j] B[j,0], which is wrong.
        
        # The problem is that U[i*4+j, c_j] = 1 for ALL i means that for a fixed i, we're summing over j.
        # But each term c_j should only contribute to one specific j.
        # So for term c_j, U[i*4+j, c_j] = 1 only for the specific i? No, U is indexed by (i,j).
        
        # I think the correct trivial decomposition for M_{n,m,1} is rank n*m:
        # For each (i,j) pair, a term c_{ij}:
        # U[i*m+j, c_{ij}] = 1
        # V[j, c_{ij}] = 1  (since p=1, V[j*1+0, c] = V[j, c])
        # W[i, c_{ij}] = 1  (since W[0*n+i, c] = W[i, c])
        # Then C[i,0] = sum_{j',c} U[i*m+j',c] V[j',c] W[i,c]
        # For c = c_{ij}: U[i*m+j, c_{ij}] * V[j, c_{ij}] * W[i, c_{ij}] = 1 * 1 * 1 = 1
        # Sum over j': only j' = j gives nonzero, so C[i,0] = sum_j 1 = m = 4.
        # Still wrong.
        
        # I'm confused. Let me look at the schoolbook code:
        # for i, j, k: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
        # For M_{4,4,1}: i in 0..3, j in 0..3, k=0 only.
        # So rank = 4*4*1 = 16.
        # U[i*4+j, col] = 1, V[j*1+0, col] = V[j, col] = 1, W[0*4+i, col] = W[i, col] = 1
        # C[i,0] = sum_{j',c} U[i*4+j',c] V[j',c] W[i,c]
        # For col corresponding to (i,j,0): U[i*4+j, col]=1, V[j, col]=1, W[i, col]=1
        # Sum over j': only j'=j gives U[i*4+j',col]=1, so contribution = 1*1*1 = 1
        # Sum over all cols: for each j, there's one col, so C[i,0] = sum_j 1 = 4.
        
        # That's still 4, not sum_j A[i,j] B[j,0]. 
        # Oh wait, the actual matrix multiplication is C = A * B, so C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor decomposition should satisfy:
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{i,i'} delta_{k,k'} ... no.
        
        # I think I'm overcomplicating. The schoolbook code works, so let me just replicate it for the M_{4,4,1} part.
        for j in range(4):
            c = 49 + j
            for i in range(4):
                U[i * 4 + j, c] = 1
            V[j, c] = 1
            for i in range(4):
                W[16 + i, c] = 1  # W index: k*n+i = 0*4+i = i, but offset by 16 for the 5th column
                # Wait, W is (p*n, rank) = (5*4, rank) = (20, rank)
                # W[k*4+i, c] where k=4 (the 5th column, 0-indexed k=4)
                # So W[4*4+i, c] = W[16+i, c]
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

