"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _build_2x2_strassen():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    Shapes: U (4,7), V (4,7), W (4,7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # The 7 Strassen products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Column 0: M1 = (A11 + A22)(B11 + B22)
    # U: A11=1, A22=1 => U[0,0]=1, U[3,0]=1
    U[0, 0] = 1; U[3, 0] = 1
    # V: B11=1, B22=1 => V[0,0]=1, V[3,0]=1
    V[0, 0] = 1; V[3, 0] = 1
    # W: C11 += 1, C22 += 1 => W[0,0]=1, W[3,0]=1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Column 1: M2 = (A21 + A22)B11
    # U: A21=1, A22=1 => U[2,1]=1, U[3,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    # V: B11=1 => V[0,1]=1
    V[0, 1] = 1
    # W: C21 += 1 => W[2,1]=1
    W[2, 1] = 1
    
    # Column 2: M3 = A11(B12 - B22)
    # U: A11=1 => U[0,2]=1
    U[0, 2] = 1
    # V: B12=1, B22=-1 => V[1,2]=1, V[3,2]=-1
    V[1, 2] = 1; V[3, 2] = -1
    # W: C12 += 1, C22 += 1 => W[1,2]=1, W[3,2]=1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Column 3: M4 = A22(B21 - B11)
    # U: A22=1 => U[3,3]=1
    U[3, 3] = 1
    # V: B21=1, B11=-1 => V[2,3]=1, V[0,3]=-1
    V[2, 3] = 1; V[0, 3] = -1
    # W: C11 += 1, C21 += 1 => W[0,3]=1, W[2,3]=1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Column 4: M5 = (A11 + A12)B22
    # U: A11=1, A12=1 => U[0,4]=1, U[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    # V: B22=1 => V[3,4]=1
    V[3, 4] = 1
    # W: C11 -= 1, C12 += 1 => W[0,4]=-1, W[1,4]=1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Column 5: M6 = (A21 - A11)(B11 + B12)
    # U: A21=1, A11=-1 => U[2,5]=1, U[0,5]=-1
    U[2, 5] = 1; U[0, 5] = -1
    # V: B11=1, B12=1 => V[0,5]=1, V[1,5]=1
    V[0, 5] = 1; V[1, 5] = 1
    # W: C22 += 1 => W[3,5]=1
    W[3, 5] = 1
    
    # Column 6: M7 = (A12 - A22)(B21 + B22)
    # U: A12=1, A22=-1 => U[1,6]=1, U[3,6]=-1
    U[1, 6] = 1; U[3, 6] = -1
    # V: B21=1, B22=1 => V[2,6]=1, V[3,6]=1
    V[2, 6] = 1; V[3, 6] = 1
    # W: C11 += 1 => W[0,6]=1
    W[0, 6] = 1
    
    return U, V, W


def _build_4x4_strassen():
    """Build 4x4x4 decomposition using 2x2 Strassen blocks.
    Returns U, V, W with shapes (16, 49), (16, 49), (16, 49).
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    U2, V2, W2 = _build_2x2_strassen()
    rank2 = 7
    rank4 = rank2 * rank2  # 49
    
    U4 = np.zeros((16, rank4), dtype=np.int64)
    V4 = np.zeros((16, rank4), dtype=np.int64)
    W4 = np.zeros((16, rank4), dtype=np.int64)
    
    # We decompose 4x4 into four 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # For each pair of Strassen terms (m1, m2), we have:
    # U4 gets contributions from U2 at positions corresponding to the A-block
    # V4 gets contributions from V2 at positions corresponding to the B-block
    # W4 gets contributions from W2 at positions corresponding to the C-block
    
    # The overall Strassen for 4x4 uses 7 macro-terms, each being a product of
    # two 2x2 matrix products. Each macro-term expands to 7x7=49 micro-terms.
    
    # Macro-term definitions (same as 2x2 Strassen but on 2x2 blocks):
    # MM1 = (A11 + A22)(B11 + B22)
    # MM2 = (A21 + A22)B11
    # MM3 = A11(B12 - B22)
    # MM4 = A22(B21 - B11)
    # MM5 = (A11 + A12)B22
    # MM6 = (A21 - A11)(B11 + B12)
    # MM7 = (A12 - A22)(B21 + B22)
    
    # Each macro-term (a,b) where a is an A-coefficient and b is a B-coefficient
    # expands using the 2x2 Strassen decomposition.
    
    # Let's define the macro-terms more carefully.
    # MM1 = (A11 + A22)(B11 + B22)
    # This is a sum of 4 products: A11*B11, A11*B22, A22*B11, A22*B22
    # Each product is a 2x2*2x2 -> 2x2 multiplication, decomposed by Strassen (7 terms).
    
    # So MM1 contributes to C11 and C22:
    # C11 += MM1, C22 += MM1
    
    # Let me restructure. The 4x4 Strassen has 7 macro-terms.
    # Each macro-term is a bilinear form on 2x2 blocks.
    # We expand each macro-term into 49 micro-terms (7 from A-side Strassen x 7 from B-side... no wait)
    
    # Actually, each macro-term like (A11 + A22)(B11 + B22) involves multiplying
    # a 2x2 matrix (sum of two 2x2 blocks) by another 2x2 matrix.
    # The product of two 2x2 matrices needs 7 Strassen multiplications.
    # So each macro-term that is a product of two linear combinations of 2x2 blocks
    # expands to 7 micro-terms.
    
    # Let me define the 7 macro-terms and their expansion:
    
    # MM1 = (A11 + A22)(B11 + B22)
    #   Let X = A11 + A22, Y = B11 + B22
    #   C11 += X*Y, C22 += X*Y
    #   X*Y expanded by Strassen: 7 terms
    
    # MM2 = (A21 + A22)B11
    #   Let X = A21 + A22, Y = B11
    #   C21 += X*Y
    #   X*Y expanded by Strassen: 7 terms
    
    # MM3 = A11(B12 - B22)
    #   Let X = A11, Y = B12 - B22
    #   C12 += X*Y, C22 += X*Y
    #   X*Y expanded by Strassen: 7 terms
    
    # MM4 = A22(B21 - B11)
    #   Let X = A22, Y = B21 - B11
    #   C11 += X*Y, C21 += X*Y
    #   X*Y expanded by Strassen: 7 terms
    
    # MM5 = (A11 + A12)B22
    #   Let X = A11 + A12, Y = B22
    #   C11 -= X*Y, C12 += X*Y
    #   X*Y expanded by Strassen: 7 terms
    
    # MM6 = (A21 - A11)(B11 + B12)
    #   Let X = A21 - A11, Y = B11 + B12
    #   C22 += X*Y
    #   X*Y expanded by Strassen: 7 terms
    
    # MM7 = (A12 - A22)(B21 + B22)
    #   Let X = A12 - A22, Y = B21 + B22
    #   C11 += X*Y
    #   X*Y expanded by Strassen: 7 terms
    
    # Total: 7 * 7 = 49 terms
    
    # For each macro-term, we need to:
    # 1. Define the A-side linear combination (which 2x2 blocks with what coefficients)
    # 2. Define the B-side linear combination
    # 3. For each of the 7 Strassen micro-terms, add to U4, V4, W4
    
    # Let's define the macro-terms:
    # Each macro-term has:
    #   a_blocks: list of (block_id, coeff) for A-side
    #   b_blocks: list of (block_id, coeff) for B-side
    #   c_blocks: list of (block_id, coeff) for C-side contribution
    
    # Block IDs: 0=A11/C11/B11, 1=A12/C12/B12, 2=A21/C21/B21, 3=A22/C22/B22
    
    macro_terms = [
        # (a_blocks, b_blocks, c_blocks)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),      # MM1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),                        # MM2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),               # MM3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),               # MM4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),               # MM5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),               # MM6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),               # MM7
    ]
    
    # For each macro-term and each Strassen micro-term, build the contribution
    col = 0
    for mt in macro_terms:
        a_blocks, b_blocks, c_blocks = mt
        
        # For each Strassen micro-term (s in 0..6):
        for s in range(7):
            # U2[., s] gives the A-side coefficients for the 2x2 Strassen term
            # V2[., s] gives the B-side coefficients
            # W2[., s] gives the C-side coefficients
            
            # A-side: for each 2x2 block position (li, lj) in the 2x2 grid of blocks
            # U2[li*2+lj, s] * a_coeff contributes to A4 at global position
            
            # The 2x2 blocks in A are at positions:
            # Block 0 (A11): rows 0-1, cols 0-1
            # Block 1 (A12): rows 0-1, cols 2-3
            # Block 2 (A21): rows 2-3, cols 0-1
            # Block 3 (A22): rows 2-3, cols 2-3
            
            # For the A-side of micro-term s:
            # U2[li*2+lj, s] where li, lj are local 2x2 indices
            # This maps to global A indices (2*block_row + li, 2*block_col + lj)
            
            # For each a_block (block_id, a_coeff):
            #   block_id 0: block_row=0, block_col=0
            #   block_id 1: block_row=0, block_col=1
            #   block_id 2: block_row=1, block_col=0
            #   block_id 3: block_row=1, block_col=1
            
            # U4 global index: (2*block_row + li)*4 + (2*block_col + lj)
            
            for block_id, a_coeff in a_blocks:
                br, bc = divmod(block_id, 2)  # block_row, block_col
                for li in range(2):
                    for lj in range(2):
                        u2_val = U2[li*2+lj, s]
                        if u2_val != 0:
                            global_i = 2*br + li
                            global_j = 2*bc + lj
                            U4[global_i*4 + global_j, col] += u2_val * a_coeff
            
            # Similarly for B-side:
            # V2[j_local*2+k_local, s] where j_local, k_local are local 2x2 indices in B
            # B block positions:
            # Block 0 (B11): rows 0-1, cols 0-1
            # Block 1 (B12): rows 0-1, cols 2-3
            # Block 2 (B21): rows 2-3, cols 0-1
            # Block 3 (B22): rows 2-3, cols 2-3
            
            for block_id, b_coeff in b_blocks:
                br, bc = divmod(block_id, 2)
                for j_local in range(2):
                    for k_local in range(2):
                        v2_val = V2[j_local*2+k_local, s]
                        if v2_val != 0:
                            global_j = 2*br + j_local
                            global_k = 2*bc + k_local
                            V4[global_j*4 + global_k, col] += v2_val * b_coeff
            
            # For C-side:
            # W2[k_local*2+i_local, s]
            # C block positions same as A
            for block_id, c_coeff in c_blocks:
                br, bc = divmod(block_id, 2)
                for k_local in range(2):
                    for i_local in range(2):
                        w2_val = W2[k_local*2+i_local, s]
                        if w2_val != 0:
                            global_k = 2*br + k_local
                            global_i = 2*bc + i_local
                            W4[global_k*4 + global_i, col] += w2_val * c_coeff
            
            col += 1
    
    return U4, V4, W4


def _build_4x4x7_decomposition():
    """Build 4x4x7 decomposition.
    Split B (4x7) into B1 (4x4) and B2 (4x3).
    Use 4x4x4 Strassen for both parts.
    Total rank: 49 + 49 = 98.
    
    Returns U, V, W with shapes (16, 98), (28, 98), (28, 98).
    Indexing: U[i*4+j, c], V[j*7+k, c], W[k*4+i, c]
    """
    U4, V4, W4 = _build_4x4_strassen()
    rank4 = 49
    
    U_out = np.zeros((16, 2*rank4), dtype=np.int64)
    V_out = np.zeros((28, 2*rank4), dtype=np.int64)
    W_out = np.zeros((28, 2*rank4), dtype=np.int64)
    
    # Part 1: B1 = B[:, 0:4] (columns 0-3)
    # V indices: j*7 + k where k in 0..3
    # W indices: k*4 + i where k in 0..3
    
    for c in range(rank4):
        U_out[:, c] = U4[:, c]
        for j in range(4):
            for k in range(4):
                V_out[j*7 + k, c] = V4[j*4 + k, c]
        for k in range(4):
            for i in range(4):
                W_out[k*4 + i, c] = W4[k*4 + i, c]
    
    # Part 2: B2 = B[:, 4:7] (columns 4-6), padded to 4x4
    # B2 has columns 4,5,6 mapped to local 0,1,2; local col 3 is 0
    # V indices: j*7 + (k+4) where k in 0..3, but only k=0,1,2 have actual B entries
    # W indices: (k+4)*4 + i where k in 0..3, but only k=0,1,2 contribute to C
    
    # Actually, for the padded part, we treat B2 as a 4x4 matrix where:
    # B2[j, k_local] = B[j, k_local + 4] for k_local in 0,1,2
    # B2[j, 3] = 0
    # C2[k_local, i] = sum_j B2[j, k_local] * A[i, j] for k_local in 0,1,2
    # C2[3, i] = 0 (but we still compute it, it will be zero)
    
    # The 4x4x4 Strassen decomposition works on any 4x4 * 4x4 -> 4x4.
    # We just need to remap indices.
    
    for c in range(rank4):
        c2 = rank4 + c
        U_out[:, c2] = U4[:, c]
        # V: local (j, k_local) -> global (j, k_local + 4)
        for j in range(4):
            for k_local in range(4):
                V_out[j*7 + (k_local + 4), c2] = V4[j*4 + k_local, c]
        # W: local (k_local, i) -> global (k_local + 4, i)
        for k_local in range(4):
            for i in range(4):
                W_out[(k_local + 4)*4 + i, c2] = W4[k_local*4 + i, c]
    
    return U_out, V_out, W_out


def construct():
    """Construct a rank-minimal (or near-minimal) tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices as lists of lists
        U: shape (16, rank)
        V: shape (28, rank)  
        W: shape (28, rank)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Build the Strassen-based decomposition
    U, V, W = _build_4x4x7_decomposition()
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
