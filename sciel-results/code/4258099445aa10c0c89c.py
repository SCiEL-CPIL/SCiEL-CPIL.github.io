import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_schoolbook_block(i_off, j_off, k_off, size_i, size_j, size_k):
        for i in range(size_i):
            for j in range(size_j):
                for k in range(size_k):
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[(i_off+i)*5 + (j_off+j)] = 1
                    v_vec[(j_off+j)*5 + (k_off+k)] = 1
                    w_vec[(k_off+k)*5 + (i_off+i)] = 1
                    add_col(u_vec, v_vec, w_vec)

    def add_strassen_2x2(i_off, j_off, k_off):
        # Adds 7 columns for a 2x2x2 block
        # Factor 0: M1 = (A00 + A11)(B00 + B11) -> C00, C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 1: M2 = (A10 + A11)B00 -> C10, C11(-1)
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+0)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        w_vec[(k_off+0)*5 + (i_off+1)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = -1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 2: M3 = A00(B01 - B11) -> C01, C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = -1
        w_vec[(k_off+1)*5 + (i_off+0)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 3: M4 = A11(B10 - B00) -> C00, C10
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+0)] = -1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        w_vec[(k_off+0)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 4: M5 = (A00 + A01)B11 -> C00(-1), C01
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+0)] = 1
        u_vec[(i_off+0)*5 + (j_off+1)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = -1
        w_vec[(k_off+1)*5 + (i_off+0)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 5: M6 = (A10 - A00)(B00 + B01) -> C11
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+1)*5 + (j_off+0)] = 1
        u_vec[(i_off+0)*5 + (j_off+0)] = -1
        v_vec[(j_off+0)*5 + (k_off+0)] = 1
        v_vec[(j_off+0)*5 + (k_off+1)] = 1
        w_vec[(k_off+1)*5 + (i_off+1)] = 1
        add_col(u_vec, v_vec, w_vec)
        
        # Factor 6: M7 = (A01 - A11)(B10 + B11) -> C00
        u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
        u_vec[(i_off+0)*5 + (j_off+1)] = 1
        u_vec[(i_off+1)*5 + (j_off+1)] = -1
        v_vec[(j_off+1)*5 + (k_off+0)] = 1
        v_vec[(j_off+1)*5 + (k_off+1)] = 1
        w_vec[(k_off+0)*5 + (i_off+0)] = 1
        add_col(u_vec, v_vec, w_vec)

    def add_block_strassen_4x4(i_off, j_off, k_off):
        # Implements rank-49 decomposition for 4x4x4 block using Block Strassen
        # The 4x4 block is divided into 2x2 blocks.
        # Local block indices: 0, 1
        # Local sub-block indices: 0, 1
        # Global index mapping:
        # i_global = i_off + 2*bi + si
        # j_global = j_off + 2*bj + sj
        # k_global = k_off + 2*bk + sk
        
        # Strassen terms for 2x2 block matrices:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # For each term, we multiply two 2x2 matrices.
        # We use the standard rank-7 Strassen decomposition for the 2x2 product.
        # The "scalars" in the 2x2 Strassen are themselves 2x2 blocks of the 4x4 matrix.
        # So for each of the 7 Strassen factors of the 2x2 block product,
        # we need to add 7 rank-1 tensors corresponding to the 2x2 sub-block product.
        # Wait, no.
        # The 7 terms $M_r$ are $2 \times 2$ matrices.
        # Let $M_r = X_r Y_r$ where $X_r, Y_r$ are $2 \times 2$ matrices.
        # We need the rank-7 decomposition of $X_r Y_r$.
        # The rank-7 decomposition of a $2 \times 2$ matrix product uses 7 rank-1 tensors.
        # Each of these rank-1 tensors corresponds to a specific entry in the $2 \times 2$ result.
        # However, the entries of $X_r$ and $Y_r$ are themselves $2 \times 2$ blocks of the original $4 \times 4$ matrices.
        # So, for a single rank-1 tensor in the $2 \times 2$ Strassen decomposition,
        # say $u_{2x2} v_{2x2} w_{2x2}$, the vectors $u_{2x2}, v_{2x2}, w_{2x2}$ have entries that are $2 \times 2$ blocks.
        # To expand this to the $4 \times 4$ level, we need to iterate over the $2 \times 2$ sub-blocks.
        # Essentially, for each of the 7 outer Strassen terms, we call the inner 2x2 Strassen logic,
        # but mapping the "scalar" indices to the $2 \times 2$ block indices.
        
        # Let's define the 7 outer Strassen terms' U, V, W block coefficients.
        # Each coefficient is a 2x2 matrix of block indices (bi, bj) etc.
        
        # Term 1: (A00 + A11)(B00 + B11)
        # U blocks: A00, A11. V blocks: B00, B11. W blocks: C00, C11.
        # We need to add the rank-7 decomposition of the product of these block sums.
        # This is equivalent to calling add_strassen_2x2 on the block indices?
        # No, add_strassen_2x2 works on scalar indices.
        # We need to expand the block indices into global indices.
        
        # Let's create a helper to add a rank-1 tensor for a specific block configuration.
        # A "block rank-1" tensor is defined by:
        # u_blocks: list of (bi, si, bj, sj) with coeff
        # v_blocks: list of (bj, sj, bk, sk) with coeff
        # w_blocks: list of (bk, sk, bi, si) with coeff
        
        # Actually, it's easier to just iterate over the 7 outer terms, and for each,
        # iterate over the 7 inner Strassen factors, and for each inner factor,
        # iterate over the 4 sub-blocks (si, sj, sk) to create the global vectors.
        
        # Outer Term 1: M1 = (A00 + A11)(B00 + B11)
        # Inner Strassen factors for 2x2 product X*Y:
        # f1: (X00+X11)(Y00+Y11) -> Z00, Z11
        # f2: (X10+X11)Y00 -> Z10, Z11(-1)
        # ...
        
        # Let's map the outer block indices to inner 2x2 indices.
        # Outer A00 corresponds to inner X00.
        # Outer A11 corresponds to inner X11.
        # Outer A10 corresponds to inner X10.
        # Outer A01 corresponds to inner X01.
        
        # So for Term 1:
        # X = [[A00, 0], [0, A11]]? No.
        # X = A00 + A11 is a sum of blocks.
        # The Strassen formula operates on the matrix of blocks.
        # Let $A_{blocks} = [[A_{00}, A_{01}], [A_{10}, A_{11}]]$.
        # $M_1 = (A_{00} + A_{11})(B_{00} + B_{11})$.
        # This is a product of two $2 \times 2$ block matrices:
        # $X_1 = [[A_{00}, 0], [0, A_{11}]]$? No.
        # $X_1 = [[A_{00}, 0], [0, A_{11}]]$ is not right.
        # $X_1$ is the matrix $[[A_{00}, 0], [0, A_{11}]]$? No.
        # The term is $(A_{00} + A_{11})$. This is a single block matrix $X_1$ where $X_{1,00} = A_{00}$ and $X_{1,11} = A_{11}$ and others 0?
        # No, $A_{00}$ and $A_{11}$ are $2 \times 2$ matrices.
        # $X_1$ is a $2 \times 2$ matrix of $2 \times 2$ blocks.
        # $X_1 = [[A_{00}, 0], [0, A_{11}]]$? No, $A_{00} + A_{11}$ is a sum.
        # In the Strassen formula for block matrices, we treat the blocks as scalars.
        # So $X_1$ has entries: $x_{00} = A_{00}, x_{11} = A_{11}, x_{01}=0, x_{10}=0$.
        # $Y_1$ has entries: $y_{00} = B_{00}, y_{11} = B_{11}, y_{01}=0, y_{10}=0$.
        # Then we compute the rank-7 decomposition of $X_1 Y_1$.
        # The entries of $X_1 Y_1$ are $2 \times 2$ matrices.
        # We then add the rank-1 tensors for these.
        
        # Let's implement a generic `add_block_product_2x2(u_blocks, v_blocks, w_blocks)`
        # where u_blocks, v_blocks, w_blocks are 2x2 lists of lists of (coeff, si, sj) tuples?
        # This is getting complicated.
        
        # Simpler approach:
        # For each of the 7 outer terms, we have specific non-zero block entries.
        # We can just call `add_strassen_2x2` for each of the 4 sub-blocks?
        # No, that would be tiling (rank 56).
        
        # Let's manually expand the 49 terms.
        # There are 7 outer terms. For each, there are 7 inner Strassen factors.
        # For each inner factor, we have 1 rank-1 tensor over the block indices.
        # But the block entries are $2 \times 2$ matrices.
        # So each "block rank-1" tensor expands to 4 global rank-1 tensors?
        # No.
        # A block rank-1 tensor is of the form $u_{block} v_{block} w_{block}$.
        # $u_{block}$ is a $2 \times 2$ matrix.
        # The rank-1 decomposition of the block product $X Y$ (where X,Y are 2x2)
        # consists of 7 terms $u_r v_r w_r$ where $u_r, v_r, w_r$ are 2x2 matrices.
        # Each such term $u_r v_r w_r$ corresponds to a sum of rank-1 tensors over the scalar entries.
        # Specifically, if $u_r$ has entries $u_{r,si,sj}$, $v_r$ has $v_{r,sj,sk}$, $w_r$ has $w_{r,sk,si}$,
        # then the term is $\sum_{si,sj,sk} u_{r,si,sj} v_{r,sj,sk} w_{r,sk,si} E_{si,sj} \otimes E_{sj,sk} \otimes E_{sk,si}$.
        # This is not a rank-1 tensor in the global sense unless we sum over si,sj,sk.
        # But we are adding rank-1 tensors.
        # So for each outer term and each inner Strassen factor, we add 1 global rank-1 tensor?
        # No.
        # The rank of the $4 \times 4$ block is 49.
        # This means we add 49 global rank-1 tensors.
        # So for each of the 7 outer terms, we add 7 global rank-1 tensors.
        # Each global rank-1 tensor is defined by sparse vectors U, V, W.
        
        # Let's define the 7 outer terms' block matrices.
        # Term 1: X = [[A00, 0], [0, A11]], Y = [[B00, 0], [0, B11]].
        # Term 2: X = [[0, 0], [A10+A11, 0]]? No.
        # Term 2: M2 = (A10 + A11)B00.
        # X = [[0, 0], [A10+A11, 0]]? No.
        # X is a 2x2 block matrix.
        # X[0,0]=0, X[0,1]=0, X[1,0]=A10+A11, X[1,1]=0.
        # Y[0,0]=B00, Y[0,1]=0, Y[1,0]=0, Y[1,1]=0.
        
        # We will iterate over the 7 outer terms.
        # For each, we define the non-zero block entries of X and Y.
        # Then we apply the 7 inner Strassen factors.
        # For each inner factor, we construct the global U, V, W vectors.
        
        # Inner Strassen factors (for 2x2 matrices):
        # f1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> indices (0,0),(1,1)
        # f2: u=(0,1,1,0), v=(1,0,0,0), w=(0,1,0,-1) -> indices (1,0),(1,1) for u, (0,0) for v, (0,1),(1,1) for w
        # ...
        
        # Let's hardcode the 49 vectors.
        
        # Helper to add a global rank-1 tensor from block definitions
        def add_global_from_blocks(u_blocks, v_blocks, w_blocks):
            # u_blocks: list of (coeff, bi, bj)
            # v_blocks: list of (coeff, bj, bk)
            # w_blocks: list of (coeff, bk, bi)
            u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
            for coeff, bi, bj in u_blocks:
                for si in range(2):
                    for sj in range(2):
                        gi = i_off + 2*bi + si
                        gj = j_off + 2*bj + sj
                        u_vec[gi*5 + gj] += coeff
            for coeff, bj, bk in v_blocks:
                for sj in range(2):
                    for sk in range(2):
                        gj = j_off + 2*bj + sj
                        gk = k_off + 2*bk + sk
                        v_vec[gj*5 + gk] += coeff
            for coeff, bk, bi in w_blocks:
                for sk in range(2):
                    for si in range(2):
                        gk = k_off + 2*bk + sk
                        gi = i_off + 2*bi + si
                        w_vec[gk*5 + gi] += coeff
            add_col(u_vec, v_vec, w_vec)

        # Outer Term 1: (A00 + A11)(B00 + B11)
        # X blocks: (1,0,0), (1,1,1). Y blocks: (1,0,0), (1,1,1).
        # Inner f1: X00+X11, Y00+Y11, Z00+Z11
        # u_blocks: (1,0,0), (1,1,1). v_blocks: (1,0,0), (1,1,1). w_blocks: (1,0,0), (1,1,1).
        add_global_from_blocks([(1,0,0), (1,1,1)], [(1,0,0), (1,1,1)], [(1,0,0), (1,1,1)])
        
        # Inner f2: (X10+X11)Y00, Z10+Z11(-1)
        # X10 is 0, X11 is 1. So u: (1,1,1). Y00 is 1. v: (1,0,0). w: (1,1,0), (-1,1,1).
        add_global_from_blocks([(1,1,1)], [(1,0,0)], [(1,1,0), (-1,1,1)])
        
        # Inner f3: X00(Y01-Y11), Z01+Z11
        # X00 is 1. u: (1,0,0). Y01 is 0, Y11 is 1. v: (-1,1,1). w: (1,0,1), (1,1,1).
        add_global_from_blocks([(1,0,0)], [(-1,1,1)], [(1,0,1), (1,1,1)])
        
        # Inner f4: X11(Y10-Y00), Z00+Z10
        # X11 is 1. u: (1,1,1). Y10 is 0, Y00 is 1. v: (-1,0,0). w: (1,0,0), (1,1,0).
        add_global_from_blocks([(1,1,1)], [(-1,0,0)], [(1,0,0), (1,1,0)])
        
        # Inner f5: (X00+X01)Y11, Z00(-1)+Z01
        # X00 is 1, X01 is 0. u: (1,0,0). Y11 is 1. v: (1,1,1). w: (-1,0,0), (1,0,1).
        add_global_from_blocks([(1,0,0)], [(1,1,1)], [(-1,0,0), (1,0,1)])
        
        # Inner f6: (X10-X00)(Y00+Y01), Z11
        # X10 is 0, X00 is 1. u: (-1,0,0). Y00 is 1, Y01 is 0. v: (1,0,0). w: (1,1,1).
        add_global_from_blocks([(-1,0,0)], [(1,0,0)], [(1,1,1)])
        
        # Inner f7: (X01-X11)(Y10+Y11), Z00
        # X01 is 0, X11 is 1. u: (-1,1,1). Y10 is 0, Y11 is 1. v: (1,1,1). w: (1,0,0).
        add_global_from_blocks([(-1,1,1)], [(1,1,1)], [(1,0,0)])

        # Outer Term 2: (A10 + A11)B00
        # X blocks: (1,1,0), (1,1,1). Y blocks: (1,0,0).
        add_global_from_blocks([(1,1,0), (1,1,1)], [(1,0,0)], [(1,1,0), (-1,1,1)])
        add_global_from_blocks([(1,1,1)], [(1,0,0)], [(1,1,0), (-1,1,1)]) # Wait, f2 uses X10+X11
        # Let's re-verify the mapping.
        # For Term 2, X = [[0,0],[A10+A11, 0]].
        # X00=0, X01=0, X10=A10+A11, X11=0.
        # Y00=B00, Y01=0, Y10=0, Y11=0.
        
        # f1: (X00+X11)(Y00+Y11) = 0
        # f2: (X10+X11)Y00 = (A10+A11)B00. Z10, Z11(-1).
        # u: X10, X11. But X11=0. So u: (1,1,0). v: (1,0,0). w: (1,1,0), (-1,1,1).
        add_global_from_blocks([(1,1,0)], [(1,0,0)], [(1,1,0), (-1,1,1)])
        
        # f3: X00(Y01-Y11) = 0
        # f4: X11(Y10-Y00) = 0
        # f5: (X00+X01)Y11 = 0
        # f6: (X10-X00)(Y00+Y01) = (A10+A11)B00. Z11.
        # u: X10. v: Y00. w: Z11.
        add_global_from_blocks([(1,1,0)], [(1,0,0)], [(1,1,1)])
        
        # f7: (X01-X11)(Y10+Y11) = 0
        
        # Outer Term 3: A00(B01 - B11)
        # X blocks: (1,0,0). Y blocks: (1,0,1), (-1,1,1).
        # f1: 0
        # f2: 0
        # f3: X00(Y01-Y11). Z01, Z11.
        add_global_from_blocks([(1,0,0)], [(1,0,1), (-1,1,1)], [(1,0,1), (1,1,1)])
        # f4: 0
        # f5: 0
        # f6: 0
        # f7: 0
        
        # Outer Term 4: A11(B10 - B00)
        # X blocks: (1,1,1). Y blocks: (1,1,0), (-1,0,0).
        # f4: X11(Y10-Y00). Z00, Z10.
        add_global_from_blocks([(1,1,1)], [(1,1,0), (-1,0,0)], [(1,0,0), (1,1,0)])
        
        # Outer Term 5: (A00 + A01)B11
        # X blocks: (1,0,0), (1,0,1). Y blocks: (1,1,1).
        # f5: (X00+X01)Y11. Z00(-1), Z01.
        add_global_from_blocks([(1,0,0), (1,0,1)], [(1,1,1)], [(-1,0,0), (1,0,1)])
        
        # Outer Term 6: (A10 - A00)(B00 + B01)
        # X blocks: (1,1,0), (-1,0,0). Y blocks: (1,0,0), (1,0,1).
        # f6: (X10-X00)(Y00+Y01). Z11.
        add_global_from_blocks([(1,1,0), (-1,0,0)], [(1,0,0), (1,0,1)], [(1,1,1)])
        
        # Outer Term 7: (A01 - A11)(B10 + B11)
        # X blocks: (1,0,1), (-1,1,1). Y blocks: (1,1,0), (1,1,1).
        # f7: (X01-X11)(Y10+Y11). Z00.
        add_global_from_blocks([(1,0,1), (-1,1,1)], [(1,1,0), (1,1,1)], [(1,0,0)])

    # Partition 4+1
    # 4x4x4: Block Strassen (rank 49)
    add_block_strassen_4x4(0, 0, 0)
    
    # 4x4x1: Schoolbook
    add_schoolbook_block(0, 0, 4, 4, 4, 1)
    
    # 4x1x4: Schoolbook
    add_schoolbook_block(0, 4, 0, 4, 1, 4)
    
    # 1x4x4: Schoolbook
    add_schoolbook_block(4, 0, 0, 1, 4, 4)
    
    # 4x1x1: Schoolbook
    add_schoolbook_block(0, 4, 4, 4, 1, 1)
    
    # 1x4x1: Schoolbook
    add_schoolbook_block(4, 0, 4, 1, 4, 1)
    
    # 1x1x4: Schoolbook
    add_schoolbook_block(4, 4, 0, 1, 1, 4)
    
    # 1x1x1: Schoolbook
    add_schoolbook_block(4, 4, 4, 1, 1, 1)

    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
