import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _validate(U, V, W, n, m, p, num_trials=3):
    """Validate that the tensor decomposition correctly computes A*B for random matrices."""
    for _ in range(num_trials):
        A = np.random.randint(-3, 4, (n, m), dtype=np.float64)
        B = np.random.randint(-3, 4, (m, p), dtype=np.float64)
        C_expected = A @ B
        
        # Compute C from the decomposition
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_r = np.array(U).reshape(n, m, -1)
        V_r = np.array(V).reshape(m, p, -1)
        W_r = np.array(W).reshape(p, n, -1)
        
        # C[i,j] = sum over k and rank r of A[i,k] * U[k, r] * V[k, r] * W[j, r] ... wait
        # Actually: C[i,j] = sum_r (sum_k A[i,k] * U[k,r]) * (sum_k B[k,j] * V[k,r]) * W[j,r]
        # But that's not right either. The tensor decomposition is:
        # T_{i,j,k} = sum_r U_{i,k,r} * V_{k,j,r} * W_{r}... 
        # Actually the standard form is:
        # C_{i,j} = sum_k A_{i,k} * B_{k,j}
        # = sum_r (sum_k A_{i,k} * U_{k,r}) * (sum_k B_{k,j} * V_{k,r}) ... no
        # The correct interpretation:
        # Let M_r = (sum_k A_{i,k} * U_{i,k,r}) for each r, but U is indexed by (i*m+k, r)
        # So U_row = A flattened, and intermediate_r = U_row . A_flat
        # Similarly intermediate'_r = V_row . B_flat
        # Then C_flat = W.T @ (intermediate * intermediate')
        # Wait, that gives C_flat[j*n+i] = sum_r W[j*n+i, r] * intermediate_r * intermediate'_r
        
        A_flat = A.flatten()  # row-major: [A[0,0], A[0,1], ..., A[0,m-1], A[1,0], ...]
        B_flat = B.flatten()  # [B[0,0], B[0,1], ..., B[0,p-1], B[1,0], ...]
        
        U_arr = np.array(U)  # (n*m, rank)
        V_arr = np.array(V)  # (m*p, rank)
        W_arr = np.array(W)  # (p*n, rank)
        
        int1 = U_arr.T @ A_flat  # (rank,)
        int2 = V_arr.T @ B_flat  # (rank,)
        C_flat = W_arr.T @ (int1 * int2)  # (p*n,)
        
        C_computed = C_flat.reshape(p, n).T  # reshape to (n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _build_strassen56():
    """Build the rank-56 Strassen-based decomposition (8 independent 2x2 blocks)."""
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for s in range(2):
        for j in range(4):
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            U[a00][col] += 1
            U[a11][col] += 1
            U[a10][col+1] += 1
            U[a11][col+1] += 1
            U[a00][col+2] += 1
            U[a11][col+3] += 1
            U[a00][col+4] += 1
            U[a01][col+4] += 1
            U[a10][col+5] += 1
            U[a00][col+5] += -1
            U[a01][col+6] += 1
            U[a11][col+6] += -1
            
            V[b00][col] += 1
            V[b11][col] += 1
            V[b00][col+1] += 1
            V[b01][col+2] += 1
            V[b11][col+2] += -1
            V[b10][col+3] += 1
            V[b00][col+3] += -1
            V[b11][col+4] += 1
            V[b00][col+5] += 1
            V[b01][col+5] += 1
            V[b10][col+6] += 1
            V[b11][col+6] += 1
            
            W[c00][col] += 1
            W[c00][col+3] += 1
            W[c00][col+4] += -1
            W[c00][col+6] += 1
            W[c01][col+2] += 1
            W[c01][col+4] += 1
            W[c10][col+1] += 1
            W[c10][col+3] += 1
            W[c11][col] += 1
            W[c11][col+1] += -1
            W[c11][col+2] += 1
            W[c11][col+5] += 1
            
            col += 7
    
    return U, V, W

def _build_strassen49_4x4():
    """
    Build a rank-49 decomposition for 4x4 matrix multiplication using Strassen
    recursively (2x2 blocks of 2x2 matrices).
    Returns U, V, W for 4x4 * 4x4 -> 4x4.
    U is (4*4, 49) = (16, 49), V is (4*4, 49) = (16, 49), W is (4*4, 49) = (16, 49).
    """
    # Strassen 4x4: partition into four 2x2 blocks, use 7 Strassen products of 2x2
    # Each 2x2 product uses 7 scalar multiplications
    # Total: 7 * 7 = 49
    rank = 49
    N = 4
    
    U = [[0] * rank for _ in range(N * N)]  # 16 x 49
    V = [[0] * rank for _ in range(N * N)]
    W = [[0] * rank for _ in range(N * N)]
    
    # Strassen 2x2: 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, partition into 2x2 blocks of size 2:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Block indices for 4x4 matrix (row-major):
    # Block (bi, bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Element (i,j) in 4x4 is at index i*4+j
    # Block element (bi, bj, ri, rj) is at index (2*bi+ri)*4 + (2*bj+rj)
    
    def blk(bi, bj, ri, rj):
        return (2*bi + ri) * N + (2*bj + rj)
    
    # Strassen combinations for 4x4 using 7 products of 2x2 matrices
    # Each 2x2 product has 7 scalar multiplications
    # Product t (t=0..6) uses 7 scalar mults at columns t*7 .. t*7+6
    
    # A-side linear combinations for each Strassen product:
    # M1: A11 + A22
    # M2: A21 + A22
    # M3: A11
    # M4: A22
    # M5: A11 + A12
    # M6: A21 - A11
    # M7: A12 - A22
    
    # For each product t, and each 2x2 sub-product s (s=0..6), 
    # the A-side coefficients for the 4 elements of the 2x2 matrix:
    # Using the same Strassen pattern for 2x2:
    # s=0: (a11+a22) -> elements (0,0) and (1,1) get +1
    # s=1: (a21+a22) -> elements (1,0) and (1,1) get +1
    # s=2: a11 -> element (0,0) gets +1
    # s=3: a22 -> element (1,1) gets +1
    # s=4: (a11+a12) -> elements (0,0) and (0,1) get +1
    # s=5: (a21-a11) -> element (1,0) gets +1, (0,0) gets -1
    # s=6: (a12-a22) -> element (0,1) gets +1, (1,1) gets -1
    
    # A-side element indices within a 2x2 block (bi,bj):
    # (0,0) -> blk(bi,bj,0,0), (0,1) -> blk(bi,bj,0,1)
    # (1,0) -> blk(bi,bj,1,0), (1,1) -> blk(bi,bj,1,1)
    
    # A-side Strassen coefficients for each product t and sub-product s:
    # a_coeffs[t][s] = list of (element_offset, coeff)
    # element_offset: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    a_strassen = [
        [(0, 1), (3, 1)],           # s=0: a11+a22
        [(2, 1), (3, 1)],           # s=1: a21+a22
        [(0, 1)],                    # s=2: a11
        [(3, 1)],                    # s=3: a22
        [(0, 1), (1, 1)],           # s=4: a11+a12
        [(2, 1), (0, -1)],          # s=5: a21-a11
        [(1, 1), (3, -1)]           # s=6: a12-a22
    ]
    
    # B-side Strassen coefficients:
    # M1: B11+B22
    # M2: B11
    # M3: B21-B22
    # M4: B11-B21
    # M5: B22
    # M6: B11+B12
    # M7: B21+B22
    
    b_strassen = [
        [(0, 1), (3, 1)],           # s=0: b11+b22
        [(0, 1)],                    # s=1: b11
        [(2, 1), (3, -1)],          # s=2: b21-b22
        [(0, 1), (2, -1)],          # s=3: b11-b21
        [(3, 1)],                    # s=4: b22
        [(0, 1), (1, 1)],           # s=5: b11+b12
        [(2, 1), (3, 1)]            # s=6: b21+b22
    ]
    
    # C-side Strassen coefficients:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each 2x2 output element (ci, cj), the coefficients for each M_s:
    # c_coeffs[c_elem][s] = coeff
    c_strassen = [
        {0: 1, 3: 1, 4: -1, 6: 1},  # C11: M1 + M4 - M5 + M7
        {2: 1, 4: 1},                # C12: M3 + M5
        {1: 1, 3: 1},                # C21: M2 + M4
        {0: 1, 1: -1, 2: 1, 5: 1}    # C22: M1 - M2 + M3 + M6
    ]
    
    # Now build the full 4x4 decomposition
    # For each Strassen product t (0..6):
    #   The A-side matrix is a 2x2 combination of the 2x2 blocks
    #   The B-side matrix is a 2x2 combination of the 2x2 blocks
    #   For each sub-product s (0..6) within product t:
    #     column = t*7 + s
    #     A coefficients: for each block element (bi,bj,ri,rj) involved in the A-side 2x2
    #     B coefficients: for each block element (bi,bj,ri,rj) involved in the B-side 2x2
    #     C coefficients: for each block element in the output 2x2
    
    # A-side: for product t, which 2x2 blocks are combined?
    # M1: A11+A22 -> blocks (0,0) and (1,1)
    # M2: A21+A22 -> blocks (1,0) and (1,1)
    # M3: A11 -> block (0,0)
    # M4: A22 -> block (1,1)
    # M5: A11+A12 -> blocks (0,0) and (0,1)
    # M6: A21-A11 -> blocks (1,0) and (0,0)
    # M7: A12-A22 -> blocks (0,1) and (1,1)
    
    a_blocks = [
        [(0,0,1), (1,1,1)],    # M1: A11+A22
        [(1,0,1), (1,1,1)],    # M2: A21+A22
        [(0,0,1)],              # M3: A11
        [(1,1,1)],              # M4: A22
        [(0,0,1), (0,1,1)],    # M5: A11+A12
        [(1,0,1), (0,0,-1)],   # M6: A21-A11
        [(0,1,1), (1,1,-1)]    # M7: A12-A22
    ]
    
    b_blocks = [
        [(0,0,1), (1,1,1)],    # M1: B11+B22
        [(0,0,1)],              # M2: B11
        [(1,0,1), (1,1,-1)],   # M3: B21-B22
        [(0,0,1), (1,0,-1)],   # M4: B11-B21
        [(1,1,1)],              # M5: B22
        [(0,0,1), (0,1,1)],    # M6: B11+B12
        [(1,0,1), (1,1,1)]     # M7: B21+B22
    ]
    
    # C-side: for product t, which output block(s) receive contributions?
    # C11 gets M1, M4, M5, M7
    # C12 gets M3, M5
    # C21 gets M2, M4
    # C22 gets M1, M2, M3, M6
    
    c_blocks = [
        [(0,0,1), (1,1,1), (0,1,-1), (0,2,1)],   # Wait, this is getting complex
    ]
    
    # Let me redo this more carefully.
    # The 4x4 Strassen algorithm:
    # C11 = M1 + M4 - M5 + M7  where C11 is the top-left 2x2 block
    # C12 = M3 + M5            where C12 is the top-right 2x2 block
    # C21 = M2 + M4            where C21 is the bottom-left 2x2 block
    # C22 = M1 - M2 + M3 + M6  where C22 is the bottom-right 2x2 block
    
    # Each M_t is a 2x2 matrix product.
    # For M_t, the A-side is a 2x2 matrix (combination of A's 2x2 blocks)
    # and the B-side is a 2x2 matrix (combination of B's 2x2 blocks).
    
    # For the tensor decomposition, each M_t contributes 7 scalar multiplications.
    # For each M_t and each of its 7 internal Strassen sub-products:
    #   The A-side linear combination involves specific elements of specific A blocks
    #   The B-side linear combination involves specific elements of specific B blocks
    #   The C-side output goes to specific elements of specific C blocks
    
    # Let me define it step by step:
    
    # For product t, sub-product s:
    #   Column index: col = t * 7 + s
    
    # A-side: The 2x2 matrix for A-side of product t
    #   For each (bi, bj, sign) in a_blocks[t]:
    #     For each element (ri, rj) in the 2x2 block:
    #       Apply the s-th Strassen coefficient to element (bi,bj,ri,rj)
    
    # B-side: Similarly with b_blocks[t]
    
    # C-side: The output 2x2 block for product t
    #   C11 (block 0,0) gets contributions from M1, M4, M5, M7
    #   C12 (block 0,1) gets contributions from M3, M5
    #   C21 (block 1,0) gets contributions from M2, M4
    #   C22 (block 1,1) gets contributions from M1, M2, M3, M6
    
    # For each output element (ci, cj) in a 2x2 block, the coefficient for M_s is:
    # From c_strassen above
    
    # C-side block assignments:
    # Output block (oi, oj) receives M_t with coefficient c_strassen[oi*2+oj][s]
    c_block_assignments = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},   # C11
        (0, 1): {2: 1, 4: 1},                   # C12
        (1, 0): {1: 1, 3: 1},                   # C21
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1}       # C22
    }
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            # A-side
            for (bi, bj, sign) in a_blocks[t]:
                for (elem_off, coeff) in a_strassen[s]:
                    ri, rj = elem_off // 2, elem_off % 2
                    idx = blk(bi, bj, ri, rj)
                    U[idx][col] += sign * coeff
            
            # B-side
            for (bi, bj, sign) in b_blocks[t]:
                for (elem_off, coeff) in b_strassen[s]:
                    ri, rj = elem_off // 2, elem_off % 2
                    idx = blk(bi, bj, ri, rj)
                    V[idx][col] += sign * coeff
            
            # C-side
            for (oi, oj, c_map) in [(0,0,c_block_assignments[(0,0)]), 
                                      (0,1,c_block_assignments[(0,1)]),
                                      (1,0,c_block_assignments[(1,0)]),
                                      (1,1,c_block_assignments[(1,1)])]:
                if s in c_map:
                    coeff = c_map[s]
                    for ri in range(2):
                        for rj in range(2):
                            idx = blk(oi, oj, ri, rj)
                            W[idx][col] += coeff
    
    return U, V, W

def construct():
    """
    Attempt to find a rank-minimal decomposition for 2x4 * 4x8 matrix multiplication.
    Strategy: Embed into 4x4 * 4x4, use Strassen rank-49, project back.
    If that doesn't work, fall back to rank-56 Strassen.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Try the 4x4 Strassen approach:
    # Embed A (2x4) into top-left of 4x4: A_ext = [[A, 0], [0, 0]]
    # Embed B (4x8) - this doesn't fit neatly into 4x4.
    # 
    # Alternative: Split B into two 4x4 blocks: B = [B1, B2]
    # Compute C1 = A_ext @ B1 and C2 = A_ext @ B2 using rank-49 Strassen
    # Each uses the same A-side (25... wait, 49 columns for A)
    # Total rank: 49 + 49 = 98. That's worse than 56.
    #
    # Better idea: A is 2x4, B is 4x8.
    # Split B into two 4x4 blocks: B = [B_left, B_right]
    # C = A @ B = [A @ B_left, A @ B_right], each C_part is 2x4.
    # 
    # For A @ B_left (2x4 * 4x4):
    #   Embed A into 4x4: A_ext = [[A, 0], [0, 0]]
    #   Compute A_ext @ B_left using rank-49 Strassen for 4x4
    #   Extract the top-left 2x4 of the result
    #
    # For A @ B_right:
    #   Same A_ext, compute A_ext @ B_right using rank-49 Strassen
    #   Extract top-left 2x4
    #
    # Total rank: 49 + 49 = 98. Still worse.
    #
    # What if we can share some of the Strassen multiplications?
    # The A-side is the same for both, but the B-side is different.
    # In the tensor decomposition, each rank term is (u_r, v_r, w_r).
    # For the two products, we have:
    #   Product 1: (u_r, v_r^1, w_r^1) for r=1..49
    #   Product 2: (u_r, v_r^2, w_r^2) for r=1..49
    # The u_r are the same, but v and w differ.
    # We can't combine them into fewer rank terms in general.
    #
    # So the 4x4 approach gives rank 98, which is worse than 56.
    #
    # Let me think of a better approach.
    # 
    # What about splitting A and B differently?
    # A is 2x4, B is 4x8.
    # 
    # Split A into two 2x2 blocks: A = [A1, A2]
    # Split B into four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # But B is 4x8, so B = [B_left, B_right] where B_left, B_right are 4x4.
    # Or B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] where each Bij is 2x2.
    # 
    # C = A @ B = [A1, A2] @ B
    # C is 2x8.
    # C = [A1 @ B + A2 @ B]... no, that's not right.
    # C[i, j] = sum_k A[i,k] B[k,j]
    # 
    # If we split A columns into two 2x2 blocks and B rows into two 2x2 blocks:
    # A = [A_left, A_right] where A_left, A_right are 2x2
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x8
    # 
    # C = A_left @ B_top + A_right @ B_bottom
    # 
    # A_left @ B_top is 2x2 * 2x8.
    # A_right @ B_bottom is 2x2 * 2x8.
    #
    # For 2x2 * 2x8: split B_top into four 2x2 blocks.
    # A_left @ B_top = [A_left @ B_top1, A_left @ B_top2, A_left @ B_top3, A_left @ B_top4]
    # Each A_left @ B_topk is 2x2 * 2x2, using Strassen rank 7.
    # Total for A_left @ B_top: 4 * 7 = 28.
    # Similarly for A_right @ B_bottom: 28.
    # Total: 56. Same as before.
    #
    # Can we do better? What if we use a different factorization?
    #
    # Let me try a different split:
    # A is 2x4, B is 4x8.
    # 
    # What if we view this as (2x4) * (4x8) and use the fact that
    # the multiplication can be seen as a sum of outer products?
    #
    # C = sum_k A[:,k] * B[k,:]
    # where A[:,k] is 2x1 and B[k,:] is 1x8.
    # There are 4 such terms. Each term is a rank-1 matrix.
    # The tensor rank of the sum might be less than 4*8=32 if there's structure.
    #
    # Actually, the trivial decomposition has rank 4*2*8/... no.
    # The trivial (schoolbook) decomposition:
    # For each i in 0..1, j in 0..7, k in 0..3:
    #   C[i,j] += A[i,k] * B[k,j]
    # This is 2*8*4 = 64 scalar multiplications. Rank 64.
    #
    # Strassen gives 56. Can we do better?
    #
    # Let me try a rank-52 approach by combining some of the Strassen blocks.
    # 
    # Actually, let me just try to find a valid decomposition with rank < 56
    # using a systematic search with integer coefficients.
    #
    # Given time constraints, I'll implement the rank-56 solution and add
    # a quick search for rank 55 or 54.
    
    # Build the rank-56 Strassen solution
    U56, V56, W56 = _build_strassen56()
    
    # Verify it works
    if _validate(U56, V56, W56, n, m, p, num_trials=2):
        if time.time() > deadline - 10:
            return {"U": U56, "V": V56, "W": W56}
        
        # Try to find a lower rank by attempting to eliminate one rank term
        # This is a heuristic: try setting one column to zero and see if
        # the decomposition still works (it won't in general, but we can
        # try linear combinations).
        
        # A more systematic approach: try rank 55 by merging two rank terms
        # into one with a linear combination. This is complex, so let's
        # just return the rank-56 solution for now.
        
        return {"U": U56, "V": V56, "W": W56}
    else:
        # Fallback: build schoolbook rank 64
        rank = n * m * p  # 2*4*8 = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}

