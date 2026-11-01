"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using Strassen on two 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block Strassen (7 products)
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1; V[2][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
        
        # Right block Strassen (7 products)
        U[0][7] = 1; U[3][7] = 1; V[4][7] = 1; V[7][7] = 1; W[4][7] = 1; W[7][7] = 1
        U[2][8] = 1; U[3][8] = 1; V[4][8] = 1; W[6][8] = 1; W[7][8] = -1
        U[0][9] = 1; V[6][9] = 1; V[7][9] = -1; W[5][9] = 1; W[7][9] = 1
        U[3][10] = 1; V[5][10] = 1; V[4][10] = -1; W[4][10] = 1; W[6][10] = 1
        U[0][11] = 1; U[1][11] = 1; V[7][11] = 1; W[4][11] = -1; W[5][11] = 1
        U[2][12] = 1; U[0][12] = -1; V[4][12] = 1; V[5][12] = 1; W[7][12] = 1
        U[1][13] = 1; U[3][13] = -1; V[6][13] = 1; V[7][13] = 1; W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_strassen_shared():
        """
        Try a rank-11 decomposition by sharing some Strassen products between
        the two 2x2 blocks. The key insight: in Strassen, M1 = (a00+a11)(b00+b11).
        If we can share the U-factor (a00+a11) across both blocks, we save rank.
        
        Actually, let's think more carefully. The two blocks share the same A matrix.
        Strassen for block 1 (B[:,0:2]):
          M1 = (a00+a11)(b00+b11)
          M2 = (a10+a11)(b00)
          M3 = a00(b10-b11)
          M4 = a11(b01-b00)
          M5 = (a00+a01)(b11)
          M6 = (a10-a00)(b00+b01)
          M7 = (a01-a11)(b10+b11)
        
        Strassen for block 2 (B[:,2:4]):
          N1 = (a00+a11)(b02+b13)
          N2 = (a10+a11)(b02)
          N3 = a00(b12-b13)
          N4 = a11(b03-b02)
          N5 = (a00+a01)(b13)
          N6 = (a10-a00)(b02+b03)
          N7 = (a01-a11)(b12+b13)
        
        The U-factors are identical between M and N pairs:
          M1 and N1 share (a00+a11)
          M2 and N2 share (a10+a11)
          M3 and N3 share a00
          M4 and N4 share a11
          M5 and N5 share (a00+a01)
          M6 and N6 share (a10-a00)
          M7 and N7 share (a01-a11)
        
        But we can't just merge them because the V and W factors differ.
        However, we can try a different approach: use a generalized Strassen
        where we combine the two blocks into a single 2x4 multiplication.
        
        Let's try a direct rank-11 construction based on known results for
        2x2 * 2x4. The border rank is known to be 11 for this case.
        
        Actually, let me try a different structured approach. We can view
        2x2 * 2x4 as computing C = A * B where B is 2x4.
        
        Let me try to construct a rank-11 decomposition manually.
        """
        R = 11
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # We'll use a hybrid: 7 for left block + 4 additional for right block
        # by sharing some structure.
        
        # Actually, let's use a known construction. For 2x2*2x4, 
        # one approach: treat B as [B1 | B2] where B1, B2 are 2x2.
        # C = A*B1 | A*B2
        # Use Strassen for A*B1 (7 mults) and a cheaper scheme for A*B2.
        # But A*B2 also needs 7 in general...
        
        # Alternative: use the fact that we can write
        # A*B = (A*B1) + (A*B2) and find a joint decomposition.
        
        # Let me try a direct algebraic construction for rank 11.
        # This is based on the work of Stothers and others.
        
        # I'll attempt a specific rank-11 construction:
        # r=0: (a00+a11)(b00+b11) -> contributes to C00, C11
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: (a10+a11)(b00) -> C10, C11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # r=2: a00(b10-b11) -> C01, C11
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # r=3: a11(b01-b00) -> C00, C10
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # r=4: (a00+a01)(b11) -> C00, C01
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # r=5: (a10-a00)(b00+b01) -> C11
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: (a01-a11)(b10+b11) -> C00
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        # Now for the right block (cols 2,3), we need to compute
        # C02, C03, C12, C13 using 4 more rank-1 terms.
        # This is not generally possible with just 4 terms, but let's see
        # if we can use the shared A-structure.
        
        # r=7: (a00+a11)(b02+b13) -> C02, C13
        U[0][7] = 1; U[3][7] = 1
        V[4][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: (a10+a11)(b02) -> C12, C13
        U[2][8] = 1; U[3][8] = 1
        V[4][8] = 1
        W[6][8] = 1; W[7][8] = -1
        
        # r=9: a00(b12-b13) -> C03, C13
        U[0][9] = 1
        V[6][9] = 1; V[7][9] = -1
        W[5][9] = 1; W[7][9] = 1
        
        # r=10: a11(b03-b02) -> C02, C12
        U[3][10] = 1
        V[5][10] = 1; V[4][10] = -1
        W[4][10] = 1; W[6][10] = 1
        
        # Check if this covers all 8 output entries
        # C00: r0(+1), r3(+1), r4(-1), r6(+1) -> a00*b00 + a11*b01 - a11*b00 + ... 
        # Let me verify numerically
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank10():
        """Try an even more aggressive rank-10 construction."""
        R = 10
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # This is speculative - trying to find a rank-10 decomposition
        # by combining left and right block Strassen more aggressively.
        
        # Use 5 for left + 5 for right with shared structure
        # r=0: (a00+a11)(b00+b11+b02+b13) -> C00, C11, C02, C13
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1; V[4][0] = 1; V[7][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[4][0] = 1; W[7][0] = 1
        
        # r=1: (a10+a11)(b00+b02) -> C10, C11, C12, C13
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1; V[4][1] = 1
        W[2][1] = 1; W[3][1] = -1; W[6][1] = 1; W[7][1] = -1
        
        # r=2: a00(b10-b11+b12-b13) -> C01, C11, C03, C13
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1; V[6][2] = 1; V[7][2] = -1
        W[1][2] = 1; W[3][2] = 1; W[5][2] = 1; W[7][2] = 1
        
        # r=3: a11(b01-b00+b03-b02) -> C00, C10, C02, C12
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1; V[5][3] = 1; V[4][3] = -1
        W[0][3] = 1; W[2][3] = 1; W[4][3] = 1; W[6][3] = 1
        
        # r=4: (a00+a01)(b11+b13) -> C00, C01, C02, C03
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1; V[7][4] = 1
        W[0][4] = -1; W[1][4] = 1; W[4][4] = -1; W[5][4] = 1
        
        # r=5: (a10-a00)(b00+b01+b02+b03) -> C11, C13
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1; V[4][5] = 1; V[5][5] = 1
        W[3][5] = 1; W[7][5] = 1
        
        # r=6: (a01-a11)(b10+b11+b12+b13) -> C00, C02
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1; V[6][6] = 1; V[7][6] = 1
        W[0][6] = 1; W[4][6] = 1
        
        # r=7: (a00+a11)(b02+b13) already covered, need more
        # r=7: (a01+a10)(b00+b11) -> cross terms
        U[1][7] = 1; U[2][7] = 1
        V[0][7] = 1; V[3][7] = 1
        W[0][7] = 0; W[3][7] = 0
        # This doesn't contribute to output... skip
        
        # Let me just verify what we have so far
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_numerical_peeling():
        """
        Use iterative rank-1 peeling to find a low-rank decomposition.
        Build the matrix multiplication tensor, then greedily peel off
        rank-1 components.
        """
        if time.time() > deadline - 10:
            return None
        
        # Build the matrix multiplication tensor T of shape (nm, mp, pn)
        # T[a, b, c] = 1 if the output entry c receives contribution from A[a]*B[b]
        # a = i*m+j (row i, col j of A), b = j*p+k (row j, col k of B), c = k*n+i (row k, col i of C)
        # Actually, the tensor is: T[(i,j), (j,k), (k,i)] = 1 for all i,j,k
        # In flattened indices: U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
        
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Greedy rank-1 peeling
        U_list = []
        V_list = []
        W_list = []
        T_work = T.copy()
        
        max_rank = 16
        tol = 1e-10
        
        for r in range(max_rank):
            if time.time() > deadline - 5:
                break
            
            # Find the largest entry in T_work
            idx = np.unravel_index(np.argmax(np.abs(T_work)), T_work.shape)
            val = T_work[idx]
            if abs(val) < tol:
                break
            
            # For the matrix multiplication tensor, rank-1 peeling is special.
            # The tensor has a specific structure. Let's use a different approach:
            # find the best rank-1 approximation.
            
            # Reshape T_work to a matrix: (nm, mp*pn)
            T_mat = T_work.reshape(nm, mp * pn)
            
            # SVD
            try:
                U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
            except:
                break
            
            if s_vals[0] < tol:
                break
            
            # The top singular vector gives the best rank-1 approximation
            u_vec = U_svd[:, 0] * s_vals[0]
            vt_vec = Vt_svd[0] * s_vals[0]
            
            # But this gives a rank-1 tensor, not necessarily a valid one for matmul.
            # For matmul tensors, rank-1 components have a specific form.
            # Let's just use the SVD-based peeling and check at the end.
            
            u_vec = u_vec / s_vals[0] if s_vals[0] != 0 else u_vec
            vt_vec = vt_vec / s_vals[0] if s_vals[0] != 0 else vt_vec
            
            U_list.append(u_vec.copy())
            V_list.append(vt_vec[:mp].copy())
            W_list.append(vt_vec[mp:].copy())
            
            # Subtract the rank-1 component
            T_work -= np.outer(u_vec, vt_vec).reshape(nm, mp, pn)
            
            # Check convergence
            if np.max(np.abs(T_work)) < tol:
                break
        
        if np.max(np.abs(T_work)) > tol:
            return None
        
        R = len(U_list)
        U_mat = np.column_stack(U_list)
        V_mat = np.column_stack(V_list)
        W_mat = np.column_stack(W_list)
        
        # Round to nearest half-integer if close
        U_rounded = np.round(U_mat * 2) / 2
        V_rounded = np.round(V_mat * 2) / 2
        W_rounded = np.round(W_mat * 2) / 2
        
        # Verify with rounded values
        U_list_r = U_rounded.tolist()
        V_list_r = V_rounded.tolist()
        W_list_r = W_rounded.tolist()
        
        if verify(U_list_r, V_list_r, W_list_r):
            return {"U": U_list_r, "V": V_list_r, "W": W_list_r}
        
        # Try without rounding
        U_list_f = U_mat.tolist()
        V_list_f = V_mat.tolist()
        W_list_f = W_mat.tolist()
        
        if verify(U_list_f, V_list_f, W_list_f):
            return {"U": U_list_f, "V": V_list_f, "W": W_list_f}
        
        return None
    
    def try_rank12_construction():
        """
        Try a specific rank-12 construction for 2x2*2x4.
        
        Key idea: use 7 Strassen products for the left 2x2 block,
        and 5 products for the right 2x2 block by sharing
        some of the A-side linear combinations.
        
        The right block needs: C02, C03, C12, C13
        C02 = a00*b02 + a01*b12
        C03 = a00*b03 + a01*b13
        C12 = a10*b02 + a11*b12
        C13 = a10*b03 + a11*b13
        
        We can compute these with 5 products:
        P1 = (a00+a11)(b02+b13) -> gives a00*b02 + a00*b13 + a11*b02 + a11*b13
        P2 = (a10+a11)(b02) -> gives a10*b02 + a11*b02
        P3 = a00*(b12-b13) -> gives a00*b12 - a00*b13
        P4 = a11*(b03-b02) -> gives a11*b03 - a11*b02
        P5 = (a00+a01)*b13 -> gives a00*b13 + a01*b13
        
        Then:
        C02 = P1 - P2 - P3 - P4 + P5 - a01*b13 - a11*b13... 
        This is getting complicated. Let me just verify numerically.
        """
        R = 12
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block: standard Strassen (r=0..6)
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1; V[2][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
        
        # Right block: 5 products (r=7..11)
        # We need: C02=a00*b02+a01*b12, C03=a00*b03+a01*b13, C12=a10*b02+a11*b12, C13=a10*b03+a11*b13
        
        # r=7: (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[4][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: (a10+a11)(b02)
        U[2][8] = 1; U[3][8] = 1
        V[4][8] = 1
        W[6][8] = 1; W[7][8] = -1
        
        # r=9: a00*(b12-b13)
        U[0][9] = 1
        V[6][9] = 1; V[7][9] = -1
        W[5][9] = 1; W[7][9] = 1
        
        # r=10: a11*(b03-b02)
        U[3][10] = 1
        V[5][10] = 1; V[4][10] = -1
        W[4][10] = 1; W[6][10] = 1
        
        # r=11: (a00+a01)*b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[5][11] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try structured decompositions in order of increasing rank
    result = try_rank10()
    if result is not None:
        return result
    
    result = try_rank12_construction()
    if result is not None:
        return result
    
    result = try_strassen_shared()
    if result is not None:
        return result
    
    result = try_strassen_rectangular()
    if result is not None:
        return result
    
    # Try numerical peeling
    result = try_numerical_peeling()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
