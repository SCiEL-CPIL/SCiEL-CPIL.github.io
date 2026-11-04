import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Known ranks for small matrix multiplication
    # M_{a,b,c}: multiply a x b matrix by b x c matrix
    # We need M_{2,3,6}
    #
    # Strategy: try to find a low-rank decomposition by leveraging
    # the structure. We know:
    # - M_{1,1,1} = rank 1
    # - M_{2,2,2} = rank 7 (Strassen)
    # - M_{1,2,2} = rank 4 (schoolbook is 4, no known improvement)
    # - M_{2,2,1} = rank 4
    # - M_{2,3,3}: unknown small rank, let's try to find it
    #
    # For M_{2,3,6}, one approach:
    # Split B into 3 blocks of 2 columns: B = [B1 | B2 | B3]
    # Then C = A*B = [A*B1 | A*B2 | A*B3]
    # Each A*Bi is a 2x2 multiplication with A being 2x3 and Bi being 3x2.
    # So we need rank of M_{2,3,2}.
    #
    # Let's try to find rank of M_{2,3,2} first, then total rank = 3 * rank(M_{2,3,2})
    #
    # For M_{2,3,2}: A is 2x3, B is 3x2, C is 2x2
    # Schoolbook rank = 2*3*2 = 12
    # Can we do better?
    #
    # Alternative: use the fact that 2x3x6 can be viewed differently.
    # Let's try a direct search for small ranks.

    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W correctly decompose the n x m * m x p multiplication."""
        rank = U.shape[1]
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U[ii * m + jj, col] = 1.0
                    V[jj * p + kk, col] = 1.0
                    W[kk * n + ii, col] = 1.0
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """Return Strassen decomposition for 2x2 * 2x2 multiplication."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
        U[0, 0] = 1; U[3, 0] = 1  # a00, a11
        V[0, 0] = 1; V[3, 0] = 1  # b00, b11
        W[0, 0] = 1; W[3, 0] = 1  # c00, c11
        
        # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a00(b10-b11) -> c01 += M3, c11 += M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = a11(b01-b00) -> c00 += M4, c01 += M4
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (a10-a00)(b00+b01) -> c11 += M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a01-a11)(b10+b11) -> c00 += M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def block_decompose(n, m, p, block_m, block_p):
        """
        Decompose M_{n,m,p} by splitting B into blocks.
        A is n x m, B is m x p.
        Split B into p//block_p blocks of size block_p, and m into m//block_m blocks of size block_m.
        For each pair of blocks, we do a (n, block_m, block_p) multiplication.
        """
        num_m_blocks = m // block_m
        num_p_blocks = p // block_p
        
        if num_m_blocks * block_m != m or num_p_blocks * block_p != p:
            return None
        
        total_rank = 0
        U_list = []
        V_list = []
        W_list = []
        
        for mb in range(num_m_blocks):
            for pb in range(num_p_blocks):
                # Get sub-tensor for A[:, mb*block_m:(mb+1)*block_m] * B[mb*block_m:(mb+1)*block_m, pb*block_p:(pb+1)*block_p]
                # This is M_{n, block_m, block_p}
                if n == 2 and block_m == 2 and block_p == 2:
                    U_sub, V_sub, W_sub = strassen_2x2x2()
                elif n == 1:
                    U_sub = np.zeros((block_m, block_m * block_p))
                    V_sub = np.zeros((block_m * block_p, block_m * block_p))
                    W_sub = np.zeros((block_p, block_m * block_p))
                    for jj in range(block_m):
                        for kk in range(block_p):
                            idx = jj * block_p + kk
                            U_sub[jj, idx] = 1.0
                            V_sub[idx, idx] = 1.0
                            W_sub[kk, idx] = 1.0
                else:
                    U_sub, V_sub, W_sub = schoolbook(n, block_m, block_p)
                
                rank_sub = U_sub.shape[1]
                
                # Map indices
                U_block = np.zeros((n * m, rank_sub))
                V_block = np.zeros((m * p, rank_sub))
                W_block = np.zeros((p * n, rank_sub))
                
                for i in range(n):
                    for jj in range(block_m):
                        for r in range(rank_sub):
                            U_block[i * m + (mb * block_m + jj), r] = U_sub[i * block_m + jj, r]
                
                for jj in range(block_m):
                    for kk in range(block_p):
                        for r in range(rank_sub):
                            V_block[(mb * block_m + jj) * p + (pb * block_p + kk), r] = V_sub[jj * block_p + kk, r]
                
                for kk in range(block_p):
                    for i in range(n):
                        for r in range(rank_sub):
                            W_block[(pb * block_p + kk) * n + i, r] = W_sub[kk * n + i, r]
                
                U_list.append(U_block)
                V_list.append(V_block)
                W_list.append(W_block)
                total_rank += rank_sub
        
        if U_list:
            U = np.hstack(U_list)
            V = np.hstack(V_list)
            W = np.hstack(W_list)
            return U, V, W
        return None

    # Try different block decompositions
    best_result = None
    best_rank = float('inf')
    
    # Option 1: Block with 2x2 Strassen
    # 2x3x6: split into 1 block of m=3 (can't split m=3 into 2s evenly)
    # Actually m=3, so we can't use 2x2 blocks for m.
    # Let's try: treat A as 2x3, B as 3x6
    # Split B into three 3x2 blocks
    # Each sub-problem is M_{2,3,2}
    
    # Let's search for a good rank for M_{2,3,2}
    # Schoolbook rank for M_{2,3,2} = 12
    # Can we do better?
    
    # Try to construct M_{2,3,2} using known techniques
    # One idea: M_{2,3,2} = M_{2,3,3} restricted? No.
    # Another: M_{2,3,2} can be seen as embedding into M_{2,3,3}
    
    # Let's try a direct construction for M_{2,3,2}
    # A is 2x3: [[a,b,c],[d,e,f]]
    # B is 3x2: [[g,h],[i,j],[k,l]]
    # C is 2x2: [[ag+bh+ck, ah+bj+cl], [dg+ei+fk, dh+ej+fl]]
    
    # We want to express this as a rank-r decomposition.
    # For rank 9:
    # This is the same as 2x2x2 but with an extra column in A and B.
    
    # Actually, let's just use the block approach with schoolbook for the sub-blocks
    # and see if we can improve.
    
    # Option: use the fact that 2x3x6 = 2x3x2 + 2x3x2 + 2x3x2 (three independent blocks)
    # If we find rank r for 2x3x2, total rank = 3r
    
    # Let's try to find a low-rank decomposition for M_{2,3,2}
    # using a search approach with small ranks
    
    def search_rank_2x3x2(max_rank=12):
        """Try to find a low-rank decomposition for M_{2,3,2}."""
        n_sub, m_sub, p_sub = 2, 3, 2
        for rank in range(6, max_rank + 1):
            if time.time() > deadline - 20:
                break
            # Try to find integer coefficients
            # This is a complex search, let's use a heuristic
            pass
        return None

    # For now, let's use the block decomposition with Strassen where possible
    # and schoolbook elsewhere
    
    # Decomposition 1: 2x3x6 as three 2x3x2 blocks
    # Each 2x3x2 has schoolbook rank 12, total = 36
    # Can we do better?
    
    # Decomposition 2: Use a different split
    # What if we view 2x3x6 as 2x(3x6) = 2x18?
    # Or 2x3x6 = (2x3)x6?
    
    # Let's try a direct construction using Strassen-like ideas
    
    # Actually, the best known approach for 2x3x6 might be to use
    # the fact that it can be embedded in a larger problem
    
    # For now, let's use the current approach: 3 blocks of 2x2x2 Strassen + handling the rest
    # A is 2x3: columns 0,1,2
    # B is 3x6: rows 0,1,2
    
    # Part 1: A[:,0] * B[0,:] -> 2x1 * 1x6 = 2x6, rank 12
    # Part 2: A[:,1:3] * B[1:3,:] -> 2x2 * 2x6 = 2x6
    #   Split B[1:3,:] into three 2x2 blocks
    #   Each 2x2*2x2 uses Strassen rank 7
    #   Total for Part 2: 3 * 7 = 21
    # Total: 12 + 21 = 33
    
    # Let's try to improve Part 1
    # A[:,0] is 2x1, B[0,:] is 1x6
    # C_part1[i,k] = A[i,0] * B[0,k]
    # This is a rank-1 matrix times a vector, which requires rank 12 in general
    # But maybe we can combine with Part 2?
    
    # Alternative: Don't split by columns of A, but by rows of B
    # B is 3x6, split into 3 blocks of 2 rows: B[0:2,:], B[1:3,:] (wait, 3 rows, can't split evenly into 2s)
    # B[0:2,:] is 2x6, B[2,:] is 1x6
    # A[:,0:2] * B[0:2,:] + A[:,2] * B[2,:]
    # First part: 2x2 * 2x6, split into 2x2*2x2 blocks (3 blocks, Strassen rank 7 each) = 21
    # Second part: 2x1 * 1x6 = 12
    # Total: 33 again
    
    # Let's try a different approach: use a known result
    # The rank of M_{2,3,6} is known to be at most 33 (from the above)
    # Is there a known better bound?
    
    # For now, let's stick with 33 and try to verify it works
    
    # Construct the 33-rank decomposition
    U = np.zeros((n * m, 33))
    V = np.zeros((m * p, 33))
    W = np.zeros((p * n, 33))
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    
    # Part 2: A[:,1:3] * B[1:3,:]
    # A_sub is 2x2: A[0,1], A[0,2], A[1,1], A[1,2]
    # B_sub is 2x6: B[1,0:6], B[2,0:6]
    # Split B_sub into three 2x2 blocks
    
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        if time.time() > deadline - 10:
            break
        
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[1 * m + 1, col] += 1.0; U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0; W[k1 * n + 1, col] += -1.0
        col += 1
        
        # M3 = A00(B10-B11)
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0; V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0; W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M4 = A11(B01-B00)
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0; V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0; W[k0 * n + 1, col] += 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[0 * m + 1, col] += 1.0; U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0; W[k1 * n + 0, col] += 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 1, col] += 1.0; U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0; V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + 2, col] += 1.0; U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0; V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1
    
    # Verify
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U_school, V_school, W_school = schoolbook(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
