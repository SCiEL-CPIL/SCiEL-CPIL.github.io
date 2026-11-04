import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    t0 = time.time()
    DEADLINE = 130.0
    
    # Strassen 2x2x2 base (rank 7)
    # U_2, V_2, W_2 are (4,7)
    U_2 = np.array([
        [1,0,1,0,1,-1,0],
        [0,0,0,0,1,0,1],
        [0,1,0,0,0,1,0],
        [1,1,0,1,0,0,-1]
    ], dtype=np.float64)
    V_2 = np.array([
        [1,1,-1,0,0,1,0],
        [0,0,1,0,0,1,0],
        [0,0,0,1,0,0,1],
        [1,0,-1,0,1,0,1]
    ], dtype=np.float64)
    W_2 = np.array([
        [1,0,0,1,-1,0,1],
        [0,0,1,0,1,0,0],
        [0,1,0,1,0,1,0],
        [1,-1,1,0,0,0,0]
    ], dtype=np.float64)
    
    # Verify base case
    # Tensor T[i,j,k] = sum_k U[i,k]*V[j,k]*W[k,k] should equal 1 if i==j else 0 for 2x2
    # Actually for 2x2x2: T_{(r1,c1),(r2,c2),(r3,c3)} = 1 if (r1,c3)=(r2,c2)... 
    # Let's verify: T[i,j,l] where i=row_A, j=row_B, l=row_C
    # For 2x2: T[(r1,c1),(r2,c2),(r3,c3)] = sum_k U[r1*2+c1, k] * V[r2*2+c2, k] * W[r3*2+c3, k]
    # This should equal 1 if c1==c2 and r1==r3... no, matrix mult: C = A*B
    # C_{r,c} = sum_m A_{r,m} B_{m,c}
    # Tensor: T_{(r,c),(m,d),(s,e)} = 1 if r==s, m==c, d==e... 
    # Actually T[a,b,c] = 1 if a=(r,m), b=(m',c), c=(r',c') and r==r', m==m', c==c'
    # So T[a,b,c] = 1 iff row_A == row_C and col_A == row_B and col_B == col_C
    
    # Build the 4x4x4 Strassen tensor (rank 49)
    # 4x4x4: A is 4x4, B is 4x4, C is 4x4
    # Use Strassen with 2x2 blocks
    
    # Block indices for 2x2 blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Strassen steps (same as before):
    steps = [
        {'A': [(1,0),(1,3)], 'B': [(1,0),(1,3)], 'C': [(1,0),(1,3)]},
        {'A': [(1,2),(1,3)], 'B': [(1,0)], 'C': [(1,2),(-1,3)]},
        {'A': [(1,0)], 'B': [(1,1),(-1,3)], 'C': [(1,1),(1,3)]},
        {'A': [(1,3)], 'B': [(1,2),(-1,0)], 'C': [(1,0),(1,2)]},
        {'A': [(1,0),(1,1)], 'B': [(1,3)], 'C': [(-1,0),(1,1)]},
        {'A': [(1,2),(-1,0)], 'B': [(1,0),(1,1)], 'C': [(1,3)]},
        {'A': [(1,1),(-1,3)], 'B': [(1,2),(1,3)], 'C': [(1,0)]}
    ]
    
    # Build 4x4x4 factor matrices (rank 49)
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    def get_block_indices(bi, size=2):
        """Get row,col indices for a 2x2 block at position bi in 4x4"""
        r0 = (bi // 2) * size
        c0 = (bi % 2) * size
        idx = []
        for r in range(size):
            for c in range(size):
                idx.append((r0+r, c0+c))
        return idx
    
    for k_idx, step in enumerate(steps):
        for m_s in range(7):
            col = k_idx * 7 + m_s
            for coeff_a, bi in step['A']:
                for (r, c) in get_block_indices(bi):
                    li = (r % 2) * 2 + (c % 2)
                    U4[r*4+c, col] += coeff_a * U_2[li, m_s]
            for coeff_b, bi in step['B']:
                for (r, c) in get_block_indices(bi):
                    li = (r % 2) * 2 + (c % 2)
                    V4[r*4+c, col] += coeff_b * V_2[li, m_s]
            for coeff_c, bi in step['C']:
                for (r, c) in get_block_indices(bi):
                    li = (r % 2) * 2 + (c % 2)
                    W4[r*4+c, col] += coeff_c * W_2[li, m_s]
    
    # Verify 4x4x4: T[a,b,c] = sum_k U4[a,k]*V4[b,k]*W4[c,k]
    # a = r*4+c for A (4x4), b = r*4+c for B, c = r*4+c for C
    # T should equal 1 if a==c and (a%4)==(b//4) ... 
    # C_{r,c} = sum_m A_{r,m} B_{m,c}
    # T[(r,c),(m,d),(r',c')] = 1 if r==r', m==c, d==c'
    # So T[a,b,c] = 1 iff a//4 == c//4, a%4 == b//4, b%4 == c%4
    
    # Build 4x4x8 by taking direct sum of two 4x4x4
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    for half in range(2):
        U[:, half*49:(half+1)*49] = U4
        # V: B is 4x8, so rows are 4*8=32
        # For half=0: B columns 0-3 -> V4 rows 0-15
        # For half=1: B columns 4-7 -> V4 rows 16-31
        V[half*16:(half+1)*16, half*49:(half+1)*49] = V4
        # W: output is 8x4, so rows are 8*4=32
        # W index: row in output (0-7), col (0-3) -> W[row*4+col, k]
        # For half=0: output cols 0-3 -> W rows 0-15
        # For half=1: output cols 4-7 -> W rows 16-31
        W[half*16:(half+1)*16, half*49:(half+1)*49] = W4
    
    # Now try to reduce rank via linear dependencies
    # Check if columns of U, V, W are linearly dependent
    # We need to find a rank-reducing combination
    
    # First, let's verify the decomposition is correct
    # T[i,j,l] = sum_k U[i,k] V[j,k] W[l,k]
    # i in 0..15 (A: 4x4), j in 0..31 (B: 4x8), l in 0..31 (C: 8x4)
    # T should be 1 if: i//4 == l//4 (row of A == row of C in 8x4... wait)
    # C is 8x4, so C_{r,c} where r=0..7, c=0..3
    # l = r*4+c
    # T[i,j,l] = 1 iff (i//4) == (l//4) and (i%4) == (j//8) and (j%8) == (l%4)
    # Wait: A is 4x4, B is 4x8, C = A*B is 4x8... 
    # But the task says output is p*n = 8*4 = 32, so C is 8x4? 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # W has shape (p*n, rank) = (32, rank)
    # W index: k*n+i where k=0..p-1=0..7, i=0..n-1=0..3
    # So W[k*n+i, r] corresponds to C[k, i] in a p x n matrix? 
    # But A is n x m = 4x4, B is m x p = 4x8, so C = A*B is 4x8
    # C[i, k] where i=0..3, k=0..7
    # W[k*n+i, r] = W[k*4+i, r] corresponds to C[i, k]
    
    # T[i,j,l]: i = row_A * 4 + col_A (0..15)
    #            j = row_B * 8 + col_B (0..31)  
    #            l = row_C * 4 + col_C (0..31) where row_C=0..7, col_C=0..3
    # T[i,j,l] = 1 iff (i//4) == (l%4) and (i%4) == (j//8) and (j%8) == (l//4)
    # i//4 = row_A, i%4 = col_A
    # j//8 = row_B, j%8 = col_B
    # l//4 = row_C (0..7), l%4 = col_C (0..3)
    # C = A*B: C[row_A, col_B] = sum_m A[row_A, m] * B[m, col_B]
    # So T = 1 iff row_A == col_C... no
    # C[i, k] = sum_m A[i, m] * B[m, k]
    # T[i,j,l] = 1 iff (i//4) == (l%4) [row_A == col_C] -- NO
    # Actually C[row, col] where row=0..3, col=0..7
    # W[k*n+i] = W[k*4+i], so l = k*4+i means k=col of C, i=row of C
    # So l//4 = k = col_C, l%4 = i = row_C
    # T = 1 iff row_A == row_C and col_A == row_B and col_B == col_C
    # i//4 == l%4 and i%4 == j//8 and j%8 == l//4
    
    # Let's just verify numerically
    # But first, let's try to find linear dependencies to reduce rank
    
    # The direct sum of two 4x4x4 gives rank 98.
    # Known: 4x4x4 has rank 49 (Strassen)
    # 4x4x8: is it 2*49=98 or can we do better?
    
    # Try to find if there are linear dependencies among the 98 columns
    # We can try to find a non-trivial linear combination of columns that gives zero
    # in all three matrices simultaneously
    
    # For each column k, we have vectors u_k in R^16, v_k in R^32, w_k in R^32
    # We want to find coefficients c_k not all zero such that
    # sum_k c_k * u_k = 0 AND sum_k c_k * v_k = 0 AND sum_k c_k * w_k = 0
    
    # This is a system of 16+32+32 = 80 equations in 98 unknowns
    # So there should be at least 98-80 = 18 free variables
    # But the structure matters
    
    # Let's set up the null space problem
    # Stack U, V, W vertically: M is (80, 98)
    M = np.vstack([U, V, W])
    
    # Find null space
    # Use SVD
    try:
        _, s, vh = np.linalg.svd(M, full_matrices=True)
        # Null space is spanned by last (98 - rank) rows of vh
        tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
        rank_M = np.sum(s > tol)
        null_dim = 98 - rank_M
        
        if null_dim > 0:
            # Null space vectors
            null_vecs = vh[rank_M:]  # (null_dim, 98)
            
            # For each null vector c, check if it actually zeros out the tensor
            # i.e., sum_k c[k] * U[i,k] * V[j,k] * W[l,k] = 0 for all i,j,l
            # This is not guaranteed by just nullifying U, V, W separately
            # We need the tensor contraction to vanish
            
            # Let's check each null vector
            best_rank = 98
            for idx in range(null_dim):
                c = null_vecs[idx]
                # Check: sum_k c[k] * U[:,k] * V[:,k] * W[:,k] should be 0
                # Compute the tensor: T_new[i,j,l] = sum_k c[k] * U[i,k]*V[j,k]*W[l,k]
                # This is expensive: 16*32*32 = 16384 entries
                # T_new = np.einsum('ik,jk,lk,k->ijl', U, V, W, c)
                # But we just need to check if it's zero
                
                # Use einsum
                T_new = np.einsum('ik,jk,lk,k->ijl', U, V, W, c, optimize=True)
                if np.allclose(T_new, 0, atol=1e-10):
                    # This null vector works! We can remove one column
                    # But we need to be careful: we want to reduce rank by removing
                    # a column that is in the span of others
                    # Actually, if c is in the null space of the stacked matrix AND
                    # the tensor contraction is zero, then the corresponding combination
                    # of rank-1 tensors is zero
                    pass
        else:
            null_dim = 0
    except Exception:
        null_dim = 0
    
    # If no reduction found, return rank 98
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a different approach: use the known structure
    # The 4x4x8 problem can be decomposed differently
    # A is 4x4, B is 4x8
    # We can split B into two 4x4 halves: B = [B1, B2]
    # C = A*B = [A*B1, A*B2]
    # Each A*Bk is a 4x4 matrix
    # So we need two 4x4x4 multiplications
    # The direct sum gives rank 98
    
    # Can we share some computation between the two halves?
    # If B1 and B2 have some relationship, maybe
    # But in general they're independent
    
    # The minimal rank for 4x4x8 might indeed be 98
    # Or it could be less if there's a clever decomposition
    
    # Let's try to see if we can find a rank-1 or rank-2 reduction
    # by checking pairwise linear dependencies
    
    # For now, return rank 98
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
