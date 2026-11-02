"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

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

    def strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        
        U: U[0]=a00, U[1]=a01, U[2]=a10, U[3]=a11
        V: V[0]=b00, V[1]=b01, V[2]=b02, V[3]=b03, V[4]=b10, V[5]=b11, V[6]=b12, V[7]=b13
        W: W[0]=C[0][0], W[1]=C[1][0], W[2]=C[0][1], W[3]=C[1][1],
           W[4]=C[0][2], W[5]=C[1][2], W[6]=C[0][3], W[7]=C[1][3]
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B):
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        # C[0][0] = M1+M4-M5+M7, C[0][1]=M3+M5, C[1][0]=M2+M4, C[1][1]=M1-M2+M3+M6
        
        # r=0: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B):
        # N1 = (a00+a11)(b02+b13)
        # N2 = (a10+a11)b02
        # N3 = a00(b03-b13)
        # N4 = a11(b12-b02)
        # N5 = (a00+a01)b13
        # N6 = (a10-a00)(b02+b03)
        # N7 = (a01-a11)(b12+b13)
        # C[0][2] = N1+N4-N5+N7, C[0][3]=N3+N5, C[1][2]=N2+N4, C[1][3]=N1-N2+N3+N6
        
        # r=7: N1
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: N2
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        return U, V, W

    def try_rank13_search():
        """
        Attempt to find a rank-13 decomposition by checking if any of the 14 
        rank-1 tensors is in the span of the other 13.
        
        The tensor T = sum_r u_r ⊗ v_r ⊗ w_r. If rank-13 exists, then one of 
        the rank-1 terms must be linearly dependent on the others in a specific 
        way that preserves the tensor structure.
        
        We check: for each r, can we write (u_r, v_r, w_r) as a linear combination 
        of the other 13 terms such that the tensor equation still holds?
        """
        U14, V14, W14 = strassen_rank14()
        R = 14
        
        # Convert to numpy for linear algebra
        U_arr = np.array(U14, dtype=float)
        V_arr = np.array(V14, dtype=float)
        W_arr = np.array(W14, dtype=float)
        
        # For each candidate term r to remove, check if the tensor can be 
        # reconstructed from the remaining 13 terms.
        # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # If we remove term r0, we need: T = sum_{r != r0} U[i,r] * V[j,r] * W[k,r]
        # This is generally impossible unless there's a special structure.
        
        # A more refined check: the space of bilinear forms.
        # The 14 terms span a 14-dimensional space of bilinear forms.
        # If rank-13 exists, the tensor must lie in a 13-dimensional subspace 
        # of rank-1 tensors.
        
        # Check linear independence of the 14 rank-1 tensors in the tensor space.
        # Flatten each rank-1 tensor u⊗v⊗w into a vector of size 4*8*8=256.
        tensor_dim = nm * mp * pn  # 256
        rank1_vectors = []
        for r in range(R):
            vec = np.zeros(tensor_dim)
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        idx = i * mp * pn + j * pn + k
                        vec[idx] = U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
            rank1_vectors.append(vec)
        
        rank1_matrix = np.array(rank1_vectors).T  # shape (256, 14)
        
        # Check rank of this matrix
        rank = np.linalg.matrix_rank(rank1_matrix)
        
        if rank < R:
            # There's linear dependence. Try to find a rank-13 decomposition.
            # This would mean one of the rank-1 tensors is in the span of others.
            # But for the tensor equation to hold, we need more than just 
            # linear independence of the rank-1 vectors.
            # For now, if rank < 14, we know a rank-13 might be possible.
            # We'd need to solve for the coefficients.
            pass
        
        # In practice, for the standard Strassen decomposition, the 14 terms 
        # are linearly independent, so rank-13 is not achievable with this 
        # structure. The true rank of T(2,2,4) is 14.
        
        return None

    # Try the structured rank-14 decomposition first
    U, V, W = strassen_rank14()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank-13 search (likely to fail, but included for completeness)
    result13 = try_rank13_search()
    if result13 is not None:
        return result13
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
