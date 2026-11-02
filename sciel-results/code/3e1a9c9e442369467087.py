import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    def strassen_2x2x2x2(U, V, W, base_col, a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        """Apply Strassen 2x2 x 2x2 at base_col. Returns 7 columns used."""
        col = base_col
        U[a11, col] += 1; U[a22, col] += 1
        V[b11, col] += 1; V[b22, col] += 1
        W[c11, col] += 1; W[c22, col] += 1

        col = base_col + 1
        U[a21, col] += 1; U[a22, col] += 1
        V[b11, col] += 1
        W[c21, col] += 1; W[c22, col] -= 1

        col = base_col + 2
        U[a11, col] += 1
        V[b12, col] += 1; V[b22, col] -= 1
        W[c12, col] += 1; W[c22, col] += 1

        col = base_col + 3
        U[a22, col] += 1
        V[b21, col] += 1; V[b11, col] -= 1
        W[c11, col] += 1; W[c21, col] += 1

        col = base_col + 4
        U[a11, col] += 1; U[a12, col] += 1
        V[b22, col] += 1
        W[c11, col] -= 1; W[c12, col] += 1

        col = base_col + 5
        U[a21, col] += 1; U[a11, col] -= 1
        V[b11, col] += 1; V[b12, col] += 1
        W[c22, col] += 1

        col = base_col + 6
        U[a12, col] += 1; U[a22, col] -= 1
        V[b21, col] += 1; V[b22, col] += 1
        W[c11, col] += 1

        return base_col + 7

    def strassen_56():
        """8 independent 2x2 x 2x2 Strassen blocks. Rank 56."""
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0
        for v in range(4):
            for half in range(2):
                a11 = 0 * m + 2 * half
                a12 = 0 * m + 2 * half + 1
                a21 = 1 * m + 2 * half
                a22 = 1 * m + 2 * half + 1

                b11 = (2 * half) * p + 2 * v
                b12 = (2 * half) * p + 2 * v + 1
                b21 = (2 * half + 1) * p + 2 * v
                b22 = (2 * half + 1) * p + 2 * v + 1

                c11 = (2 * v) * n + 0
                c12 = (2 * v + 1) * n + 0
                c21 = (2 * v) * n + 1
                c22 = (2 * v + 1) * n + 1

                col = strassen_2x2x2x2(U, V, W, col, a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22)

        return U, V, W, rank

    def try_rank_reduced():
        """Try to find a lower rank by sharing Strassen factors.
        
        Key idea: The 4 output blocks C[:, 2v:2v+2] each depend on all 4 rows of B.
        For output block v, we need:
          C[0, 2v] = A[0,:]*B[:,2v] = sum_j A[0,j]*B[j,2v]
          C[0, 2v+1] = A[0,:]*B[:,2v+1]
          C[1, 2v] = A[1,:]*B[:,2v]
          C[1, 2v+1] = A[1,:]*B[:,2v+1]
        
        Let's try a different partition: group A columns differently.
        
        Alternative: use the fact that A is 2x4. We can write:
        C = A @ B where A has shape (2,4) and B has shape (4,8).
        
        Partition B into 4 blocks of 4x2: B = [B0 B1 B2 B3]
        C = [A@B0, A@B1, A@B2, A@B3]
        
        Each A@Bv is 2x2. The tensor for 2x4 x 4x2 has rank at most 8 (schoolbook).
        Can we do better? The 2x2 x 2x2 tensor has rank 7.
        But 2x4 x 4x2 is not 2x2 x 2x2.
        
        Actually, 2x4 x 4x2: we can partition A = [A1 A2] (each 2x2) and B = [B1; B2] (each 2x2).
        Then C = A1*B1 + A2*B2.
        The rank of the sum tensor is at most rank(tensor1) + rank(tensor2) = 7 + 7 = 14.
        But can we do better by sharing?
        
        Let's try: for each output column pair v, we need 4 entries of C.
        C[0,2v] = sum_{j=0}^{3} A[0,j]*B[j,2v]
        C[0,2v+1] = sum_{j=0}^{3} A[0,j]*B[j,2v+1]
        C[1,2v] = sum_{j=0}^{3} A[1,j]*B[j,2v]
        C[1,2v+1] = sum_{j=0}^{3} A[1,j]*B[j,2v+1]
        
        Using Strassen on A1*B1 + A2*B2 for each v: 7*4 = 28? No, that's 14 per v, 56 total.
        
        Let me try a different approach: use 3x3x3 or other small tensor decompositions.
        Actually, let's just try to verify rank 48 by removing redundant terms.
        """
        # Try rank 48: see if we can eliminate 8 of the 56 terms
        # This is a heuristic attempt - try various subsets
        U56, V56, W56, r56 = strassen_56()
        
        # Try to find linear dependencies among the 56 rank-1 terms
        # Each term t is defined by (U_col, V_col, W_col)
        # Two terms are linearly dependent if one is a scalar multiple of the other
        # For exact rank reduction, we need to find a linear combination that cancels
        
        # Let's try a different structural decomposition
        # Partition: A is 2x4, B is 4x8
        # Write A = [A_left | A_right] where each is 2x2
        # Write B = [B_top; B_bottom] where each is 2x8
        # C = A_left @ B_top + A_right @ B_bottom
        # Each A_* @ B_* is 2x2 x 2x8
        # 2x2 x 2x8 can be done with 2 * (2x2 x 2x2) = 2*7 = 14? No...
        # 2x2 x 2x8: partition B_* into 4 blocks of 2x2
        # C = sum_{v=0}^{3} A_left @ B_top[:, 2v:2v+2] + A_right @ B_bottom[:, 2v:2v+2]
        # Each A @ B_block is 2x2 x 2x2, rank 7
        # Total: 8 * 7 = 56. Same as before.
        
        # Let's try yet another approach: use the fact that n=2 is small
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
        # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
        
        # Define intermediate sums:
        # S0 = A[0,0]*B[0,:] + A[0,1]*B[1,:]  (row vector, length 8)
        # S1 = A[0,2]*B[2,:] + A[0,3]*B[3,:]
        # S2 = A[1,0]*B[0,:] + A[1,1]*B[1,:]
        # S3 = A[1,2]*B[2,:] + A[1,3]*B[3,:]
        # C[0,:] = S0 + S1
        # C[1,:] = S2 + S3
        # Each S* requires 2 scalar multiplications per output element = 2*8 = 16
        # Total: 4*16 = 64. Worse.
        
        # The Strassen approach is optimal within this family.
        # Let's just return the 56-rank solution.
        return U56, V56, W56, r56

    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_56()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
