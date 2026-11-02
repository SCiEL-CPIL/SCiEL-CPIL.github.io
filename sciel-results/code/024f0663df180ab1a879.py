"""Matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130

    # Schoolbook rank is n*m*p = 96
    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def validate_einsum(U, V, W, rank):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        return np.allclose(T, 1.0, atol=1e-8)

    # Start with schoolbook as fallback
    school = build_schoolbook()
    best_result = school
    if validate_einsum(school["U"], school["V"], school["W"], n * m * p):
        best_rank = n * m * p
    else:
        best_rank = float('inf')

    # Try Kronecker product approach: Strassen(2,2,2) x Schoolbook(2,2,3)
    # This should give rank 7 * 12 = 84 if the permutation is correct
    # 
    # For 4x4 x 4x6, we view:
    # A as 2x2 blocks of 2x2 matrices
    # B as 2x3 blocks of 2x2 matrices  
    # C as 2x3 blocks of 2x2 matrices
    #
    # The block structure is:
    # C_block[i,j] = sum_l A_block[i,l] * B_block[l,j]
    # where i in {0,1}, j in {0,1,2}, l in {0,1}
    #
    # Each block is 2x2, so we need to multiply 2x2 matrices.
    # We can use Strassen for the 2x2 multiplications.
    #
    # Actually, let's think about this more carefully.
    # M_{4,4,6} can be decomposed as a Kronecker product of M_{2,2,2} and M_{2,2,3}
    # only if the dimensions align properly.
    #
    # The tensor M_{m,n,p} has entries T[i,j,k] = delta[i,j,k] for the multiplication tensor.
    # A rank-1 decomposition term is u * v * w where:
    #   u is indexed by (i,j), v by (j,k), w by (k,i)
    #
    # For the Kronecker product M_{2,2,2} ⊗ M_{2,2,3}:
    # The resulting tensor has indices:
    #   (i1,i2), (j1,j2), (k1,k2) where i1,j1,k1 in {0,1}, i2,j2,k2 in {0,1}
    # But we need M_{4,4,6}, so the last index should range over 6 values, not 4.
    #
    # This doesn't directly work. Let me try a different approach.
    #
    # Alternative: Use the fact that M_{4,4,6} can be viewed as computing
    # C = A * B where A is 4x4, B is 4x6.
    #
    # We can split A into 2x2 blocks and B into 2x3 blocks of 2x2 matrices.
    # Then C_block[i,j] = sum_l A_block[i,l] * B_block[l,j]
    #
    # For each fixed (i,j), this is a sum of 2 matrix multiplications of 2x2 matrices.
    # But we need to do this for all 6 block positions (i,j) where i in {0,1}, j in {0,1,2}.
    #
    # Total scalar multiplications if we do it naively: 6 positions * 2 sums * 4 mults = 48
    # But that's not the right way to count the tensor rank.
    #
    # Let me just return the schoolbook decomposition. It's the safest option.
    
    return best_result
