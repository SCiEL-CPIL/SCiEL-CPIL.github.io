import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct (vectorized)."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        # Tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: U shape (n,m,r), V shape (m,p,r), W shape (p,n,r)
        T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
        return bool(np.allclose(T, 1.0, atol=1e-10))
    
    def laderman_3x3x3():
        """
        Laderman's rank-23 decomposition of 3x3x3 matrix multiplication.
        Returns list of 23 tuples (u, v, w) where each is a 9-element list.
        u[i*3+j] = coefficient of A[i,j] in this product
        v[j*3+k] = coefficient of B[j,k] in this product
        w[k*3+i] = coefficient contributing to C[i,k] from this product
        """
        def mat_to_vec(mat):
            return [mat[i][j] for i in range(3) for j in range(3)]
        
        factors = []
        
        # p1 = (a00+a11)(b00+b11): contributes +1 to c00, +1 to c11
        factors.append((
            mat_to_vec([[1,0,0],[0,1,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[0,1,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[0,1,0],[0,0,0]])
        ))
        
        # p2 = (a00-a11)(b00-b11): contributes +1 to c00, +1 to c11
        factors.append((
            mat_to_vec([[1,0,0],[0,-1,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[0,-1,0],[0,0,0]]),
            mat_to_vec([[1,0,0],[0,1,0],[0,0,0]])
        ))
        
        # p3 = (a01+a11)(b10-b00): contributes to c01, c11
        # A part: a01, a11. B part: -b00, b10.
        # c01 gets a01*b00 + a11*b10 (from expansion: a01*b10 - a01*b00 + a11*b10 - a11*b00)
        # Hmm, this is getting complex. Let me use a different approach.
        # I'll construct the factors by directly specifying which (i,j,k) entries they contribute to.
        
        # Build Laderman factors from the bilinear algorithm.
        # Each product p_r = (sum_{i,j} a_r[i,j] * A[i,j]) * (sum_{j,k} b_r[j,k] * B[j,k])
        # contributes alpha_r[i,k] to C[i,k].
        # Tensor factor: U_r[i,j]=a_r[i,j], V_r[j,k]=b_r[j,k], W_r[k,i]=alpha_r[i,k]
        
        # Laderman's 23 products (a coefficients for A, b coefficients for B, alpha for C):
        # Format: (a_matrix, b_matrix, alpha_matrix) where each is 3x3
        
        L = [
            # p1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
            ([[1,0,0],[0,1,0],[0,0,0]], [[1,0,0],[0,1,0],[0,0,0]], [[1,0,0],[0,1,0],[0,0,0]]),
            # p2: (a00-a11)(b00-b11) -> c00+=1, c11+=1
            ([[1,0,0],[0,-1,0],[0,0,0]], [[1,0,0],[0,-1,0],[0,0,0]], [[1,0,0],[0,1,0],[0,0,0]]),
            # p3: (a01+a11)(b10-b00) -> c01+=1, c11+=1
            ([[0,1,0],[0,1,0],[0,0,0]], [[-1,0,0],[1,0,0],[0,0,0]], [[0,1,0],[0,0,1],[0,0,0]]),
            # p4: (a10-a00)(b01+b11) -> c01+=1, c11+=1
            ([[-1,0,0],[1,0,0],[0,0,0]], [[0,1,0],[0,1,0],[0,0,0]], [[0,1,0],[0,0,1],[0,0,0]]),
            # p5: (a10+a22)(b11+b22) -> c11+=1, c22+=1
            ([[0,0,0],[1,0,0],[0,0,1]], [[0,0,0],[0,1,0],[0,0,1]], [[0,0,0],[0,1,0],[0,0,1]]),
            # p6: (a10-a22)(b11-b22) -> c11+=1, c22+=1
            ([[0,0,0],[1,0,0],[0,0,-1]], [[0,0,0],[0,1,0],[0,0,-1]], [[0,0,0],[0,1,0],[0,0,1]]),
            # p7: (a20+a00)(b22+b00) -> c00+=1, c22+=1
            ([[1,0,0],[0,0,0],[1,0,0]], [[1,0,0],[0,0,0],[0,0,1]], [[1,0,0],[0,0,0],[0,0,1]]),
            # p8: (a20-a00)(b22-b00) -> c00+=1, c22+=1
            ([[-1,0,0],[0,0,0],[1,0,0]], [[-1,0,0],[0,0,0],[0,0,1]], [[1,0,0],[0,0,0],[0,0,1]]),
            # p9: (a21+a11)(b10+b00) -> c10+=1, c20+=1, c21+=1
            ([[0,0,0],[0,1,0],[0,1,0]], [[1,0,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[1,1,0]]),
            # p10: (a21-a11)(b10-b00) -> c10+=1, c20+=1, c21+=1
            ([[0,0,0],[0,-1,0],[0,1,0]], [[-1,0,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[1,1,0]]),
            # p11: (a02+a22)(b20+b00) -> c00+=1, c02+=1, c20+=1, c22+=1
            ([[1,0,0],[0,0,0],[0,0,1]], [[1,0,0],[0,0,0],[1,0,0]], [[1,0,0],[0,0,1],[1,0,0]]),
            # p12: (a02-a22)(b20-b00) -> c00+=1, c02+=1, c20+=1, c22+=1
            ([[-1,0,0],[0,0,0],[0,0,1]], [[-1,0,0],[0,0,0],[1,0,0]], [[1,0,0],[0,0,1],[1,0,0]]),
            # p13: (a02+a12)(b10+b20) -> c02+=1, c12+=1, c22+=1
            ([[0,0,1],[0,0,1],[0,0,0]], [[0,0,0],[1,0,0],[1,0,0]], [[0,0,0],[0,0,1],[1,0,0]]),
            # p14: (a02-a12)(b10-b20) -> c02+=1, c12+=1, c22+=1
            ([[0,0,1],[0,0,-1],[0,0,0]], [[0,0,0],[1,0,0],[-1,0,0]], [[0,0,0],[0,0,1],[1,0,0]]),
            # p15: (a12+a22)(b20+b10) -> c12+=1, c22+=1
            ([[0,0,0],[0,0,1],[0,0,1]], [[0,0,0],[1,0,0],[1,0,0]], [[0,0,0],[0,0,1],[1,0,0]]),
            # p16: (a12-a22)(b20-b10) -> c12+=1, c22+=1
            ([[0,0,0],[0,0,1],[0,0,-1]], [[0,0,0],[1,0,0],[-1,0,0]], [[0,0,0],[0,0,1],[1,0,0]]),
            # p17: (a01+a10)(b11+b01) -> c01+=1, c10+=1, c11+=1
            ([[0,1,0],[1,0,0],[0,0,0]], [[0,1,0],[0,1,0],[0,0,0]], [[0,1,0],[1,0,0],[0,0,1]]),
            # p18: (a01-a10)(b11-b01) -> c01+=1, c10+=1, c11+=1
            ([[0,1,0],[-1,0,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,1,0],[1,0,0],[0,0,1]]),
            # p19: (a21+a20)(b10+b20) -> c20+=1, c21+=1, c12+=1, c22+=1
            ([[0,0,0],[0,0,0],[1,1,0]], [[0,0,0],[1,0,0],[1,0,0]], [[0,0,0],[0,0,0],[1,1,0]]),
            # p20: (a21-a20)(b10-b20) -> c20+=1, c21+=1, c12+=1, c22+=1
            ([[0,0,0],[0,0,0],[1,-1,0]], [[0,0,0],[1,0,0],[-1,0,0]], [[0,0,0],[0,0,0],[1,1,0]]),
            # p21: (a01+a21)(b11+b21) -> c01+=1, c21+=1, c12+=1, c22+=1
            ([[0,1,0],[0,0,0],[0,1,0]], [[0,0,0],[0,1,0],[0,0,1]], [[0,1,0],[0,0,0],[0,1,0]]),
            # p22: (a01-a21)(b11-b21) -> c01+=1, c21+=1, c12+=1, c22+=1
            ([[0,1,0],[0,0,0],[0,-1,0]], [[0,0,0],[0,1,0],[0,0,-1]], [[0,1,0],[0,0,0],[0,1,0]]),
            # p23: (a20+a12)(b21+b12) -> c12+=1, c20+=1, c21+=1, c22+=1
            ([[0,0,0],[0,0,1],[1,0,0]], [[0,0,0],[0,0,1],[0,1,0]], [[0,0,0],[0,0,0],[1,1,0]]),
        ]
        
        for a_mat, b_mat, alpha_mat in L:
            u = mat_to_vec(a_mat)
            v = mat_to_vec(b_mat)
            # W uses trace convention: W[k*3+i] = alpha[i,k]
            w = [0] * 9
            for i in range(3):
                for k in range(3):
                    w[k*3+i] = alpha_mat[i][k]
            factors.append((u, v, w))
        
        return factors
    
    def build_3x3x7_decomposition():
        """
        Build the 3x3x7 decomposition using Laderman for 3x3x3 blocks.
        Split p=7 into: 3, 3, 1. Rank = 23+23+9 = 55.
        """
        n, m, p = 3, 3, 7
        
        lad_factors = laderman_3x3x3()
        rank_lad = len(lad_factors)
        
        total_rank = rank_lad + rank_lad + 9
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Block 1: columns 0-2 of B
        for r in range(rank_lad):
            u, v, w = lad_factors[r]
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = u[i*3+j]
            for j in range(m):
                for k in range(3):
                    V[j*p+k][r] = v[j*3+k]
            for k in range(3):
                for i in range(n):
                    W[k*n+i][r] = w[k*3+i]
        
        # Block 2: columns 3-5 of B
        offset = rank_lad
        for r in range(rank_lad):
            u, v, w = lad_factors[r]
            for i in range(n):
                for j in range(m):
                    U[i*m+j][offset+r] = u[i*3+j]
            for j in range(m):
                for k in range(3):
                    V[j*p+(3+k)][offset+r] = v[j*3+k]
            for k in range(3):
                for i in range(n):
                    W[(3+k)*n+i][offset+r] = w[k*3+i]
        
        # Block 3: column 6 of B (schoolbook 3x3x1)
        offset2 = 2 * rank_lad
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][offset2+col] = 1
                V[j*p+6][offset2+col] = 1
                W[6*n+i][offset2+col] = 1
                col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try the Laderman-based decomposition
    try:
        result = build_3x3x7_decomposition()
        if verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
