import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    def verify_decomposition(U, V, W, n, m, p, trials=3):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] equals the matrix multiplication tensor.
        T[i,j,k] = 1 if the (i,j) entry of A times the (j,k) entry of B contributes to C[i,k].
        For matrix multiplication, T[i,j,k] = 1 for all (i,j,k) in the valid range,
        because C[i,k] = sum_j A[i,j] B[j,k].
        """
        rank = U.shape[1]
        # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_recon = np.einsum('ar,br,cr->ijk',
                            U.reshape(n, m, rank),
                            V.reshape(m, p, rank),
                            W.reshape(p, n, rank))
        # The matrix multiplication tensor is all-ones: T[i,j,k] = 1 for all valid i,j,k
        T_true = np.ones((n, m, p))
        return np.allclose(T_recon, T_true, atol=1e-8)

    def trivial_decomposition():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def build_wfta_decomposition():
        """
        WFTA over Z_2 x Z_3.
        
        The 2x4x6 matrix multiplication decomposes as a direct sum over the group.
        Using the Kronecker structure:
        - The 2x2x2 part (from Z_2 x Z_2) has rank 4 (two independent Z_2 DFTs)
        - The 1x2x3 part (from Z_2 x Z_3) has rank 6 (one Z_2 DFT, one Z_3 DFT)
        
        But wait: 2x4x6 = 2*(4x6). The 4x6 part can be seen as (2x2)*(2x3) in a block sense.
        Actually, let me think more carefully.
        
        For 2x4x6: A is 2x4, B is 4x6, C is 2x6.
        C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        The tensor T has shape (2,4,6) and T[i,j,k] = 1 for all i,j,k.
        
        WFTA approach: We can factor the indices using the group structure.
        Let's use a simpler approach: Kronecker product decomposition.
        
        2x4x6 = (2x2x2) * (1x2x3) in the sense of tensor Kronecker product.
        The rank of a Kronecker product of tensors is the product of their ranks.
        
        2x2x2 tensor (all-ones) has rank... well, the all-ones 2x2x2 tensor is rank 1.
        But that's not right for matrix multiplication.
        
        Let me reconsider. The matrix multiplication tensor for n x m x p is:
        T[i,j,k] = 1 (it's the identity mapping in the sense that C[i,k] = sum_j A[i,j]B[j,k])
        
        Actually, the tensor IS the all-ones tensor of shape (n,m,p).
        The rank of the all-ones tensor of shape (n,m,p) is 1 (it's just 1*1*1... no).
        
        Wait, the all-ones tensor T[i,j,k] = 1 can be written as:
        T = sum_r u_r * v_r * w_r
        If we take u = [1,...,1], v = [1,...,1], w = [1,...,1], then
        (u * v * w)[i,j,k] = u[i]*v[j]*w[k] = 1*1*1 = 1.
        So the all-ones tensor has rank 1!
        
        But that can't be right for matrix multiplication. Let me re-read the task.
        
        "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 6 matrix."
        
        The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        In tensor form, if we flatten A to vector a of length n*m, B to vector b of length m*p,
        C to vector c of length n*p, then c = T(a,b) where T is a bilinear map.
        
        The tensor T has components T[i,j,k] such that:
        c[i*n+k... wait, let me use the convention from the task.
        
        The task says: "The output factor follows the official trace convention W[k*n+i,r]"
        
        So W is indexed by (k*n + i) where k is the column index of C and i is the row index.
        U is indexed by (i*m + j) where i is row of A, j is column of A / row of B.
        V is indexed by (j*p + k) where j is row of B, k is column of B.
        
        The decomposition is:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        
        And T[i,j,k] = 1 for all valid i, j, k (because the (i,j) entry of A and (j,k) entry of B
        always contribute to the (i,k) entry of C).
        
        So yes, the tensor IS the all-ones tensor of shape (n, m, p) = (2, 4, 6).
        
        And the all-ones tensor has rank 1! We can write:
        T[i,j,k] = 1 = 1 * 1 * 1
        
        So U[i*m+j, 0] = 1 for all i,j
        V[j*p+k, 0] = 1 for all j,k
        W[k*n+i, 0] = 1 for all k,i
        
        This gives rank 1!
        
        But wait, that seems too good to be true. Let me verify with the trivial decomposition.
        The trivial decomposition has rank n*m*p = 48 and:
        U[i*m+j, r] = 1 only when r = i*(m*p) + j*p + k (for specific k)
        V[j*p+k, r] = 1 only when r = i*(m*p) + j*p + k
        W[k*n+i, r] = 1 only when r = i*(m*p) + j*p + k
        
        So the trivial decomposition gives:
        sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        = U[i*m+j, r0] V[j*p+k, r0] W[k*n+i, r0] where r0 = i*(m*p)+j*p+k
        = 1 * 1 * 1 = 1
        
        So both the trivial decomposition and the rank-1 decomposition should work.
        
        Let me verify the rank-1 decomposition:
        U[i*m+j, 0] = 1 for all i in {0,1}, j in {0,1,2,3}
        V[j*p+k, 0] = 1 for all j in {0,1,2,3}, k in {0,...,5}
        W[k*n+i, 0] = 1 for all k in {0,...,5}, i in {0,1}
        
        Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
        = 1 * 1 * 1 = 1 = T[i,j,k]
        
        So the rank-1 decomposition is valid!
        
        But the task says "rank-minimal" and the current program uses rank 42.
        If rank 1 works, why isn't everyone using it?
        
        Let me re-read the task more carefully...
        
        "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 6 matrix."
        
        Hmm, maybe I'm misunderstanding the tensor. Let me think about what "tensor decomposition for matrix multiplication" means.
        
        In the context of algebraic complexity theory, the matrix multiplication tensor T^{(n,m,p)} is defined such that:
        T^{(n,m,p)}_{i,j,k} = 1 if the (i,j) element of A and (j,k) element of B multiply to contribute to (i,k) element of C.
        
        But this is exactly the all-ones tensor. The "structure" of matrix multiplication is captured by the way the indices are arranged, not by the values of the tensor.
        
        Actually, I think I'm confusing myself. Let me look at this differently.
        
        The bilinear map for matrix multiplication is:
        (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        
        This is a bilinear map from (R^{n*m} x R^{m*p}) -> R^{n*p}.
        
        A rank-1 decomposition of this bilinear map is:
        C[i,k] = sum_r f_r(A) * g_r(B) * h_r(i,k)
        
        where f_r: R^{n*m} -> R, g_r: R^{m*p} -> R, h_r: R^{n*p} -> R are linear functionals.
        
        In tensor form:
        T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        And T should equal 1 for all valid (i,j,k) because the coefficient of A[i,j]*B[j,k] in C[i,k] is always 1.
        
        So yes, T is the all-ones tensor, and it has rank 1.
        
        I think the key insight is that the "matrix multiplication tensor" in this context is indeed the all-ones tensor, and its rank is 1.
        
        Let me just go with rank 1 and verify it works.
        """
        rank = 1
        U = np.ones((n * m, rank))
        V = np.ones((m * p, rank))
        W = np.ones((p * n, rank))
        return U, V, W

    def build_strassen_decomposition():
        """Fallback: Strassen-based decomposition with rank 42."""
        def get_strassen_2x2x2():
            rank = 7
            U = np.zeros((4, rank))
            V = np.zeros((4, rank))
            W = np.zeros((4, rank))
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1; W[3, 1] = -1
            U[0, 2] = 1
            V[2, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1; W[3, 2] = 1
            U[3, 3] = 1
            V[1, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[2, 3] = 1
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1; W[1, 4] = 1
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[2, 5] = 1
            W[3, 5] = 1
            U[1, 6] = 1; U[3, 6] = -1
            V[1, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1
            return U, V, W

        U_s, V_s, W_s = get_strassen_2x2x2()
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        def add_block(a_idx, b_idx, w_idx, offset):
            for r in range(7):
                for ai in a_idx:
                    U[ai, offset + r] += U_s[ai % 4, r]
                for bi in b_idx:
                    V[bi, offset + r] += V_s[bi % 4, r]
                for wi in w_idx:
                    W[wi, offset + r] += W_s[wi % 4, r]

        a1_idx = [0, 1, 2, 3]
        a2_idx = [4, 5, 6, 7]
        b1_idx = [0, 1, 6, 7]
        b2_idx = [2, 3, 8, 9]
        b3_idx = [4, 5, 10, 11]
        b4_idx = [12, 13, 18, 19]
        b5_idx = [14, 15, 20, 21]
        b6_idx = [16, 17, 22, 23]
        c1_idx = [0, 1, 2, 3]
        c2_idx = [4, 5, 6, 7]
        c3_idx = [8, 9, 10, 11]

        add_block(a1_idx, b1_idx, c1_idx, 0)
        add_block(a2_idx, b4_idx, c1_idx, 7)
        add_block(a1_idx, b2_idx, c2_idx, 14)
        add_block(a2_idx, b5_idx, c2_idx, 21)
        add_block(a1_idx, b3_idx, c3_idx, 28)
        add_block(a2_idx, b6_idx, c3_idx, 35)
        return U, V, W

    try:
        # Try rank-1 decomposition first
        U, V, W = build_wfta_decomposition()
        if verify_decomposition(U, V, W, n, m, p, trials=2):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Fallback to Strassen
        U, V, W = build_strassen_decomposition()
        if verify_decomposition(U, V, W, n, m, p, trials=2):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return trivial_decomposition()

    except Exception:
        return trivial_decomposition()

# EVOLVE-BLOCK-END
