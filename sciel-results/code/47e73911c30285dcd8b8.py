import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx][r] = 1
                V_school[v_idx][r] = 1
                W_school[w_idx][r] = 1
                r += 1
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(rank):
                        s += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                    if s != 1:
                        return False
        return True

    def try_winograd():
        """
        Winograd convolution approach for 4x5x5.
        
        We view matrix multiplication as computing C[i,k] = sum_j A[i,j]*B[j,k].
        
        Using cyclic convolution over Z[x]/(x^5 - 1), we can factor x^5 - 1
        over the rationals: x^5 - 1 = (x-1)(x^4 + x^3 + x^2 + x + 1)
        
        The rank-5 approach: evaluate at the 5th roots of unity.
        For the cyclic convolution of length 5, we need 5 evaluation points.
        
        Actually, let's think more carefully. The matrix multiplication tensor
        T[i,j,k] = delta_{i,k} * delta_{j,j} ... no.
        
        T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
        
        The tensor is T[i,j,k] where the multiplication is:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this sense? No!)
        
        Actually the matrix multiplication tensor M[i,j,k] = 1 if and only if
        the (i,k) entry of C depends on A[i,j] and B[j,k]. So M[i,j,k] = 1 for all i,j,k.
        
        Wait, that's not right either. The tensor is:
        M[i,j,k] = 1 if (i,j) is an index of A and (j,k) is an index of B
        such that their product contributes to C[i,k].
        
        For standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        So M[i,j,k] = 1 for all valid i, j, k.
        
        This is just the all-ones tensor of shape (4,5,5).
        
        The rank of the all-ones tensor (n,m,p) is max(n,m,p) if we can use
        rank-1 tensors that are products of indicator vectors.
        
        Actually no. The all-ones tensor [1]_{i,j,k} has rank 1: it's u_i * v_j * w_k
        where u_i = 1, v_j = 1, w_k = 1 for all i,j,k.
        
        Wait, that's wrong. Let me reconsider.
        
        The matrix multiplication tensor is NOT the all-ones tensor.
        
        For C = AB where A is n×m and B is m×p:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        The tensor T has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
        multiply to contribute to C[i,k].
        
        So T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
        
        Yes! The matrix multiplication tensor IS the all-ones tensor of shape (n,m,p).
        
        And the rank of the all-ones tensor of shape (n,m,p) is 1!
        
        Because: T[i,j,k] = 1 = (1)(1)(1) = u_i * v_j * w_k
        where u_i = 1, v_j = 1, w_k = 1.
        
        So rank 1?? That can't be right because then any matrix multiplication
        could be done with 1 multiplication.
        
        I'm confusing myself. Let me think again.
        
        The matrix multiplication tensor T is defined such that:
        C[i,k] = sum_r U[i,r] V[r,j] ... no.
        
        The standard formulation: We want to decompose the tensor T such that
        T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  ... this isn't standard.
        
        Let me use the convention from the task:
        The output factor follows W[k*n+i,r]
        
        So the decomposition is:
        For each rank term r:
          U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for all i, j, k.
        
        So yes, the tensor is all-ones of shape (n*m, m*p, p*n) with the constraint
        that the indices are linked: (i,j), (j,k), (k,i).
        
        The tensor T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k.
        
        This is NOT the full all-ones tensor of shape (20, 25, 20) - it's a
        structured tensor where only certain index combinations are nonzero.
        
        Specifically, T[a, b, c] = 1 if there exist i,j,k such that:
        a = i*m + j, b = j*p + k, c = k*n + i
        
        And T[a,b,c] = 0 otherwise.
        
        So the tensor has exactly n*m*p = 100 nonzero entries (all equal to 1).
        
        Now, what's the rank of this tensor?
        
        For the Winograd approach, we can use the fact that this is related to
        cyclic convolution.
        
        Let's try a different approach: use the Strassen-based block decomposition
        but optimize the boundary blocks.
        
        Actually, let me try a rank-20 approach based on the following idea:
        
        For the 4x5x5 case, we can write:
        A (4x5) = [A1 | A2] where A1 is 4x4 and A2 is 4x1
        B (5x5) = [B1; B2] where B1 is 4x5 and B2 is 1x5
        
        C = A1*B1 + A2*B2
        
        If we can compute A1*B1 with rank 49 (Strassen 4x4x4) and A2*B2 with rank 4,
        that gives 53. But we need to be more clever.
        
        Alternative: use the fact that 5 = 4+1 and try to share computations.
        
        Let me just implement a careful Strassen-based approach and verify it works.
        """
        
        # Strassen 2x2x2 terms
        strassen_2x2x2 = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        def tensor_product_vec(v1, v2):
            result = [0] * 16
            for a1 in range(2):
                for b1 in range(2):
                    for a2 in range(2):
                        for b2 in range(2):
                            idx = (a1*2+a2)*4 + (b1*2+b2)
                            result[idx] = v1[a1*2+b1] * v2[a2*2+b2]
            return result
        
        strassen_4x4x4 = []
        for t1 in strassen_2x2x2:
            for t2 in strassen_2x2x2:
                u = tensor_product_vec(t1[0], t2[0])
                v = tensor_product_vec(t1[1], t2[1])
                w = tensor_product_vec(t1[2], t2[2])
                strassen_4x4x4.append((u, v, w))
        
        # Total rank: 49 (for 4x4x4) + 16 (4x1x4) + 16 (1x4x4) + 4 (1x1x4) = 85
        rank = 49 + 16 + 16 + 4
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Add Strassen 4x4x4 terms (handles j<4, k<4)
        for r_idx, (u, v, w) in enumerate(strassen_4x4x4):
            for i in range(4):
                for j in range(4):
                    U[i*5+j][r_idx] = u[i*4+j]
            for j in range(4):
                for k in range(4):
                    V[j*5+k][r_idx] = v[j*4+k]
            for k in range(4):
                for i in range(4):
                    W[k*4+i][r_idx] = w[k*4+i]
        
        # Add 4x1x4 block (j<4, k==4): schoolbook
        r_offset = 49
        for i in range(4):
            for j in range(4):
                U[i*5+j][r_offset] = 1
                V[j*5+4][r_offset] = 1
                W[4*4+i][r_offset] = 1
                r_offset += 1
        
        # Add 1x4x4 block (j==4, k<4): schoolbook
        for i in range(4):
            for k in range(4):
                U[i*5+4][r_offset] = 1
                V[4*5+k][r_offset] = 1
                W[k*4+i][r_offset] = 1
                r_offset += 1
        
        # Add 1x1x4 block (j==4, k==4): schoolbook
        for i in range(4):
            U[i*5+4][r_offset] = 1
            V[4*5+4][r_offset] = 1
            W[4*4+i][r_offset] = 1
            r_offset += 1
        
        if verify(U, V, W):
            return {'U': U, 'V': V, 'W': W}
        
        return None

    def try_optimized_block():
        """
        Try to reduce the boundary block ranks by sharing terms.
        
        The 4x1x4 block has 16 terms, 1x4x4 has 16 terms, 1x1x4 has 4 terms.
        Can we do better?
        
        For 4x1x4: C[i,k] = A[i,4] * B[4,k] for i in 0..3, k in 0..3
        This is an outer product of a 4-vector and a 4-vector, rank 4!
        
        For 1x4x4: C[i,k] = A[i,4] * B[4,k] - same thing!
        
        Wait, 4x1x4 means: i in 0..3, j=4, k in 0..3
        C[i,k] += A[i,4] * B[4,k]
        
        1x4x4 means: i in 0..3, j=4, k in 0..3
        C[i,k] += A[i,4] * B[4,k]
        
        These are the SAME terms! So we only need 16 terms total for both,
        not 16+16=32.
        
        Let me re-examine. The blocks are:
        - j<4, k<4: 4x4x4, handled by Strassen (49 terms)
        - j<4, k==4: i in 0..3, j in 0..3, k=4: C[i,4] += A[i,j]*B[j,4] (16 terms)
        - j==4, k<4: i in 0..3, j=4, k in 0..3: C[i,k] += A[i,4]*B[4,k] (16 terms)
        - j==4, k==4: i in 0..3, j=4, k=4: C[i,4] += A[i,4]*B[4,4] (4 terms)
        
        So the blocks are:
        Block 1: i in 0..3, j in 0..3, k in 0..3 (4x4x4)
        Block 2: i in 0..3, j in 0..3, k=4 (4x4x1)
        Block 3: i in 0..3, j=4, k in 0..3 (4x1x4)
        Block 4: i in 0..3, j=4, k=4 (4x1x1)
        
        Block 2: C[i,4] += sum_{j=0}^3 A[i,j]*B[j,4]
        This is a 4x4x1 tensor. Rank can be 4 (each i gets one term).
        
        Block 3: C[i,k] += A[i,4]*B[4,k]
        This is a 4x1x4 tensor. Rank can be 4.
        
        Block 4: C[i,4] += A[i,4]*B[4,4]
        This is a 4x1x1 tensor. Rank can be 1.
        
        Total: 49 + 4 + 4 + 1 = 58 terms!
        """
        
        # Strassen 2x2x2 terms
        strassen_2x2x2 = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
            ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
        
        def tensor_product_vec(v1, v2):
            result = [0] * 16
            for a1 in range(2):
                for b1 in range(2):
                    for a2 in range(2):
                        for b2 in range(2):
                            idx = (a1*2+a2)*4 + (b1*2+b2)
                            result[idx] = v1[a1*2+b1] * v2[a2*2+b2]
            return result
        
        strassen_4x4x4 = []
        for t1 in strassen_2x2x2:
            for t2 in strassen_2x2x2:
                u = tensor_product_vec(t1[0], t2[0])
                v = tensor_product_vec(t1[1], t2[1])
                w = tensor_product_vec(t1[2], t2[2])
                strassen_4x4x4.append((u, v, w))
        
        rank = 49 + 4 + 4 + 1
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Add Strassen 4x4x4 terms (handles j<4, k<4)
        for r_idx, (u, v, w) in enumerate(strassen_4x4x4):
            for i in range(4):
                for j in range(4):
                    U[i*5+j][r_idx] = u[i*4+j]
            for j in range(4):
                for k in range(4):
                    V[j*5+k][r_idx] = v[j*4+k]
            for k in range(4):
                for i in range(4):
                    W[k*4+i][r_idx] = w[k*4+i]
        
        # Block 2: 4x4x1 (i in 0..3, j in 0..3, k=4)
        # C[i,4] += A[i,j] * B[j,4]
        # Rank 4: one term per i
        r_offset = 49
        for i in range(4):
            for j in range(4):
                U[i*5+j][r_offset] = 1
            V[4*5+4][r_offset] = 1  # B[4,4]... wait
            W[4*4+i][r_offset] = 1
            r_offset += 1
        
        # Hmm, that's not right. Let me think again.
        # For block 2: sum_j A[i,j]*B[j,4]
        # We need: sum_r U[i*5+j, r] * V[j*5+4, r] * W[4*4+i, r] = 1 for all i,j
        # This is a 4x4x1 tensor with all ones.
        # Rank 4: use u_i = 1, v_j = 1, w = 1? No, that gives rank 1.
        # Actually the tensor T[i,j] = 1 for all i,j has rank 4 (or 1?).
        # T[i,j] = 1 = u_i * v_j where u_i=1, v_j=1. Rank 1!
        
        # Wait, but we need the product U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # to sum to 1 for each (i,j,k) in the block.
        
        # For block 2: (i,j,4) where i in 0..3, j in 0..3
        # We need sum_r U[i*5+j, r] * V[j*5+4, r] * W[4*4+i, r] = 1
        
        # If we use rank 1: U[i*5+j, r0] = 1, V[j*5+4, r0] = 1, W[4*4+i, r0] = 1
        # Then the sum is 1 for all i,j. But this also affects other blocks!
        # Because U[i*5+j, r0] = 1 for all i,j in 0..3, which is in the Strassen block.
        
        # So we can't just use rank 1 because it interferes.
        
        # The issue is that the factor matrices are shared across blocks.
        # We need to be careful.
        
        # Let me just use schoolbook for the boundary blocks to be safe.
        # Total: 49 + 16 + 16 + 4 = 85
        
        return None

    result = try_winograd()
    if result is not None:
        return result
    
    return {'U': U_school, 'V': V_school, 'W': W_school}
