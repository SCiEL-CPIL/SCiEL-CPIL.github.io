import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Schoolbook fallback (rank 100)
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

    def try_winograd_f5():
        """
        Winograd F5 cyclic convolution for each of the 4 rows.
        For each row i (output row), we compute:
          C[t] = sum_j A[i,j] * B[j,t]  for t=0..4
        
        This is a cyclic convolution of vector a = [A[i,0],...,A[i,4]] with
        each column of B, but we can batch: for each row i, the 5 output
        values C[0..4] are cyclic convolutions of a with 5 different vectors
        (columns of B). But actually, it's simpler: for fixed i, the mapping
        (A[i,:], B) -> C[0..4] is linear in both, so we can use Winograd
        F5 which decomposes length-5 cyclic convolution into 5 multiplications
        in the ring Z[x]/(x^5-1) using the CRT factorization.
        
        x^5 - 1 = (x-1)(x^4+x^3+x^2+x+1)
        
        The Winograd F5 scheme:
        - Transform a (length 5) via a 5x5 matrix M_a to 5 intermediate values
        - Transform b (length 5) via a 5x5 matrix M_b to 5 intermediate values  
        - Multiply element-wise (5 scalar multiplications)
        - Transform result via a 5x5 matrix M_c back to 5 outputs
        
        For matrix multiplication row i:
          a = A[i, 0..4]  (5 values from A)
          For each column k of B, b_k = B[0..4, k] (5 values)
          c_k = cyclic_conv(a, b_k) = C[i, k]
        
        But we need to handle all 5 columns simultaneously. The standard Winograd
        F5 for convolution of two length-5 vectors uses 5 multiplications.
        For 5 different b vectors, we get 5*5 = 25 multiplications per row... 
        that's worse than schoolbook (20 per row).
        
        Let me reconsider. The correct approach:
        
        For the 4x5x5 multiplication tensor, we can view it as 4 independent
        1x5x5 tensors (one per output row). Each 1x5x5 tensor computes:
          C[k] = sum_j A[j] * B[j,k]  for k=0..4
          
        This is equivalent to: for each k, C[k] = dot(A, B_col_k)
        
        Using Winograd F5 cyclic convolution:
        The tensor T_{j,k} = sum_j A_j B_{j,k} can be decomposed using
        the F5 Winograd scheme which requires only 5 multiplications instead of 5.
        
        Wait, that's not right either. Let me think more carefully.
        
        Actually, the Winograd F5 method for computing y = M*x where M is a
        5x5 matrix with a specific circulant structure uses 5 multiplications.
        But our B matrix is arbitrary, not circulant.
        
        Let me use a different approach: the tensor rank of 1x5x5 matrix
        multiplication (which is just the dot product repeated 5 times with
        different B columns) is actually 5 (it's just 5 independent dot products,
        each of rank 1... no, each dot product A·b_k has rank 5 in the tensor
        sense since it's sum_j A_j * b_{j,k}).
        
        Hmm, let me reconsider the whole strategy. The tensor for 1x5x5
        "multiplication" is T_{j,k} = sum_j A_j B_{j,k}. This is actually
        just the outer product structure: T = A ⊗ B^T flattened. The rank of
        this tensor as a 5x5x1 tensor is 5 (since it's a rank-5 matrix).
        
        So 4 rows * 5 = 20 is already optimal for the row-by-row decomposition.
        The Winograd approach won't help here because the B matrix is not
        circulant.
        
        Let me try a different structural approach: use the fact that 4x5x5
        can be decomposed using a 4x4x4 core (Strassen, rank 49 or 28) plus
        cross terms, but optimize the cross terms using rank-revealing
        decompositions.
        """
        
        # Let me try a completely different approach: use the known optimal
        # or near-optimal decompositions from the literature for small matrices.
        # 
        # For 4x5x5, one approach is to use the "laser method" or to find
        # a decomposition via the tensor rank bounds.
        #
        # Actually, let me try a simpler structural improvement:
        # Decompose 4x5x5 as:
        #   - 4x4x4 core: use the best known 4x4x4 decomposition
        #   - 4x1x4 strip: rank 4 (since it's A[i,j]*B[4,k] for i<4,j<4,k<4... 
        #     wait, that's 4*4*4=64 terms schoolbook, but actually for fixed j=4:
        #     C[i,k] += A[i,4]*B[4,k], which is outer product, rank 4*4=16? No,
        #     it's sum over nothing, just A[i,4]*B[4,k] for each (i,k), so it's
        #     a rank-1 matrix in the (i,k) plane, meaning 1 term per (i,k) pair = 16 terms
        #   - 1x4x4 strip: similarly 16 terms
        #   - 1x1x4 corner: 4 terms
        #
        # Total with schoolbook cross: 49 + 16 + 16 + 4 = 85 (same as current)
        #
        # Can we do better on the cross terms? The 4x1x4 strip computes:
        #   C[i,k] += A[i,4] * B[4,k]  for i=0..3, k=0..3
        # This is the outer product of vector a=[A[i,4]] and b=[B[4,k]],
        # which has tensor rank 4 (not 16!) since it's a rank-1 matrix.
        # Wait, in the tensor decomposition sense:
        #   T_{i,4,k} = A[i,4] * B[4,k]
        # This is a single term: U_{i,4} * V_{4,k} * W_{k,i} where
        # U_{i,4}=A[i,4], V_{4,k}=B[4,k], W_{k,i}=1
        # So it's rank 1! Not 16!
        
        # Let me redo this properly.
        # The 4x1x4 block: for i in 0..3, j=4, k in 0..3:
        #   T_{i,4,k} = A[i,4] * B[4,k]
        # This is a single rank-1 tensor term:
        #   U[i*5+4] = A[i,4] (but A[i,4] is a variable, not a constant)
        # 
        # Wait, I'm confusing the tensor entries with the variables.
        # In the tensor decomposition, we need:
        #   T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T_{i,j,k} = delta_{i,j,k} in the sense that the multiplication
        # tensor has T_{i,j,k} = 1 if the output (i,k) receives contribution
        # from A[i,j]*B[j,k], else 0.
        
        # For the 4x1x4 block (j=4, i<4, k<4):
        #   T_{i,4,k} = 1 (contribution to C[i,k] from A[i,4]*B[4,k])
        # We need: sum_r U[i,4,r] * V[4,k,r] * W[k,i,r] = 1 for all i<4, k<4
        # 
        # This is the all-ones 4x4 matrix in the (i,k) plane, which has rank 4.
        # So we need 4 terms: for r=0..3, set U[i,4,r]=delta_{i,r}, V[4,k,r]=delta_{k,r}, W[k,i,r]=1
        # Check: sum_r delta_{i,r} * delta_{k,r} * 1 = delta_{i,k}. That's not 1!
        #
        # I need sum_r U[i,4,r] * V[4,k,r] * W[k,i,r] = 1 for ALL i,k in 0..3.
        # This means the matrix M_{i,k} = sum_r U[i,4,r] * V[4,k,r] * W[k,i,r] must be all-ones.
        # 
        # If I set W[k,i,r] = 1 for all k,i,r, then I need sum_r U[i,4,r]*V[4,k,r] = 1.
        # This is the outer product of U_col and V_row, which can only give rank-1 matrices.
        # So I need at least 4 terms to get a rank-4 (all-ones) matrix.
        #
        # With 4 terms: U[i,4,r] = delta_{i,r}, V[4,k,r] = delta_{k,r}, W[k,i,r] = 1
        # Then sum_r delta_{i,r}*delta_{k,r}*1 = delta_{i,k}. Still not all-ones!
        #
        # The issue is that W also depends on (k,i). Let me think again.
        # I need: sum_r U[i,4,r] * V[4,k,r] * W[k,i,r] = 1 for all i,k.
        # 
        # Try: 4 terms with U[i,4,r]=1, V[4,k,r]=1, W[k,i,r]=delta_{i+r mod 4, k}
        # No, this is getting complicated. Let me just use 16 terms (schoolbook) for the cross.
        
        # Actually, the minimum rank of the all-ones 4x4x4 tensor (where the third
        # index is tied to the first two via W[k,i,r]) is more subtle.
        # Let me just verify that 16 terms work and move on.
        
        # New strategy: use Strassen 4x4x4 (rank 28, not 49!) for the core.
        # The current code uses 49 (7*7 tensor product of 2x2x2 Strassen),
        # but the actual Strassen 4x4x4 rank is 28 (since Strassen 2x2x2 has rank 7,
        # and 4x4x4 can be done with 7*7=49, but there are better 4x4x4 decompositions).
        #
        # Actually, the best known 4x4x4 tensor rank is 49 (from Strassen 2x2x2 tensor
        # product), but there might be better. Let me check: the tensor rank of
        # 4x4x4 matrix multiplication is known to be at most 49 (Strassen) and
        # at least 49? No, it's at most 49 and the lower bound is not tight.
        # 
        # Actually, for 4x4x4, the best known upper bound is 49 (Strassen) and
        # lower bound is 49? Let me just use 49 and focus on optimizing the cross terms.
        
        # Let me try a different structural approach: use the "laser method"
        # or just try to find a better decomposition via random search /
        # rank-revealing factorization.
        
        # For now, let me try: 4x4x4 Strassen (49) + optimized cross terms.
        # The cross terms:
        #   Block (i<4, j=4, k<4): T_{i,4,k}=1 for i,k<4. Need rank of this "slice".
        #   Block (i<4, j=4, k=4): T_{i,4,4}=1 for i<4. 
        #   Block (i=4, j<4, k<4): T_{4,j,k}=1 for j,k<4.
        #   Block (i=4, j=4, k<4): T_{4,4,k}=1 for k<4.
        #   Block (i=4, j<4, k=4): T_{4,j,4}=1 for j<4.
        #   Block (i=4, j=4, k=4): T_{4,4,4}=1.
        
        # Wait, I think I mislabeled. Let me re-read the problem.
        # A is 4x5 (n=4 rows, m=5 cols), B is 5x5 (m=5 rows, p=5 cols).
        # C = A*B is 4x5 (n=4 rows, p=5 cols).
        # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
        #
        # The tensor T has indices (i, j, k) with i in 0..3, j in 0..4, k in 0..4.
        # T_{i,j,k} = 1 (each term A[i,j]*B[j,k] contributes to C[i,k]).
        #
        # Decomposition:
        #   Core: i<4, j<4, k<4 -> 4x4x4, rank 49 (Strassen)
        #   Strip1: i<4, j=4, k<4 -> 4x1x4, T_{i,4,k}=1
        #   Strip2: i<4, j<4, k=4 -> 4x4x1, T_{i,j,4}=1  
        #   Strip3: i<4, j=4, k=4 -> 4x1x1, T_{i,4,4}=1
        #   Strip4: i=4... wait, i only goes 0..3 (n=4). So no i=4.
        #
        # Ah, I see. n=4, m=5, p=5. So i in 0..3, j in 0..4, k in 0..4.
        # There's no i=4! The "1x4x4" and "1x1x4" blocks in the current code
        # are wrong - they should be for j=4, not i=4.
        
        # Correct decomposition:
        #   Core: i in 0..3, j in 0..3, k in 0..3 -> 4x4x4
        #   Strip_j: i in 0..3, j=4, k in 0..3 -> 4x1x4
        #   Strip_k: i in 0..3, j in 0..3, k=4 -> 4x4x1
        #   Corner: i in 0..3, j=4, k=4 -> 4x1x1
        
        # For Strip_j (4x1x4): T_{i,4,k}=1 for i,k in 0..3
        #   Need: sum_r U[i,4,r] * V[4,k,r] * W[k,i,r] = 1 for all i,k
        #   This is equivalent to: the 4x4 matrix M_{i,k} = sum_r U[i,4,r]*V[4,k,r]*W[k,i,r] = 1
        #   If we set W[k,i,r] = 1 for all, then we need sum_r U[i,4,r]*V[4,k,r] = 1
        #   which is rank 1, impossible for all-ones.
        #   But W can depend on (k,i). 
        #   Actually, this is the "matrix multiplication" tensor of 4x1 times 1x4,
        #   which is just the outer product, rank 4.
        #   Wait, 4x1x1x4... no. The tensor is 4x1x4 (three indices: i, j=4 fixed, k).
        #   It's a 4x4 matrix (i,k) with all entries 1, which has rank 4.
        #   So we need 4 terms.
        #
        #   4 terms: for r=0..3:
        #     U[i,4,r] = delta_{i,r}
        #     V[4,k,r] = delta_{k,r}  
        #     W[k,i,r] = 1
        #   Check: sum_r delta_{i,r} * delta_{k,r} * 1 = delta_{i,k}. NOT 1!
        #
        #   I need a different W. Let me try:
        #     U[i,4,r] = 1 for all i,r
        #     V[4,k,r] = 1 for all k,r
        #     W[k,i,r] = delta_{i,k} ... no, that gives delta_{i,k} again.
        #
        #   Hmm. Let me think of it as: I need 4 vectors u_r (length 4), v_r (length 4),
        #   and w_r (length 16, indexed by (k,i)) such that
        #   sum_r u_r[i] * v_r[k] * w_r[k,i] = 1 for all i,k.
        #
        #   If I set w_r[k,i] = 1 for all, then I need sum_r u_r[i]*v_r[k] = 1,
        #   which is a rank-1 matrix, impossible.
        #
        #   If I set u_r[i] = delta_{i,r}, v_r[k] = 1, w_r[k,i] = delta_{k,r}:
        #   sum_r delta_{i,r} * 1 * delta_{k,r} = delta_{i,k}. Still not 1.
        #
        #   If I set u_r[i] = 1, v_r[k] = delta_{k,r}, w_r[k,i] = 1:
        #   sum_r 1 * delta_{k,r} * 1 = 1. YES! This works!
        #   So: U[i,4,r] = 1, V[4,k,r] = delta_{k,r}, W[k,i,r] = 1 for r=0..3.
        #   Check: for fixed (i,k), sum_r U[i,4,r]*V[4,k,r]*W[k,i,r] = sum_r 1*delta_{k,r}*1 = 1. Correct!
        
        # Similarly for Strip_k (4x4x1): T_{i,j,4}=1 for i,j in 0..3
        #   Need: sum_r U[i,j,r] * V[j,4,r] * W[4,i,r] = 1 for all i,j
        #   Set: U[i,j,r] = 1, V[j,4,r] = delta_{j,r}, W[4,i,r] = 1 for r=0..3.
        #   Check: sum_r 1*delta_{j,r}*1 = 1. Correct!
        
        # For Corner (4x1x1): T_{i,4,4}=1 for i in 0..3
        #   Need: sum_r U[i,4,r] * V[4,4,r] * W[4,i,r] = 1 for all i
        #   Set: U[i,4,r] = delta_{i,r}, V[4,4,r] = 1, W[4,i,r] = 1 for r=0..3.
        #   Check: sum_r delta_{i,r}*1*1 = 1. Correct!
        
        # Total rank: 49 + 4 + 4 + 4 = 61. Better than 85!
        
        # Now let me implement this.
        
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
        
        rank = 49 + 4 + 4 + 4  # 61
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Add Strassen 4x4x4 terms (handles i<4, j<4, k<4)
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
        
        # Add Strip_j: i<4, j=4, k<4 (4 terms)
        r_offset = 49
        for r in range(4):
            for i in range(4):
                U[i*5+4][r_offset + r] = 1
            for k in range(4):
                V[4*5+k][r_offset + r] = (1 if k == r else 0)
            for k in range(4):
                for i in range(4):
                    W[k*4+i][r_offset + r] = 1
        
        # Add Strip_k: i<4, j<4, k=4 (4 terms)
        r_offset = 53
        for r in range(4):
            for i in range(4):
                for j in range(4):
                    U[i*5+j][r_offset + r] = 1
            for j in range(4):
                V[j*5+4][r_offset + r] = (1 if j == r else 0)
            for i in range(4):
                W[4*4+i][r_offset + r] = 1
        
        # Add Corner: i<4, j=4, k=4 (4 terms)
        r_offset = 57
        for r in range(4):
            for i in range(4):
                U[i*5+4][r_offset + r] = (1 if i == r else 0)
            V[4*5+4][r_offset + r] = 1
            for i in range(4):
                W[4*4+i][r_offset + r] = 1
        
        if verify(U, V, W):
            return {'U': U, 'V': V, 'W': W}
        
        return None

    result = try_winograd_f5()
    if result is not None:
        return result
    
    return {'U': U_school, 'V': V_school, 'W': W_school}
