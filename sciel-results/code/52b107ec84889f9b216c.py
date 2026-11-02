"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Check diagonal: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i3][c]
                                if abs(s) > tol:
                                    return False
        return True
    
    def schoolbook(n, m, p):
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
        return U, V, W
    
    # Strategy: Treat the 2x5 matrix as two independent 1x5 row vectors.
    # For each row i, we need to compute a 1x5 vector (row i of A) times a 5x5 matrix B.
    # This is a 1x5x5 tensor with rank at most 5 (schoolbook for 1xm x mxp is m*p = 25,
    # but we can do better: the rank of 1xm x mxp is exactly m since we can factor as
    # sum_k e_k^T * B_k where B_k is the k-th column... wait no.
    # Actually for 1x5 x 5x5: output is 1x5, input is 5x5.
    # Tensor T[a,j,k] = delta(a,0) * (B[j,k])... no, T maps (a_vec, B) -> a_vec @ B.
    # T[a, j, k] where a is row index (only 0), j is column of a / row of B, k is column of B.
    # T[0, j, k] = 1 (the j-th element of a times the j-th row... wait.
    # (a @ B)[k] = sum_j a[j] * B[j,k]
    # So T[a_idx, j, k] = delta(a_idx, 0) * 1... no.
    # The tensor is: for output index k, input indices are (a[j], B[j,k]).
    # T[k, j, a_idx]... let me use the convention: T[out, in1, in2].
    # out = k (0..4), in1 = (a_idx, j) but a_idx is always 0, so in1 = j (0..4), in2 = (j2, k2) = (j, k)
    # T[k, j, j2, k2]... this is getting confusing.
    # 
    # Let me use the standard tensor decomposition convention for matrix multiplication:
    # (A @ B)[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 if this is the (i,j,k) component.
    # For 2x5 @ 5x5: i in 0..1, j in 0..4, k in 0..4. T has 2*5*5 = 50 non-zero entries, all 1.
    # Rank of this tensor: we want sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k]
    # 
    # Key insight: For fixed i, the sub-tensor T_i[j,k] = T[i,j,k] is just the 5x5 identity-like
    # tensor where T_i[j,k] = 1 for all j,k (since A[i,:] is a 5-vector and B is 5x5).
    # Wait no: T[i,j,k] = 1 for all i,j,k in the matrix multiplication tensor.
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # and 0 for all off-diagonal terms.
    #
    # For n=2, m=p=5: we can try to find a rank-25 decomposition (already done) or lower.
    # The minimum rank for 1x5 x 5x5 (i.e., vector-matrix mult) is 5 (since it's equivalent to
    # computing 5 dot products of length 5, which has rank 5).
    # So for 2x5 x 5x5, if the two rows are independent, the rank should be at least 10.
    # Can we achieve rank 10? Let's try.
    
    # For rank 10: c = 0..9. Let c = i0 * 5 + k0 where i0 in {0,1}, k0 in 0..4.
    # U[i*m+j, c] = U[i*m+j, i0*5+k0]
    # V[j*p+k, c] = V[j*p+k, i0*5+k0]
    # W[k*n+i, c] = W[k*n+i, i0*5+k0]
    #
    # Structure: U[i*m+j, i0*5+k0] = delta(i, i0) * delta(j, k0)
    #            V[j*p+k, i0*5+k0] = 1 (for all j,k)
    #            W[k*n+i, i0*5+k0] = delta(i, i0) * delta(k, k0)
    #
    # Check diagonal: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # = sum_{i0,k0} delta(i,i0)*delta(j,k0) * 1 * delta(i,i0)*delta(k,k0)
    # = delta(i,i)*delta(j,j)*delta(k,k) = 1 (when i=i, j=j, k=k). Good.
    #
    # Check off-diagonal: sum_c U[i*m+j,c]*V[j2*p+k2,c]*W[k3*n+i3,c]
    # = sum_{i0,k0} delta(i,i0)*delta(j,k0) * 1 * delta(i3,i0)*delta(k3,k0)
    # = delta(i,i3)*delta(j,k0)*delta(k3,k0) summed over k0
    # = delta(i,i3) * [sum_{k0} delta(j,k0)*delta(k3,k0)]
    # = delta(i,i3) * delta(j,k3)
    # This should be 0 when (i,j,k) != (i3,j2,k3). But we have delta(j,k3) which is not
    # delta(j,j2)*delta(k,k3). This doesn't work.
    
    # Let me try a different structure. For the 1x5 x 5x5 case:
    # We want: sum_c U[j,c] * V[j,k,c] * W[k,c] = 1 for all j,k (diagonal)
    # and sum_c U[j,c] * V[j2,k2,c] * W[k2,c] = 0 for (j,k) != (j2,k2).
    # Rank 5: c = 0..4.
    # U[j,c] = delta(j,c), V[j,k,c] = delta(j,c), W[k,c] = delta(k,c).
    # Check: sum_c delta(j,c)*delta(j2,c)*delta(k,c)*delta(k2,c) = delta(j,j2)*delta(k,k2). 
    # Wait: sum_c delta(j,c)*delta(j2,c)*delta(k,c)*delta(k2,c)
    # = delta(j,j2) * [sum_c delta(j,c)*delta(k,c)*delta(k2,c)]... no.
    # = sum_c delta(j,c)*delta(j2,c)*delta(k,c)*delta(k2,c)
    # For this to be non-zero, we need j=j2=k=k2=c. So it's 1 if j=j2=k=k2, else 0.
    # But we want 1 for all (j,k) pairs on the diagonal, i.e., for each (j,k) pair,
    # sum_c U[j,c]*V[j,k,c]*W[k,c] = 1.
    # With U[j,c]=delta(j,c), V[j,k,c]=delta(j,c), W[k,c]=delta(k,c):
    # sum_c delta(j,c)*delta(j,c)*delta(k,c) = sum_c delta(j,c)*delta(k,c) = delta(j,k).
    # This gives 1 only when j=k, not for all j,k. So this doesn't work.
    
    # Correct rank-5 decomposition for 1x5 x 5x5:
    # (a @ B)[k] = sum_j a[j]*B[j,k]
    # Tensor: T[j, j2, k2] where output is k2, and inputs are a[j] and B[j2,k2].
    # T[j, j2, k2] = delta(j, j2). (The j-th element of a multiplies the j-th row of B.)
    # Decomposition: sum_c U[j,c]*V[j2,k2,c]*W[k2,c] = delta(j,j2)
    # Rank 5: c = 0..4.
    # U[j,c] = delta(j,c), V[j2,k2,c] = delta(j2,c), W[k2,c] = 1.
    # Check: sum_c delta(j,c)*delta(j2,c)*1 = delta(j,j2). Correct!
    # Off-diagonal: sum_c U[j,c]*V[j2,k2,c]*W[k2,c] = delta(j,j2) which is 0 when j!=j2. Correct.
    
    # Now for 2x5 x 5x5 with rank 10:
    # c = i0*5 + k0 where i0 in {0,1}, k0 in 0..4.
    # U[i*m+j, c] = U[i*5+j, i0*5+k0] = delta(i, i0) * delta(j, k0)
    # V[j*p+k, c] = V[j*5+k, i0*5+k0] = delta(j, k0)  [independent of i0]
    # W[k*n+i, c] = W[k*2+i, i0*5+k0] = delta(i, i0)
    #
    # Check diagonal: sum_c U[i*5+j,c]*V[j*5+k,c]*W[k*2+i,c]
    # = sum_{i0,k0} delta(i,i0)*delta(j,k0) * delta(j,k0) * delta(i,i0)
    # = sum_{i0,k0} delta(i,i0)^2 * delta(j,k0)^2
    # = delta(i,i) * delta(j,j) = 1. Good.
    #
    # Check off-diagonal: sum_c U[i*5+j,c]*V[j2*5+k2,c]*W[k3*2+i3,c]
    # = sum_{i0,k0} delta(i,i0)*delta(j,k0) * delta(j2,k0) * delta(i3,i0)
    # = delta(i,i3) * sum_{k0} delta(j,k0)*delta(j2,k0)
    # = delta(i,i3) * delta(j,j2)
    # This should be 0 when (i,j,k) != (i3,j2,k3). We have delta(i,i3)*delta(j,j2) which
    # is 1 when i=i3 and j=j2, regardless of k vs k3. So for i=i3, j=j2, k!=k3, this gives 1,
    # but it should be 0. So this doesn't work.
    
    # The issue is that W doesn't depend on k, so we can't distinguish different k values.
    # We need W to depend on k. Let me try:
    # W[k*n+i, c] = W[k*2+i, i0*5+k0] = delta(i, i0) * delta(k, k0)
    # Then: sum_c U[i*5+j,c]*V[j*5+k,c]*W[k*2+i,c]
    # = sum_{i0,k0} delta(i,i0)*delta(j,k0) * delta(j,k0) * delta(i,i0)*delta(k,k0)
    # = delta(i,i) * delta(j,j) * delta(k,k) = 1. Good.
    # Off-diagonal: sum_c U[i*5+j,c]*V[j2*5+k2,c]*W[k3*2+i3,c]
    # = sum_{i0,k0} delta(i,i0)*delta(j,k0) * delta(j2,k0) * delta(i3,i0)*delta(k3,k0)
    # = delta(i,i3) * sum_{k0} delta(j,k0)*delta(j2,k0)*delta(k3,k0)
    # = delta(i,i3) * delta(j,j2) * delta(j,k3)
    # This is 1 when i=i3, j=j2, j=k3. But we want 0 when (i,j,k) != (i3,j2,k3).
    # When i=i3, j=j2, k=k3: we get delta(j2,k3) which is not necessarily 1.
    # So this still doesn't work.
    
    # The fundamental issue: for 1x5 x 5x5, the rank is 5, and the decomposition is:
    # U[j,c] = delta(j,c), V[j2,k2,c] = delta(j2,c), W[k2,c] = 1.
    # The W factor doesn't depend on k2 in a way that distinguishes it.
    # For 2x5 x 5x5, we need to combine two such decompositions.
    # c = i0*5 + c0 where c0 in 0..4.
    # U[i*5+j, i0*5+c0] = delta(i,i0) * delta(j,c0)
    # V[j*5+k, i0*5+c0] = delta(j,c0)
    # W[k*2+i, i0*5+c0] = delta(i,i0)
    # This gives rank 10. Let's verify more carefully.
    # Diagonal (i,j,k): sum_c U[i*5+j,c]*V[j*5+k,c]*W[k*2+i,c]
    # = sum_{i0,c0} delta(i,i0)*delta(j,c0) * delta(j,c0) * delta(i,i0)
    # = delta(i,i) * sum_{c0} delta(j,c0)^2 = 1 * 1 = 1. Good.
    # Off-diagonal (i,j,k) vs (i3,j2,k3) where not all equal:
    # sum_c U[i*5+j,c]*V[j2*5+k2,c]*W[k3*2+i3,c]
    # = sum_{i0,c0} delta(i,i0)*delta(j,c0) * delta(j2,c0) * delta(i3,i0)
    # = delta(i,i3) * sum_{c0} delta(j,c0)*delta(j2,c0)
    # = delta(i,i3) * delta(j,j2)
    # This is 1 when i=i3 and j=j2, regardless of k,k3. So if i=i3, j=j2, k!=k3,
    # we get 1 but should get 0. FAIL.
    
    # The problem is that W doesn't encode k information. In the 1x5 x 5x5 case,
    # W[k2,c] = 1 works because there's only one output dimension. But here we have
    # two rows, and the W factor needs to encode which row AND which output column.
    # 
    # Actually, I think the minimum rank for 2x5 x 5x5 is 25, not 10.
    # The tensor is T[i,j,k] = 1 for all i in {0,1}, j in {0..4}, k in {0..4}.
    # This is a 2x5x5 tensor. Its rank is at least max(2, 5, 5) = 5? No, tensor rank
    # is not bounded by max dimension. The rank of a 2x5x5 all-ones tensor...
    # Actually, the matrix multiplication tensor for 2x5 x 5x5 has a specific structure.
    # Let me just stick with rank 25 which we know works.
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            for i in range(n):
                U[i * m + j0][c] = 1
                W[k0 * n + i][c] = 1
            V[j0 * p + k0][c] = 1
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
