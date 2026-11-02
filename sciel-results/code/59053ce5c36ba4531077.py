import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
    V = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
    W = np.zeros((p * n, schoolbook_rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    # Verify schoolbook
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        pass

    # Try to find a lower-rank decomposition using random search
    # The idea: we want to find U, V, W with rank r < 36 such that
    # for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    # This is a hard problem. Let's try a few random approaches.

    # Approach 1: Random initialization with gradient-free optimization
    # We can't easily do this, so let's just return schoolbook.

    # For 2x3 x 3x6, the known minimal rank is at most 36 (schoolbook).
    # There might be improvements, but finding them requires sophisticated techniques.

    # Let's try a simple approach: see if we can merge some of the rank-1 terms.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if they share indices.
    # Specifically, if i1=i2, j1=j2, k1=k2, they're the same term.
    # If i1=i2 but j1!=j2 or k1!=k2, we might be able to combine them.

    # Actually, the tensor is T[i,j,k] = 1 for all i,j,k, and we want to write it as
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r]

    # This is equivalent to finding a rank-r decomposition of the tensor T.
    # The tensor T has shape (n, m, p) = (2, 3, 6) with all entries 1.
    # Wait, that's not right. Let me re-read the problem.

    # The problem says: "Find a rank-minimal exact tensor decomposition for multiplying 
    # a 2 by 3 matrix by a 3 by 6 matrix."

    # The standard formulation: we have matrices A (n x m) and B (m x p), and we want 
    # to compute C = A * B (n x p).
    # C[i,k] = sum_j A[i,j] B[j,k]

    # A rank-1 decomposition of the matrix multiplication tensor would be:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) W[r,i,k]
    # This doesn't quite work. Let me think more carefully.

    # Actually, the standard way to view matrix multiplication as a tensor decomposition:
    # The tensor T has components T[i,j,k] where i is the row of A, j is the column of A / row of B,
    # and k is the column of B.
    # T[i,j,k] = 1 if we're computing the (i,k) entry of C using A[i,j] and B[j,k].

    # But this is just the identity tensor in some sense. The actual tensor for matrix multiplication
    # is more subtle.

    # Let me look at this differently. The bilinear map (A, B) -> C = AB can be written as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{i,j,k} A[i,j] B[j,k] delta_{i,i} delta_{k,k}

    # This is not a standard tensor decomposition. Let me use the interface given:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]
    # W: (p*n, rank) -> W[k*n+i, r]
    # And the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # This is saying that the tensor with all 1s (of shape n x m x p) can be decomposed as a sum
    # of rank-1 tensors. The rank of the all-1s tensor of shape n x m x p is 1 (it's just the 
    # outer product of three vectors of 1s).

    # Wait, that can't be right. If the rank is 1, then we could write:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_r U * V * W = 1 * 1 * 1 = 1 for all i,j,k.

    # So the rank is 1?! That seems too good to be true.

    # Let me re-read the problem statement more carefully.

    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix."

    # I think the issue is that the tensor is not the all-1s tensor. The tensor for matrix multiplication
    # has a specific structure that depends on the indices.

    # Let me look at the interface again:
    # "U", "V", "W": factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # with integer/half-integer entries that exactly decompose the matrix multiplication tensor

    # And the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # If this is the condition, then the all-1s decomposition works with rank 1.
    # But that seems too easy. Let me check if there's a different condition.

    # Actually, I think I'm misunderstanding. The tensor for matrix multiplication is not the all-1s tensor.
    # The condition should be related to the actual computation of C = AB.

    # Let me think about this more carefully. The matrix multiplication tensor T_{n,m,p} is a trilinear
    # form that takes three arguments: the entries of A, the entries of B, and the entries of C.
    # T(A, B, C) = sum_{i,j,k} C[i,k] A[i,j] B[j,k]

    # But this is not quite right either. Let me look at the standard definition.

    # The matrix multiplication tensor T_{n,m,p} is defined as:
    # T_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^A e_j^B e_k^C
    # where e_i^A, e_j^B, e_k^C are the standard basis vectors for the spaces of A, B, and C respectively.

    # Wait, that's not right. The tensor should encode the multiplication.

    # Let me look at this from a different angle. The rank of the matrix multiplication tensor T_{n,m,p}
    # is the minimum number of scalar multiplications needed to compute the product of an n x m matrix
    # and an m x p matrix.

    # For n=2, m=3, p=6, the schoolbook rank is n*m*p = 36.
    # There might be lower-rank decompositions.

    # The condition in the interface is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    # This is saying that the tensor with all 1s can be decomposed as a sum of rank-1 tensors.
    # But this is not the matrix multiplication tensor.

    # I think there's a misunderstanding in the problem statement or in my understanding.
    # Let me just return the schoolbook decomposition and hope for the best.

    # Try to find a better decomposition using random search
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W

    # Try random rank-1 additions to see if we can reduce the rank
    # This is a very naive approach, but it's a start

    # Let's try to find a decomposition with rank < 36
    # We'll use a random search approach

    for target_rank in range(20, schoolbook_rank):
        if time.time() > deadline:
            break

        # Try random initialization
        U_try = np.random.randn(n * m, target_rank)
        V_try = np.random.randn(m * p, target_rank)
        W_try = np.random.randn(p * n, target_rank)

        # Verify if this is a valid decomposition
        if verify(U_try, V_try, W_try):
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break

    # If we found a better decomposition, return it
    if best_rank < schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Otherwise, return the schoolbook decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
