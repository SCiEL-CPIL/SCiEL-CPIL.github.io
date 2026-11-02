"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from functools import lru_cache

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build the matrix multiplication tensor T[i,j,k] = delta[i,j,k]
    # In the decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid else 0
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid (i,j,k) since it's the identity in the "multiplication" sense
    
    # Actually, the matrix multiplication tensor has entries:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor can be viewed as a 3-way tensor. We can "matricize" it.
    # One common matricization: for each k, form a matrix M_k where M_k[i*n+j, ...] 
    # But let's think differently.
    
    # The tensor T has shape (n*m) x (m*p) x (p*n) in the U,V,W indexing.
    # T_{(i*m+j), (j*p+k), (k*n+i)} = 1 for valid i,j,k, else 0.
    
    # We can form a matrix by fixing one mode. Let's fix the W index.
    # For each (k,n+i) pair in W, we get a matrix of size (n*m) x (m*p).
    # But we need consistency across all three modes.
    
    # Alternative: use the fact that the matrix multiplication tensor is the 
    # "structure tensor" and can be decomposed using known algorithms.
    
    # For 3x3x3, Strassen gives rank 7. For 3x3x5, we can embed.
    # Let's try a direct approach: construct the tensor and use SVD on a matricization.
    
    # Matricize the tensor by flattening the first two indices:
    # M[k*n+i, i*m+j] = T[i,j,k*n+i] -- this doesn't quite work due to index coupling.
    
    # Better: The tensor is T[i,j,k] = 1 for all i,j,k in their ranges.
    # This is actually the all-ones tensor! Wait, no.
    
    # Let me re-read the problem. The tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    # This means the tensor is simply the all-ones tensor of shape (n,m,p)!
    # That can't be right because then rank would be 1.
    
    # I think I'm misunderstanding. The tensor decomposition is:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (using the trace convention)
    # where T[i,j,k] = 1 if the indices are "compatible" for matrix mult.
    
    # Actually for standard matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all i,j,k (it's the identity map on the multiplication)
    # The decomposition finds rank-r factors such that the bilinear form matches.
    
    # Let me just use the schoolbook as baseline and try to improve via SVD.
    
    # The tensor T can be represented as a matrix by fixing one index.
    # Fix k: M_k[i*m+j, k*n+i] -- but i appears in both, so this is tricky.
    
    # Let's use a different matricization. The tensor has 3 modes:
    # Mode 1: i*m+j (size n*m)
    # Mode 2: j*p+k (size m*p)  
    # Mode 3: k*n+i (size p*n)
    
    # T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T is nonzero only when the indices are consistent (same i,j,k).
    
    # For a given (i,j,k): a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1
    
    # We can form a matrix by "unfolding" along mode 3:
    # For each c = k*n+i, form matrix M_c of size (n*m) x (m*p)
    # M_c[a,b] = T[a,b,c]
    
    # For fixed c = k*n+i:
    # a = i*m+j for j in 0..m-1
    # b = j'*p+k for j' in 0..m-1, but we need j'=j for consistency
    # So M_c[i*m+j, j*p+k] = 1 if j'=j else 0
    # This gives M_c as a diagonal-like matrix with 1s on the "consistent" positions.
    
    # Actually, for fixed (i,k), M_c[i*m+j, j*p+k] = 1 for all j.
    # So M_c has 1s at positions (i*m+j, j*p+k) for j=0..m-1.
    
    # Let's build this matrix and use SVD.
    
    size1 = n * m
    size2 = m * p
    size3 = p * n
    
    # Build the matricized tensor: for each c, a matrix of size (size1 x size2)
    # Stack them to form a big matrix of size (size1 * size3) x size2 or similar.
    
    # Actually, let's form a 2D matrix by combining modes 1 and 3:
    # BigM[i*m+j, c] = T[i*m+j, ?, c] -- but we need mode 2 too.
    
    # Let me try: form a matrix of size (size1 * size2) x size3
    # M[(a,b), c] = T[a,b,c]
    
    # This is a valid matricization. Its rank gives a lower bound on tensor rank.
    # And SVD gives a decomposition.
    
    M = np.zeros((size1 * size2, size3), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a * size2 + b, c] = 1.0
    
    # SVD of M
    U_svd, s, Vt_svd = np.linalg.svd(M, full_matrices=False)
    
    # The rank of M is the number of nonzero singular values
    tol = 1e-10
    rank_M = np.sum(s > tol)
    
    # Now we need to convert this back to a tensor decomposition.
    # M = U_svd @ diag(s) @ Vt_svd
    # M[(a,b), c] = sum_r U_svd[(a,b), r] * s[r] * Vt_svd[r, c]
    
    # But we need: T[a,b,c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # This is a 3-way decomposition, not a 2-way.
    
    # The SVD gives us a 2-way decomposition of the matricization, which corresponds
    # to a tensor decomposition where one factor is "absorbed".
    
    # Specifically, if M[(a,b), c] = sum_r X[(a,b), r] * Y[c, r],
    # then we can set: U[a, r] * V[b, r] = X[(a,b), r] and W[c, r] = Y[c, r]
    # But this requires X to be rank-1 in the (a,b) indices for each r, which it's not.
    
    # So this approach doesn't directly give a tensor decomposition.
    
    # Let me try a different approach: use the known Strassen algorithm for 3x3x3
    # and extend it to 3x3x5.
    
    # For 3x3x3, Strassen's algorithm has rank 7.
    # For 3x3x5, we can think of it as 5 copies of 3x3x1 or similar.
    
    # Actually, let's just try to use a greedy or iterative approach to find a lower rank.
    
    # Given time constraints, let's return the schoolbook decomposition for now,
    # but try a simple improvement: check if we can merge some terms.
    
    # For safety, return schoolbook if we're running out of time.
    if time.time() > deadline - 5:
        return _schoolbook(n, m, p)
    
    # Try a simple rank reduction: the tensor T[i,j,k]=1 for all i,j,k
    # is actually the outer product of three vectors of all ones!
    # T[i,j,k] = 1 = 1*1*1
    # So rank 1?! That can't be right for matrix multiplication.
    
    # I think I've been misunderstanding the tensor. Let me re-read.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 5 matrix"
    # The output factor follows the official trace convention W[k*n+i,r].
    
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # It's the tensor that encodes the bilinear map (A,B) -> C where C = A*B.
    # As a 3-way tensor, T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid (i,j,k).
    
    # But this IS the all-ones tensor, which has rank 1.
    # That contradicts known results (Strassen rank 7 for 3x3x3).
    
    # I think the issue is that the tensor is not T[i,j,k] but rather
    # the structure tensor where the indices are coupled differently.
    
    # Let me just return the schoolbook decomposition. It's correct, if not optimal.
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
