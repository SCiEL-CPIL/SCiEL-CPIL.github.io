"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (28, rank)
    # W shape: (p*n, rank) = (14, rank)
    # Tensor: T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # But the convention is W[k*n + i, r]
    
    # The multiplication tensor for (n x m) * (m x p) -> (n x p)
    # T[i, j, k] = 1 if we multiply A[i,j] * B[j,k] to get C[i,k]
    # Actually the tensor is: sum_j A[i,j] B[j,k] = C[i,k]
    # So the tensor T has indices (i, j, k) where T[i,j,k] = delta (it's just 1 for each (i,j,k))
    # Wait, no. The matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T_{i,j,k} = 1 (for all i,j,k) represents the structure
    # Actually T[i,j,k] = delta_{i,k}... no.
    
    # Let me re-read. The task says:
    # "U[i*m+j, r]", "V[j*p+k, r]", "W[k*n+i, r]"
    # So U is indexed by (i,j) with i in [0,n), j in [0,m)
    # V is indexed by (j,k) with j in [0,m), k in [0,p)
    # W is indexed by (k,i) with k in [0,p), i in [0,n)
    
    # The decomposition is: for all (i,j,k),
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # What is T[i,j,k]? For matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor that encodes this is T[i,j,k] = 1 for all valid (i,j,k)
    # because each term A[i,j]*B[j,k] contributes to C[i,k].
    
    # Wait, actually the standard way: the matrix multiplication tensor of order (n,m,p)
    # is a 3-tensor T of size n x m x p where T[i,j,k] = 1 if and only if... 
    # Actually it's just all ones? No.
    
    # Let me think again. The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j]B[j,k]
    # This can be written as: C[i,k] = sum_{j,r} A[i,j] * (sum over r of something) * B[j,k]
    # 
    # The standard tensor decomposition: T[i,j,k] where the "tensor" represents the
    # multiplication structure. For matrix multiplication, T[i,j,k] = 1 for all (i,j,k)
    # in the valid range, because every (i,j,k) triple contributes.
    
    # Actually no. Let me look at the schoolbook solution:
    # For each (i,j,k): U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1
    # So sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for the specific r corresponding to (i,j,k)
    # and 0 otherwise. So T[i,j,k] = 1 for all (i,j,k).
    
    # So the tensor is all-ones of shape (n, m, p) = (2, 4, 7).
    # We need rank-minimal decomposition where:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # with T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,7)
    
    # The rank is at least max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # because we need to represent 28 distinct (j,k) pairs in V.
    # Actually the lower bound is more subtle.
    
    # Known results: For 2x4x7, the known upper bounds...
    # Schoolbook: 2*4*7 = 56
    # The observed improvement went from 56 to 50 using Strassen-like blocks.
    
    # Strategy: Use Strassen's algorithm on sub-blocks.
    # 2x4 = (2x2 blocks of 2x2)
    # 4x7 = (2x2 blocks of 2x4... hmm, 4x7 doesn't split evenly into 2x2)
    # 
    # Let me think of a block decomposition:
    # A is 2x4, B is 4x7
    # Split A into 2 blocks of 2x2: A = [A11 A12; A21 A22] where each is 2x2
    # Split B into 2x2 and 2x3: B = [B11 B12; B21 B22] where B11,B12 are 2x2,2x3... 
    # This is getting complicated.
    
    # Alternative: try to find a rank < 56 decomposition systematically.
    # 
    # Known: For 2x2x2, Strassen gives rank 7.
    # For 2x4x4, we can use Strassen on 2x2 blocks: rank = 7*2 = 14? 
    # Actually 2x4x4: split into two 2x2x2 problems, rank = 7+7 = 14.
    # But 2x4x7: split 4x7 into 4x4 + 4x3, giving 2x4x4 + 2x4x3.
    # 2x4x4 has rank at most 14 (two Strassen 2x2x2).
    # 2x4x3: this is trickier.
    
    # Let me try a different approach: use the known Strassen-based decomposition
    # for 2x2x2 and extend it.
    
    # Actually, let me just try to construct a good decomposition.
    # 
    # For 2x4x7, one approach:
    # Think of it as multiplying [2x4] by [4x7].
    # We can write this as: [2x2 | 2x2] * [2x2; 2x2 | 2x2; 2x2]... 
    # 
    # Let me use a block Strassen approach:
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B = [[B11, B12], [B21, B22]] where B11 is 2x4, B12 is 2x3, B21 is 2x4, B22 is 2x3
    # 
    # This doesn't split evenly. Let me try:
    # A is 2x4, B is 4x7.
    # Split the middle dimension 4 into 2+2:
    # A = [A_left, A_right] where A_left is 2x2, A_right is 2x2
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # C = A_left * B_top + A_right * B_bottom
    # 
    # Each 2x2 * 2x7 multiplication: we can use Strassen-like for 2x2 * 2x2 blocks.
    # 2x2 * 2x7 = 2x2 * [2x2 | 2x2 | 2x1]... still uneven.
    
    # Let me just try to find the best known rank for 2x4x7.
    # The trivial bound is 56. Strassen-based might give less.
    # 
    # Actually, let me just implement a search for a good decomposition.
    # Since the tensor is all-ones (n x m x p), we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a 3-way tensor decomposition problem.
    # 
    # Let me try a structured approach:
    # For each j (middle index, 0..3), we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,k
    # 
    # If we fix j, this becomes a rank-1 decomposition of the all-ones matrix of size n x p
    # weighted by the r-th component.
    
    # Key insight: If we can make the decomposition separable in j, we might do better.
    
    # Let me try: use rank 28 (the lower bound from m*p).
    # For each (j,k) pair, have a rank-1 term.
    # Term (j,k): U[i*m+j, r_{j,k}] = a_{i,j,k}, V[j*p+k, r_{j,k}] = b_{j,k}, W[k*n+i, r_{j,k}] = c_{k,i}
    # We need: sum over r... but if each (j,k) has its own r, then:
    # For fixed (i,j,k): only r_{j,k} contributes, so a_{i,j,k} * b_{j,k} * c_{k,i} = 1
    # This requires a_{i,j,k} * b_{j,k} * c_{k,i} = 1 for all i,j,k.
    # We can set b_{j,k} = 1, then a_{i,j,k} * c_{k,i} = 1.
    # Set a_{i,j,k} = 1, c_{k,i} = 1.
    # This gives rank = m*p = 28!
    
    # Wait, let me verify:
    # For each (j,k) in [0,4)x[0,7), assign a unique rank index r.
    # U[i*m+j, r_{j,k}] = 1 for all i
    # V[j*p+k, r_{j,k}] = 1
    # W[k*n+i, r_{j,k}] = 1 for all i
    # 
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = U[i*m+j,r_{j,k}]*V[j*p+k,r_{j,k}]*W[k*n+i,r_{j,k}] = 1*1*1 = 1.
    # 
    # And for other r values (corresponding to different (j',k')), the product is 0
    # because either V[j*p+k, r_{j',k'}] = 0 (if j' != j or k' != k) or...
    # Wait, V[j*p+k, r_{j',k'}] is nonzero only if (j,k) = (j',k'), so yes.
    
    # So rank 28 works! Let me verify the dimensions:
    # U: 8 x 28, V: 28 x 28, W: 14 x 28
    # U[i*m+j, r_{j,k}] = 1 means for fixed (j,k), all 8 rows... no, U has 8 rows (i*m+j for i in 0..1, j in 0..3)
    # For fixed (j,k), U[i*m+j, r_{j,k}] = 1 for i=0,1 (two entries in column r_{j,k})
    # V[j*p+k, r_{j,k}] = 1 (one entry)
    # W[k*n+i, r_{j,k}] = 1 for i=0,1 (two entries in column r_{j,k})
    
    # Let me code this up:
    
    rank = m * p  # 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
                W[k * n + i][col] = 1
            V[j * p + k][col] = 1
            col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    # Fallback to schoolbook
                    return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

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
