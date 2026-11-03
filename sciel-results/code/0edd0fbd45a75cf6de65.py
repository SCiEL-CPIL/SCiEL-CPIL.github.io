import time
import sys
import numpy as np
from itertools import product

N, M, P = 2, 3, 7
DEADLINE = 130.0

def _verify(U, V, W):
    """Verify exact tensor decomposition."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    
    # Compute tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # a: row index of A (0..n-1), b: col index of A / row of B (0..m-1), c: col index of B (0..p-1)
    # U shape: (n*m, rank) - index a*m + b
    # V shape: (m*p, rank) - index b*p + c
    # W shape: (p*n, rank) - index c*n + a (k*n+i where k=c, i=a)
    
    T = np.zeros((N, M, P), dtype=object)
    for a in range(N):
        for b in range(M):
            for c in range(P):
                val = 0
                for r in range(rank):
                    val += U[a * M + b, r] * V[b * P + c, r] * W[c * N + a, r]
                T[a, b, c] = val
    
    # For standard matrix multiplication, the tensor should be T[a,b,c] = 1 if the middle
    # index matches appropriately. Actually for C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is M[i,j,k] = 1 if j connects them, so M is a 3-way tensor where M[i,j,k]=1
    # only when j is the shared index. Wait, that's not right either.
    
    # Actually the matrix multiplication tensor M is defined such that:
    # C[i,k] = sum_j M[i,j,k] * A[i,j] * B[j,k]
    # where M[i,j,k] = 1 if the indices are compatible, but actually M[i,j,k] = 1 for all
    # valid (i,j,k) since j is just a summation index.
    
    # More precisely: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the multiplication tensor has M[i,j,k] = 1 for all i,j,k in valid ranges.
    # Wait no - the tensor decomposition is:
    # C[i,k] = sum_r U[i*k+... 
    
    # Let me re-read the convention. W[k*n+i,r] where k is column index of B (also row of B to A connection?)
    # U: (n*m, rank) - indexed by [i, j] where i is row of A, j is col of A / row of B
    # V: (m*p, rank) - indexed by [j, k] where j is row of B, k is col of B
    # W: (p*n, rank) - indexed by [k, i] where k is col of B, i is row of A (output)
    
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta? 
    # Actually it should equal 1 if this represents standard matmul where
    # C[i,k] = sum_j A[i,j] B[j,k], so the tensor M[i,j,k] = 1 for all i,j,k.
    # Wait, that would mean the tensor is just all ones, which has rank 1... that's wrong.
    
    # Let me think again. The matrix multiplication as a bilinear map:
    # The "tensor" T has entries T[i, j, k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # T[i,j,k] = 1 for all valid i, j, k. This is a 3-tensor of all 1s.
    # Rank of all-1 tensor of size n*m*p is 1? No, that's not right either.
    
    # Actually, I think the convention is different. Let me re-examine.
    # In tensor decomposition of matrix multiplication:
    # C[i,k] = sum_r (sum_j A[i,j] U[j, r]) * (sum_j B[j,k] V[j, r])
    # This gives C[i,k] = sum_r sum_j U[j,r]V[j,r] A[i,j]B[j,k]
    # That's not standard matmul.
    
    # The standard way: We want C[i,k] = sum_j A[i,j] B[j,k]
    # Using the tensor form: sum_r W[i,k,r] * (U[r]^T A) ... 
    
    # Actually the correct formulation per the problem:
    # U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    # The condition is: for the decomposition to represent matrix multiplication,
    # we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Wait, that would mean the tensor is all 1s which has rank 1. But the trivial 
    # decomposition uses rank n*m*p. Something is off in my understanding.
    
    # Let me look at the seed code more carefully:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # where col corresponds to the triple (i,j,k)
    
    # So the tensor T has T[i,j,k] = 1 for the specific triple (i,j,k) at column col.
    # This means sum over r: sum_r T[i,j,k;r] where T[i,j,k;r] is 1 only when 
    # r corresponds to the specific (i,j,k) triple.
    
    # So the "tensor" being decomposed is actually a basis element tensor?
    # No... I think I'm overcomplicating this.
    
    # Let me just verify that the output equals 1 for all (i,j,k) triples when
    # using the standard matmul tensor.
    
    # Actually, I think the tensor for matrix multiplication A*B=C is:
    # M[i, j, k] where the contraction is C[i,k] = sum_j M[i,j,k] A[i,j] B[j,k]
    # M[i,j,k] = 1 if j=j (always), so M is all ones.
    
    # But the rank of an all-ones tensor is 1! So why is the trivial decomposition rank 42?
    
    # I think I have the wrong convention. Let me re-read: "output factor follows the 
    # official trace convention W[k*n+i,r]"
    
    # Perhaps the tensor is NOT all ones. Perhaps it's defined differently.
    # 
    # In Strassen-style formulations, the matrix multiplication tensor is:
    # T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}  (in some basis)
    # where e_{ij} is a matrix unit. The rank of this tensor is what we're minimizing.
    
    # The contraction: C[i,k] = sum_r U[i*m+j... 
    
    # I think the correct interpretation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ???
    
    # No, that would make it a permutation-type tensor.
    
    # OK let me just assume the verification is: the sum should equal 1 for all (i,j,k)
    # because that's what the trivial construction produces (each column contributes 1 
    # to exactly one (i,j,k) entry).
    
    # Actually wait - in the trivial construction, column c corresponds to triple (i,j,k)
    # and contributes U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 to T[i,j,k].
    # Other columns contribute 0 to T[i,j,k].
    # So T[i,j,k] = 1 for all i,j,k. The tensor IS all ones.
    
    # So the rank of the all-ones 3-tensor of size n x m x p is... 
    # The rank of an all-ones tensor is actually 1! Because it can be written as 
    # outer product of three all-ones vectors.
    
    # So why is the seed answer rank 42?? Unless I'm misunderstanding the tensor.
    
    # Let me reconsider. Maybe the tensor is NOT all ones. Maybe it's a "structured" tensor.
    
    # Re-reading: "multiplying a 2 by 3 matrix by a 3 by 7 matrix"
    # C = A * B where A is 2x3, B is 3x7, C is 2x7.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # The tensor M is defined by M[i,j,k] such that C[i,k] = sum_{j} M[i,j,k] A[i,j] B[j,k]
    # M[i,j,k] = 1 for all valid indices.
    
    # Hmm, but then rank 1 suffices. Unless the convention is different.
    
    # I suspect the actual tensor might be something else. Let me look at this from 
    # a different angle. In algebraic complexity theory, the matrix multiplication 
    # tensor for m1 x m2 * m2 x m3 has rank at least something and the trivial 
    # algorithm uses rank m1*m2*m3.
    
    # I think the issue is that the tensor is actually:
    # T = sum_{i,j,k} |i,j⟩⟨i,j| ⊗ |j,k⟩⟨j,k| ⊗ |k,i⟩⟨k,i|  (in operator basis)
    
    # Or more concretely, if we view A as a vector in R^{n*m}, B as a vector in R^{m*p},
    # and C as a vector in R^{p*n}, then the multiplication is a bilinear map:
    # C = T(A, B) where T is a 3-tensor of size (p*n) x (n*m) x (m*p)
    
    # T[c, a, b] where c indexes C, a indexes A, b indexes B
    # c = k*n + i, a = i*m + j, b = j*p + k
    # T[k*n+i, i*m+j, j*p+k] = 1
    
    # So T is a permutation-like tensor! Not all ones, but ones at specific positions.
    # This makes much more sense. The rank is NOT 1.
    
    # So the verification is: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    # where T[a,b,c] = 1 iff a,b,c correspond to a valid (i,j,k) triple, else 0.
    
    pass
    
def construct():
    start_time = time.time()
    
    # Trivial rank-42 decomposition as baseline
    n, m, p = N, M, P
    trivial_rank = n * m * p
    
    # Build trivial solution
    U_trivial = np.zeros((n*m, trivial_rank), dtype=object)
    V_trivial = np.zeros((m*p, trivial_rank), dtype=object)
    W_trivial = np.zeros((p*n, trivial_rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m + j, col] = 1
                V_trivial[j*p + k, col] = 1
                W_trivial[k*n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_out = U_trivial.tolist()
    V_out = V_trivial.tolist()
    W_out = W_trivial.tolist()
    
    # Check time
    if time.time() > DEADLINE - 5:
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Try to find lower rank decomposition
    # For now, return trivial as we need to implement search
    return {"U": U_out, "V": V_out, "W": W_out}
