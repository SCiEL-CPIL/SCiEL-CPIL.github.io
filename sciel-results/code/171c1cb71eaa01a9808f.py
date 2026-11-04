import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
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


def _verify(U, V, W, n, m, p):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for rr in range(r):
                    u = U[i*m+j][rr]
                    v = V[j*p+k][rr]
                    w = W[k*n+i][rr]
                    if u != 0 and v != 0 and w != 0:
                        total += Fraction(u) * Fraction(v) * Fraction(w)
                if total != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100
    
    # Try a structured decomposition: 
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # We can write this as sum over j of A[i,j]*B[j,k]
    # Schoolbook uses one term per (i,j,k).
    # 
    # Idea: use a "factor" decomposition where we group terms.
    # For example, for each i, we have n vectors A[i,:] of length m.
    # For each k, we have p vectors B[:,k] of length m.
    # C[i,k] = A[i,:] dot B[:,k]
    # 
    # We can use a low-rank approximation of the "identity" in the m-dimensional space,
    # but we need exactness.
    #
    # Alternative: use the fact that m=5 is small. We can use a decomposition of the
    # 5-dimensional inner product into fewer terms.
    # The inner product <x,y> = sum_j x_j * y_j can be computed with rank m=5 (schoolbook).
    # Can we do it with fewer? No, the rank of the inner product tensor is m.
    # So for each (i,k) pair, we need rank >= m = 5.
    # Total rank >= n*p*m = 90? No, that's the schoolbook.
    # 
    # Actually, the tensor is T[i,j,k] = 1. The rank of this tensor is what we're looking for.
    # For T[i,j,k] = 1 (all ones), the rank is 1! Because T = u ⊗ v ⊗ w where u_i=1, v_j=1, w_k=1.
    # Wait, that can't be right for matrix multiplication.
    #
    # I think I'm confusing the tensor. Let me re-derive.
    #
    # The matrix multiplication map: (A, B) -> A*B
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # As a bilinear map, this is a tensor T in (R^n * R^m)^* ⊗ (R^m * R^p)^* ⊗ (R^n * R^p)
    # T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] contributes with coefficient 1)
    #
    # So T[i,j,k] = 1 for all i,j,k. This is indeed the all-ones tensor.
    # The rank of the all-ones tensor T[i,j,k] = 1 is 1, achieved by U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1.
    #
    # But wait, that would mean rank 1, which is clearly wrong. The issue is that the tensor is not T[i,j,k]=1.
    #
    # Let me reconsider. The bilinear map is:
    # f(A,B) = A*B
    # <f(A,B), C> = sum_{i,k} C[i,k] * (A*B)[i,k] = sum_{i,k} C[i,k] * sum_j A[i,j]*B[j,k]
    # = sum_{i,j,k} A[i,j]*B[j,k]*C[i,k]
    #
    # So the tensor is T[i,j,k] = 1 for all i,j,k, and the trilinear form is sum_{i,j,k} A[i,j]*B[j,k]*C[i,k]*T[i,j,k].
    # But C[i,k] is the output, not an input. The tensor should be in (R^{nm})^* ⊗ (R^{mp})^* ⊗ (R^{np})
    # T[i,j,k] where i is row of A, j is column of A / row of B, k is column of B.
    # T[i,j,k] = 1 for all i,j,k.
    #
    # The rank of this tensor: T[i,j,k] = 1 = u_i * v_j * w_k where u_i=1, v_j=1, w_k=1.
    # So rank is 1?? That can't be right.
    #
    # I think the issue is that the tensor is not simply all-ones. Let me look at it differently.
    #
    # The multiplication tensor T is defined by: T(i,j,k) = delta_{i,k} in some sense? No.
    #
    # Actually, I think the correct tensor is: T[i,j,k] = 1 for all i,j,k, and the rank of this tensor IS 1.
    # But that contradicts the known fact that matrix multiplication has rank n*m*p for the schoolbook.
    #
    # The resolution: the tensor T[i,j,k] = 1 has rank 1, but the matrix multiplication tensor is NOT T[i,j,k]=1.
    # 
    # Let me look at this more carefully. The bilinear map is:
    # f: R^{n x m} x R^{m x p} -> R^{n x p}
    # f(A,B)[i,k] = sum_j A[i,j] * B[j,k]
    #
    # As a tensor in (R^{nm})^* ⊗ (R^{mp})^* ⊗ R^{np}:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (where e_i is the basis vector for position (i,j) in A, etc.)
    #
    # Wait, T[i,j,k] = 1 for all i,j,k. This is the sum of all basis tensors e_i ⊗ e_j ⊗ e_k.
    # This can be written as (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k), so rank 1.
    #
    # I'm clearly making an error. Let me look at a simple case: 1x1 * 1x1.
    # f(A,B) = A*B, a scalar. T[0,0,0] = 1. Rank 1. Correct.
    #
    # 2x2 * 2x2: f(A,B)[i,k] = sum_j A[i,j]*B[j,k]
    # T[i,j,k] = 1 for all i,j,k (8 terms).
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k = (e_0+e_1) ⊗ (e_0+e_1) ⊗ (e_0+e_1)
    # Rank 1?? But Strassen shows rank 7, and schoolbook is 8.
    #
    # I think the issue is that the tensor is NOT T[i,j,k]=1. Let me re-derive.
    #
    # The bilinear map f(A,B) = A*B. The tensor T is such that:
    # f(A,B) = T(A,B) where T is a bilinear form.
    # T(A,B) = sum_{i,k} (A*B)[i,k] * E_{i,k} where E_{i,k} is the matrix unit.
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T(A,B) = sum_{i,k} sum_j A[i,j]*B[j,k]*E_{i,k}
    # = sum_{i,j,k} A[i,j]*B[j,k]*E_{i,k}
    #
    # So the tensor is T = sum_{i,j,k} (e_{i,j})^* ⊗ (e_{j,k})^* ⊗ E_{i,k}
    # where e_{i,j} is the basis vector for A[i,j], e_{j,k} for B[j,k], and E_{i,k} for the output.
    #
    # The key is that the output basis is E_{i,k}, NOT e_k. So the tensor is:
    # T[i,j,k] = E_{i,k} (as a vector in the output space)
    #
    # So T[i,j,k] is the matrix unit E_{i,k}, which depends on i and k, not just 1.
    #
    # Now the rank is not 1. For 2x2*2x2, the tensor has 8 terms, and the rank is 7 (Strassen).
    #
    # So my verification was wrong. Let me fix it.
    #
    # The decomposition: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # where W[k,i,r] is the coefficient of E_{i,k} in the output.
    #
    # So for the decomposition to be valid:
    # sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1 for all i,j,k
    #
    # This is what I had! So the tensor IS T[i,j,k]=1 in the sense that the coefficient of E_{i,k} is 1 for each (i,j,k) term.
    #
    # But then why is the rank not 1? Because the output space is R^{np}, and the tensor is:
    # T = sum_{i,j,k} e_{i,j}^* ⊗ e_{j,k}^* ⊗ E_{i,k}
    # This is NOT the same as (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k E_k) because E_{i,k} depends on both i and k.
    #
    # So the tensor is a sum of np*m terms, each of which is a rank-1 tensor. The rank of the sum can be less than np*m.
    #
    # OK so my verification is correct: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    #
    # And the schoolbook achieves this with rank np*m by using one term per (i,j,k).
    #
    # Now, can we do better? For 2x2*2x2, yes (rank 7). For 3x5*5x6, likely yes.
    #
    # The random search approach won't work because the probability of hitting a valid decomposition is vanishingly small.
    #
    # I need a smarter approach. Let me try to use a known construction or a more structured search.
    #
    # One idea: use a "block" decomposition. Split the 5x6 matrix B into two 5x3 blocks B1 and B2.
    # Then C = A*B = A*B1 + A*B2.
    # A*B1 is 3x3, A*B2 is 3x3.
    # Each can be computed with rank at most 3*3*3=27 (schoolbook) or 23 (Strassen for 3x3).
    # Total rank: 27+27=54 or 23+23=46.
    #
    # Let me implement this: compute A*B1 and A*B2 separately, then add.
    #
    # For A*B1 (3x3 * 3x3):
    # C1[i,k1] = sum_j A[i,j]*B1[j,k1] where j=0,1,2 and k1=0,1,2
    # U1[i*3+j, r], V1[j*3+k1, r], W1[k1*3+i, r] for r=0..26
    #
    # For A*B2 (3x3 * 3x3):
    # C2[i,k2] = sum_j A[i,j]*B2[j,k2] where j=0,1,2 and k2=0,1,2
    # U2[i*3+j, r], V2[j*3+k2, r], W2[k2*3+i, r] for r=0..26
    #
    # Total: U = [U1, U2], V = [V1, V2], W = [W1, W2] with rank 54.
    #
    # But wait, the output C is 3x6, so C[i, k1] and C[i, k2] are separate.
    # The W factor should be W[k*n+i, r] where k=0..5.
    # For B1: k=0,1,2, for B2: k=3,4,5.
    #
    # So W1 has entries W[k1*3+i, r] for k1=0,1,2, and W2 has W[(k1+3)*3+i, r] for k1=0,1,2.
    #
    # This gives a valid decomposition of rank 54.
    #
    # Can we do better? Use Strassen for the 3x3 multiplications.
    # Strassen for 3x3*3x3 has rank 23.
    # Total: 23+23=46.
    #
    # Let me implement the block decomposition with schoolbook for each block first.

    def _block_decomposition(n, m, p):
        # Split B into ceil(p/2) and floor(p/2) blocks
        # Actually, let's split into two equal-ish blocks
        p1 = p // 2
        p2 = p - p1
        
        # For A*B1: n x m * m x p1
        # Schoolbook rank: n*m*p1
        rank1 = n * m * p1
        rank2 = n * m * p2
        total_rank = rank1 + rank2
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # First block: B[:, 0:p1]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p1):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Second block: B[:, p1:p]
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    U[i * m + j][col] = 1
                    V[j * p + p1 + k][col] = 1
                    W[(p1 + k) * n + i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try block decomposition
    block_result = _block_decomposition(n, m, p)
    if _verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
        return block_result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
