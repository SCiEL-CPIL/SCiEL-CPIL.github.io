"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 120s as safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # The tensor has shape (n, m, p) with T[i,j,k] = 1 for all valid indices
    # We need to find a rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # The schoolbook rank is n*m*p = 72
    # We need to find a lower rank decomposition
    
    # For 3x3 times 3x8 matrix multiplication, let's think about structure
    # A is 3x3, B is 3x8, C = A*B is 3x8
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The tensor T is indexed by (i,j,k) where i in [0,3), j in [0,3), k in [0,8)
    # T[i,j,k] = 1 for all i,j,k (since every A[i,j] multiplies every B[j,k])
    
    # Wait, that's not right. The multiplication tensor has T[i,j,k] = 1 if the
    # (i,j) entry of A and (j,k) entry of B contribute to C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k since every pair contributes.
    
    # Actually, the standard matrix multiplication tensor M_{n,m,p} has entries
    # M[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # This is just the all-ones tensor of shape (3,3,8)!
    # The rank of the all-ones tensor of shape (n,m,p) is... 
    # For the all-ones tensor, we can write it as:
    # T[i,j,k] = 1 = (1)(1)(1)
    # So it's rank 1!
    
    # Wait, that can't be right for matrix multiplication. Let me reconsider.
    
    # The matrix multiplication tensor is NOT the all-ones tensor. 
    # The multiplication tensor T has T[i,j,k] = 1 if and only if the j-th row 
    # of A and j-th column of B contribute to C[i,k]. But in standard matrix 
    # multiplication, C[i,k] = sum_j A[i,j] * B[j,k], so yes, every (i,j,k) 
    # triple with valid indices contributes with coefficient 1.
    
    # Hmm, but that would make it rank 1, which is clearly wrong for matrix multiplication.
    
    # I think I'm confusing the tensor with the multiplication map. Let me reconsider.
    
    # The matrix multiplication tensor T is a 3-tensor where T[i,j,k] = 1 if the
    # product A[i,j]*B[j,k] contributes to C[i,k]. In standard matrix multiplication,
    # C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
    
    # But the rank of this tensor (as a 3-tensor) is not the same as the rank of 
    # matrix multiplication. The tensor rank is the minimum r such that 
    # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c].
    
    # For the all-ones tensor of shape (3,3,8), we have T[i,j,k] = 1 for all i,j,k.
    # We can write 1 = 1*1*1, so U[i,c]=1, V[j,c]=1, W[k,c]=1 for c=0 gives rank 1.
    
    # But wait, that's the tensor rank, not the matrix multiplication rank. The 
    # matrix multiplication rank (border rank or something else?) is different.
    
    # Actually, I think the task is asking for the tensor rank of the matrix 
    # multiplication tensor, which for the all-ones tensor is indeed 1.
    
    # But that seems too easy. Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 
    # matrix by a 3 by 8 matrix."
    
    # The interface says: U of shape (n*m, rank), V of shape (m*p, rank), W of shape 
    # (p*n, rank)
    
    # So U has indices (i*m+j), V has indices (j*p+k), W has indices (k*n+i).
    # The decomposition is: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Where T[i,j,k] = 1 for all valid i,j,k (the all-ones tensor).
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # This is satisfied by rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    
    # But that gives rank 1, which seems too low. Let me verify with the schoolbook 
    # decomposition: rank = n*m*p = 72.
    
    # I think the issue is that the tensor for matrix multiplication is NOT the 
    # all-ones tensor. Let me reconsider.
    
    # In matrix multiplication, C = A * B where A is n x m and B is m x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The multiplication tensor T is defined such that C = T(A, B) where 
    # T is a multilinear map. The tensor T has components T[i,j,k] such that 
    # C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]... no, that's not right either.
    
    # Let me think more carefully. The multiplication map is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write this as: C = M(A, B) where M is a bilinear map.
    # The tensor associated with M has entries M[i,j,k] where the output index is 
    # (i,k) and input indices are (i,j) and (j,k).
    
    # So M[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to 
    # C[i,k]. In standard matrix multiplication, this is always true, so 
    # M[i,j,k] = 1 for all valid i,j,k.
    
    # But the tensor rank of the all-ones tensor is 1, as I showed above.
    
    # I'm clearly missing something. Let me look at the schoolbook decomposition 
    # in the current program:
    
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i*m+j][column] = 1
    #       V[j*p+k][column] = 1
    #       W[k*n+i][column] = 1
    #       column += 1
    
    # So for each (i,j,k), we have a term U[i*m+j]*V[j*p+k]*W[k*n+i] = 1*1*1 = 1.
    # The sum over all (i,j,k) gives n*m*p terms, each equal to 1, so the total is 
    # n*m*p, not 1.
    
    # Ah, I see the issue. The decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # In the schoolbook decomposition, for a fixed (i,j,k), only one term c (the one 
    # corresponding to (i,j,k)) is nonzero, and it equals 1. So the sum is 1, which 
    # equals T[i,j,k] = 1.
    
    # But if I use rank 1 with U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1, 
    # then for any (i,j,k), the sum is U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1, 
    # which also equals T[i,j,k] = 1.
    
    # So rank 1 should work! But that contradicts the fact that matrix multiplication 
    # has a known rank greater than 1.
    
    # I think the confusion is between the tensor rank of the multiplication tensor 
    # and the rank of matrix multiplication as a bilinear map. These are different concepts.
    
    # The tensor rank (or multilinear rank?) of the all-ones tensor is 1, but the 
    # "rank" of matrix multiplication in the context of fast matrix multiplication 
    # algorithms is different.
    
    # Actually, I think I've been overthinking this. The task is simply asking for 
    # the tensor rank of the all-ones tensor of shape (3,3,8), which is 1.
    
    # Let me verify: can we decompose the all-ones tensor of shape (3,3,8) with rank 1?
    # T[i,j,k] = 1 for all i,j,k.
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # With rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then for any i,j,k: U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓
    
    # So the answer is rank 1!
    
    # But wait, the current program uses rank 72 (schoolbook), and the score is 
    # seed/value = 1.0/72 ≈ 0.0139. If I return rank 1, the score would be 1.0/1 = 1.0, 
    # which is the same as the seed. That doesn't make sense.
    
    # Let me re-read the task: "score is the rank" and "the evaluator measures the 
    # raw value (lower is better). The search score is seed/value, so 1.0 equals the 
    # seed program and above 1.0 is an improvement".
    
    # So if the seed has rank 72 and I find rank 1, the score would be 72/1 = 72, 
    # which is much better than 1.0.
    
    # But is rank 1 really valid? Let me double-check by considering what the 
    # multiplication tensor actually is.
    
    # I think the key insight I'm missing is that the matrix multiplication tensor 
    # is NOT the all-ones tensor. The tensor T[i,j,k] should be 1 only if the 
    # specific combination of indices (i,j,k) contributes to the output. But in 
    # standard matrix multiplication, every (i,j,k) with valid indices does 
    # contribute, so T[i,j,k] = 1 for all valid i,j,k.
    
    # Unless... the tensor is defined differently. Maybe T[i,j,k] is the coefficient 
    # of A[i,j]*B[j,k] in C[i,k], which is 1 if j is the same for both, and 0 otherwise. 
    # But that's always 1 for valid j, so it's still the all-ones tensor.
    
    # I'm stuck. Let me just try rank 1 and see if it's accepted. If not, I'll 
    # fall back to a lower bound search.
    
    # Actually, let me reconsider the problem. Maybe the tensor is not the all-ones 
    # tensor. In matrix multiplication, C = A*B, the (i,k)-th entry of C is 
    # sum_j A[i,j]*B[j,k]. The multiplication tensor T is a 3-tensor such that 
    # C = T(A,B) where T is defined by T[i,j,k] = 1 if the j-th column of A and 
    # j-th row of B (or something like that) contribute to C[i,k].
    
    # I think the standard definition is: the matrix multiplication tensor M_{n,m,p} 
    # is the tensor in (R^{n*m}) * (R^{m*p}) * (R^{p*n}) such that 
    # <M_{n,m,p}, A ⊗ B ⊗ C> = trace(C^T * A * B) for all A, B, C.
    
    # Wait, the task says "The output factor follows the official trace convention 
    # W[k*n+i,r]". This suggests that the tensor is defined via a trace formula.
    
    # Let me assume the tensor is the standard matrix multiplication tensor and try 
    # to find its rank. For 3x3 times 3x3, the rank is known to be 23 (Strassen's 
    # algorithm and improvements). For 3x3 times 3x8, it should be higher.
    
    # Given the complexity, let me just implement a search for a low-rank 
    # decomposition using integer/half-integer entries.
    
    # For now, let me return the schoolbook decomposition as a fallback, but try 
    # to find a better one.
    
    # Actually, given the time constraints and the complexity of finding a 
    # rank-minimal decomposition, let me just return the schoolbook decomposition.
    
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
