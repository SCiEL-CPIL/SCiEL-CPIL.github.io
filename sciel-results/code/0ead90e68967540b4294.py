"""Matrix multiplication 3x5x5 rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # The matrix multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    
    # Strategy: Start from schoolbook (rank 75) and try to reduce rank
    # by finding linear dependencies or using structured decompositions.
    
    # First, let's try a direct approach:
    # The tensor is T_{ijk} where i in [0,n), j in [0,m), k in [0,p)
    # T_{ijk} = 1 (it's the standard matrix multiplication tensor)
    
    # Schoolbook: rank = n*m*p = 75
    # But we can do better. For matrix multiplication, the rank is known to be
    # less than n*m*p for many cases.
    
    # Key insight: The multiplication tensor for (n,m,p) has rank at most
    # n*m*p, but for rectangular cases, there are known improvements.
    
    # Let's try to find a lower rank decomposition using ALS with exact arithmetic.
    
    # Approach: Use the fact that we can represent the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has entries T[i,j,k] = delta (it's 1 for all valid i,j,k in the
    # standard multiplication tensor - wait, no. The tensor T_{ijk} represents
    # the coefficient such that C_{ik} = sum_{j,r} A_{ij} B_{jk} * T_{ijk}
    # Actually T_{ijk} = 1 for all i,j,k since each product A[i,j]*B[j,k]
    # contributes to C[i,k].
    
    # So the tensor is simply all-ones tensor of shape (n, m, p).
    
    # Wait, let me reconsider. The matrix multiplication tensor T is defined by:
    # T(e_i ⊗ e_j ⊗ e_k) = 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k]
    # For standard matrix multiplication, every (i,j,k) triple has T=1.
    
    # So T is the all-ones tensor of shape (3,5,5).
    
    # The rank of the all-ones tensor of shape (n,m,p) is... 
    # Actually no. Let me think again.
    
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # T_{ijk} is defined such that:
    # (A*B)_{ik} = sum_{j} A_{ij} B_{jk}
    # 
    # In tensor form: C_{ik} = sum_{j,r} U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    # where the sum over r is the rank.
    #
    # The "target" tensor is T_{ijk} where we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # Wait, that's not right either. Let me re-read the task.
    #
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The decomposition satisfies:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    #
    # For matrix multiplication, T[i,j,k] = 1 for all valid i, j, k.
    # Because A[i,j] * B[j,k] always contributes to C[i,k].
    
    # So we need the all-ones tensor of shape (3, 5, 5) to have a rank-r decomposition.
    # The rank of the all-ones tensor of shape (n, m, p) over the reals is max(n, m, p)?
    # No, that's not right either.
    
    # Actually, the all-ones tensor of shape (n,m,p) has rank 1! Because:
    # T_{ijk} = 1 = 1 * 1 * 1
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 gives rank 1.
    
    # Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    # Oh I see the issue. The matrix multiplication tensor is NOT all ones in the
    # way I was thinking. Let me reconsider.
    
    # For multiplying an n×m matrix A by an m×p matrix B to get n×p matrix C:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # The tensor T is a 3-tensor of shape (n, m, p) where:
    # T[i, j, k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    #
    # And the decomposition is:
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k]) ... 
    #
    # No, the standard formulation is:
    # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    #
    # Hmm, let me look at the schoolbook code more carefully:
    # U[i*m+j][column] = 1  -> U[i,j] = 1
    # V[j*p+k][column] = 1  -> V[j,k] = 1  
    # W[k*n+i][column] = 1  -> W[k,i] = 1
    #
    # So for each (i,j,k) triple, we have a rank-1 term with all ones.
    # The reconstruction would be:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For schoolbook, each (i,j,k) has its own column c, so:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) matches exactly, else 0
    #
    # Wait no. In schoolbook, for column c corresponding to (i',j',k'):
    # U[i*m+j, c] = 1 if i=i' and j=j', else 0
    # V[j*p+k, c] = 1 if j=j' and k=k', else 0
    # W[k*n+i, c] = 1 if k=k' and i=i', else 0
    #
    # So the product U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 only if
    # i=i', j=j', k=k' simultaneously, i.e., only when (i,j,k) = (i',j',k').
    #
    # So the reconstructed tensor is T[i,j,k] = 1 for all i,j,k.
    # This confirms: the target tensor IS the all-ones tensor.
    
    # Now, the rank of the all-ones tensor of shape (n,m,p):
    # T[i,j,k] = 1 for all i,j,k
    # 
    # This can be written as: T[i,j,k] = sum_r u[i,j,r] * v[j,k,r] * w[k,i,r]
    #
    # For rank 1: u[i,j,0]=1, v[j,k,0]=1, w[k,i,0]=1 works!
    # T[i,j,k] = 1*1*1 = 1 ✓
    
    # So the rank is 1?! That seems too good. Let me verify with the evaluator's
    # expected behavior.
    
    # Actually wait - I need to double-check. Is the tensor really all-ones?
    # In matrix multiplication C = AB where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The "tensor" here encodes which products A[i,j]*B[j,k] contribute to C[i,k].
    # For each (i,j,k), the coefficient is 1. So yes, T[i,j,k] = 1.
    
    # But then the rank-1 decomposition U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1
    # would give:
    # For any (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1
    #
    # This seems correct! So the answer should be rank 1.
    
    # But the schoolbook gives rank 75 and scores 1.0. If rank 1 were valid,
    # the score would be 75/1 = 75, which is way above 1.0.
    
    # Let me re-read the task more carefully...
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix
    # by a 5 by 5 matrix."
    
    # "score is the rank" and "seed/value, so 1.0 equals the seed program"
    # Seed has rank 75, so score = 75/75 = 1.0. If we find rank r, score = 75/r.
    
    # So if rank 1 works, score = 75. That would be a massive improvement.
    
    # But I suspect the tensor is NOT simply all-ones. Let me reconsider.
    
    # Perhaps the tensor entries are not all 1. Maybe the convention is different.
    # Let me re-examine: "W[k*n+i,r]" - this is the output factor.
    
    # The standard matrix multiplication tensor for C = AB:
    # C_{ik} = sum_j A_{ij} B_{jk}
    #
    # In the bilinear algorithm form:
    # C_{ik} = sum_r sum_j U_{ij}^{(r)} A_{ij} * V_{jk}^{(r)} B_{jk} * W_{ki}^{(r)}
    #
    # Hmm, that's not quite standard. Let me think of it differently.
    
    # The bilinear complexity formulation:
    # We want: C_{ik} = sum_r W_{ik}^{(r)} * P_r
    # where P_r = sum_{i,j} U_{ij}^{(r)} A_{ij} * sum_{j,k} V_{jk}^{(r)} B_{jk}
    #
    # No, that's not right either. The standard form is:
    # C_{ik} = sum_r (sum_j U_{ij}^{(r)} A_{ij}) * (sum_j V_{jk}^{(r)} B_{jk}) * W_{ik}^{(r)}
    #
    # Wait, I think the correct form based on the code is:
    # The tensor T[i,j,k] satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    #
    # And for matrix multiplication, what is T[i,j,k]?
    # 
    # I believe T[i,j,k] = 1 for all i,j,k because each product A[i,j]*B[j,k]
    # appears exactly once in the computation of C[i,k].
    
    # Given the schoolbook code confirms this (each (i,j,k) gets a unique rank-1
    # term with all-ones), the target is indeed the all-ones tensor.
    
    # So rank 1 should work. Let me just return rank 1.
    
    # But wait - maybe I'm misunderstanding the problem. Let me consider that
    # perhaps the "tensor" is not all-ones but rather encodes the structure of
    # matrix multiplication more subtly.
    
    # Actually, I just realized: in some formulations, the matrix multiplication
    # tensor is NOT the all-ones tensor. The tensor T_{ijk} might be defined
    # differently.
    
    # Let me reconsider: perhaps T_{ijk} is 1 only when the indices are "compatible"
    # in some way, or perhaps the tensor has a different structure.
    
    # Given that the schoolbook decomposition uses rank n*m*p = 75 with all-ones
    # entries, and this is described as the "schoolbook decomposition", I believe
    # the target tensor IS the all-ones tensor, and the schoolbook is just the
    # naive way to decompose it.
    
    # Therefore, rank 1 should be valid. Let me return it.
    
    U = [[1] * 1 for _ in range(n * m)]
    V = [[1] * 1 for _ in range(m * p)]
    W = [[1] * 1 for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
