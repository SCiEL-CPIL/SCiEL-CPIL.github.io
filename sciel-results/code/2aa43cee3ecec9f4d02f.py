"""Tensor decomposition for 2x3 x 3x10 matrix multiplication."""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _build_strassen_2x2x2(offset_i, offset_j, offset_k, n, m, p):
    """
    Return list of (u_dict, v_dict, w_dict) for Strassen 2x2x2.
    Each dict maps (local_idx, col) -> value for the U/V/W matrices restricted to the 2x2 block.
    
    Actually, let's think in terms of the global indices.
    For a 2x2x2 block with A rows i0,i1; cols j0,j1; B rows j0,j1; cols k0,k1; C rows i0,i1; cols k0,k1.
    
    Strassen 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mt is a rank-1 tensor. For Mt, we need:
    U_t[i,j] * V_t[j,k] * W_t[k,i] summed over the rank index gives the contribution.
    
    For M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]):
    U has: (i0,j0)=1, (i1,j1)=1
    V has: (j0,k0)=1, (j1,k1)=1
    W has: (k0,i0)=1, (k0,i1)=1, (k1,i0)=1, (k1,i1)=1
    But wait, that would give 4 terms in the product, not 1.
    
    Actually, in tensor decomposition, a rank-1 term is:
    sum_{i,j,k} U[i*m+j] * V[j*p+k] * W[k*n+i] * A[i,j] * B[j,k]
    should equal the coefficient for C[i,k].
    
    So for a single rank-1 component, we have fixed vectors u, v, w and:
    contribution to C[i,k] = u_{i,j1} * v_{j1,k} * w_{k,i} ... no, that's not right either.
    
    The tensor T[i,j,k] for matrix multiplication is 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    So T[i,j,k] = 1 for all valid (i,j,k).
    
    A rank-1 decomposition is: T = sum_{t=1}^{rank} u_t ⊗ v_t ⊗ w_t
    where (u_t ⊗ v_t ⊗ w_t)[i,j,k] = u_t[i,j] * v_t[j,k] * w_t[k,i]
    
    Wait, the indices don't quite work that way. Let me reconsider.
    
    The matrix multiplication tensor has indices (i, j, k) where:
    - i: row of A (and C), 0..n-1
    - j: col of A / row of B, 0..m-1  
    - k: col of B (and C), 0..p-1
    
    T[i,j,k] = 1 (it's the identity tensor for matrix mult)
    
    A rank-1 decomposition: T[i,j,k] = sum_t U_t[i,j] * V_t[j,k] * W_t[k,i]
    
    Wait, W should have indices (k,i) to match the output C[i,k].
    
    So: sum_t U_t[i*m+j] * V_t[j*p+k] * W_t[k*n+i] = 1 for all (i,j,k)
    
    For Strassen 2x2x2, the 7 rank-1 terms are well-known. Let me just hardcode them.
    
    For the 2x2 case (i in {0,1}, j in {0,1}, k in {0,1}):
    
    Term 1: U: {(0,0):1, (1,1):1}, V: {(0,0):1, (1,1):1}, W: {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    Check: sum over t of U_t[i,j]*V_t[j,k]*W_t[k,i]
    For (i,j,k)=(0,0,0): U(0,0)*V(0,0)*W(0,0) = 1*1*1 = 1 ✓
    For (i,j,k)=(0,0,1): U(0,0)*V(0,1)*W(1,0) = 1*0*1 = 0 ✗ (should be 1)
    
    Hmm, that's not right. Let me look up the actual Strassen tensor decomposition.
    
    Actually, I think the issue is that each Strassen "product" Mt corresponds to a single rank-1 term, but the U, V, W vectors are sparse in a specific way.
    
    Let me just use the known result that 2x2x2 has rank 7 and construct it properly.
    
    The 7 rank-1 tensors for 2x2x2 (using 0-indexed):
    
    t=1: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}, W={(0,0):1,(1,1):1}
         This gives: A00*B00*C00 + A11*B11*C11
         
    t=2: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}, W={(0,1):1,(1,0):1}
         This gives: A00*B00*C01 + A11*B11*C10
         
    Hmm, this is getting complicated. Let me just use a different approach.
    
    For now, let me just return the schoolbook decomposition which is always valid.
    """
    pass

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Start with schoolbook rank = n*m*p = 60
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce rank using Strassen-like tricks
    # For 2x3x10, we can split into blocks
    # M_{2,3,10} = M_{2,3,5} + M_{2,3,5} (splitting the 10 columns into two 5-column blocks)
    # M_{2,3,5} can be further decomposed
    
    # Let's try a simpler approach: use the fact that n=2 is small
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # The tensor is: sum_{i,j,k} A[i,j]*B[j,k]*C[i,k]
    # = sum_{j,k} (A[0,j]*C[0,k] + A[1,j]*C[1,k]) * B[j,k]
    
    # This suggests we might be able to share some rank-1 terms between i=0 and i=1
    
    # For now, return the schoolbook decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
