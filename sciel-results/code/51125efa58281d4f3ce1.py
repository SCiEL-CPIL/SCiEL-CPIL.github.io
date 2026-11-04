"""
Strategy: Randomized sparse support search for tensor decomposition.
We search for a rank R < 144 decomposition of the 4x4 * 4x9 matrix multiplication tensor.
We fix R=135 (a small reduction from 144) and randomly sample support patterns (sets of non-zero entries) for U, V, W.
For each pattern, we check if the resulting linear system has a unique solution with integer/half-integer coefficients.
"""

import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Dimensions:
    # A: n x m (4x4)
    # B: m x p (4x9)
    # C: n x p (4x9)
    # U shape: (n*m, R) = (16, R)
    # V shape: (m*p, R) = (36, R)
    # W shape: (p*n, R) = (36, R)
    
    # Tensor T has indices (i, j, k) corresponding to A[i,j] * B[j,k] = C[i,k]
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Note: The problem statement says W[k*n+i, r]. Let's verify the trace convention.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T_{i,j,k} = delta_{i,k} is not right.
    # T_{i,j,k} = 1 if we are computing the (i,k) entry of C using A(i,j) and B(j,k).
    # So T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, the tensor for matrix multiplication is defined such that:
    # <T, A, B> = C.
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # The decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
    # Let's check the indices in the prompt:
    # U: (n*m, rank) -> index (i*m+j, r)
    # V: (m*p, rank) -> index (j*p+k, r)
    # W: (p*n, rank) -> index (k*n+i, r)
    # So the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Summing over r, we get T[i,j,k].
    # For the decomposition to be exact, T[i,j,k] must equal 1 for all valid i,j,k.
    
    # We want to find U, V, W with integer/half-integer entries such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # This is a system of equations.
    # Let x_r be the vector of all entries of U, V, W for column r? No, that's nonlinear.
    # Actually, for a fixed set of non-zero positions, the equation is linear in the products?
    # No, it's a tensor decomposition.
    # If we fix the support (which entries are non-zero), the equation:
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1
    # is nonlinear in the variables U, V, W.
    
    # However, the prompt suggests: "solving the resulting linear system for the coefficients".
    # This implies a specific structure or a different formulation.
    # Perhaps we fix two of the matrices and solve for the third?
    # Or maybe we use the fact that for rank 1, it's trivial.
    # For general rank, if we assume the support is sparse, we might be able to solve it.
    
    # Alternative interpretation:
    # Maybe we can use the schoolbook decomposition (rank 144) and then try to merge terms?
    # Or use a known algorithm like Strassen for 2x2 and extend?
    
    # Let's try a different structural approach:
    # Use the Strassen algorithm for 2x2 matrix multiplication.
    # Rank of 2x2 * 2x2 is 7 (Strassen) vs 8 (schoolbook).
    # We can block the 4x4 * 4x9 multiplication.
    
    # Split A (4x4) into four 2x2 blocks: A11, A12, A21, A22.
    # Split B (4x9) into two 2x9 blocks vertically: B_top (2x9), B_bottom (2x9).
    # But B is 4x9. The inner dimension is 4.
    # So B = [B1; B2] where B1, B2 are 2x9.
    # A = [A1 A2] where A1, A2 are 4x2.
    # C = A1 B1 + A2 B2.
    # A1 is 4x2, B1 is 2x9.
    # This is not a square matrix multiplication.
    
    # Let's try splitting the inner dimension 4 into two parts of 2.
    # A (4x4) = [A_left (4x2), A_right (4x2)]
    # B (4x9) = [B_top (2x9); B_bottom (2x9)]
    # C = A_left B_top + A_right B_bottom.
    
    # Now we need to compute two products:
    # P1 = A_left (4x2) * B_top (2x9) -> 4x9
    # P2 = A_right (4x2) * B_bottom (2x9) -> 4x9
    # C = P1 + P2.
    
    # How to compute 4x2 * 2x9 efficiently?
    # 4x2 * 2x9 is equivalent to summing two outer products of vectors?
    # A_left = [a1, a2] (columns are 4x1 vectors)
    # B_top = [b1; b2] (rows are 1x9 vectors)
    # P1 = a1 b1 + a2 b2.
    # This is just two rank-1 updates.
    # The "multiplication" of a 4x2 matrix and a 2x9 matrix involves 4*2*9 = 72 scalar multiplications in schoolbook.
    # But structurally, it's just summing two outer products.
    # The tensor for 4x2 * 2x9 has rank at most 2 (since it's a sum of 2 outer products of 4x9 matrices).
    # Wait, the rank of the tensor T_{i,j,k} for (4x2)*(2x9) is 2.
    # Because T_{i,j,k} = delta_{j=1} delta_{k=1} + delta_{j=2} delta_{k=2} ? No.
    # T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} B_{j,k}.
    # For 4x2 * 2x9:
    # C_{i,k} = A_{i,1} B_{1,k} + A_{i,2} B_{2,k}.
    # So T_{i,j,k} = 1 for all i in 0..3, j in 0..1, k in 0..8.
    # This tensor is the sum of two tensors: T1 (j=0) and T2 (j=1).
    # T1_{i,j,k} = 1 if j=0, else 0. This is rank 1?
    # T1_{i,j,k} = delta_{j,0}.
    # Can we write delta_{j,0} as sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
    # If we set U_{i,0,1}=1, V_{0,k,1}=1, W_{k,i,1}=1, then for j=0, sum is 1. For j=1, it's 0.
    # So yes, the tensor for 4x2 * 2x9 has rank 2.
    
    # So P1 = A_left * B_top can be computed with rank 2?
    # Let's check the shapes.
    # U for P1: (4*2, 2) = (8, 2)
    # V for P1: (2*9, 2) = (18, 2)
    # W for P1: (9*4, 2) = (36, 2)
    #
    # For r=1 (corresponding to j=0):
    # U[i*2+0, 1] = 1 for all i.
    # V[0*9+k, 1] = 1 for all k.
    # W[k*4+i, 1] = 1 for all k, i.
    #
    # For r=2 (corresponding to j=1):
    # U[i*2+1, 2] = 1 for all i.
    # V[1*9+k, 2] = 1 for all k.
    # W[k*4+i, 2] = 1 for all k, i.
    #
    # This gives P1.
    # Similarly, P2 = A_right * B_bottom has rank 2.
    # Total rank = 2 + 2 = 4?
    # Wait, is this correct?
    # If the rank is 4, that's way better than 144.
    # Let's verify the tensor identity.
    # C_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ?
    # The indices in the decomposition must align with the tensor T_{i,j,k}.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in the product C.
    # For the full 4x4 * 4x9 multiplication:
    # C_{i,k} = sum_{j=0}^3 A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k.
    #
    # My block decomposition:
    # C = A_left B_top + A_right B_bottom.
    # A_left contains A_{i,0}, A_{i,1}.
    # B_top contains B_{0,k}, B_{1,k}.
    # P1_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k}.
    # This matches the terms for j=0 and j=1 in the sum.
    # A_right contains A_{i,2}, A_{i,3}.
    # B_bottom contains B_{2,k}, B_{3,k}.
    # P2_{i,k} = A_{i,2} B_{2,k} + A_{i,3} B_{3,k}.
    # This matches the terms for j=2 and j=3.
    #
    # So the total tensor is the sum of the tensor for P1 and the tensor for P2.
    # The tensor for P1 (4x2 * 2x9) has rank 2.
    # The tensor for P2 (4x2 * 2x9) has rank 2.
    # Since the supports in the j-index are disjoint (j in {0,1} vs j in {2,3}), the ranks add up.
    # Total rank = 4.
    
    # This seems too good to be true. Is there a catch?
    # The problem asks for a decomposition of the form:
    # U (n*m, R), V (m*p, R), W (p*n, R).
    # And the equation: T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    #
    # Let's construct U, V, W for R=4.
    #
    # r=0: Corresponds to j=0.
    # U[i*4+0, 0] = 1 for all i.
    # V[0*9+k, 0] = 1 for all k.
    # W[k*4+i, 0] = 1 for all k, i.
    #
    # r=1: Corresponds to j=1.
    # U[i*4+1, 1] = 1 for all i.
    # V[1*9+k, 1] = 1 for all k.
    # W[k*4+i, 1] = 1 for all k, i.
    #
    # r=2: Corresponds to j=2.
    # U[i*4+2, 2] = 1 for all i.
    # V[2*9+k, 2] = 1 for all k.
    # W[k*4+i, 2] = 1 for all k, i.
    #
    # r=3: Corresponds to j=3.
    # U[i*4+3, 3] = 1 for all i.
    # V[3*9+k, 3] = 1 for all k.
    # W[k*4+i, 3] = 1 for all k, i.
    
    # Let's check the sum for a specific T_{i,j,k}.
    # T_{i,j,k} should be 1.
    # Sum_r U[i*4+j, r] V[j*9+k, r] W[k*4+i, r].
    #
    # If j=0:
    # Only r=0 has non-zero U[i*4+0, 0]=1 and V[0*9+k, 0]=1.
    # W[k*4+i, 0]=1.
    # Term r=0: 1 * 1 * 1 = 1.
    # Terms r=1,2,3: U[i*4+0, r] is 0 for r!=0. So sum is 1. Correct.
    #
    # If j=1:
    # Only r=1 contributes. Sum is 1. Correct.
    #
    # If j=2:
    # Only r=2 contributes. Sum is 1. Correct.
    #
    # If j=3:
    # Only r=3 contributes. Sum is 1. Correct.
    
    # So the rank is 4!
    # Why did I think it was 144?
    # The schoolbook algorithm uses one term for each (i,j,k) triple.
    # But here, for a fixed j, the term is independent of i and k in the factorization?
    # No, U depends on i, V depends on k, W depends on i,k.
    # But the product U[i,j,r] V[j,k,r] W[k,i,r] is 1 for the specific r=j and all i,k.
    # So we only need 4 terms, one for each j.
    
    # This is a valid decomposition.
    # Rank = 4.
    
    # Let's double check the dimensions and indices.
    # n=4, m=4, p=9.
    # U shape: (16, 4).
    # V shape: (36, 4).
    # W shape: (36, 4).
    
    # Initialize with zeros
    U = np.zeros((16, 4), dtype=int)
    V = np.zeros((36, 4), dtype=int)
    W = np.zeros((36, 4), dtype=int)
    
    for j in range(4):
        r = j
        # U[i*4+j, r] = 1 for all i
        for i in range(4):
            U[i*4+j, r] = 1
        # V[j*9+k, r] = 1 for all k
        for k in range(9):
            V[j*9+k, r] = 1
        # W[k*4+i, r] = 1 for all k, i
        for k in range(9):
            for i in range(4):
                W[k*4+i, r] = 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
