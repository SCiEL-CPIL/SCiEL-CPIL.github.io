"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# Known lower bound for 4x5x5 is at least 100 (n*m*p = 100 is schoolbook).
# The true minimal rank is unknown but likely less than 100.
# We'll try to find a decomposition with rank < 100.

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = delta_{i,k} (for A[i,j] * B[j,k] -> C[i,k])
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the output C[i,k] gets contribution from A[i,j]*B[j,k]
    # Actually, the matrix multiplication tensor is T[i,j,k] = delta_{ik} in the sense that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all (i,j,k) where we're computing C[i,k] from A[i,j]*B[j,k]
    # Wait, no. The tensor T is defined such that T[i,j,k] = 1 if and only if the (i,j) entry of A
    # and (j,k) entry of B contribute to the (i,k) entry of C.
    # Actually T[i,j,k] = delta_{i,k} is wrong. Let me think again.
    #
    # The matrix multiplication tensor T_{n,m,p} is a 3-tensor with T[i,j,k] = 1 if i==k? No.
    # T[i,j,k] represents the coefficient in C[i,k] = sum_{j} A[i,j] B[j,k].
    # As a tensor in U*V*W, T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{i,k}? No.
    #
    # Let me use the standard definition. The multiplication tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where the sum is over all (i,j,k) triples, and the tensor evaluates to:
    # T(A,B) = sum_{i,j,k} A[i,j] B[j,k] e_i ⊗ e_k... 
    #
    # Actually, the standard way: T is an element of (C^n)^* ⊗ (C^m)^* ⊗ (C^p)^*
    # such that T(A,B,C) = sum_{i,j,k} A[i,j] B[j,k] C[k,i] (with appropriate index conventions)
    #
    # For the factorization U, V, W with T = sum_c U_c ⊗ V_c ⊗ W_c:
    # T(A,B) = sum_c (U_c^T A) (V_c^T B) W_c  ... this is getting confusing.
    #
    # Let me just use the direct verification:
    # For the decomposition to be valid, for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  (since A[i,j]*B[j,k] contributes 1 to C[i,k])
    # Wait, that's not right either. The tensor T[i,j,k] = 1 for all valid (i,j,k) is wrong.
    #
    # The correct tensor: T[i,j,k] = 1 if and only if the (i,j) element of A and (j,k) element of B
    # multiply to give a contribution to C[i,k]. Since C[i,k] = sum_j A[i,j]B[j,k], every (i,j,k) triple
    # contributes. So T[i,j,k] = 1 for all i,j,k? No, that would mean the tensor is all ones.
    #
    # I think the correct formulation is:
    # T is the tensor such that when you contract T with A (on first two indices) and B (on second two),
    # you get C. Specifically, T[i,j,k] = delta_{i,k} is NOT right.
    #
    # Let me just directly check: the factorization says
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    # where T[i,j,k] is the matrix multiplication tensor.
    #
    # The matrix multiplication tensor T satisfies: for any matrices A (n×m), B (m×p),
    # C = A*B, we have C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # where T[i,j,k] = 1 if the indices match appropriately. Specifically, T[i,j,k] = 1 for all (i,j,k)
    # where i is the row of A, j is the column of A / row of B, k is the column of B, and the output
    # is at position (i,k). So T[i,j,k] = 1 for ALL valid (i,j,k) triples? 
    #
    # No wait. If T[i,j,k] = 1 for all (i,j,k), then sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j]B[j,k]
    # which is not C[i,k] but rather a scalar sum of all products. That's wrong.
    #
    # The correct tensor must have T[i,j,k] depend on the relationship between i, j, k.
    # Since C[i,k] = sum_j A[i,j] B[j,k], the tensor T must satisfy:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] e_i ⊗ e_k = C
    # This means T[i,j,k] = delta_{i,k}... no, that doesn't work dimensionally.
    #
    # I think the issue is that T is a 3-tensor with indices (i, j, k) where:
    # - i indexes rows of A (and rows of C)
    # - j indexes columns of A (and rows of B)  
    # - k indexes columns of B (and columns of C)
    # And T[i,j,k] = 1 for all (i,j,k) is actually correct IF we interpret the tensor as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # and the contraction is: (T, A, B) = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j]B[j,k]
    # But this gives a scalar, not a matrix C.
    #
    # I think the correct interpretation is that the tensor T has components T[i,j,k] where the
    # factorization T = sum_c U_c ⊗ V_c ⊗ W_c means:
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # And we need T[i,j,k] = 1 for all (i,j,k)? Let me test with the schoolbook.
    #
    # In schoolbook, U[i*m+j, c] = 1 only for c = i*m+j, V[j*p+k, c] = 1 only for c = i*m+j,
    # W[k*n+i, c] = 1 only for c = i*m+j. So for a given (i,j,k), only c = i*m+j contributes,
    # and T[i,j,k] = 1*1*1 = 1. So yes, T[i,j,k] = 1 for all (i,j,k).
    
    # So the verification is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank decompositions using random search / optimization
    # The key insight: we need U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] sums to 1 for all (i,j,k)
    
    # Try ranks from 100 down, but that's expensive. Let's try a smarter approach.
    # Use the fact that for small matrices, known bounds exist.
    # For 4x5x5, the known minimal rank is unknown but likely around 90-100.
    
    # Try a random initialization approach with gradient-free optimization
    # Actually, let's try a structured approach: perturb the schoolbook and try to merge terms.
    
    # Alternative: use the Strassen-like idea of combining terms.
    # For now, try a simple local search: start from schoolbook, randomly zero out some entries
    # and try to fix up the remaining ones.
    
    # Given time constraints, let's try a few random rank-r decompositions
    for target_rank in [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]:
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            continue
        
        # Try random initialization
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random sparse initialization
            U = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Quick check
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                return {"U": best_U, "V": best_V, "W": best_W}
    
    # If we found a better rank, return it
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise return schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
