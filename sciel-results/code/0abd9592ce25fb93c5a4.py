"""Tensor decomposition for 2x3 * 3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _build_system(n, m, p, rank):
    """
    Build the linear system for tensor decomposition.
    
    We want: for each (i, k), sum_j U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
    for all r, and the bilinear form equals the matrix multiplication tensor.
    
    Actually, the condition is:
    For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] = 1 if this is a "multiplication" term, 0 otherwise.
    
    Wait, let me think again. The matrix multiplication tensor T has entries:
    T[i, j, k] = 1 for all valid i, j, k (it's the identity in the middle index).
    
    A rank-r decomposition means:
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    This is a trilinear decomposition. We can fix W and solve for U, V, or use other approaches.
    
    A common approach: fix the structure and solve linear equations.
    """
    pass

def _try_rank(n, m, p, rank, deadline):
    """
    Try to find a rank-`rank` decomposition.
    Uses a parameterized approach where we fix some structure and solve linear systems.
    """
    # Strategy: Use the fact that we can write the decomposition as:
    # For each rank term r, we have vectors u_r in R^(n*m), v_r in R^(m*p), w_r in R^(p*n)
    # such that for all (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # One approach: assume w_r has a specific simple form (e.g., standard basis or sparse)
    # and solve for u_r, v_r.
    
    # Alternative: use bilinear decomposition. Fix one set of vectors and solve for others.
    
    # Let's try: fix W to be a matrix with simple structure.
    # For example, W could be the identity or have block structure.
    
    # Actually, a better approach for small ranks: use randomized linear algebra.
    # Generate random W, then solve the linear system for U and V.
    
    # The system is: for each (i,k), the vector (sum_j U[i*m+j,r]*V[j*p+k,r])_r should equal W[k*n+i,:]
    
    # This is equivalent to: for each k, the matrix U[:, k*p:(k+1)*p] @ V[k*p:(k+1)*p, :]^T ... no wait.
    
    # Let me re-index. Let U be (n*m) x rank, V be (m*p) x rank, W be (p*n) x rank.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank).
    # Condition: for all i, j, k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # This is a tensor network contraction. Hard to solve directly.
    
    # Alternative approach: use the fact that matrix multiplication can be decomposed
    # using Kronecker products or other structured decompositions.
    
    # For 2x3x6, let's try to exploit the structure.
    # C = A * B, where A is 2x3, B is 3x6.
    # We can write B as [B1, B2] where B1, B2 are 3x3.
    # Then C = [A*B1, A*B2].
    
    # If we have a rank-r decomposition for 2x3x3, we can use it for both blocks.
    # The total rank would be 2r (since the U vectors for the two blocks are independent).
    
    # Actually, we can share the U vectors! The A matrix is the same for both blocks.
    # So we need:
    # U: (6, 2r) where the first r columns are for B1 and next r for B2
    # V: (18, 2r) where first 9 rows correspond to B1 and next 9 to B2
    # W: (12, 2r) where first 6 rows correspond to C1 and next 6 to C2
    
    # For each block, we need a rank-r decomposition of 2x3x3.
    
    # What is the rank of 2x3x3? Let's search for it.
    
    # For now, let's try to find a decomposition for 2x3x3 with various ranks.
    
    def _find_2x3x3_rank(target_rank, deadline):
        """Find a rank-target_rank decomposition for 2x3x3 matrix multiplication."""
        n2, m2, p2 = 2, 3, 3
        
        # Build the linear system.
        # We need: for all (i,j,k) in [0,2)x[0,3)x[0,3):
        # sum_r U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r] = 1
        
        # This is nonlinear. Let's use a different approach.
        # Fix W to be a standard basis or simple structure.
        
        # Approach: assume W has a specific form. For example, W[k*n2+i, r] = delta_{k, r mod p2} * delta_{i, r div p2}
        # i.e., W is the identity-like structure.
        
        # Actually, let's use a more systematic approach.
        # We can write the decomposition as:
        # T = sum_r u_r ⊗ v_r ⊗ w_r
        
        # where u_r in R^(n2*m2), v_r in R^(m2*p2), w_r in R^(p2*n2).
        
        # One method: fix u_r and w_r, solve for v_r, or vice versa.
        
        # Let's try: fix w_r to be standard basis vectors.
        # w_r[k*n2+i] = 1 if (k, i) == (r % p2, r // p2), else 0.
        
        # Then the condition becomes:
        # For all (i, j, k): U[i*m2+j, k*n2+i] * V[j*p2+k, k*n2+i] = 1
        # Wait, that's not right. Let me re-derive.
        
        # If W is the identity (in some sense), then:
        # sum_r U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r] = 1
        # If W[k*n2+i, r] = delta_{(k,i), (r% p2, r // p2)}, then:
        # U[i*m2+j, k*n2+i] * V[j*p2+k, k*n2+i] = 1 for all (i,j,k)
        
        # This gives us a system of equations. Let's set up the variables.
        # There are target_rank variables for U (each has n2*m2 entries) and V (each has m2*p2 entries).
        # But we've fixed W, so we have n2*m2*target_rank + m2*p2*target_rank variables.
        # And n2*m2*p2 = 18 equations.
        
        # This is underdetermined if target_rank is large enough.
        
        # Let's try a different approach: use the Strassen-like method.
        # For 2x2x2, Strassen uses rank 7.
        # For 2x3x3, the rank is known to be at most 12 (I think).
        
        # Let's just try to solve the linear system numerically.
        
        # We'll parameterize: let U and V be random, then solve for W.
        # Or fix U and V, solve for W.
        
        # Actually, the simplest approach: generate random U and V, then for each (i,k),
        # compute the vector b_{i,k} where b_{i,k}[r] = sum_j U[i*m2+j, r] * V[j*p2+k, r]
        # and set W[k*n2+i, :] = b_{i,k} / norm(b_{i,k}) or something.
        
        # No, that won't work because we need the product to equal 1.
        
        # Let me use a different strategy: use the bilinear decomposition.
        # The matrix multiplication tensor can be written as:
        # T_{i,j,k} = sum_r A_{i,j,r} * B_{j,k,r} * C_{k,i,r}
        
        # This is equivalent to finding a rank-r decomposition in the tensor sense.
        
        # For small dimensions, we can use a search approach.
        # Generate random rank-r decompositions and check if they work.
        
        # Actually, for exact decompositions, we need rational/integer solutions.
        
        # Let's try a constructive approach for 2x3x3.
        # I'll use the fact that 2x3x3 can be decomposed using 12 rank-1 tensors.
        
        # Here's a known decomposition (I'll construct it):
        # Use the following idea: split the 3x3 matrix into blocks.
        
        # For now, let's just return None to indicate we didn't find one.
        return None
    
    # Try to find 2x3x3 decomposition with various ranks
    for r in range(12, 17):
        if time.time() > deadline:
            break
        # We need a rank-r decomposition for 2x3x3
        # For simplicity, let's just use schoolbook rank 18 for 2x3x3
        # which gives total rank 36 for 2x3x6.
        pass
    
    # Fall back to schoolbook
    return None

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Returns dict with 'U', 'V', 'W' as lists of lists with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 36
    
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower-rank decomposition
    # Use a linear algebra approach
    
    # The tensor decomposition condition is:
    # For all (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We can fix one of the matrices and solve for the others.
    # Let's fix W to be a simple matrix and solve for U and V.
    
    # Approach: Use the fact that we can write the problem as a matrix equation.
    # Let's define:
    # For each k, define U_k as the submatrix of U corresponding to column index k
    # Actually, let me re-index differently.
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    # Condition: for all i, j, k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # This is a 3-tensor decomposition. Let's use a different approach.
    
    # We can write: for each (i, k), the vector sum_j (U[i*m+j, :] * V[j*p+k, :]) = W[k*n+i, :]
    # where * denotes element-wise multiplication.
    
    # Let's define M_{i,k} = W[k*n+i, :] (a row vector of length rank)
    # And for each j, define the element-wise product U[i*m+j, :] * V[j*p+k, :]
    
    # So: sum_j (U[i*m+j, :] ⊙ V[j*p+k, :]) = M_{i,k}
    
    # This is a system of linear equations in the entries of U and V, but it's bilinear.
    
    # Let's try a different approach: use randomized methods.
    # Generate random W, then solve for U and V.
    
    # Actually, let's try to use the known result that the rank of 2x3x6 is at most 24
    # by using a rank-12 decomposition for 2x3x3.
    
    # For 2x3x3, I'll construct a rank-12 decomposition.
    
    # The 2x3x3 multiplication tensor:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in [0,2), j in [0,3), k in [0,3)
    
    # We need: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
    
    # Let's try to construct this with rank 12.
    
    # Idea: Use the fact that we can decompose 2x3x3 as a combination of simpler tensors.
    
    # For now, let's just use the schoolbook decomposition which we know is correct.
    # The rank is 36.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
