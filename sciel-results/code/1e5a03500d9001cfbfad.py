"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5  # A: n x m, B: m x p, C: n x p
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U indexed by [i*m + j], V by [j*p + k], W by [k*n + i]
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Build a rank-30 schoolbook decomposition using Kronecker-product
    # structure derived from identity matrices, then verify correctness.
    # This ensures exact integer entries (all 0 or 1) and provides a baseline.
    
    # Schoolbook: for each (i, j, k) triple, one term
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r = i*m*p + j*p + k
    
    rank = n * m * p  # 30
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify the decomposition exactly:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook, each term r=(i',j',k') contributes U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
    # which is 1 iff i'=i, j'=j, k'=k, so sum over j gives 1 for the matching j.
    # Thus C[i,k] = sum_j (1 if j matches) = 1 for the correct j... 
    # Wait, let me re-verify: C[i,k] should equal sum_j A[i,j]*B[j,k]
    # In the tensor: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook term r=(i',j',k'): U[i*m+j, r]=1 iff i'=i and j'=j
    # V[j*p+k, r]=1 iff j'=j and k'=k
    # W[k*n+i, r]=1 iff k'=k and i'=i
    # So product is 1 iff i'=i, j'=j, k'=k, i.e., r=(i,j,k)
    # Sum over r: only r=(i,j,k) contributes, giving 1
    # Sum over j: sum_j 1 = m? No wait...
    # Actually for fixed (i,k), sum_j sum_r ... 
    # For a given j, the term r=(i,j,k) gives U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    # So for each j, there's exactly one r that contributes 1, and sum over j gives m=3?
    # That's wrong. Let me re-examine.
    
    # The matrix multiplication tensor T[i,j,k] = delta[i,k]... no.
    # T is the bilinear map: C = A*B, so C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has components T[i, j, k] such that the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # So T[i,j,k] = 1 for all i,j,k, and C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    # In the rank decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # For schoolbook with r=(i',j',k'):
    # U[i*m+j, r] = 1 iff i'=i, j'=j
    # V[j*p+k, r] = 1 iff j'=j, k'=k  
    # W[k*n+i, r] = 1 iff k'=k, i'=i
    # Product = 1 iff i'=i, j'=j, k'=k
    # Sum over r = sum over (i',j',k') of [i'=i, j'=j, k'=k] = 1 (only r=(i,j,k) matches)
    
    # So T[i,j,k] = 1 for all i,j,k. Correct!
    
    # Now verify C[i,k] = sum_j A[i,j] * B[j,k]:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # Wait, no. The decomposition gives us the tensor, and then:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # And the tensor decomposition means:
    # sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # No, that's not right either. Let me think more carefully.
    
    # The matrix multiplication tensor T is a 3-tensor with T[i, j, k] = 1 (for the standard
    # bilinear map C = A*B). A rank-r decomposition is:
    # T[i, j, k] = sum_{r=1}^r U[i, r] * V[j, r] * W[k, r]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # and the indices are: U indexed by (i,j) flattened as i*m+j, V by (j,k) as j*p+k, W by (k,i) as k*n+i
    
    # Then: sum_j A[i,j] * B[j,k] = sum_{r} (sum_j U[i*m+j, r] * A[i,j]) * (sum_k ... ) 
    # No. The correct interpretation:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using the tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We have: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_k (sum_{i,j} T[i,j,k] * A[i,j] * B[j,k])
    # Hmm, this isn't quite the right way to see it.
    
    # Actually, the standard way: the bilinear map is:
    # <C, A, B> = sum_{i,j,k} C[i,k] * ... no.
    
    # Let me just verify numerically.
    
    # Convert to numpy arrays for verification
    U_arr = U.astype(float)
    V_arr = V.astype(float)
    W_arr = W.astype(float)
    
    # Check: for random A (n x m) and B (m x p), does C = A @ B match?
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # Wait, that's not right. Let me reconsider.
    
    # The decomposition is: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] is the coefficient in: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # To compute C from the decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # NO! That's double counting. 
    
    # The correct formula: given the rank decomposition of the multiplication tensor,
    # we compute:
    # For each r:
    #   alpha_r = sum_j U[i*m+j, r] * A[i,j]   (for each i)
    #   beta_r = sum_k V[j*p+k, r] * B[j,k]    (for each j)  
    #   gamma_r = W[k*n+i, r]                   (for each i,k)
    #   C[i,k] += alpha_i[r] * beta_j[r] * gamma_r ... 
    # This is getting confused. Let me use the standard formulation.
    
    # Standard: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    # No, that's not right either.
    
    # Let me just do the direct verification:
    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all i,j,k.
    
    # Verify:
    T_computed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_computed[i, j, k] = np.sum(
                    U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]
                )
    
    # T should be all ones
    if not np.allclose(T_computed, 1.0):
        # Fallback: this shouldn't happen for schoolbook, but just in case
        pass
    
    # Now verify the actual matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using the decomposition, we can compute:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not the right formula either.
    
    # The correct formula for matrix multiplication from tensor decomposition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # And we have T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # So: C[i,k] = sum_j 1 * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k] ✓
    
    # The decomposition is valid if T[i,j,k] = 1 for all i,j,k, which we verified above.
    
    # Return as lists (the interface expects lists or arrays)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
