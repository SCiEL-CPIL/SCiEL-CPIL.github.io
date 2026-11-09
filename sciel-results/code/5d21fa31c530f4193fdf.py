"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if i,j,k are all valid, 0 otherwise
    # Actually, the tensor for matrix multiplication: T_{i,j,k} = delta_{i,j,k} in some sense
    # For C = A*B: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (decomposition terms)
    # The decomposition: C[i,k] = sum_r U[i*m+j_i, r] * V[j_i*p+k, r] * W[k*n+i, r]
    # More precisely, for each rank-1 term r:
    #   It contributes U[i*m+j, r] * V[j*p+k, r] to C[i,k]
    # The full sum over r should give: C[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: for all i,k: sum_r sum_j U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] ... 
    # Wait, I need to re-read the convention.
    
    # W[k*n+i, r] - so W is indexed by (k, i) -> flattened as k*n+i
    # U[i*m+j, r] - indexed by (i, j) -> flattened as i*m+j  
    # V[j*p+k, r] - indexed by (j, k) -> flattened as j*p+k
    
    # The decomposition says: for the matrix multiplication tensor,
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should reconstruct the product
    # Actually, the standard form is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) (sum_j V[j*p+k,r] B[j,k]) * W[k*n+i,r]
    # No, that's not right either.
    
    # Let me think again. The tensor T has components T[i,j,k] for the multiplication.
    # The rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r]
    # And T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For matrix multiplication, T[i,j,k] = 1 if the indices correspond to a valid multiplication term
    # Actually, the matrix multiplication tensor is: T[i,j,k] = δ (the identity)
    # More precisely, for C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: T_{i,k}^{j} or something... 
    
    # I think the correct interpretation is:
    # The multiplication tensor M has M[i,j,k] = 1 for all valid i,j,k (it's the "structure tensor")
    # No, that's not right.
    
    # Let me just verify numerically: pick random A and B, compute C = A@B,
    # then compute C' using the decomposition and check C == C'.
    
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Hmm, that doesn't look right dimensionally.
    
    # Actually, I think the correct formula is:
    # C[i,k] = sum_r U[i*m + j1, r] * V[j2*p + k, r] * W[k*n + i, r] * A[i,j1] * B[j2,k]
    # summed over j1, j2, r. But that's 3 sums.
    
    # I think the standard decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # No wait, that would be sum_r [ (U_r · A_i) * (V_r · B_k) * W_r[i,k] ]
    
    # Let me just test with the trivial decomposition.
    
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for i in range(n):
        for k in range(p):
            val = 0.0
            for r in range(rank):
                # Sum over j for U part
                u_part = sum(U_arr[i*m + j, r] * A[i, j] for j in range(m))
                # Sum over j for V part  
                v_part = sum(V_arr[j*p + k, r] * B[j, k] for j in range(m))
                val += u_part * v_part * W_arr[k*n + i, r]
            C_computed[i, k] = val
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition
    U, V, W = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank by attempting to eliminate terms
    # Strategy: try rank from lower values up, use numerical optimization
    
    # For now, return the trivial decomposition as baseline
    # Try a few specific ranks that might work
    target_ranks = [45, 48, 50, 52, 53]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition with target_rank
        # Use a simple approach: start with trivial, try to merge terms
        # This is complex, so for now just return trivial if we can't find better
        
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
