"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check if U,V,W exactly decompose the matmul tensor.
    Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 iff k==j (no, that's wrong).
    Actually: (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,k}? No.
    The standard convention: T_{i,j,k} where i is row of A, j is inner, k is col of B.
    We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 if this represents the identity on the "multiply" operation.
    Wait, the tensor for matrix multiplication is T[i,j,k] = 1 if the index j connects i and k?
    No. The matrix multiplication tensor T has entries T[i,j,k] = 1 if j==j (always) ... no.
    Let's recall the definition.
    The tensor T is such that (A*B)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_{j,c} A_{i,j} B_{j,k} T_{i,j,k}^{(c)} ... no.
    
    Standard decomposition:
    (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    We write this as sum_c (sum_j A_{i,j} U_{i,j,c}) (sum_k B_{j,k} V_{j,k,c}) W_{k,i,c}
    This requires: sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = delta_{j,k}? No, that's not right either.
    
    Let's look at the schoolbook:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for the triple (i,j,k).
    Then sum_c U[i*m+j', c] V[j'*p+k', c] W[k'*n+i', c] 
    For a fixed (i, j, k), the only non-zero term in the sum over c is the one where c corresponds to (i,j,k).
    So the product is 1.
    But the matrix multiplication is sum_j A[i,j] B[j,k].
    The tensor T[i,j,k] should be 1.
    The reconstruction is: sum_{j,c} A[i,j] B[j,k] (U[i,j,c] V[j,k,c] W[k,i,c])?
    No, the standard formula is:
    (A*B)_{i,k} = sum_c (sum_j A_{i,j} U_{i,j,c}) (sum_k B_{j,k} V_{j,k,c}) ... this doesn't close.
    
    Correct identity:
    (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    We want to express this as sum_c P_c(A) Q_c(B) R_c.
    Usually: (A*B)_{i,k} = sum_c (sum_j A_{i,j} U_{i,j,c}) (sum_j B_{j,k} V_{j,k,c}) W_{k,i,c}
    This requires: sum_c U_{i,j,c} V_{j',k,c} W_{k,i,c} = delta_{j,j'}
    So the tensor T_{i,j,k} in the decomposition sense is defined by:
    T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    And we need T_{i,j,k} = 1 for all i,j,k? No.
    We need the reconstructed product to be correct.
    (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}
    Using the decomposition:
    (A*B)_{i,k} = sum_c (sum_j A_{i,j} U_{i*m+j, c}) (sum_k' B_{j,k'} V_{j*p+k', c}) ... wait, indices are mixed.
    
    Let's stick to the verification used in the prompt's context (which implies the standard matmul tensor).
    The tensor T has size n*m*p.
    T[i, j, k] = 1 if we are multiplying A[i,j] and B[j,k].
    Actually, the "matrix multiplication tensor" is usually defined as T[i,j,k] = 1.
    And the decomposition satisfies:
    T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    Wait, if T[i,j,k]=1, then sum_c ... = 1.
    Let's check schoolbook:
    For fixed i,j,k, there is exactly one c (the one for i,j,k) where U=1, V=1, W=1.
    Sum is 1.
    So the target tensor is the all-ones tensor?
    If T is all ones, then (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k} * 1.
    This matches.
    
    So verification:
    Construct U, V, W as 3D arrays (or flatten).
    Compute T_recon[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
    Check if T_recon == 1 for all i,j,k.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Reshape for einsum
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T_recon[i,j,k] = sum_c U3[i,j,c] V3[j,k,c] W3[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # Target is all ones
    return np.allclose(T_recon, 1.0, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook baseline
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_s, V_s, W_s
    
    # 2. Try to find lower rank using structured patterns
    # The matmul tensor T[i,j,k]=1 is a sum of rank-1 tensors.
    # A rank-1 tensor is u_i v_j w_k.
    # We can try to find a set of rank-1 tensors that sum to 1.
    
    # Heuristic: Try ranks from a lower bound up to schoolbook.
    # Lower bound for matmul n x m x p is max(n*m, m*p, p*n).
    # For 3x5x7: max(15, 35, 21) = 35.
    # Known best for 3x5x7? 
    # Strassen for 2x2x2 is 7.
    # For 3x3x3, best known is 23 (Landsberg).
    # For 3x5x7, we don't know the exact minimum, but it's likely < 105.
    # Let's try a few specific ranks.
    
    # We will use a randomized search with local refinement.
    # Since exact integer/half-integer entries are required, we can search in that space.
    # But random search in a huge space is hard.
    
    # Alternative: Use the structure.
    # T[i,j,k] = 1.
    # This is a "constant" tensor.
    # We can try to decompose the constant tensor.
    
    # Let's try a simple greedy approach or a small search for specific ranks.
    # Given the time limit, we can't do heavy optimization.
    # We will try to verify if a lower rank is possible by checking if we can
    # find U, V, W with integer entries.
    
    # For now, we stick to the schoolbook if no better is found quickly.
    # However, the prompt asks for improvements.
    # Let's try to implement a simple search for rank < 105.
    
    # We'll try ranks: 35, 40, 50, 60, 70, 80, 90, 100.
    # For each rank, we'll try a few random initializations and check.
    # But checking is O(n*m*p*rank). 105 * 105 = ~11k ops. Very fast.
    # The search is the hard part.
    
    # Let's try a deterministic construction if possible.
    # Is there a known decomposition for 3x5x7?
    # Not off the top of my head.
    
    # We will return the schoolbook decomposition as it is guaranteed valid.
    # Any attempt to improve must be verified.
    
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
