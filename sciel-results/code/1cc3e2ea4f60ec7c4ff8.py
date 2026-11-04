"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check if the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    # Build the tensor T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 for all valid (i,j,k)
    # Actually, the matrix multiplication tensor is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition must satisfy: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    # Compute the reconstructed tensor
    # T_recon[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can do this efficiently with einsum or by reshaping
    
    # Reshape U, V, W to 3D
    # U has shape (n*m, rank) -> reshape to (n, m, rank)
    # V has shape (m*p, rank) -> reshape to (m, p, rank)
    # W has shape (p*n, rank) -> reshape to (p, n, rank)
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T_recon[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    T_recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # The target tensor is all ones
    target = np.ones((n, m, p))
    
    # Check exact equality (with small tolerance for floating point)
    return np.allclose(T_recon, target, atol=1e-9)

def _verify_fast(U, V, W):
    """Faster verification using random testing."""
    n, m, p = DIMENSIONS
    # Test with a few random matrices
    for _ in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        # Compute A*B directly
        AB_direct = A @ B
        # Compute using decomposition
        # AB[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] * A[i,j]*B[j,k]
        # Wait, that's not right. Let me think again.
        
        # The decomposition says: for the bilinear form, we have
        # AB[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
        # No wait, that's not right either.
        
        # Let me reconsider. The tensor decomposition for matrix multiplication is:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # where T[i,j,k] = 1 for all i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])
        
        # The bilinear form is:
        # (A*B)[i,k] = sum_j A[i,j]*B[j,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
        
        # Wait, I need to be more careful. Let me re-derive.
        # We have: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # And we want: (A*B)[i,k] = sum_c W[k*n+i,c] * f_c(A) * g_c(B)
        # where f_c(A) = sum_j U[i*m+j,c]*A[i,j]... no that doesn't work because i is free.
        
        # Actually, the standard way: 
        # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
        # No, that's not right. Let me look at this more carefully.
        
        # The tensor T is defined such that T[i,j,k] = 1 for all valid indices.
        # The decomposition is T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # Then (A*B)[i,k] = sum_j A[i,j]*B[j,k]*T[i,j,k] = sum_j A[i,j]*B[j,k]*sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # = sum_c W[k,i,c] * (sum_j A[i,j]*U[i,j,c]) * (sum_j B[j,k]*V[j,k,c])
        
        # So the algorithm is:
        # For each c:
        #   f_c = A * U_c  (where U_c is the c-th column of U, reshaped to (n,m))
        #   g_c = V_c * B  (where V_c is the c-th column of V, reshaped to (m,p))
        #   (A*B)[i,k] += W[k*n+i,c] * f_c[i] * g_c[k]
        
        # Let's implement this:
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        
        # f_c[i] = sum_j U3[i,j,c]*A[i,j] -> this is element-wise multiply then sum over j
        # Actually: f_c = np.einsum('ijc,ij->ic', U3, A)
        # g_c[k] = sum_j V3[j,k,c]*B[j,k] -> g_c = np.einsum('jkc,jk->kc', V3, B)
        # Then AB_decomp[i,k] = sum_c W3[k,i,c]*f_c[i]*g_c[k]
        # = np.einsum('kic,ic,kc->ik', W3, f, g)
        
        f = np.einsum('ijc,ij->ic', U3, A)
        g = np.einsum('jkc,jk->kc', V3, B)
        AB_decomp = np.einsum('kic,ic,kc->ik', W3, f, g)
        
        if not np.allclose(AB_direct, AB_decomp, atol=1e-8):
            return False
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Start with schoolbook decomposition
    U, V, W = _schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by finding linear dependencies
    # The idea: if columns of the "coefficient tensor" are linearly dependent, we can reduce rank
    
    # For now, return the schoolbook decomposition as a baseline
    # A proper optimization would use tensor decomposition algorithms
    
    # Check if we have time for a basic optimization
    if time.time() < deadline:
        # Try to find a lower rank decomposition using iterative methods
        # This is a simplified approach; a real implementation would use more sophisticated methods
        
        # For now, just return the schoolbook decomposition
        pass
    
    # Convert to lists for the expected format
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
