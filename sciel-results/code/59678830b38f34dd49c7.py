"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.linalg as sla

DIMENSIONS = [4, 5, 6]

def _canonical_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k,
    arranged as a 3D array with dimensions (n, p, m) for CP decomposition."""
    # T[i, k, j] = 1 for all i in [0,n), k in [0,p), j in [0,m)
    # The multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor entry is T[i, k, j] = 1
    return np.ones((n, p, m))

def _cp_decompose(tensor, rank):
    """CP decomposition via ALS (Alternating Least Squares)."""
    n, p, m = tensor.shape
    # Initialize factors randomly
    np.random.seed(42)
    A = np.random.randn(n, rank)
    B = np.random.randn(p, rank)
    C = np.random.randn(m, rank)
    
    tol = 1e-10
    max_iter = 1000
    
    for iteration in range(max_iter):
        # Update A: fix B, C
        # tensor[i,k,j] = sum_r A[i,r] B[k,r] C[j,r]
        # For each i: sum_{k,j} tensor[i,k,j] B[k,:] C[j,:] = A[i,:] @ diag(...) 
        # More precisely: A[i,:] = (sum_{k,j} tensor[i,k,j] B[k,:] * C[j,:]) / (sum_{k,j} (B[k,:] * C[j,:])^2)
        # Let's use the standard ALS update:
        
        # Update A
        BC = np.einsum('kr,jr->kj r', B, C)  # BC[k,j,r] = B[k,r]*C[j,r]
        # Actually, let's do it more carefully
        # For A: tensor[i,k,j] = sum_r A[i,r] * (B[k,r] * C[j,r])
        # So A[i,:] satisfies: sum_{k,j} tensor[i,k,j] B[k,r] C[j,r] = sum_r A[i,r] sum_{k,j} B[k,r'] C[j,r'] B[k,r] C[j,r]
        # Let's use matrix form:
        # Reshape: T_flat[i, (k,j)] = sum_r A[i,r] M[(k,j), r] where M[(k,j),r] = B[k,r]*C[j,r]
        # M has shape (p*m, rank)
        M = np.einsum('kr,jr->kjr', B, C).reshape(p*m, rank)
        T_flat = tensor.reshape(n, p*m)
        # A[i,:] = T_flat[i,:] M^{-1} (pseudo-inverse)
        A_new, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
        A = A_new.T
        
        # Update B
        AC = np.einsum('ir,jr->ijr', A, C).reshape(n*m, rank)
        T_flat_b = tensor.reshape(p, n*m)
        # tensor[i,k,j] -> for B: tensor[k,i,j] = sum_r B[k,r] (A[i,r]*C[j,r])
        # M_b[(i,j), r] = A[i,r]*C[j,r], shape (n*m, rank)
        M_b = np.einsum('ir,jr->ijr', A, C).reshape(n*m, rank)
        T_flat_b = np.einsum('ikj->kij', tensor).reshape(p, n*m)
        B_new, _, _, _ = np.linalg.lstsq(M_b, T_flat_b.T, rcond=None)
        B = B_new.T
        
        # Update C
        AB = np.einsum('ir,kr->ikr', A, B).reshape(n*p, rank)
        # tensor[i,k,j] -> for C: tensor[j,i,k] = sum_r C[j,r] (A[i,r]*B[k,r])
        M_c = np.einsum('ir,kr->ikr', A, B).reshape(n*p, rank)
        T_flat_c = np.einsum('ikj->jki', tensor).reshape(m, n*p)
        C_new, _, _, _ = np.linalg.lstsq(M_c, T_flat_c.T, rcond=None)
        C = C_new.T
        
        # Check convergence
        if iteration > 0:
            recon = np.einsum('ir,kr,jr->ikj', A, B, C)
            err = np.linalg.norm(recon - tensor) / np.linalg.norm(tensor)
            if err < tol:
                break
    
    return A, B, C

def _snap_to_halfint(values):
    """Snap values to nearest half-integer (i.e., multiples of 0.5)."""
    return np.round(values * 2) / 2.0

def construct():
    n, m, p = DIMENSIONS
    tensor = _canonical_tensor(n, m, p)
    
    # Try increasing ranks until we get an exact decomposition
    # The theoretical lower bound for matrix multiplication rank is at least max(n*p, m*p, n*m)
    # For 4x5x6: lower bound = max(24, 30, 20) = 30
    # Schoolbook rank = 120
    
    best_rank = n * m * p
    best_factors = None
    
    for rank in range(30, 121):
        try:
            A, B, C = _cp_decompose(tensor, rank)
            # Snap to half-integers
            A_s = _snap_to_halfint(A)
            B_s = _snap_to_halfint(B)
            C_s = _snap_to_halfint(C)
            
            # Verify exactness after snapping
            recon = np.einsum('ir,kr,jr->ikj', A_s, B_s, C_s)
            if np.allclose(recon, tensor, atol=1e-8):
                best_rank = rank
                best_factors = (A_s, B_s, C_s)
                break
        except Exception:
            continue
    
    if best_factors is None:
        # Fallback to schoolbook
        best_rank = n * m * p
        A_s = np.zeros((n, best_rank))
        B_s = np.zeros((p, best_rank))
        C_s = np.zeros((m, best_rank))
        col = 0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    A_s[i, col] = 1
                    B_s[k, col] = 1
                    C_s[j, col] = 1
                    col += 1
        best_factors = (A_s, B_s, C_s)
    
    A_s, B_s, C_s = best_factors
    rank = A_s.shape[1]
    
    # Convert to the required format:
    # U has shape (n*m, rank): U[i*m + j, r] = ?
    # The tensor is T[i,k,j] with i in [0,n), k in [0,p), j in [0,m)
    # U corresponds to A (shape n x rank) but we need n*m entries
    # V corresponds to B (shape p x rank) but we need m*p entries  
    # W corresponds to C (shape m x rank) but we need p*n entries
    
    # Wait, let me re-read the contract:
    # U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for the multiplication tensor
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    
    # So we need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    
    # Let me redefine:
    # U[i*m+j, r] corresponds to A[i,j] part
    # V[j*p+k, r] corresponds to B[j,k] part  
    # W[k*n+i, r] corresponds to C[k,i] part (but with trace convention W[k*n+i, r])
    
    # The CP decomposition I did was: T[i,k,j] = sum_r A[i,r] B[k,r] C[j,r]
    # So: U[i*m+j, r] should be... hmm, the indices are coupled.
    
    # Let me just construct it directly:
    # For the schoolbook-like approach but with lower rank, I need to find
    # U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
    # such that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # This is a tensor decomposition of the tensor T[i,j,k] = 1
    # Let me just use the schoolbook for safety:
    rank_final = n * m * p
    U = [[0] * rank_final for _ in range(n * m)]
    V = [[0] * rank_final for _ in range(m * p)]
    W = [[0] * rank_final for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
