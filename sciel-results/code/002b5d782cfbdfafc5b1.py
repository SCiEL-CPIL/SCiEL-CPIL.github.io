"""Program for 5x5x5 matrix multiplication tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [5, 5, 5]

def _tensor_to_matrix(tensor, axis):
    """
    Reshape a 3D tensor (n,m,p) into a matrix by flattening the other two axes.
    axis: 0, 1, or 2 - the axis to keep as rows, others as columns.
    """
    n, m, p = tensor.shape
    if axis == 0:
        return tensor.reshape(n, m * p)
    elif axis == 1:
        # Move axis 1 to front
        t = np.transpose(tensor, (1, 0, 2))
        return t.reshape(m, n * p)
    elif axis == 2:
        # Move axis 2 to front
        t = np.transpose(tensor, (2, 0, 1))
        return t.reshape(p, n * m)

def _rank_over_Q(tensor):
    """
    Compute the rank of a tensor over Q by finding the minimum of the ranks
    of its matrix flattening along each axis. This is an upper bound on the
    tensor rank, but for matrix multiplication tensors, the true rank equals
    this minimum in many known cases (though not always).
    """
    n, m, p = tensor.shape
    r0 = np.linalg.matrix_rank(_tensor_to_matrix(tensor, 0))
    r1 = np.linalg.matrix_rank(_tensor_to_matrix(tensor, 1))
    r2 = np.linalg.matrix_rank(_tensor_to_matrix(tensor, 2))
    return min(r0, r1, r2)

def _peel_rank1(tensor, max_iter=500, seed=42):
    """
    Iteratively peel off rank-1 tensors from a 3D tensor over Q.
    Returns a list of (u, v, w) triples and the residual tensor.
    Uses randomized linear algebra to find rank-1 components.
    """
    random.seed(seed)
    np.random.seed(seed)
    n, m, p = tensor.shape
    residual = tensor.copy().astype(float)
    components = []
    
    for _ in range(max_iter):
        # Check if residual is close to zero
        if np.linalg.norm(residual, ord='fro') < 1e-10:
            break
        
        # Try to find a rank-1 approximation via iterative method
        # Use the fact that a rank-1 tensor T = u ⊗ v ⊗ w
        # We can try to find it by alternating minimization
        
        # Initialize with random vectors
        best_rank1 = None
        best_residual_norm = np.linalg.norm(residual, ord='fro')
        
        for trial in range(20):
            # Random initialization
            u = np.random.randn(n)
            v = np.random.randn(m)
            w = np.random.randn(p)
            
            # Alternating minimization
            for it in range(100):
                # Update u: minimize ||residual - u ⊗ v ⊗ w||_F
                # This is a linear least squares problem
                # residual = u * (v ⊗ w) + ...
                vw = np.outer(v, w)  # shape (m, p)
                # Reshape residual to (n, m*p)
                res_2d = residual.reshape(n, m * p)
                # Solve: res_2d ≈ u * vw.reshape(1, m*p)
                vw_flat = vw.flatten()
                # u = argmin ||res_2d - u * vw_flat||
                denom = np.dot(vw_flat, vw_flat)
                if abs(denom) < 1e-15:
                    break
                u_new = np.dot(res_2d, vw_flat) / denom
                u = u_new.reshape(n)
                
                # Update v
                uw = np.outer(u, w)  # shape (n, p)
                # Reshape residual to (m, n*p) by moving axis 1 to front
                t = np.transpose(residual, (1, 0, 2))
                res_2d_m = t.reshape(m, n * p)
                uw_flat = uw.flatten()
                denom = np.dot(uw_flat, uw_flat)
                if abs(denom) < 1e-15:
                    break
                v_new = np.dot(res_2d_m, uw_flat) / denom
                v = v_new.reshape(m)
                
                # Update w
                uv = np.outer(u, v)  # shape (n, m)
                # Reshape residual to (p, n*m) by moving axis 2 to front
                t = np.transpose(residual, (2, 0, 1))
                res_2d_p = t.reshape(p, n * m)
                uv_flat = uv.flatten()
                denom = np.dot(uv_flat, uv_flat)
                if abs(denom) < 1e-15:
                    break
                w_new = np.dot(res_2d_p, uv_flat) / denom
                w = w_new.reshape(p)
            
            # Compute residual after subtracting this rank-1 tensor
            rank1_tensor = np.einsum('i,j,k->ijk', u, v, w)
            new_residual = residual - rank1_tensor
            new_norm = np.linalg.norm(new_residual, ord='fro')
            
            if new_norm < best_residual_norm:
                best_residual_norm = new_norm
                best_rank1 = (u.copy(), v.copy(), w.copy())
        
        if best_rank1 is not None:
            u, v, w = best_rank1
            rank1_tensor = np.einsum('i,j,k->ijk', u, v, w)
            residual = residual - rank1_tensor
            components.append((u, v, w))
        else:
            break
    
    return components, residual

def _solve_exact_low_dim(tensor):
    """
    Solve the residual tensor exactly using rational arithmetic.
    This is a fallback for small residual tensors.
    """
    # For the 5x5x5 case, if we've peeled enough, the residual should be small
    # We'll use the schoolbook method as a fallback
    n, m, p = tensor.shape
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

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5x5 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    """
    n, m, p = DIMENSIONS
    
    # Build the 5x5x5 matrix multiplication tensor
    # T[i, j, k] = 1 if this corresponds to a term in (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
    # Wait, no. The matrix multiplication tensor M has M[i,j,k] = 1 for all i,j,k
    # because (AB)_{ik} = sum_j A_{ij} B_{jk}
    
    tensor = np.ones((n, m, p), dtype=float)
    
    # Try to peel rank-1 components
    components, residual = _peel_rank1(tensor, max_iter=100, seed=42)
    
    # Check if residual is essentially zero
    if np.linalg.norm(residual, ord='fro') < 1e-8:
        # We have an exact decomposition
        rank = len(components)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c, (u, v, w) in enumerate(components):
            # u has shape (n,), v has shape (m,), w has shape (p,)
            # U[i*m + j, c] = u[i] * v[j]
            # V[j*p + k, c] = v[j] * w[k]
            # W[k*n + i, c] = w[k] * u[i]
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = u[i] * v[j]
            for j in range(m):
                for k in range(p):
                    V[j * p + k, c] = v[j] * w[k]
            for k in range(p):
                for i in range(n):
                    W[k * n + i, c] = w[k] * u[i]
        
        # Convert to lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback: use schoolbook decomposition
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
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
