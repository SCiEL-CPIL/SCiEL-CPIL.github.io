"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 by 3x7 matrix multiplication.
    Uses a non-linear optimization approach to find a decomposition of rank R < n*m*p.
    """
    n, m, p = DIMENSIONS
    target_rank = 30  # Try to find a decomposition of rank 30
    
    # Define the tensor T_{i,j,k} = 1 if it corresponds to a term in the sum?
    # The tensor for matrix multiplication C_{i,k} = sum_j A_{i,j} B_{j,k}
    # is T_{i,j,k} = 1 for all i,j,k?
    # No, the decomposition is C_{i,k} = sum_r U_{i,r} V_{k,r} W_{j,r}?
    # Standard convention:
    # C_{i,k} = sum_r U_{i,r} V_{k,r} W_{j,r} is not right.
    # The tensor is T_{i,j,k}.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We want C_{i,k} = sum_r U_{i,r} V_{k,r} W_{j,r}? No.
    # The factors are U (n x R), V (m x R), W (p x R)?
    # The task says: U (n*m, R), V (m*p, R), W (p*n, R).
    # This implies the indices are flattened.
    # U_{i*m+j, r}, V_{j*p+k, r}, W_{k*n+i, r}.
    # The term is U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}.
    # Sum over r.
    # This sum must equal the matrix multiplication tensor.
    # What is the matrix multiplication tensor?
    # It is the tensor T such that T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # Actually, the "tensor" is the object being decomposed.
    # For matrix multiplication, the tensor is T_{i,j,k} = 1 for all i,j,k?
    # No, the tensor is defined by the structure of the multiplication.
    # The multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be written as C_{i,k} = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} \delta_{?}
    # Actually, the standard tensor decomposition of matrix multiplication is:
    # M_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is not a tensor decomposition in the sense of $T = \sum u \otimes v \otimes w$ where $T_{ijk}$ are the coefficients.
    # The "tensor" is the multiplication map.
    # The rank is the number of scalar multiplications.
    # The schoolbook decomposition has terms:
    # For each j, we multiply A_{:,j} and B_{j,:}.
    # This involves n*p scalar multiplications.
    # Total n*m*p.
    
    # The interface asks for U, V, W.
    # U is (n*m, R). V is (m*p, R). W is (p*n, R).
    # The decomposition is:
    # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = T_{i,j,k}
    # What is T_{i,j,k}?
    # It must be the tensor that represents the multiplication.
    # The multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be seen as a bilinear map.
    # The tensor T_{i,j,k} is 1 if the indices match the multiplication pattern?
    # Actually, for the standard multiplication, T_{i,j,k} = 1 for all i,j,k?
    # No.
    # Let's check the schoolbook code.
    # U[i*m+j, col] = 1.0
    # V[j*p+k, col] = 1.0
    # W[k*n+i, col] = 1.0
    # Sum over col (r).
    # For a fixed i,j,k, the sum is 1 if there is a col such that
    # U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1.
    # In schoolbook, col corresponds to (i,j,k).
    # So for a specific (i,j,k), only the term where col=(i,j,k) is non-zero.
    # So the sum is 1.
    # Thus, the target tensor T_{i,j,k} is 1 for all i,j,k.
    
    # So we need to find U, V, W such that:
    # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k.
    
    # This is a system of n*m*p = 42 equations.
    # Variables: R*(n*m + m*p + p*n) = R*(6+21+14) = 41R.
    
    # We can solve this using least_squares.
    
    def residual(x):
        # x is flattened [U_flat, V_flat, W_flat]
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U+size_V]
        W_flat = x[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape((n*m, target_rank))
        V = V_flat.reshape((m*p, target_rank))
        W = W_flat.reshape((p*n, target_rank))
        
        # Compute sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
        # We need to check if it equals 1 for all i,j,k.
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], np.dot(V[v_idx, :], W[w_idx, :]))
                    # Wait, dot(A, dot(B, C)) is not sum_r A_r B_r C_r.
                    # sum_r U_r V_r W_r = U @ diag(V) @ W? No.
                    # It is the Hadamard product sum.
                    # val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    res[i, j, k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        
        # Target is 1.
        return (res - 1.0).flatten()
        
    # Initial guess: Schoolbook
    init_U = np.zeros((n*m, target_rank))
    init_V = np.zeros((m*p, target_rank))
    init_W = np.zeros((p*n, target_rank))
    
    # Fill first n*m*p columns with schoolbook?
    # We only have target_rank columns.
    # If target_rank < n*m*p, we can't use schoolbook directly.
    # We can use a random initialization.
    
    np.random.seed(42)
    x0 = np.random.randn(target_rank * (n*m + m*p + p*n))
    
    # Solve
    result = least_squares(residual, x0, method='lm', max_nfev=1000)
    
    if result.success and np.allclose(residual(result.x), 0.0, atol=1e-9):
        U_flat, V_flat, W_flat = result.x[:n*m], result.x[n*m:n*m+m*p], result.x[n*m+m*p:]
        U = U_flat.reshape((n*m, target_rank)).tolist()
        V = V_flat.reshape((m*p, target_rank)).tolist()
        W = W_flat.reshape((p*n, target_rank)).tolist()
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
