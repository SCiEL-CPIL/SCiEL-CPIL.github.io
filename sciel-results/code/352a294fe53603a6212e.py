"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # For matrix multiplication: T[i*m+j, j*p+k, k*n+i] = 1
    # Actually, the tensor for matrix multiplication:
    # (C = A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has T[i, j, k] = 1 for the (i,j,k) index
    # In the factorization: sum_c U[a,c] V[b,c] W[c_idx,c] = T[a,b,c_idx]
    # where a indexes A, b indexes B, c_idx indexes C
    
    # Let's define:
    # A has indices (i, j) with i in [0,n), j in [0,m) -> a = i*m + j
    # B has indices (j, k) with j in [0,m), k in [0,p) -> b = j*p + k  
    # C has indices (k, i) with k in [0,p), i in [0,n) -> c = k*n + i
    # T[a, b, c] = 1 if and only if the j from A matches the j from B and the i,k are consistent
    # Specifically: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from 35 upward
    for rank in [35, 36, 37, 38, 39, 40]:
        if time.time() > deadline - 20:
            break
        
        # Try to find a decomposition of given rank
        result = _try_find_decomposition(T, rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, T, n, m, p):
                # Convert to integer/half-integer
                U_int = _to_half_int(U)
                V_int = _to_half_int(V)
                W_int = _to_half_int(W)
                if U_int is not None and V_int is not None and W_int is not None:
                    # Final verification with integer values
                    if _verify(U_int, V_int, W_int, T, n, m, p):
                        return {"U": U_int, "V": V_int, "W": W_int}
    
    # Fallback: schoolbook rank 40
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(T, rank, n, m, p, deadline):
    """Try to find a rank-r decomposition using random restarts and optimization."""
    np.random.seed(42)
    attempts = 0
    max_attempts = 500
    
    while attempts < max_attempts and time.time() < deadline - 10:
        attempts += 1
        
        # Initialize with random half-integer values
        scale = 2.0
        U = np.round(np.random.randn(n*m, rank) * scale) / 2.0
        V = np.round(np.random.randn(m*p, rank) * scale) / 2.0
        W = np.round(np.random.randn(p*n, rank) * scale) / 2.0
        
        # Use least squares to solve for W given U, V
        # We want: sum_c U[a,c] V[b,c] W[c_idx,c] = T[a,b,c_idx]
        # For each c_idx: sum_c U[a,c] V[b,c] W[c_idx,c] = T[a,b,c_idx]
        # This is linear in W for fixed U, V
        
        # Build the linear system for W
        # For each c_idx, we have n*m*(m*p) equations but that's too many
        # Instead, use the fact that:
        # Let M_c = U[:,c] * V[:,c]^T (outer product, shape n*m x m*p)
        # Then T[:, :, c_idx] = sum_c M_c * W[c_idx, c]
        
        # So for each c_idx, T[:,:,c_idx] is in the span of {M_c}
        # We can solve: T[:,:,c_idx] = sum_c W[c_idx,c] * M_c
        
        # Flatten: for each c_idx, we have (n*m)*(m*p) equations
        # Let's solve least squares for W
        
        # Build matrix A: shape (n*m*m*p, rank) where A[flat_idx, c] = M_c[flat_idx]
        A_mat = np.zeros((n*m*m*p, rank))
        for c in range(rank):
            M_c = np.outer(U[:, c], V[:, c])  # shape (n*m, m*p)
            A_mat[:, c] = M_c.flatten()
        
        # Target: T flattened, shape (p*n, n*m*m*p)
        T_flat = T.transpose(2, 0, 1).reshape(p*n, -1)  # shape (p*n, n*m*m*p)
        
        # Solve: T_flat[c_idx, :] = W[c_idx, :] @ A_mat
        # This is: T_flat = W @ A_mat (where W is p*n x rank)
        # So W = T_flat @ A_mat^+ (pseudo-inverse)
        
        try:
            A_pinv = np.linalg.pinv(A_mat)  # shape (rank, n*m*m*p)
            W_sol = T_flat @ A_pinv.T  # shape (p*n, rank)
        except:
            continue
        
        # Round W to half-integers
        W_rounded = np.round(W_sol * 2) / 2.0
        
        # Verify
        if _verify(U, V, W_rounded, T, n, m, p):
            # Try to also round U and V
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            if _verify(U_rounded, V_rounded, W_rounded, T, n, m, p):
                return (U_rounded, V_rounded, W_rounded)
        
        # Also try without rounding U, V but with rounded W
        if _verify(U, V, W_rounded, T, n, m, p):
            return (U, V, W_rounded)
    
    return None

def _verify(U, V, W, T, n, m, p):
    """Verify that U, V, W decompose T."""
    # Compute sum_c U[:,c] * V[:,c]^T * W[c_idx,c] and compare to T
    # More efficiently: for each c_idx, check T[:,:,c_idx] == sum_c W[c_idx,c] * outer(U[:,c], V[:,c])
    tol = 1e-9
    for c_idx in range(p * n):
        result = np.zeros((n*m, m*p))
        for c in range(len(U[0])):
            if W[c_idx, c] != 0:
                result += W[c_idx, c] * np.outer(U[:, c], V[:, c])
        if not np.allclose(result, T[:, :, c_idx], atol=tol):
            return False
    return True

def _to_half_int(arr):
    """Convert array to half-integer values if close enough."""
    arr2 = arr * 2
    if np.allclose(arr2, np.round(arr2), atol=1e-6):
        return np.round(arr2).astype(int) / 2.0
    return None

# EVOLVE-BLOCK-END
