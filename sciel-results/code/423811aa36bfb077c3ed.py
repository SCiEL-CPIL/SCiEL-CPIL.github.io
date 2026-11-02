"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def _build_tensor():
    """Build the matrix multiplication tensor for n x m times m x p."""
    n, m, p = DIMENSIONS
    # T[i, j, l, k] = 1 if j == l, else 0
    # Where i: row of A (n), j: col of A / row of B (m), l: row of B (m), k: col of B (p)
    # The multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So tensor T[i, j, j, k] = 1
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0
    return T

def _try_decomposition(rank, T, n, m, p, max_iter=5000, tol=1e-10):
    """Try to find a rank-r decomposition. Returns (U, V, W) as lists or None."""
    T_flat = T.flatten()
    total_elements = T_flat.size
    
    def objective(params):
        # params: U (n*m * rank), V (m*p * rank), W (p*n * rank)
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        
        U = params[:U_size].reshape(n * m, rank)
        V = params[U_size:U_size + V_size].reshape(m * p, rank)
        W = params[U_size + V_size:U_size + V_size + W_size].reshape(p * n, rank)
        
        # Compute reconstruction: for each (i,j,l,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Note: W index is k*n + i per the trace convention W[k*n+i, r]
        recon = np.zeros_like(T)
        for c in range(rank):
            uc = U[:, c]  # (n*m,)
            vc = V[:, c]  # (m*p,)
            wc = W[:, c]  # (p*n,)
            # uc[i*m+j], vc[j*p+k], wc[k*n+i]
            # Reshape to 4D
            u4 = uc.reshape(n, m)
            v4 = vc.reshape(m, p)
            w4 = wc.reshape(p, n)  # w4[k, i]
            
            # For each (i,j,k): u4[i,j] * v4[j,k] * w4[k,i]
            # This is a 4D tensor: result[i,j,k] = u4[i,j] * v4[j,k] * w4[k,i]
            # But we need result[i,j,j,k] pattern... 
            # Actually the full 4D tensor is:
            # R[i, j, l, k] = sum_c U[i*m+j, c] * V[l*p+k, c] * W[k*n+i, c]
            # Wait, V should be indexed by (row_of_B, col_of_B) = (l, k)
            # So V[l*p + k, c]
            # And the tensor element T[i, j, l, k] should equal sum_c U[i*m+j,c] * V[l*p+k,c] * W[k*n+i,c]
            
            # Let me redo: R[i,j,l,k] = U[i*m+j, c] * V[l*p+k, c] * W[k*n+i, c]
            # We want R to match T where T[i,j,l,k] = delta(j,l)
            
            # Build R as outer products
            # u4[i,j], v4[l,k], w4[k,i]
            # R_c[i,j,l,k] = u4[i,j] * v4[l,k] * w4[k,i]
            # This is a 4D tensor. Let's compute it.
            # u4 shape: (n, m), v4 shape: (m, p), w4 shape: (p, n)
            # R_c[i,j,l,k] = u4[i,j] * v4[l,k] * w4[k,i]
            # = u4[i,j] * (v4[l,k] * w4[k,i])
            # For fixed i: sum over j,l,k: u4[i,j] * v4[l,k] * w4[k,i]
            # This is like a 4D tensor. Let me just compute it directly.
            
            # R_c[i, j, l, k] = u4[i,j] * v4[l,k] * w4[k,i]
            # Reshape to use broadcasting
            # u4: (n, m, 1, 1)
            # v4: (1, 1, m, p) -> but v4 is (m, p) so v4[l,k]
            # w4: (n, 1, 1, p) -> w4[k,i] means index [k, i] so shape (p, n)
            # R_c[i,j,l,k] = u4[i,j] * v4[l,k] * w4[k,i]
            
            u4_reshaped = u4[:, :, np.newaxis, np.newaxis]  # (n, m, 1, 1)
            v4_reshaped = v4[np.newaxis, np.newaxis, :, :]  # (1, 1, m, p)
            # w4[k,i] -> we need shape (1, 1, 1, p) indexed by [k, i]
            # w4 is (p, n), so w4[k, i] for k in range(p), i in range(n)
            # We want R_c[i,j,l,k] to use w4[k,i]
            # So w4_reshaped should be (n, 1, 1, p) where w4_reshaped[i, 0, 0, k] = w4[k, i]
            w4_reshaped = w4.T[np.newaxis, np.newaxis, :, :]  # w4.T is (n, p), so (1, 1, n, p)
            # Wait, w4 is (p, n), w4.T is (n, p). w4.T[i, k] = w4[k, i]. Good.
            # So w4_reshaped[i, 0, 0, k] = w4.T[i, k] = w4[k, i]. Correct.
            
            recon += u4_reshaped * v4_reshaped * w4_reshaped
        
        return (recon - T_flat).flatten()
    
    # Initialize with random values
    np.random.seed(42)
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    total_params = U_size + V_size + W_size
    
    best_result = None
    best_cost = np.inf
    
    for trial in range(3):
        params0 = np.random.randn(total_params) * 0.5
        try:
            result = least_squares(objective, params0, method='lm', max_nfev=2000)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_result = result
        except Exception:
            continue
    
    if best_result is None or best_cost > tol:
        return None
    
    params = best_result.x
    U_size = n * m * rank
    V_size = m * p * rank
    W_size = p * n * rank
    
    U = params[:U_size].reshape(n * m, rank)
    V = params[U_size:U_size + V_size].reshape(m * p, rank)
    W = params[U_size + V_size:U_size + V_size + W_size].reshape(p * n, rank)
    
    # Snap to nearest half-integer
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Verify exactness after snapping
    recon = np.zeros((n, m, m, p))
    for c in range(rank):
        u4 = U_snap[:, c].reshape(n, m)
        v4 = V_snap[:, c].reshape(m, p)
        w4 = W_snap[:, c].reshape(p, n)
        u4_r = u4[:, :, np.newaxis, np.newaxis]
        v4_r = v4[np.newaxis, np.newaxis, :, :]
        w4_r = w4.T[np.newaxis, np.newaxis, :, :]
        recon += u4_r * v4_r * w4_r
    
    if np.allclose(recon, T, atol=1e-10):
        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    # Try without snapping if it was already close to integers/half-integers
    if np.allclose(recon, T, atol=1e-10):
        # Check if values are close to half-integers
        if np.allclose(U * 2, np.round(U * 2), atol=1e-6) and \
           np.allclose(V * 2, np.round(V * 2), atol=1e-6) and \
           np.allclose(W * 2, np.round(W * 2), atol=1e-6):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    deadline = time.time() + 140  # Safety margin
    
    # Try ranks from low to high
    for rank in range(1, 19):
        if time.time() > deadline:
            break
        
        result = _try_decomposition(rank, T, n, m, p)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
