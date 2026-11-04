import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n, m, p) with T[i,j,k] = 1 if i,j,k correspond to the multiplication term?
    # No, the tensor T represents the matrix multiplication map.
    # T[i, j, k] is the coefficient of A[i,j] * B[j,k] in C[i,k].
    # In standard matrix multiplication C = A B, C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T has T[i, j, k] = 1 for all i, j, k?
    # Wait. The tensor is T_{i, j, k} such that C_{i, k} = \sum_{j} T_{i, j, k} A_{i, j} B_{j, k}?
    # No, the bilinear form is \sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^*?
    # The task says: "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor $M_{n \times m \times p}$ has entries $M_{i,j,k} = 1$ if the index $j$ is shared?
    # Actually, the tensor is defined by $T_{i, j, k} = 1$ for all $i \in [0,n), j \in [0,m), k \in [0,p)$.
    # Because $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # So the term $A_{i,j} B_{j,k}$ appears in $C_{i,k}$ with coefficient 1.
    # Thus, $T_{i,j,k} = 1$ for all valid i, j, k.
    
    T = np.ones((n, m, p))
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition using ALS
    # We search for rank R
    # Time limit guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_rank = rank_schoolbook
    best_factors = None
    
    # Lower bound heuristic
    # For 3x3x7, we expect rank around 49 or lower
    # Try ranks from 40 to 62
    for R in range(40, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            U = np.random.randn(n, R)
            V = np.random.randn(m, R)
            W = np.random.randn(p, R)
            
            # ALS iterations
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                # Solve for U
                # T[i, j, k] = sum_r U[i, r] * (V[j, r] * W[k, r])
                # Reshape T to (n, m*p)
                T_flat = T.reshape(n, m * p)
                # Construct matrix M_VW of shape (m*p, R)
                # M_VW[j*p + k, r] = V[j, r] * W[k, r]
                M_VW = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        M_VW[j * p + k, :] = V[j, :] * W[k, :]
                
                # Solve U @ M_VW.T = T_flat.T ?
                # T_flat[i, jp+k] = sum_r U[i, r] * M_VW[jp+k, r]
                # This is U @ M_VW.T = T_flat.T
                # We want U such that U @ M_VW.T approx T_flat.T
                # U = T_flat.T @ pinv(M_VW.T)
                # Or use lstsq: M_VW.T @ U.T = T_flat.T.T?
                # Let's use lstsq on M_VW.T (shape R x m*p) and T_flat.T (shape m*p x n)
                # We want X (R x n) such that M_VW.T @ X = T_flat.T
                # X = lstsq(M_VW.T, T_flat.T)
                U_T, _, _, _ = np.linalg.lstsq(M_VW.T, T_flat.T, rcond=None)
                U = U_T.T
                
                # Solve for V
                # T[i, j, k] = sum_r (U[i, r] * W[k, r]) * V[j, r]
                # Reshape T to (m, n*p)
                T_flat_m = T.transpose(1, 0, 2).reshape(m, n * p)
                M_UW = np.zeros((n * p, R))
                for i in range(n):
                    for k in range(p):
                        M_UW[i * p + k, :] = U[i, :] * W[k, :]
                
                # V @ M_UW.T = T_flat_m.T
                V_T, _, _, _ = np.linalg.lstsq(M_UW.T, T_flat_m.T, rcond=None)
                V = V_T.T
                
                # Solve for W
                # T[i, j, k] = sum_r (U[i, r] * V[j, r]) * W[k, r]
                # Reshape T to (p, n*m)
                T_flat_p = T.transpose(2, 0, 1).reshape(p, n * m)
                M_UV = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        M_UV[i * m + j, :] = U[i, :] * V[j, :]
                
                # W @ M_UV.T = T_flat_p.T
                W_T, _, _, _ = np.linalg.lstsq(M_UV.T, T_flat_p.T, rcond=None)
                W = W_T.T
            
            # Check if we have a solution
            # Reconstruct T from U, V, W
            T_recon = np.zeros((n, m, p))
            for r in range(R):
                T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Check error
            error = np.max(np.abs(T_recon - T))
            if error < 1e-6:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                T_recon_snap = np.zeros((n, m, p))
                for r in range(R):
                    T_recon_snap += np.einsum('i,j,k->ijk', U_snap[:, r], V_snap[:, r], W_snap[:, r])
                
                error_snap = np.max(np.abs(T_recon_snap - T))
                if error_snap < 1e-9:
                    # Found a valid half-integer decomposition
                    best_rank = R
                    best_factors = (U_snap, V_snap, W_snap)
                    found = True
                    break
        
        if found:
            break
    
    if best_factors is not None:
        U, V, W = best_factors
        # Construct output matrices
        U_out = np.zeros((n * m, best_rank))
        V_out = np.zeros((m * p, best_rank))
        W_out = np.zeros((p * n, best_rank))
        
        for i in range(n):
            for j in range(m):
                U_out[i * m + j, :] = U[i, :]
                
        for j in range(m):
            for k in range(p):
                V_out[j * p + k, :] = V[j, :]
                
        for k in range(p):
            for i in range(n):
                W_out[k * n + i, :] = W[k, :]
                
        # Convert to lists of lists for the interface
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist()
        }
    else:
        # Fallback to schoolbook
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
