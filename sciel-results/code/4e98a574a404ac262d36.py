"""Bootstrap direct approach for 2x3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if (i,j,k) corresponds to a multiplication term, else 0
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_decomposition(n, m, p, target_rank, max_iters=1000, seed=42):
    """Try to find a decomposition of given rank using random initialization and Newton-like refinement."""
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # We want: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of n*m*p equations in n*m*rank + m*p*rank + p*n*rank variables
    
    # Use a simple iterative approach: fix two matrices, solve for the third
    # This is a bilinear/trilinear problem, so we can use alternating least squares
    
    for it in range(max_iters):
        # Fix V and W, solve for U
        # For each (i,j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # This is a linear system in U[i*m+j,:] for each (i,j)
        
        # Build the system for U
        # For fixed (i,j), we need: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k in range(p)
        # Let M_U[i,j] be a p x rank matrix where M_U[i,j][k,r] = V[j*p+k,r] * W[k*n+i,r]
        # Then M_U[i,j] @ U[i*m+j,:] = ones(p)
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M @ u = ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        # Fix U and W, solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        # Fix U and V, solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err < 1e-8:
            # Snap to rational values if possible
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify the snapped version
            if _verify_decomposition(
                U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p
            ):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
            
            # If snapping doesn't work, try returning the float version (but task requires integer/half-integer)
            # So we need to find a rational approximation
            # For now, try with higher precision snapping
            for scale in [4, 8, 16, 32]:
                U_snap = np.round(U * scale) / scale
                V_snap = np.round(V * scale) / scale
                W_snap = np.round(W * scale) / scale
                if _verify_decomposition(
                    U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), n, m, p
                ):
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
        
        # Early termination if not improving
        if it > 0 and max_err > 1.0:
            break
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Try structured approaches to find lower rank decompositions
    # For 2x3x7, known lower bounds suggest rank at least max(2*7, 3*7, 2*3) = 21
    # But there might be constructions with rank around 18-20
    
    # Try various target ranks from low to high
    target_ranks = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:  # Keep 20s buffer
            break
        
        # Try multiple random seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_decomposition(n, m, p, target_rank, max_iters=200, seed=seed)
            if result is not None:
                U_res, V_res, W_res = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_res, "V": V_res, "W": W_res}
                    # If we found a good one, we can stop early
                    if target_rank <= 20:
                        break
        if target_rank <= 20 and best_rank == target_rank:
            break

    return best_result
# EVOLVE-BLOCK-END
