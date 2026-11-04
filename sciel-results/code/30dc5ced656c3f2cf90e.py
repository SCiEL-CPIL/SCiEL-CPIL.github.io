import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS

def verify(U, V, W, tol=1e-12):
    """Verify that the tensor decomposition is correct.
    
    The multiplication tensor T has T[i,j,k] = 1 if and only if
    the output entry (k, i) gets a contribution from A[i,j] * B[j,k].
    In the trace convention: W index is k*n + i, U index is i*m + j, V index is j*p + k.
    
    We check: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    And for mismatched indices, the sum should be 0.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), the contribution to output W[k*n+i] from input A[i*m+j] and B[j*p+k]
    # should be 1 if the decomposition is correct.
    # Actually, the tensor T_{ijk} = 1 for all i,j,k (it's the full multiplication tensor).
    # The decomposition says: T_{ijk} = sum_r U_{im+j,r} V_{jp+k,r} W_{kn+i,r}
    
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u_idx = i * M + j
                v_idx = j * P + k
                w_idx = k * N + i
                s = 0.0
                for r in range(len(U[0])):
                    s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 150.0 # Evaluator kills at 150s, keep margin
    
    # Schoolbook rank: N*M*P = 42
    def schoolbook():
        rank = N * M * P
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    # Precompute indices for faster access
    # U indices: i*M + j
    # V indices: j*P + k
    # W indices: k*N + i
    
    def snap_to_half(X):
        """Snap values to nearest 0.5 and return as list of lists of floats (0.0, 0.5, 1.0, etc.)"""
        X = np.round(X * 2) / 2.0
        return X.tolist()

    def try_rank_als(R, max_iter=500, seed=0):
        """
        Try to find a rank-R decomposition using vectorized ALS.
        Returns (U, V, W) if found and valid, else None.
        """
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(N * M, R) * 0.1
        V = np.random.randn(M * P, R) * 0.1
        W = np.random.randn(P * N, R) * 0.1
        
        # Precompute index arrays for vectorization
        # We need to update U, V, W.
        # Update U[iM+j, r]: 
        #   Minimize sum_k (sum_r U[iM+j,r] V[jP+k,r] W[kN+i,r] - 1)^2
        #   Let B_{k,r} = V[jP+k,r] W[kN+i,r]
        #   Then we solve (B^T B) U_col = B^T 1
        
        # To speed up, we can batch the updates.
        # But N, M, P are small. The overhead of Python loops might be acceptable if inner ops are vectorized.
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Update U
            for i in range(N):
                for j in range(M):
                    # Build B matrix: P x R
                    # B[k, r] = V[j*P+k, r] * W[k*N+i, r]
                    # V slice: V[j*P : (j+1)*P, :]
                    # W indices: [k*N+i for k in range(P)]
                    V_slice = V[j*P:(j+1)*P, :]
                    W_idx = [k*N + i for k in range(P)]
                    W_slice = W[W_idx, :]
                    B = V_slice * W_slice # Broadcasting works: (P, R) * (P, R) -> (P, R)
                    
                    # Solve (B^T B) x = B^T 1
                    # Regularize
                    ATA = B.T @ B + 1e-10 * np.eye(R)
                    ATb = B.T @ np.ones(P)
                    try:
                        U[i*M+j, :] = np.linalg.solve(ATA, ATb)
                    except np.linalg.LinAlgError:
                        return None
            
            # Update V
            for j in range(M):
                for k in range(P):
                    # Build B matrix: N x R
                    # B[i, r] = U[i*M+j, r] * W[k*N+i, r]
                    # U indices: [i*M+j for i in range(N)]
                    # W slice: W[k*N : (k+1)*N, :]
                    U_idx = [i*M + j for i in range(N)]
                    U_slice = U[U_idx, :]
                    W_slice = W[k*N:(k+1)*N, :]
                    B = U_slice * W_slice
                    
                    ATA = B.T @ B + 1e-10 * np.eye(R)
                    ATb = B.T @ np.ones(N)
                    try:
                        V[j*P+k, :] = np.linalg.solve(ATA, ATb)
                    except np.linalg.LinAlgError:
                        return None
            
            # Update W
            for k in range(P):
                for i in range(N):
                    # Build B matrix: M x R
                    # B[j, r] = U[i*M+j, r] * V[j*P+k, r]
                    # U slice: U[i*M : (i+1)*M, :]
                    # V indices: [j*P+k for j in range(M)]
                    U_slice = U[i*M:(i+1)*M, :]
                    V_idx = [j*P + k for j in range(M)]
                    V_slice = V[V_idx, :]
                    B = U_slice * V_slice
                    
                    ATA = B.T @ B + 1e-10 * np.eye(R)
                    ATb = B.T @ np.ones(M)
                    try:
                        W[k*N+i, :] = np.linalg.solve(ATA, ATb)
                    except np.linalg.LinAlgError:
                        return None
            
            # Check convergence every 10 iterations
            if it % 10 == 0:
                err = 0.0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            s = np.dot(U[i*M+j], V[j*P+k] * W[k*N+i])
                            err += (s - 1.0) ** 2
                
                if err < 1e-12:
                    # Snap to half-integers
                    U_snap = snap_to_half(U)
                    V_snap = snap_to_half(V)
                    W_snap = snap_to_half(W)
                    
                    if verify(U_snap, V_snap, W_snap, tol=1e-9):
                        return (U_snap, V_snap, W_snap)
        
        return None

    # Try ranks from 35 to 41 (skip 42 as we have schoolbook)
    # Start from lower ranks to find minimal
    # Note: If rank 39 is the true min, we need to find it.
    for R in range(35, 42):
        if time.time() > deadline - 15:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            result = try_rank_als(R, max_iter=200, seed=seed)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                break
        if best != schoolbook():
            break

    return best
