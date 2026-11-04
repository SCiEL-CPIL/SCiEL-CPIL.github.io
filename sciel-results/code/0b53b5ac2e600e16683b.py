import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the tensor."""
    target = _build_target_tensor(n, m, p)
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(recon, target, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # m2 = (c+d)e
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # m3 = a(f-h)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # m4 = d(g-e)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # m5 = (a+b)h
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # m6 = (c-a)(e+f)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # m7 = (b-d)(g+h)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[3,6]=1
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build 2x4x6 decomposition using Strassen for 2x2x2 blocks.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    S_U, S_V, S_W = _strassen_2x2x2()
    
    sub_problems = [
        (0, 0, 0, 0),  # A1 * B11 -> C cols 0,1
        (0, 0, 2, 2),  # A1 * B12 -> C cols 2,3
        (0, 0, 4, 4),  # A1 * B13 -> C cols 4,5
        (2, 2, 0, 0),  # A2 * B21 -> C cols 0,1
        (2, 2, 2, 2),  # A2 * B22 -> C cols 2,3
        (2, 2, 4, 4),  # A2 * B23 -> C cols 4,5
    ]
    
    for sp_idx, (a_start, b_row_start, b_col_start, c_col_start) in enumerate(sub_problems):
        base = sp_idx * 7
        for c in range(7):
            col = base + c
            for i in range(2):
                for j in range(2):
                    U[i*4 + (a_start + j), col] = S_U[i*2 + j, c]
            for j in range(2):
                for k in range(2):
                    V[(b_row_start + j)*6 + (b_col_start + k), col] = S_V[j*2 + k, c]
            for k in range(2):
                for i in range(2):
                    W[(c_col_start + k)*2 + i, col] = S_W[k*2 + i, c]
    
    return U, V, W

def _als_decomposition(target, n, m, p, rank, rng, max_iters=200, tol=1e-8):
    """
    Alternating Least Squares with L1 sparsity regularization.
    """
    # Initialize with random values
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    # Normalize to prevent divergence
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    lambda_l1 = 0.01
    
    for it in range(max_iters):
        if _deadline_reached():
            break
            
        # Update U
        # T[a,:,:] = U[a,:] @ (V[:,None,:] * W[None,:,:]).reshape(rank, -1)
        # P[c, b, d] = V[b,c] * W[d,c]
        P = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        # U[a,:] = T[a,:,:].reshape(-1) @ P.T @ (P @ P.T + lambda * I)^-1
        # For sparsity, we can use soft-thresholding after least squares
        try:
            PTP = P @ P.T + lambda_l1 * np.eye(rank)
            PTY = P @ target.reshape(n*m, -1).T
            U_new = np.linalg.solve(PTP, PTY).T
            # Soft thresholding
            U = np.sign(U_new) * np.maximum(np.abs(U_new) - lambda_l1*0.5, 0)
        except:
            break
            
        # Update V
        Q = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        try:
            QTP = Q @ Q.T + lambda_l1 * np.eye(rank)
            QTY = Q @ target.transpose(1, 0, 2).reshape(m*p, -1).T
            V_new = np.linalg.solve(QTP, QTY).T
            V = np.sign(V_new) * np.maximum(np.abs(V_new) - lambda_l1*0.5, 0)
        except:
            break
            
        # Update W
        R = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        try:
            RTP = R @ R.T + lambda_l1 * np.eye(rank)
            RTY = R @ target.transpose(2, 0, 1).reshape(p*n, -1).T
            W_new = np.linalg.solve(RTP, RTY).T
            W = np.sign(W_new) * np.maximum(np.abs(W_new) - lambda_l1*0.5, 0)
        except:
            break
            
        # Check convergence
        recon = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.linalg.norm(target - recon)
        if err < tol:
            break
            
    return U, V, W

def _try_lower_rank_search(n, m, p, max_rank, max_iters=200, seed=42):
    rng = np.random.RandomState(seed)
    target = _build_target_tensor(n, m, p)
    
    best_U, best_V, best_W = None, None, None
    best_rank = n*m*p
    
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n*m*p
    
    # Try ranks from 30 to 41
    for rank in range(30, min(best_rank, max_rank + 1)):
        if _deadline_reached():
            break
            
        for attempt in range(10):
            if _deadline_reached():
                break
                
            U, V, W = _als_decomposition(target, n, m, p, rank, rng, max_iters=max_iters)
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                if rank < best_rank:
                    best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                    best_rank = rank
                    break # Found a valid one for this rank, move to lower rank
        
    if best_rank < n*m*p:
        return best_U, best_V, best_W, best_rank
    
    return None, None, None, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the Strassen-based 42-rank decomposition
    U, V, W = _build_2x4x6_decomposition()
    
    if _verify_decomposition(U, V, W, n, m, p):
        # Try to improve with search
        if not _deadline_reached():
            U2, V2, W2, rank2 = _try_lower_rank_search(n, m, p, max_rank=42, max_iters=100, seed=42)
            if U2 is not None and rank2 < 42:
                U, V, W = U2, V2, W2
    
    # Fallback to schoolbook if anything fails
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
