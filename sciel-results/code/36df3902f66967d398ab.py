import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
            
    # Build the target tensor T as a 3D array or just check the sum
    # T[a, b, c] = 1 if (a, b, c) is a valid index, else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid.add((a, b, c))
                
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                expected = 1.0 if (a, b, c) in valid else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            for i in range(4):
                U[i][col] = u[i]
                
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def snap_to_half_integers(mat, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    snapped = []
    for row in mat:
        new_row = []
        for val in row:
            # Nearest half integer
            nearest = round(val * 2) / 2.0
            if abs(val - nearest) < tol:
                new_row.append(nearest)
            else:
                new_row.append(val)
        snapped.append(new_row)
    return snapped

def search_rank(rank, time_limit=5.0):
    """
    Search for a decomposition of the given rank using ALS.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    nm, mp, pn = NM, MP, PN
    
    # Target tensor T as a 3D numpy array
    T = np.zeros((nm, mp, pn))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
                
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Scale to avoid overflow/underflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_err = np.inf
        for it in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Solve for U
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # For fixed a: T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r])
            # Let M_r = V[:, r] * W[:, r] (element-wise product, shape mp x pn)
            # vec(T[a, :, :]) = sum_r U[a, r] * vec(M_r)
            # X is mp*pn x rank matrix with columns vec(M_r)
            # y is mp*pn vector T[a, :, :] flattened
            # U[a, :] = (X^T X)^{-1} X^T y
            
            X = np.zeros((mp * pn, rank))
            for r in range(rank):
                M_r = V[:, r] * W[:, r]
                X[:, r] = M_r.flatten()
                
            # Solve least squares for each a
            # U[a, :] = np.linalg.lstsq(X, T[a, :, :].flatten())[0]
            # To speed up, precompute (X^T X)^{-1} X^T
            try:
                XtX_inv = np.linalg.pinv(X.T @ X)
                Xt = X.T
            except np.linalg.LinAlgError:
                break
                
            for a in range(nm):
                y = T[a, :, :].flatten()
                U[a, :] = XtX_inv @ (Xt @ y)
                
            # Solve for V
            # T[b, a, c] = sum_r V[b, r] * (U[a, r] * W[c, r])
            X = np.zeros((nm * pn, rank))
            for r in range(rank):
                M_r = U[:, r] * W[:, r] # shape nm x pn
                X[:, r] = M_r.flatten()
                
            try:
                XtX_inv = np.linalg.pinv(X.T @ X)
                Xt = X.T
            except np.linalg.LinAlgError:
                break
                
            for b in range(mp):
                # T[b, a, c] -> need to reshape T to (mp, nm, pn)
                # T is (nm, mp, pn). So T[:, b, :] is nm x pn.
                y = T[:, b, :].flatten()
                V[b, :] = XtX_inv @ (Xt @ y)
                
            # Solve for W
            # T[c, a, b] = sum_r W[c, r] * (U[a, r] * V[b, r])
            X = np.zeros((nm * mp, rank))
            for r in range(rank):
                M_r = U[:, r] * V[:, r] # shape nm x mp
                X[:, r] = M_r.flatten()
                
            try:
                XtX_inv = np.linalg.pinv(X.T @ X)
                Xt = X.T
            except np.linalg.LinAlgError:
                break
                
            for c in range(pn):
                # T[c, a, b] -> T is (nm, mp, pn). So T[:, :, c] is nm x mp.
                y = T[:, :, c].flatten()
                W[c, :] = XtX_inv @ (Xt @ y)
                
            # Check error
            # Compute reconstructed tensor
            # T_rec[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # This is expensive to do fully. Just check a few points or use trace?
            # For now, skip full error check in loop, do it after.
            pass
            
        # After ALS, check if solution is valid
        # Snap to half-integers
        U_snap = snap_to_half_integers(U.tolist())
        V_snap = snap_to_half_integers(V.tolist())
        W_snap = snap_to_half_integers(W.tolist())
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
            
    return None

def construct():
    """Entry point: return the best decomposition found."""
    # Deadline guard
    start_time = time.time()
    deadline = 145.0 # 150s limit, keep 5s margin
    
    # Try to find lower rank
    # Ranks to try: 27, 26, 25, 24
    for rank in [27, 26, 25, 24]:
        if time.time() - start_time > deadline - 10:
            break
        U, V, W = search_rank(rank, time_limit=10.0)
        if U is not None:
            return {"U": U, "V": V, "W": W}
            
    # Fallback to rank 28
    U, V, W = strassen_decomposition_228()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
