import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Efficient check using einsum
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Note: W index is k*n+i, so W3[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _snap_to_rational(arr, tol=1e-9):
    """Snap values to nearest integer or half-integer if close enough."""
    result = np.array(arr, dtype=float)
    # Try integers first
    int_snapped = np.round(result)
    if np.max(np.abs(result - int_snapped)) < tol:
        return int_snapped.astype(int).tolist()
    # Try half-integers
    half_snapped = np.round(result * 2) / 2
    if np.max(np.abs(result - half_snapped)) < tol:
        # Convert to list, using int where possible
        lst = []
        for row in half_snapped:
            r = []
            for v in row:
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(int(round(v * 2)) / 2.0)
            lst.append(r)
        return lst
    # Return as-is (floats)
    return result.tolist()

def _construct_rank5_2x3x3():
    """
    Construct a rank-5 decomposition for 2x3 * 3x3 = 2x3 multiplication.
    """
    n, m, p = 2, 3, 3
    rank = 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Known rank-5 solution for 2x3x3
    # We can hardcode a known solution or find it quickly.
    # Let's try to find it with a fixed seed for reliability.
    np.random.seed(42)
    
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    for it in range(1000):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Update W
        M = np.einsum('ijc,jkc->kic', U3, V3)
        for k in range(p):
            for i in range(n):
                vec = M[k, i, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    W[k*n+i, :] = vec / norm_sq
        
        # Update U
        W3 = W.reshape(p, n, rank)
        N = np.einsum('jkc,kic->ijc', V3, W3)
        for i in range(n):
            for j in range(m):
                vec = N[i, j, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    U[i*m+j, :] = vec / norm_sq
        
        # Update V
        U3 = U.reshape(n, m, rank)
        P = np.einsum('ijc,kic->jkc', U3, W3)
        for j in range(m):
            for k in range(p):
                vec = P[j, k, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    V[j*p+k, :] = vec / norm_sq
        
        if it % 100 == 0:
            recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.max(np.abs(recon - 1.0))
            if err < 1e-6:
                break

    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    err = np.max(np.abs(recon - 1.0))
    if err < 1e-6:
        return U, V, W, True
    return None, None, None, False

def _construct_rank16_2x3x7():
    """
    Construct a rank-16 decomposition for 2x3 * 3x7 = 2x7 multiplication.
    Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1).
    Use rank-5 for 2x3x3 and rank-6 for 2x3x1.
    """
    n, m, p = 2, 3, 7
    rank_total = 16
    nm, mp, pn = n*m, m*p, p*n
    
    U5, V5, W5, ok5 = _construct_rank5_2x3x3()
    if not ok5:
        return None, None, None, False
    
    # Rank 6 for 2x3x1 is just schoolbook
    n1, m1, p1 = 2, 3, 1
    rank6 = 6
    U6 = np.zeros((n1*m1, rank6))
    V6 = np.zeros((m1*p1, rank6))
    W6 = np.zeros((p1*n1, rank6))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U6[i*m1+j, col] = 1.0
                V6[j*p1+k, col] = 1.0
                W6[k*n1+i, col] = 1.0
                col += 1
    
    U_comp = np.zeros((nm, rank_total))
    V_comp = np.zeros((mp, rank_total))
    W_comp = np.zeros((pn, rank_total))
    
    # Block 1: A * B1 (cols 0-2)
    U_comp[:, 0:5] = U5
    for j in range(3):
        for k in range(3):
            V_comp[j*7+k, 0:5] = V5[j*3+k, :]
    for k in range(3):
        for i in range(2):
            W_comp[k*2+i, 0:5] = W5[k*2+i, :]
            
    # Block 2: A * B2 (cols 3-5)
    U_comp[:, 5:10] = U5
    for j in range(3):
        for k in range(3, 6):
            V_comp[j*7+k, 5:10] = V5[j*3+(k-3), :]
    for k in range(3):
        for i in range(2):
            W_comp[k*2+i, 5:10] = W5[k*2+i, :]
            
    # Block 3: A * B3 (col 6)
    U_comp[:, 10:16] = U6
    for j in range(3):
        V_comp[j*7+6, 10:16] = V6[j*1+0, :]
    for i in range(2):
        W_comp[6*2+i, 10:16] = W6[0*2+i, :]
        
    return U_comp, V_comp, W_comp, True

def _als_decompose(n, m, p, rank, max_iters=200, seed=0):
    """
    Run ALS to find a decomposition of rank 'rank' for M_{n,m,p}.
    Returns (U, V, W, success).
    """
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(seed)
    
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    for it in range(max_iters):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # Update W
        M = np.einsum('ijc,jkc->kic', U3, V3)
        for k in range(p):
            for i in range(n):
                vec = M[k, i, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    W[k*n+i, :] = vec / norm_sq
        
        # Update U
        W3 = W.reshape(p, n, rank)
        N = np.einsum('jkc,kic->ijc', V3, W3)
        for i in range(n):
            for j in range(m):
                vec = N[i, j, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    U[i*m+j, :] = vec / norm_sq
        
        # Update V
        U3 = U.reshape(n, m, rank)
        P = np.einsum('ijc,kic->jkc', U3, W3)
        for j in range(m):
            for k in range(p):
                vec = P[j, k, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    V[j*p+k, :] = vec / norm_sq
                    
        if it % 50 == 0:
            recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.max(np.abs(recon - 1.0))
            if err < 1e-8:
                return U, V, W, True

    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    err = np.max(np.abs(recon - 1.0))
    if err < 1e-6:
        return U, V, W, True
    return U, V, W, False

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # 1. Get the baseline rank-16 solution
    U16, V16, W16, ok16 = _construct_rank16_2x3x7()
    
    best_U, best_V, best_W = None, None, None
    best_rank = 42 # Schoolbook
    
    if ok16:
        U_list = _snap_to_rational(U16)
        V_list = _snap_to_rational(V16)
        W_list = _snap_to_rational(W16)
        if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, 16, tol=1e-4):
            best_U, best_V, best_W = U_list, V_list, W_list
            best_rank = 16
    
    # 2. Try to find lower ranks (15, 14, ...)
    # We have plenty of time.
    for rank in range(best_rank - 1, 10, -1):
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try multiple seeds
        max_attempts = 50 if rank >= 14 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            # Use a seed based on rank and attempt
            seed = rank * 1000 + attempt
            U, V, W, success = _als_decompose(n, m, p, rank, max_iters=300, seed=seed)
            
            if success:
                # Snap and verify
                U_list = _snap_to_rational(U)
                V_list = _snap_to_rational(V)
                W_list = _snap_to_rational(W)
                
                if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank, tol=1e-4):
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = rank
                    found = True
                    break
                    
        if found:
            continue
            
    # 3. If no better found, return best (which is likely 16)
    if best_U is None:
        # Fallback to schoolbook if something went wrong
        U_sb = np.zeros((nm, 42))
        V_sb = np.zeros((mp, 42))
        W_sb = np.zeros((pn, 42))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1.0
                    V_sb[j*p+k, col] = 1.0
                    W_sb[k*n+i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
        best_rank = 42

    return {"U": best_U, "V": best_V, "W": best_W}
