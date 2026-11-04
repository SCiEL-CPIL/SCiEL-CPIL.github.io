import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _snap_to_rational(arr, tol=1e-9):
    """Snap values to nearest integer or half-integer if close enough."""
    result = np.array(arr, dtype=float)
    int_snapped = np.round(result)
    if np.max(np.abs(result - int_snapped)) < tol:
        return int_snapped.astype(int).tolist()
    half_snapped = np.round(result * 2) / 2
    if np.max(np.abs(result - half_snapped)) < tol:
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
    return result.tolist()

def _als_decompose(n, m, p, rank, max_iters=1000, seed=0, tol=1e-8):
    """
    Run ALS to find a decomposition of rank 'rank' for M_{n,m,p}.
    Returns (U, V, W, success).
    """
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(seed)
    
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
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
                    
        if it % 10 == 0:
            recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            err = np.max(np.abs(recon - 1.0))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < tol:
                return U, V, W, True

    recon = np.einsum('ijc,jkc,kic->ijk', best_U.reshape(n, m, rank), 
                       best_V.reshape(m, p, rank), 
                       best_W.reshape(p, n, rank))
    err = np.max(np.abs(recon - 1.0))
    if err < 1e-6:
        return best_U, best_V, best_W, True
    return best_U, best_V, best_W, False

def _find_decomposition(n, m, p, target_rank, max_attempts=20, deadline=None):
    """
    Try to find a decomposition of rank 'target_rank' for M_{n,m,p}.
    Returns (U, V, W, success).
    """
    for attempt in range(max_attempts):
        if deadline and time.time() > deadline - 10:
            break
        seed = attempt * 7919 + 1
        U, V, W, success = _als_decompose(n, m, p, target_rank, max_iters=500, seed=seed)
        if success:
            U_list = _snap_to_rational(U)
            V_list = _snap_to_rational(V)
            W_list = _snap_to_rational(W)
            if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, target_rank, tol=1e-4):
                return U_list, V_list, W_list, True
    return None, None, None, False

def _construct_block_decomposition(n, m, p):
    """
    Construct a block decomposition for M_{n,m,p} by splitting p into chunks.
    For 2x3x7, split into 2x3x3, 2x3x3, 2x3x1.
    """
    # Find rank-5 for 2x3x3
    U5, V5, W5, ok5 = _find_decomposition(2, 3, 3, 5, max_attempts=50, deadline=time.time()+20)
    if not ok5:
        # Fallback to schoolbook for 2x3x3 (rank 9) if 5 fails
        U5, V5, W5, ok5 = _find_decomposition(2, 3, 3, 9, max_attempts=10, deadline=time.time()+5)
        if not ok5:
            return None, None, None, False
        rank5 = 9
    else:
        rank5 = 5

    # Find rank-6 for 2x3x1
    U1, V1, W1, ok1 = _find_decomposition(2, 3, 1, 6, max_attempts=10, deadline=time.time()+5)
    if not ok1:
        # Fallback to schoolbook for 2x3x1 (rank 6) - should always work
        U1 = np.zeros((2*3, 6))
        V1 = np.zeros((3*1, 6))
        W1 = np.zeros((1*2, 6))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(1):
                    U1[i*3+j, col] = 1.0
                    V1[j*1+k, col] = 1.0
                    W1[k*2+i, col] = 1.0
                    col += 1
        U1 = U1.tolist()
        V1 = V1.tolist()
        W1 = W1.tolist()
        rank1 = 6
    else:
        rank1 = 6

    # Assemble 2x3x7
    n, m, p = 2, 3, 7
    nm, mp, pn = n*m, m*p, p*n
    total_rank = rank5 + rank5 + rank1
    U_total = np.zeros((nm, total_rank))
    V_total = np.zeros((mp, total_rank))
    W_total = np.zeros((pn, total_rank))
    
    U5 = np.array(U5)
    V5 = np.array(V5)
    W5 = np.array(W5)
    U1 = np.array(U1)
    V1 = np.array(V1)
    W1 = np.array(W1)
    
    # Block 1: cols 0-2
    U_total[:, 0:rank5] = U5
    for j in range(m):
        for k in range(3):
            V_total[j*p+k, 0:rank5] = V5[j*3+k, :]
    for k in range(3):
        for i in range(n):
            W_total[k*n+i, 0:rank5] = W5[k*n+i, :]
            
    # Block 2: cols 3-5
    offset = rank5
    U_total[:, offset:offset+rank5] = U5
    for j in range(m):
        for k in range(3, 6):
            V_total[j*p+k, offset:offset+rank5] = V5[j*3+(k-3), :]
    for k in range(3):
        for i in range(n):
            W_total[k*n+i, offset:offset+rank5] = W5[k*n+i, :]
            
    # Block 3: col 6
    offset2 = 2 * rank5
    U_total[:, offset2:offset2+rank1] = U1
    for j in range(m):
        V_total[j*p+6, offset2:offset2+rank1] = V1[j*1+0, :]
    for i in range(n):
        W_total[6*n+i, offset2:offset2+rank1] = W1[0*n+i, :]
        
    U_list = _snap_to_rational(U_total)
    V_list = _snap_to_rational(V_total)
    W_list = _snap_to_rational(W_total)
    
    if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, total_rank, tol=1e-4):
        return U_list, V_list, W_list, True
        
    return None, None, None, False

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    best_U, best_V, best_W = None, None, None
    best_rank = 42  # Schoolbook
    
    # 1. Try block decomposition (likely rank 16)
    U_block, V_block, W_block, ok_block = _construct_block_decomposition(n, m, p)
    if ok_block:
        best_U, best_V, best_W = U_block, V_block, W_block
        best_rank = len(U_block[0]) if isinstance(U_block, list) and U_block else 16
        
    # 2. Try to find a better rank directly using ALS on the full problem
    # Start from best_rank - 1 down to 10
    for rank in range(best_rank - 1, 9, -1):
        if time.time() > deadline - 20:
            break
            
        # Use the block solution as a warm start? Or just random?
        # Random is simpler but might miss the structure.
        # Let's try a few random restarts.
        found = False
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            seed = rank * 1000 + attempt
            U, V, W, success = _als_decompose(n, m, p, rank, max_iters=300, seed=seed)
            if success:
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
            
    # 3. Fallback to schoolbook
    if best_U is None:
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
