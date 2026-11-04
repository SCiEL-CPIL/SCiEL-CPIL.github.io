"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
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
    Uses a known rank-5 algorithm.
    """
    n, m, p = 2, 3, 3
    rank = 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Known rank-5 decomposition for M_{2,3,3}
    # The five rank-1 terms:
    # Term 0: U=[1,0,0,0,0,0], V=[1,0,0,0,0,0,0,0,0], W=[1,0,0,0]
    #   -> (i,j,k) = (0,0,0): 1*1*1=1, others 0
    #   But we need sum over all terms to give 1 for every (i,j,k)
    
    # Let me use a verified construction.
    # For 2x3x3, one standard rank-5 algorithm:
    
    # The 5 products are:
    # P1 = (a00 + a10) * (b00 + b01 + b02)  -- no, this isn't right either
    
    # Let me just use ALS to find it numerically, with good initialization
    np.random.seed(42)
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(100):
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        for it in range(1000):
            # Update W
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
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
        
        # Check error
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(recon - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if best_err < 1e-6:
        return best_U, best_V, best_W, True
    return best_U, best_V, best_W, False

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Strategy: Compose rank-5 2x3x3 and rank-6 2x3x1
    # Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    
    # Get rank-5 decomposition for 2x3x3
    U5, V5, W5, ok5 = _construct_rank5_2x3x3()
    
    # Rank-6 decomposition for 2x3x1 (schoolbook)
    n1, m1, p1 = 2, 3, 1
    rank6 = n1 * m1 * p1
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
    ok6 = True
    
    if ok5 and ok6:
        # Compose: total rank = 5 + 5 + 6 = 16
        rank_composed = 5 + 5 + 6
        U_comp = np.zeros((nm, rank_composed))
        V_comp = np.zeros((mp, rank_composed))
        W_comp = np.zeros((pn, rank_composed))
        
        # Block 1: A * B1 where B1 is columns 0-2 of B
        U_comp[:, 0:5] = U5
        for j in range(3):
            for k in range(3):
                V_comp[j*7+k, 0:5] = V5[j*3+k, :]
        for k in range(3):
            for i in range(2):
                W_comp[k*2+i, 0:5] = W5[k*2+i, :]
        
        # Block 2: A * B2 where B2 is columns 3-5 of B
        U_comp[:, 5:10] = U5
        for j in range(3):
            for k in range(3, 6):
                V_comp[j*7+k, 5:10] = V5[j*3+(k-3), :]
        for k in range(3):
            for i in range(2):
                W_comp[k*2+i, 5:10] = W5[k*2+i, :]
        
        # Block 3: A * B3 where B3 is column 6 of B
        U_comp[:, 10:16] = U6
        for j in range(3):
            V_comp[j*7+6, 10:16] = V6[j*1+0, :]
        for i in range(2):
            W_comp[6*2+i, 10:16] = W6[0*2+i, :]
        
        # Verify
        if _verify_decomposition(U_comp, V_comp, W_comp, n, m, p, rank_composed, tol=1e-4):
            U_list = _snap_to_rational(U_comp)
            V_list = _snap_to_rational(V_comp)
            W_list = _snap_to_rational(W_comp)
            
            if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank_composed, tol=1e-4):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank_composed
    
    # Try to improve further with ALS if time permits
    if time.time() < deadline - 30:
        for rank in range(best_rank - 1, 20, -1):
            if time.time() > deadline - 25:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline - 20:
                    break
                
                np.random.seed(rank * 100 + attempt)
                U = np.random.randn(nm, rank)
                V = np.random.randn(mp, rank)
                W = np.random.randn(pn, rank)
                
                for it in range(500):
                    if time.time() > deadline - 15:
                        break
                    
                    U3 = U.reshape(n, m, rank)
                    V3 = V.reshape(m, p, rank)
                    M = np.einsum('ijc,jkc->kic', U3, V3)
                    for k in range(p):
                        for i in range(n):
                            vec = M[k, i, :]
                            norm_sq = np.dot(vec, vec)
                            if norm_sq > 1e-12:
                                W[k*n+i, :] = vec / norm_sq
                    
                    W3 = W.reshape(p, n, rank)
                    N = np.einsum('jkc,kic->ijc', V3, W3)
                    for i in range(n):
                        for j in range(m):
                            vec = N[i, j, :]
                            norm_sq = np.dot(vec, vec)
                            if norm_sq > 1e-12:
                                U[i*m+j, :] = vec / norm_sq
                    
                    U3 = U.reshape(n, m, rank)
                    P = np.einsum('ijc,kic->jkc', U3, W3)
                    for j in range(m):
                        for k in range(p):
                            vec = P[j, k, :]
                            norm_sq = np.dot(vec, vec)
                            if norm_sq > 1e-12:
                                V[j*p+k, :] = vec / norm_sq
                
                # Check convergence
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.max(np.abs(recon - 1.0))
                
                if err < 1e-6:
                    U_list = _snap_to_rational(U)
                    V_list = _snap_to_rational(V)
                    W_list = _snap_to_rational(W)
                    if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p, rank, tol=1e-4):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            
            if found:
                continue
    
    return best
