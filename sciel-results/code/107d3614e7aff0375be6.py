import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def _round_to_half_integer(M):
    """Round matrix entries to the nearest half-integer."""
    return np.round(M * 2) / 2.0

def _find_decomposition_3x3x3(rank, time_budget=10):
    """
    Attempt to find a decomposition of rank 'rank' for 3x3x3 matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    n = 3
    # U: (9, rank), V: (9, rank), W: (9, rank)
    # Target: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    start = time.time()
    
    # Try multiple random initializations
    max_trials = 50
    for trial in range(max_trials):
        if time.time() - start > time_budget:
            break
            
        np.random.seed(trial * 1000 + 42)
        U = np.random.randn(9, rank) * 0.5
        V = np.random.randn(9, rank) * 0.5
        W = np.random.randn(9, rank) * 0.5
        
        for it in range(100):
            if time.time() - start > time_budget:
                break
                
            # Update U
            for a in range(9):
                i, j = divmod(a, 3)
                M = np.zeros((3, rank))
                for k in range(3):
                    M[k, :] = V[j*3+k, :] * W[k*3+i, :]
                try:
                    U[a, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                except:
                    pass
                    
            # Update V
            for b in range(9):
                j, k = divmod(b, 3)
                M = np.zeros((3, rank))
                for i in range(3):
                    M[i, :] = U[i*3+j, :] * W[k*3+i, :]
                try:
                    V[b, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                except:
                    pass
                    
            # Update W
            for c in range(9):
                k, i = divmod(c, 3)
                M = np.zeros((3, rank))
                for j in range(3):
                    M[j, :] = U[i*3+j, :] * V[j*3+k, :]
                try:
                    W[c, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                except:
                    pass
                    
            # Check error
            err = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        total = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                        err += (total - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if err < 1e-10:
                break
                
        if best_err < 1e-10:
            break

    if best_err < 1e-10:
        # Round to half-integers and verify
        U_r = _round_to_half_integer(best_U)
        V_r = _round_to_half_integer(best_V)
        W_r = _round_to_half_integer(best_W)
        
        # Verify rounded solution
        valid = True
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = np.sum(U_r[i*3+j, :] * V_r[j*3+k, :] * W_r[k*3+i, :])
                    if abs(total - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_r, V_r, W_r
            
    return None

def _schoolbook_3x3x6():
    n, m, p = 3, 3, 6
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return get_schoolbook()
    
    # Try to find a decomposition for the 3x3x3 block
    # We try ranks in increasing order: 23, 24, 25, 26
    # If we find rank R, the total rank for 3x3x6 is 2*R
    
    for rank_3x3 in [23, 24, 25, 26]:
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        # Allocate time budget based on difficulty
        # Rank 23 is hardest, give it more time? Or just fixed time?
        time_budget = 10 if rank_3x3 == 23 else 5
        
        try:
            result = _find_decomposition_3x3x3(rank_3x3, time_budget)
            if result is not None:
                U1, V1, W1 = result
                # Construct 3x3x6 decomposition
                # U: (9, 2*rank_3x3)
                # V: (18, 2*rank_3x3)
                # W: (18, 2*rank_3x3)
                
                total_rank = 2 * rank_3x3
                U = np.zeros((9, total_rank))
                V = np.zeros((18, total_rank))
                W = np.zeros((18, total_rank))
                
                # Block 1
                U[:, 0:rank_3x3] = U1
                V[0:9, 0:rank_3x3] = V1
                W[0:9, 0:rank_3x3] = W1
                
                # Block 2
                U[:, rank_3x3:2*rank_3x3] = U1
                V[9:18, rank_3x3:2*rank_3x3] = V1
                W[9:18, rank_3x3:2*rank_3x3] = W1
                
                # Final verification
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
            
    # Fallback
    return get_schoolbook()
