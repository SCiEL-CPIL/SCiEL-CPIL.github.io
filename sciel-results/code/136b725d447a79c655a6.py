"""
Combined program: Strassen-based block decomposition for 2x5x6 matrix multiplication
with systematic rank reduction via greedy column elimination.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    And 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                # Check consistency
                if i == i2 and j == j2 and k == k2:
                    # Valid case, already checked
                    continue
                else:
                    s = 0.0
                    for r in range(rank):
                        s += U[y, r] * V[z, r] * W[x, r]
                    if abs(s) > tol:
                        return False
    return True

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # 7 products for Strassen 2x2x2 block
            
            # P1
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P2
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # P3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # P5
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # P6
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # P7
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
            
    # Remaining j=4
    for k in range(p):
        # C_0,k = A_0,4 B_4,k
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        # C_1,k = A_1,4 B_4,k
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
        
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _check_valid_entries(U, V, W):
    """Check that all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # Check if val is integer or half-integer
                if abs(val * 2 - round(val * 2)) > 1e-12:
                    return False
    return True

def _verify_fast(Ua, Va, Wa, n=2, m=5, p=6, tol=1e-9):
    """Vectorized verification of decomposition."""
    rank = Ua.shape[1]
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Build index arrays
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
    idx_u = np.array([x[0] for x in indices])
    idx_v = np.array([x[1] for x in indices])
    idx_w = np.array([x[2] for x in indices])
    # U[idx_u, :] has shape (n*m*p, rank)
    # V[idx_v, :] has shape (n*m*p, rank)
    # W[idx_w, :] has shape (n*m*p, rank)
    # Element-wise product then sum along axis 1
    vals = Ua[idx_u, :] * Va[idx_v, :] * Wa[idx_w, :]
    totals = vals.sum(axis=1)
    return np.all(np.abs(totals - 1.0) < tol)

def _try_local_search(U, V, W, time_limit=90):
    """Try to reduce rank via greedy column elimination with random restarts."""
    start_time = time.time()
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    n, m, p = 2, 5, 6
    current_rank = Ua.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    # Precompute indices for fast verification
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
    idx_u = np.array([x[0] for x in indices])
    idx_v = np.array([x[1] for x in indices])
    idx_w = np.array([x[2] for x in indices])
    
    def _verify_with_indices(Ur, Vr, Wr):
        vals = Ur[idx_u, :] * Vr[idx_v, :] * Wr[idx_w, :]
        totals = vals.sum(axis=1)
        return np.all(np.abs(totals - 1.0) < 1e-9)
    
    def _can_remove_fast(Ur, Vr, Wr, col):
        rank = Ur.shape[1]
        mask = np.ones(rank, dtype=bool)
        mask[col] = False
        return _verify_with_indices(Ur[:, mask], Vr[:, mask], Wr[:, mask])
    
    # Phase 1: Greedy elimination (try all columns, remove first valid one, repeat)
    improved = True
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        for col in range(Ua.shape[1]):
            if time.time() - start_time > time_limit - 5:
                break
            if _can_remove_fast(Ua, Va, Wa, col):
                mask = np.ones(Ua.shape[1], dtype=bool)
                mask[col] = False
                Ua = Ua[:, mask]
                Va = Va[:, mask]
                Wa = Wa[:, mask]
                best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
                best_rank = Ua.shape[1]
                improved = True
                break
    
    # Phase 2: Random restarts with different elimination orders
    remaining = time_limit - (time.time() - start_time)
    if remaining > 2:
        np.random.seed(42)
        for restart in range(20):
            if time.time() - start_time > time_limit - 3:
                break
            # Start from best found so far
            Ua = best_U.copy()
            Va = best_V.copy()
            Wa = best_W.copy()
            
            # Try random order of columns
            cols = np.random.permutation(Ua.shape[1])
            for col in cols:
                if time.time() - start_time > time_limit - 3:
                    break
                if col >= Ua.shape[1]:
                    continue
                if _can_remove_fast(Ua, Va, Wa, col):
                    mask = np.ones(Ua.shape[1], dtype=bool)
                    mask[col] = False
                    Ua = Ua[:, mask]
                    Va = Va[:, mask]
                    Wa = Wa[:, mask]
                    if Ua.shape[1] < best_rank:
                        best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
                        best_rank = Ua.shape[1]
    
    if best_rank < current_rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Try Strassen-based rank-54 decomposition
    try:
        U, V, W = _build_54_rank(2, 5, 6)
        
        # Verify the initial decomposition
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), 2, 5, 6):
            if _check_valid_entries(U.tolist(), V.tolist(), W.tolist()):
                # Try to reduce rank via local search
                remaining_time = deadline - time.time() - 5
                if remaining_time > 5:
                    reduced = _try_local_search(U.tolist(), V.tolist(), W.tolist(), time_limit=remaining_time)
                    if reduced is not None:
                        ru, rv, rw = reduced
                        if _verify_decomposition(ru, rv, rw, 2, 5, 6) and _check_valid_entries(ru, rv, rw):
                            return {'U': ru, 'V': rv, 'W': rw}
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
