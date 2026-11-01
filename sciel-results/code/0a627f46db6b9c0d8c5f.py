"""Combined program: Strassen blocks + integer pattern search for 2x2x3 rank-10."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 15
    deadline = time.time() + 140  # 10s safety margin

    # Build rank-53 baseline (7 Strassen blocks + 1 schoolbook)
    U53, V53, W53 = _build_strassen_53(n, m, p)
    
    if not _verify_int(U53, V53, W53, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    best_rank = 53
    best_U, best_V, best_W = U53, V53, W53

    # Try to find rank-10 decomposition for 2x2x3
    if time.time() < deadline - 10:
        U10, V10, W10 = _find_2x2x3_rank10(deadline)
        if U10 is not None:
            U52, V52, W52 = _build_combined_52(n, m, p, U10, V10, W10)
            if _verify_int(U52, V52, W52, n, m, p):
                best_rank = 52
                best_U, best_V, best_W = U52, V52, W52

    return {"U": best_U, "V": best_V, "W": best_W}


def _build_strassen_53(n, m, p):
    """Build rank-53: 7 Strassen blocks (rank 7 each) + 1 schoolbook (rank 4)."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,)]
    
    all_U, all_V, all_W = [], [], []

    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1)
        else:
            k = block[0]
            terms = _schoolbook_terms(k)
        
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _strassen_terms(k0, k1):
    """Strassen rank-7 terms for 2x2x2 block with k indices k0, k1."""
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    return [
        # M1
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        # M3
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        # M4
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,k0):1, (1,k1):1}, {(k0,0):1}),
    ]


def _schoolbook_terms(k):
    """Schoolbook rank-4 terms for 2x2x1 block."""
    return [
        ({(0,0):1}, {(0,k):1}, {(k,0):1}),
        ({(0,1):1}, {(1,k):1}, {(k,0):1}),
        ({(1,0):1}, {(0,k):1}, {(k,1):1}),
        ({(1,1):1}, {(1,k):1}, {(k,1):1}),
    ]


def _build_schoolbook(n, m, p):
    """Build schoolbook rank-60 decomposition."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _verify_int(U, V, W, n, m, p):
    """Verify exact integer decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i*m + j
                v_row = j*p + k
                w_row = k*n + i
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if val != 1:
                    return False
    return True


def _build_combined_52(n, m, p, U10, V10, W10):
    """Build rank-52: 6 Strassen blocks + rank-10 2x2x3 block."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
    
    all_U, all_V, all_W = [], [], []

    for block in blocks:
        k0, k1 = block
        terms = _strassen_terms(k0, k1)
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)

    # Add rank-10 terms for columns 12, 13, 14
    for r in range(10):
        u_entries = []
        v_entries = []
        w_entries = []
        
        for i in range(n):
            for j in range(m):
                val = U10[i*m+j, r]
                if val != 0:
                    u_entries.append((i*m+j, val))
        
        for j in range(m):
            for k_local in range(3):
                val = V10[j*3 + k_local, r]
                if val != 0:
                    k_global = 12 + k_local
                    v_entries.append((j*p + k_global, val))
        
        for k_local in range(3):
            for i in range(n):
                val = W10[k_local*n + i, r]
                if val != 0:
                    k_global = 12 + k_local
                    w_entries.append((k_global*n + i, val))
        
        all_U.append(u_entries)
        all_V.append(v_entries)
        all_W.append(w_entries)

    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val

    return U, V, W


def _find_2x2x3_rank10(deadline):
    """
    Search for rank-10 decomposition of 2x2x3 matrix multiplication.
    Uses structured enumeration of small integer patterns.
    
    The 2x2x3 tensor has 12 entries (all = 1). We need 10 rank-1 terms.
    Strategy: enumerate support patterns and solve for coefficients.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Try a structured ansatz: 7 Strassen terms for k=0,1 + 3 coupling terms for k=2
    # This is a specific algebraic construction attempt.
    
    # First, let's try the known rank-10 construction if it exists.
    # For 2x2x3, the rank is believed to be 10 (or possibly 11).
    # Let's try to find it by solving a constrained system.
    
    # Ansatz: Use 7 terms that are Strassen-like for the 2x2x2 part (k=0,1)
    # and 3 additional terms that handle k=2.
    
    # Strassen for 2x2x2 (k=0,1):
    # p1 = (A00+A11)(B00+B11)
    # p2 = (A10+A11)B00
    # p3 = A00(B10-B00)
    # p4 = A11(B01-B00)
    # p5 = (A00+A01)B11
    # p6 = (A10-A00)(B00+B01)
    # p7 = (A01-A11)(B10+B11)
    
    # For 2x2x3, B has columns 0,1,2.
    # Let's try to extend Strassen to include k=2.
    
    # One approach: use 4 schoolbook terms for k=2, but try to share
    # with the Strassen terms for k=0,1.
    
    # Actually, let's try a direct numerical approach with a better initialization.
    
    np.random.seed(42)
    best_error = float('inf')
    best_solution = None
    
    # Try multiple random restarts with ALS
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        
        # Random initialization with small values
        U = np.random.randn(4, rank) * 0.5
        V = np.random.randn(6, rank) * 0.5
        W = np.random.randn(6, rank) * 0.5
        
        # ALS iterations
        for it in range(300):
            if time.time() > deadline - 20:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B_mat = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            B_mat[r, k] = V[j*p + k, r] * W[k*n + i, r]
                    try:
                        U[i*m + j, :] = np.linalg.lstsq(B_mat, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    C_mat = np.zeros((rank, n*m))
                    for r in range(rank):
                        for i in range(n):
                            C_mat[r, i*m + j] = U[i*m + j, r] * W[k*n + i, r]
                    try:
                        V[j*p + k, :] = np.linalg.lstsq(C_mat, np.ones(n*m), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    D_mat = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            D_mat[r, j] = U[i*m + j, r] * V[j*p + k, r]
                    try:
                        W[k*n + i, :] = np.linalg.lstsq(D_mat, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-14:
                break
    
    if best_solution is not None and best_error < 1e-10:
        U, V, W = best_solution
        # Try to round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify_2x2x3(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try rounding to integers
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if _verify_2x2x3(U_r, V_r, W_r):
            return U_r, V_r, W_r
    
    return None, None, None


def _verify_2x2x3(U, V, W, tol=1e-9):
    """Verify 2x2x3 decomposition."""
    n, m, p = 2, 2, 3
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
