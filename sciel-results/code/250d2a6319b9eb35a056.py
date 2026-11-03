"""Program for matrix multiplication 3x4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# Strassen decomposition for 2x2x2
# Each entry is (u_coeffs, v_coeffs, w_coeffs) where coeffs map local index -> coefficient
# Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
STRASSEN_DECOMP = [
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
    ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
    ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
    ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
    ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
]

def _build_strassen_3x4x8():
    """Build the block-Strassen decomposition: rank 88.
    
    Structure:
    - Split middle dimension m=4 into two blocks of 2
    - Split p=8 into four blocks of 2
    - Top 2 rows of A: use Strassen (7 mults) for each 2x2x2 block -> 2*4*7 = 56
    - Bottom 1 row of A: schoolbook (4 mults) for each 1x2x2 block -> 2*4*4 = 32
    Total: 88
    """
    n, m, p = DIMENSIONS
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for j1 in range(2):  # 2 blocks of 2 in m dimension
        # Top part: rows 0,1 of A (2x2x2 blocks)
        for l in range(4):  # 4 blocks of 2 in p dimension
            for u_coeffs, v_coeffs, w_coeffs in STRASSEN_DECOMP:
                # U (A): local 2x2 -> global
                for li, coeff in u_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i  # 0 or 1
                    global_j = 2 * j1 + li_j
                    U[global_i * m + global_j][col] += coeff
                
                # V (B): local 2x2 -> global
                for li, coeff in v_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_j = 2 * j1 + li_i
                    global_k = 2 * l + li_j
                    V[global_j * p + global_k][col] += coeff
                
                # W (C): local 2x2 -> global
                for li, coeff in w_coeffs.items():
                    li_i, li_j = li // 2, li % 2
                    global_i = li_i
                    global_k = 2 * l + li_j
                    W[global_k * n + global_i][col] += coeff
                
                col += 1
        
        # Bottom part: row 2 of A (1x2x2 blocks), schoolbook
        for l in range(4):
            for j2 in range(2):
                for k_off in range(2):
                    global_i = 2
                    global_j = 2 * j1 + j2
                    global_k = 2 * l + k_off
                    U[global_i * m + global_j][col] = 1
                    V[global_j * p + global_k][col] = 1
                    W[global_k * n + global_i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_full(U, V, W, tol=1e-9):
    """Verify exact tensor decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1"""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_quick(U, V, W, tol=1e-9):
    """Quick verification on a subset of entries."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check a representative subset
    for i in range(n):
        for j in range(0, m, 2):
            for k in range(0, p, 2):
                s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integers or half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _try_remove_columns(U, V, W, deadline):
    """Try to remove linearly dependent columns to reduce rank.
    
    Strategy: iteratively try removing each column and check if the
    decomposition remains valid. If a column is a linear combination
    of others in a way that preserves the tensor identity, it can be removed.
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    best_rank = rank
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Shuffle column order for better exploration
    np.random.seed(42)
    order = list(range(rank))
    np.random.shuffle(order)
    
    removed = set()
    
    for col_idx in order:
        if time.time() > deadline - 3:
            break
        if col_idx in removed:
            continue
        
        # Try removing this column
        keep = [i for i in range(rank) if i != col_idx]
        if not keep:
            break
        
        U_test = U_arr[:, keep]
        V_test = V_arr[:, keep]
        W_test = W_arr[:, keep]
        
        # Quick check first
        if not _verify_quick(U_test.tolist(), V_test.tolist(), W_test.tolist()):
            continue
        
        # Full verification
        if not _verify_full(U_test.tolist(), V_test.tolist(), W_test.tolist()):
            continue
        
        # Check half-integer constraint
        if not _verify_half_integer(U_test.tolist(), V_test.tolist(), W_test.tolist()):
            continue
        
        # Successfully removed a column
        removed.add(col_idx)
        best_U = U_test
        best_V = V_test
        best_W = W_test
        rank = rank - 1
        best_rank = rank
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

def _schoolbook():
    """Fallback schoolbook decomposition, rank 96."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3 by 4 matrix by a 4 by 8 matrix.
    
    Uses block Strassen decomposition (rank 88) then attempts
    column elimination to reduce rank further.
    """
    start_time = time.time()
    DEADLINE = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Build the Strassen-based decomposition (rank 88)
    result = _build_strassen_3x4x8()
    
    # Verify correctness
    if not _verify_full(result['U'], result['V'], result['W']):
        # Fallback: schoolbook
        result = _schoolbook()
        return result
    
    # Try to reduce rank by removing linearly dependent columns
    if time.time() < DEADLINE - 10:
        U_red, V_red, W_red, new_rank = _try_remove_columns(
            result['U'], result['V'], result['W'], DEADLINE
        )
        if new_rank < 88:
            result = {"U": U_red, "V": V_red, "W": W_red}
    
    return result
