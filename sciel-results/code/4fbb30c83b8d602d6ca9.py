"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal tensor decomposition for 2x2 by 2x14 matrix multiplication.
    Tries ranks from 28 upward, using a constructive approach.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Try ranks from theoretical minimum upward
    for rank in range(28, 50):
        if time.time() > deadline - 5:
            break
        result = _try_construct_rank(rank, n, m, p)
        if result is not None:
            if _verify_decomposition_full(result['U'], result['V'], result['W'], n, m, p):
                return result

    # Fallback: Schoolbook decomposition (rank 56)
    return _construct_schoolbook(n, m, p)

def _try_construct_rank(rank, n, m, p):
    """
    Try to construct a rank-`rank` decomposition using a structured approach.
    Uses the idea that we can pair up the 14 columns of B with the 2 rows of A.
    """
    # The tensor entries are at (a,b,c) = (i*m+j, j*p+k, k*n+i) for i in [0,n), j in [0,m), k in [0,p)
    # n=2, m=2, p=14: 56 entries
    # U: (4, rank), V: (28, rank), W: (28, rank)
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strategy: For each of the 7 pairs of columns (2r, 2r+1) of B,
    # use a 4x4 structure. The 2x2 A matrix times a 2x2 block of B gives a 2x2 block of C.
    # For 2x2 * 2x2, the minimum rank is 7 (Strassen).
    # But we have 14 columns, so 7 blocks of 2x2.
    # Total rank = 7 * 7 = 49 (block Strassen)
    
    # Alternative: Try to find a better structure.
    # Let's try a rank-28 approach using the fact that n=2, m=2.
    # 
    # Actually, let me try a different approach: use the known result that
    # for 2x2 * 2xp, the minimum rank is at most 7*ceil(p/2) using block Strassen,
    # but potentially lower with a more global approach.
    
    # For now, let's try the block Strassen construction which gives rank 49.
    if rank == 49:
        return _construct_strassen_49(n, m, p)
    
    # For other ranks, we need a different approach.
    # Let's try a greedy/randomized approach for smaller ranks.
    if rank < 49:
        return _try_randomized_construct(rank, n, m, p)
    
    return None

def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    """
    num_blocks = p // 2
    rank = num_blocks * 7

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen linear forms for A (2x2)
    U_forms = np.array([
        [1, 0, 0, 1],   # s=0: A00+A11
        [0, 0, 1, 1],   # s=1: A10+A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00+A01
        [-1, 0, 1, 0],  # s=5: A10-A00
        [0, 1, 0, -1]   # s=6: A01-A11
    ], dtype=np.int64)

    for r in range(num_blocks):
        idx_e = 2 * r           # B[0, 2r]
        idx_f = 2 * r + 1       # B[0, 2r+1]
        idx_g = p + 2 * r       # B[1, 2r]
        idx_h = p + 2 * r + 1   # B[1, 2r+1]

        # W index: k*n + i where k is column of C, i is row of C
        w_idx_c00 = 4 * r       # C[0, 2r]: k=2r, i=0
        w_idx_c11 = 4 * r + 1   # C[1, 2r]: k=2r, i=1
        w_idx_c01 = 4 * r + 2   # C[0, 2r+1]: k=2r+1, i=0
        w_idx_c10 = 4 * r + 3   # C[1, 2r+1]: k=2r+1, i=1

        for s in range(7):
            c = 7 * r + s

            U[0, c] = U_forms[s, 0]
            U[1, c] = U_forms[s, 1]
            U[2, c] = U_forms[s, 2]
            U[3, c] = U_forms[s, 3]

            if s == 0:
                V[idx_e, c] = 1
                V[idx_h, c] = 1
            elif s == 1:
                V[idx_e, c] = 1
            elif s == 2:
                V[idx_f, c] = 1
                V[idx_h, c] = -1
            elif s == 3:
                V[idx_e, c] = -1
                V[idx_g, c] = 1
            elif s == 4:
                V[idx_h, c] = 1
            elif s == 5:
                V[idx_e, c] = 1
                V[idx_f, c] = 1
            elif s == 6:
                V[idx_g, c] = 1
                V[idx_h, c] = 1

            if s == 0:
                W[w_idx_c00, c] = 1
                W[w_idx_c11, c] = 1
            elif s == 1:
                W[w_idx_c11, c] = 1
                W[w_idx_c10, c] = -1
            elif s == 2:
                W[w_idx_c01, c] = 1
                W[w_idx_c10, c] = 1
            elif s == 3:
                W[w_idx_c00, c] = 1
                W[w_idx_c11, c] = 1
            elif s == 4:
                W[w_idx_c00, c] = -1
                W[w_idx_c01, c] = 1
            elif s == 5:
                W[w_idx_c10, c] = 1
            elif s == 6:
                W[w_idx_c00, c] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_randomized_construct(rank, n, m, p):
    """
    Try to construct a rank-`rank` decomposition using a structured greedy approach.
    This is experimental and may not succeed.
    """
    # For now, return None for ranks < 49 as we don't have a known construction
    return None

def _verify_decomposition_full(U, V, W, n, m, p, tol=1e-9):
    """
    Full verification: check all tensor entries.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    # Check all nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0.0
                for r in range(rank):
                    total += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                if abs(total - 1.0) > tol:
                    return False

    # Check zero entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if not (j == j2 and k == k2 and i == i2):
                    total = 0.0
                    for r in range(rank):
                        total += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                    if abs(total) > tol:
                        return False

    return True

def _construct_schoolbook(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
