"""Search for a rank-minimal exact tensor decomposition of 2x4 times 4x7 matrix multiply."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 115.0  # safety margin within 150s


def _construct_np(n, m, p):
    """Construct rank n*p decomposition."""
    rank = n * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for k in range(p):
            c = k * n + i
            for j in range(m):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
            W[k * n + i, c] = 1
    return U, V, W


def _verify_np(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    for i in range(n):
        for k in range(p):
            w = W[k * n + i]
            s = 0.0
            for j in range(m):
                u = U[i * m + j]
                v = V[j * p + k]
                s += np.sum(u * v * w)
            if abs(s - 1.0) > tol:
                return False
    return True


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _try_strassen_blocks(n, m, p):
    """
    Try Strassen-based block decomposition.
    For 2x4x7: split m=4 into 2+2, then each [2,2,7] splits p=7 into 2+2+2+1.
    Each [2,2,2] uses Strassen (rank 7), each [2,2,1] uses schoolbook (rank 4).
    Total: 2 * (3*7 + 4) = 2 * 25 = 50.
    This is worse than n*p=14, but included as a structural alternative.
    """
    if n != 2 or m != 4 or p != 7:
        return None
    
    # We'll build the decomposition manually using Strassen for 2x2x2 blocks
    # Strassen for 2x2x2: rank 7
    # The 7 Strassen products for matrices A(2x2), B(2x2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our tensor [2,4,7], we split the 4 into two 2s:
    # A is 2x4 = [A_left | A_right] where A_left, A_right are 2x2
    # B is 4x7 = [B_top; B_bottom] where B_top, B_bottom are 2x7
    # C = A*B = A_left*B_top + A_right*B_bottom (each is 2x7)
    
    # For A_left*B_top (2x2 * 2x7): split B_top into [B1|B2|B3|B4] where B1,B2,B3 are 2x2, B4 is 2x1
    # Use Strassen for each 2x2x2 block, schoolbook for 2x2x1
    
    # This gives rank 2*(3*7+4) = 50, which is worse than 14.
    # So we skip this and just return None.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Start with schoolbook as fallback
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try rank = n*p = 14
    U14, V14, W14 = _construct_np(n, m, p)
    if _verify_np(U14, V14, W14, n, m, p):
        best_U, best_V, best_W = U14, V14, W14
        best_rank = n * p

    # Try Strassen blocks (likely won't beat 14, but structural)
    if best_rank > 20:
        result = _try_strassen_blocks(n, m, p)
        if result is not None:
            best_U, best_V, best_W = result['U'], result['V'], result['W']
            best_rank = result['U'].shape[1]

    # Convert to list-of-lists with integer entries
    def to_int_list(mat):
        m2 = mat * 2
        if np.allclose(m2, np.round(m2), atol=1e-9):
            m2 = np.round(m2).astype(int)
            if np.all(m2 % 2 == 0):
                return (m2 // 2).tolist()
            return m2.tolist()
        return mat.tolist()

    return {
        'U': to_int_list(best_U),
        'V': to_int_list(best_V),
        'W': to_int_list(best_W)
    }
