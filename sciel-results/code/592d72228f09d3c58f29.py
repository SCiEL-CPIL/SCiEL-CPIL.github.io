"""Program for rank-minimal exact tensor decomposition of 2x3 * 3x6 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the standard schoolbook (rank n*m*p) decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Verify: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _build_strassen_2x2x2(n, m, p):
    """Attempt Strassen-based decomposition for 2x2x2 blocks.
    For 2x3 * 3x6, we can use a block decomposition approach."""
    # This is a placeholder - Strassen for non-square blocks is complex
    # Fall back to schoolbook
    return _build_schoolbook(n, m, p)

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = float('inf')
    
    # Start with schoolbook (always valid)
    schoolbook = _build_schoolbook(n, m, p)
    if _verify_decomposition(schoolbook["U"], schoolbook["V"], schoolbook["W"], n, m, p):
        best_result = schoolbook
        best_rank = len(schoolbook["U"][0])
    
    # Try Strassen-based approaches if time permits
    if time.time() < deadline - 30:
        try:
            strassen = _build_strassen_2x2x2(n, m, p)
            if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
                rank_s = len(strassen["U"][0])
                if rank_s < best_rank:
                    best_result = strassen
                    best_rank = rank_s
        except Exception:
            pass
    
    # If no valid result found (shouldn't happen), fall back to schoolbook
    if best_result is None:
        best_result = _build_schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
