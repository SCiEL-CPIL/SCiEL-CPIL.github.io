import numpy as np
import time

DIMENSIONS = [5, 5, 5]
DEADLINE_SECONDS = 140.0

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
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

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # r=0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=1: M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # r=2: M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # r=3: M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # r=4: M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # r=5: M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=6: M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _rect_block_decomp(n, m, p):
    """
    Decompose n x m x p matrix multiplication.
    Uses splitting strategies to minimize rank.
    Returns U (n*m, rank), V (m*p, rank), W (p*n, rank).
    """
    if m == 0 or n == 0 or p == 0:
        return [], [], 0
    
    # Base cases
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2()
        return U, V, 7
        
    # If one dimension is 1, it's just outer products
    if n == 1:
        rank = m * p
        U = [[0] * rank for _ in range(m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, rank
        
    if m == 1:
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(n * p)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[i * p + k][col] = 1
                col += 1
        return U, V, rank
        
    if p == 1:
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank for _ in range(n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, rank

    # Recursive splitting
    # Try splitting m (inner dimension)
    best_rank = n * m * p
    best_U = None
    best_V = None
    
    # Split m into m1 + m2
    for m1 in range(1, m):
        m2 = m - m1
        # Recurse
        U1, V1, r1 = _rect_block_decomp(n, m1, p)
        U2, V2, r2 = _rect_block_decomp(n, m2, p)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            # Construct U, V, W
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            # Copy U1, V1
            for i in range(n):
                for j in range(m1):
                    for c in range(r1):
                        U[i * m + j][c] = U1[i * m1 + j][c]
            for j in range(m1):
                for k in range(p):
                    for c in range(r1):
                        V[j * p + k][c] = V1[j * p + k][c]
            for k in range(p):
                for i in range(n):
                    for c in range(r1):
                        W[k * n + i][c] = W1[k * n + i][c]
                        
            # Copy U2, V2 (offset indices)
            for i in range(n):
                for j in range(m2):
                    for c in range(r2):
                        U[i * m + (m1 + j)][c + r1] = U2[i * m2 + j][c]
            for j in range(m2):
                for k in range(p):
                    for c in range(r2):
                        V[(m1 + j) * p + k][c + r1] = V2[j * p + k][c]
            for k in range(p):
                for i in range(n):
                    for c in range(r2):
                        W[k * n + i][c + r1] = W2[k * n + i][c]
                        
            best_U = U
            best_V = V
            best_W = W

    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1, r1 = _rect_block_decomp(n1, m, p)
        U2, V2, W2, r2 = _rect_block_decomp(n2, m, p)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            for i in range(n1):
                for j in range(m):
                    for c in range(r1):
                        U[i * m + j][c] = U1[i * m + j][c]
            for i in range(n2):
                for j in range(m):
                    for c in range(r2):
                        U[(n1 + i) * m + j][c + r1] = U2[i * m + j][c]
                        
            for j in range(m):
                for k in range(p):
                    for c in range(r1):
                        V[j * p + k][c] = V1[j * p + k][c]
                    for c in range(r2):
                        V[j * p + k][c + r1] = V2[j * p + k][c]
                        
            for k in range(p):
                for i in range(n1):
                    for c in range(r1):
                        W[k * n + i][c] = W1[k * n1 + i][c]
                for i in range(n2):
                    for c in range(r2):
                        W[k * n + (n1 + i)][c + r1] = W2[k * n2 + i][c]
                        
            best_U = U
            best_V = V
            best_W = W

    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1, r1 = _rect_block_decomp(n, m, p1)
        U2, V2, W2, r2 = _rect_block_decomp(n, m, p2)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            U = [[0] * r for _ in range(n * m)]
            V = [[0] * r for _ in range(m * p)]
            W = [[0] * r for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for c in range(r1):
                        U[i * m + j][c] = U1[i * m + j][c]
                    for c in range(r2):
                        U[i * m + j][c + r1] = U2[i * m + j][c]
                        
            for j in range(m):
                for k in range(p1):
                    for c in range(r1):
                        V[j * p + k][c] = V1[j * p1 + k][c]
                for k in range(p2):
                    for c in range(r2):
                        V[j * p + (p1 + k)][c + r1] = V2[j * p2 + k][c]
                        
            for k in range(p1):
                for i in range(n):
                    for c in range(r1):
                        W[k * n + i][c] = W1[k * n + i][c]
            for k in range(p2):
                for i in range(n):
                    for c in range(r2):
                        W[(p1 + k) * n + i][c + r1] = W2[k * n + i][c]
                        
            best_U = U
            best_V = V
            best_W = W

    if best_U is None:
        # Fallback to schoolbook
        res = _schoolbook(n, m, p)
        return res["U"], res["V"], n*m*p

    return best_U, best_V, best_W

def _construct_3_2_partition():
    n, m, p = 5, 5, 5
    # Blocks
    # A: 3x3, 3x2, 2x3, 2x2
    # B: 3x3, 3x2, 2x3, 2x2
    
    blocks = [
        (0, 3, 0, 3, 0, 3, 0, 3), # A[0:3,0:3] B[0:3,0:3] -> C[0:3,0:3]
        (0, 3, 0, 3, 0, 3, 3, 5), # A[0:3,0:3] B[0:3,3:5] -> C[0:3,3:5]
        (0, 3, 3, 5, 3, 5, 0, 3), # A[0:3,3:5] B[3:5,0:3] -> C[0:3,0:3]
        (0, 3, 3, 5, 3, 5, 3, 5), # A[0:3,3:5] B[3:5,3:5] -> C[0:3,3:5]
        (3, 5, 0, 3, 0, 3, 0, 3), # A[3:5,0:3] B[0:3,0:3] -> C[3:5,0:3]
        (3, 5, 0, 3, 0, 3, 3, 5), # A[3:5,0:3] B[0:3,3:5] -> C[3:5,3:5]
        (3, 5, 3, 5, 3, 5, 0, 3), # A[3:5,3:5] B[3:5,0:3] -> C[3:5,0:3]
        (3, 5, 3, 5, 3, 5, 3, 5), # A[3:5,3:5] B[3:5,3:5] -> C[3:5,3:5]
    ]
    
    total_rank = 0
    for b in blocks:
        # Calculate local rank
        ni = b[1] - b[0]
        mj = b[3] - b[2]
        pk = b[5] - b[4]
        # Wait, the block definition is:
        # A[i_start:i_end, k_start:k_end] * B[k_start:k_end, j_start:j_end] -> C[i_start:i_end, j_start:j_end]
        # So:
        # i_start, i_end, k_start, k_end, j_start, j_end
        # A is i x k. B is k x j.
        # ni = i_end - i_start
        # mk = k_end - k_start
        # pj = j_end - j_start
        
        ni = b[1] - b[0]
        mk = b[3] - b[2]
        pj = b[5] - b[4]
        
        U, V, r = _rect_block_decomp(ni, mk, pj)
        total_rank += r

    U_global = [[0] * total_rank for _ in range(n * m)]
    V_global = [[0] * total_rank for _ in range(m * p)]
    W_global = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for b in blocks:
        i_start, i_end, k_start, k_end, j_start, j_end = b
        ni = i_end - i_start
        mk = k_end - k_start
        pj = j_end - j_start
        
        U, V, r = _rect_block_decomp(ni, mk, pj)
        
        for i in range(ni):
            for k in range(mk):
                for c in range(r):
                    U_global[(i_start + i) * m + (k_start + k)][col + c] += U[i * mk + k][c]
                    
        for k in range(mk):
            for j in range(pj):
                for c in range(r):
                    V_global[(k_start + k) * p + (j_start + j)][col + c] += V[k * pj + j][c]
                    
        for j in range(pj):
            for i in range(ni):
                for c in range(r):
                    W_global[(j_start + j) * n + (i_start + i)][col + c] += W[j * ni + i][c]
                    
        col += r
        
    return {"U": U_global, "V": V_global, "W": W_global}

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    
    for i in range(n):
        for j in range(p):
            s = 0.0
            for r in range(rank):
                f_r = 0.0
                for k in range(m):
                    f_r += U[i*m+k][r] * A[i,k]
                g_r = 0.0
                for k in range(m):
                    g_r += V[k*p+j][r] * B[k,j]
                s += W[j*n+i][r] * f_r * g_r
            C_computed[i,j] = s
            
    if np.max(np.abs(C_computed - C_expected)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Fallback
    fallback = _schoolbook(n, m, p)
    
    # Try 3+2 partition
    if time.time() < deadline:
        try:
            result = _construct_3_2_partition()
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
        except Exception as e:
            print(f"3+2 partition construction failed: {e}")
            
    return fallback

# EVOLVE-BLOCK-END
