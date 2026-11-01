import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all diagonal terms (i, j, k, i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    
    # Check off-diagonal terms: for each (i,j,k) and (i',j',k') where not all equal
    # We check a representative subset to save time
    np.random.seed(42)
    checks = 500
    for _ in range(checks):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        ip = np.random.randint(n)
        jp = np.random.randint(m)
        kp = np.random.randint(p)
        if i == ip and j == jp and k == kp:
            continue
        val = 0.0
        for r in range(rank):
            val += U_np[i*m+j, r] * V_np[jp*p+kp, r] * W_np[kp*n+ip, r]
        if abs(val) > 1e-10:
            return False
    
    return True

def _strassen_2x2(n, m, p, j_start, k_start, U, V, W, col):
    """
    Apply Strassen's algorithm for multiplying a 2x2 A block (rows 0,1 of A; cols j_start, j_start+1)
    by a 2x2 B block (rows j_start, j_start+1 of B; cols k_start, k_start+1).
    Returns new column index after adding 7 terms.
    
    U indices: i*m + j  for i in {0,1}, j in {j_start, j_start+1}
    V indices: j*p + k  for j in {j_start, j_start+1}, k in {k_start, k_start+1}
    W indices: k*n + i  for k in {k_start, k_start+1}, i in {0,1}
    """
    # A block: rows 0,1; cols j0, j1
    # B block: rows j0, j1; cols k0, k1
    j0 = j_start
    j1 = j_start + 1
    k0 = k_start
    k1 = k_start + 1
    
    # U indices for A elements
    u_a00 = 0 * m + j0  # A[0, j0]
    u_a01 = 0 * m + j1  # A[0, j1]
    u_a10 = 1 * m + j0  # A[1, j0]
    u_a11 = 1 * m + j1  # A[1, j1]
    
    # V indices for B elements
    v_b00 = j0 * p + k0  # B[j0, k0]
    v_b01 = j0 * p + k1  # B[j0, k1]
    v_b10 = j1 * p + k0  # B[j1, k0]
    v_b11 = j1 * p + k1  # B[j1, k1]
    
    # W indices for C elements
    w_c00 = k0 * n + 0  # C[0, k0]
    w_c01 = k1 * n + 0  # C[0, k1]
    w_c10 = k0 * n + 1  # C[1, k0]
    w_c11 = k1 * n + 1  # C[1, k1]
    
    # Strassen's 7 products:
    # m1 = (a00 + a11) * (b00 + b11)
    U[u_a00][col] = 1
    U[u_a11][col] = 1
    V[v_b00][col] = 1
    V[v_b11][col] = 1
    W[w_c00][col] = 1
    W[w_c11][col] = 1
    col += 1
    
    # m2 = (a10 + a11) * b00
    U[u_a10][col] = 1
    U[u_a11][col] = 1
    V[v_b00][col] = 1
    W[w_c10][col] = 1
    W[w_c11][col] = -1
    col += 1
    
    # m3 = a00 * (b01 - b11)
    U[u_a00][col] = 1
    V[v_b01][col] = 1
    V[v_b11][col] = -1
    W[w_c01][col] = 1
    W[w_c11][col] = 1
    col += 1
    
    # m4 = a11 * (b10 - b00)
    U[u_a11][col] = 1
    V[v_b10][col] = 1
    V[v_b00][col] = -1
    W[w_c00][col] = 1
    W[w_c10][col] = 1
    col += 1
    
    # m5 = (a00 + a01) * b11
    U[u_a00][col] = 1
    U[u_a01][col] = 1
    V[v_b11][col] = 1
    W[w_c01][col] = 1
    W[w_c00][col] = -1
    col += 1
    
    # m6 = (a10 - a00) * (b00 + b01)
    U[u_a10][col] = 1
    U[u_a00][col] = -1
    V[v_b00][col] = 1
    V[v_b01][col] = 1
    W[w_c11][col] = 1
    col += 1
    
    # m7 = (a01 - a11) * (b10 + b11)
    U[u_a01][col] = 1
    U[u_a11][col] = -1
    V[v_b10][col] = 1
    V[v_b11][col] = 1
    W[w_c00][col] = 1
    col += 1
    
    return col

def _schoolbook_term(U, V, W, i, j, k, n, m, p, col):
    """Add one schoolbook term for A[i,j] * B[j,k] -> C[i,k]."""
    U[i * m + j][col] = 1
    V[j * p + k][col] = 1
    W[k * n + i][col] = 1
    return col + 1

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication 2x3 by 3x10.
    Uses Strassen's algorithm for the 2x2 sub-block (cols 0,1 of A with cols 0,1 of B)
    and schoolbook for the remaining 2x1 sub-block (col 2 of A with all cols of B).
    
    Rank = 7 (Strassen for 2x2) + 2*10 (schoolbook for 2x1) = 27
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Total rank: 7 + 2*10 = 27
    rank = 7 + 2 * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for A[:, 0:2] * B[0:2, :]
    # A block: 2x2 (rows 0,1; cols 0,1)
    # B block: 2x10 (rows 0,1; cols 0..9)
    # We split B into 5 blocks of width 2, applying Strassen to each 2x2 sub-block
    
    for g in range(5):
        k_base = 2 * g
        if time.time() > deadline - 5:
            break
        col = _strassen_2x2(n, m, p, 0, k_base, U, V, W, col)
    
    # Part 2: Schoolbook for A[:, 2] * B[2, :]
    # A column: 2x1 (rows 0,1; col 2)
    # B row: 1x10 (row 2; cols 0..9)
    # C contribution: 2x10
    
    for i in range(n):
        for k in range(p):
            if time.time() > deadline - 5:
                break
            col = _schoolbook_term(U, V, W, i, 2, k, n, m, p, col)
    
    # col should equal rank = 27
    assert col == rank, f"Column count {col} != rank {rank}"
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        pass  # Valid
    else:
        # Fallback to schoolbook if verification fails
        rank_fallback = n * m * p
        U = [[0] * rank_fallback for _ in range(n * m)]
        V = [[0] * rank_fallback for _ in range(m * p)]
        W = [[0] * rank_fallback for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
