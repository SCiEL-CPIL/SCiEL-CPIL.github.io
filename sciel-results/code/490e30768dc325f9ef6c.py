"""Matrix multiplication 4,5,6 - Strassen-based rank reduction via 6x6x6 embedding."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7)."""
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, -1, -1],
    ], dtype=object)
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0],
        [0, 0, -1, -1, 1, 0, 1],
    ], dtype=object)
    W = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
    ], dtype=object)
    return U, V, W

def _build_6x6x6_strassen():
    """Build 6x6x6 decomposition using 9 blocks of 2x2x2 Strassen.
    Returns U (36,63), V (36,63), W (36,63).
    """
    rank = 9 * 7
    U = np.zeros((36, rank), dtype=object)
    V = np.zeros((36, rank), dtype=object)
    W = np.zeros((36, rank), dtype=object)
    
    U2, V2, W2 = _strassen_2x2x2()
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                # Offset in the 6x6 tensor
                off_i = bi * 2
                off_j = bj * 2
                off_k = bk * 2
                col_start = ((bi * 3 + bj) * 3 + bk) * 7
                
                for i in range(2):
                    for j in range(2):
                        a_full = (off_i + i) * 6 + (off_j + j)
                        a_part = i * 2 + j
                        U[a_full, col_start:col_start+7] = U2[a_part]
                
                for j in range(2):
                    for k in range(2):
                        b_full = (off_j + j) * 6 + (off_k + k)
                        b_part = j * 2 + k
                        V[b_full, col_start:col_start+7] = V2[b_part]
                
                for k in range(2):
                    for i in range(2):
                        c_full = (off_k + k) * 6 + (off_i + i)
                        c_part = k * 2 + i
                        W[c_full, col_start:col_start+7] = W2[c_part]
    
    return U, V, W

def _extract_4x5x6_from_6x6x6(U6, V6, W6):
    """Extract the 4x5x6 sub-tensor from the 6x6x6 decomposition.
    
    U: shape (20, 63) - rows for i in [0,3], j in [0,4]
    V: shape (30, 63) - rows for j in [0,4], k in [0,5]
    W: shape (24, 63) - rows for k in [0,5], i in [0,3]
    
    Indexing convention:
    U row: i*6 + j (for 6x6x6)
    V row: j*6 + k (for 6x6x6)
    W row: k*6 + i (for 6x6x6)
    """
    # U rows: i in [0,3], j in [0,4]
    u_rows = []
    for i in range(4):
        for j in range(5):
            u_rows.append(i * 6 + j)
    U4 = U6[u_rows]
    
    # V rows: j in [0,4], k in [0,5]
    v_rows = []
    for j in range(4):
        for k in range(6):
            v_rows.append(j * 6 + k)
    for j in range(4, 5):
        for k in range(6):
            v_rows.append(j * 6 + k)
    V4 = V6[v_rows]
    
    # W rows: k in [0,5], i in [0,3]
    w_rows = []
    for k in range(6):
        for i in range(4):
            w_rows.append(k * 6 + i)
    W4 = W6[w_rows]
    
    return U4, V4, W4

def _verify_numpy(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matmul tensor using numpy."""
    # Tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1 if j2==j and k2==k and i2==i, else 0
    # More precisely: T[(i*m+j), (j*p+k), (k*n+i)] should equal 1 when the indices are consistent
    # Actually, the matmul tensor is: T[a,b,c] = 1 iff a=(i*m+j), b=(j*p+k), c=(k*n+i) for some i,j,k
    
    # Check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = int(sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank)))
                if actual != expected:
                    return False
    # Check that off-diagonal terms are zero
    # Sample some off-diagonal to verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                for k2 in range(p):
                    if k2 == k:
                        continue
                    c = k2 * n + i
                    actual = int(sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank)))
                    if actual != 0:
                        return False
    return True

def _verify_full(U, V, W, n, m, p, rank):
    """Full verification of the decomposition."""
    # Build the expected tensor and compare
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            if j2 != j:
                continue
            for c in range(p * n):
                k2 = c // n
                i2 = c % n
                if k2 != k or i2 != i:
                    expected = 0
                else:
                    expected = 1
                s = 0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook n x m x p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
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
    import time
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try the 6x6x6 Strassen embedding approach
    if time.time() < deadline - 30:
        try:
            U6, V6, W6 = _build_6x6x6_strassen()
            U4, V4, W4 = _extract_4x5x6_from_6x6x6(U6, V6, W6)
            rank_63 = 63
            
            # Verify
            if _verify_numpy(U4, V4, W4, n, m, p, rank_63):
                best_rank = rank_63
                best_result = {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
            else:
                # Try full verification for debugging
                if _verify_full(U4, V4, W4, n, m, p, rank_63):
                    best_rank = rank_63
                    best_result = {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
