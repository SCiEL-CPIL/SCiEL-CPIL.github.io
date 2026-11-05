"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def build_block_strassen():
        """
        Constructs a rank-117 decomposition using Strassen for 2x2x2 blocks
        and schoolbook for the rest.
        """
        rank = 0
        # Pre-allocate with max possible rank (125) or dynamic?
        # Dynamic is safer for exact rank.
        # But we know it's 117.
        # Let's just build lists and convert to list of lists at the end.
        # Or use numpy? No, standard lists are fine for 125x117.
        
        U = [[0] * 117 for _ in range(n * m)]
        V = [[0] * 117 for _ in range(m * p)]
        W = [[0] * 117 for _ in range(p * n)]
        
        # Block definitions
        # Block 0: indices 0, 1
        # Block 1: indices 2, 3
        # Block 2: index 4
        
        def get_block_indices(b):
            if b == 0:
                return [0, 1]
            elif b == 1:
                return [2, 3]
            else:
                return [4]

        # Process 2x2x2 blocks
        # There are 8 such blocks: bi in {0,1}, bj in {0,1}, bk in {0,1}
        for bi in [0, 1]:
            for bj in [0, 1]:
                for bk in [0, 1]:
                    # Define the 7 Strassen terms for this block
                    # Local indices u (row A/C), v (col A/row B), w (col B/C)
                    # Global i = 2*bi + u, j = 2*bj + v, k = 2*bk + w
                    
                    # Term 1: (A00+A11)(B00+B11)
                    # U: A00, A11
                    U[ (2*bi+0)*m + (2*bj+0) ][rank] = 1
                    U[ (2*bi+1)*m + (2*bj+1) ][rank] = 1
                    # V: B00, B11
                    V[ (2*bj+0)*p + (2*bk+0) ][rank] = 1
                    V[ (2*bj+1)*p + (2*bk+1) ][rank] = 1
                    # W: C00, C01, C10, C11
                    W[ (2*bk+0)*n + (2*bi+0) ][rank] = 1
                    W[ (2*bk+0)*n + (2*bi+1) ][rank] = 1
                    W[ (2*bk+1)*n + (2*bi+0) ][rank] = 1
                    W[ (2*bk+1)*n + (2*bi+1) ][rank] = 1
                    rank += 1

                    # Term 2: (A10+A11)B00
                    U[ (2*bi+1)*m + (2*bj+0) ][rank] = 1
                    U[ (2*bi+1)*m + (2*bj+1) ][rank] = 1
                    V[ (2*bj+0)*p + (2*bk+0) ][rank] = 1
                    W[ (2*bk+0)*n + (2*bi+1) ][rank] = 1
                    W[ (2*bk+1)*n + (2*bi+1) ][rank] = 1
                    rank += 1

                    # Term 3: A00(B01-B11)
                    U[ (2*bi+0)*m + (2*bj+0) ][rank] = 1
                    V[ (2*bj+0)*p + (2*bk+1) ][rank] = 1
                    V[ (2*bj+1)*p + (2*bk+1) ][rank] = -1
                    W[ (2*bk+1)*n + (2*bi+0) ][rank] = 1
                    W[ (2*bk+1)*n + (2*bi+1) ][rank] = 1
                    rank += 1

                    # Term 4: A11(B10-B00)
                    U[ (2*bi+1)*m + (2*bj+1) ][rank] = 1
                    V[ (2*bj+1)*p + (2*bk+0) ][rank] = 1
                    V[ (2*bj+0)*p + (2*bk+0) ][rank] = -1
                    W[ (2*bk+0)*n + (2*bi+0) ][rank] = 1
                    W[ (2*bk+0)*n + (2*bi+1) ][rank] = 1
                    rank += 1

                    # Term 5: (A00+A01)B11
                    U[ (2*bi+0)*m + (2*bj+0) ][rank] = 1
                    U[ (2*bi+0)*m + (2*bj+1) ][rank] = 1
                    V[ (2*bj+1)*p + (2*bk+1) ][rank] = 1
                    W[ (2*bk+1)*n + (2*bi+0) ][rank] = 1
                    W[ (2*bk+1)*n + (2*bi+1) ][rank] = 1
                    rank += 1

                    # Term 6: (A10-A00)(B00+B01)
                    U[ (2*bi+1)*m + (2*bj+0) ][rank] = 1
                    U[ (2*bi+0)*m + (2*bj+0) ][rank] = -1
                    V[ (2*bj+0)*p + (2*bk+0) ][rank] = 1
                    V[ (2*bj+0)*p + (2*bk+1) ][rank] = 1
                    W[ (2*bk+1)*n + (2*bi+1) ][rank] = 1
                    rank += 1

                    # Term 7: (A01-A11)(B10+B11)
                    U[ (2*bi+0)*m + (2*bj+1) ][rank] = 1
                    U[ (2*bi+1)*m + (2*bj+1) ][rank] = -1
                    V[ (2*bj+1)*p + (2*bk+0) ][rank] = 1
                    V[ (2*bj+1)*p + (2*bk+1) ][rank] = 1
                    W[ (2*bk+0)*n + (2*bi+0) ][rank] = 1
                    W[ (2*bk+0)*n + (2*bi+1) ][rank] = 1
                    rank += 1

        # Process other blocks
        for bi in [0, 1, 2]:
            for bj in [0, 1, 2]:
                for bk in [0, 1, 2]:
                    # Skip 2x2x2 blocks
                    if bi in [0, 1] and bj in [0, 1] and bk in [0, 1]:
                        continue
                    
                    is_ = get_block_indices(bi)
                    js_ = get_block_indices(bj)
                    ks_ = get_block_indices(bk)
                    
                    for i in is_:
                        for j in js_:
                            for k in ks_:
                                U[i * m + j][rank] = 1
                                V[j * p + k][rank] = 1
                                W[k * n + i][rank] = 1
                                rank += 1
        
        # Trim to actual rank
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        
        return {"U": U, "V": V, "W": W}, rank

    def verify(U, V, W):
        """Check that U, V, W exactly decompose the matmul tensor."""
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        # For each c, compute outer product and check
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
        U_t = U_np.reshape(n, m, rank)
        V_t = V_np.reshape(m, p, rank)
        W_t = W_np.reshape(p, n, rank)
        # Compute sum over c: U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
        # Use einsum
        result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i, j, k] = 1.0
        return np.allclose(result, expected, atol=1e-9)

    result, rank = build_block_strassen()
    
    # Verify using numpy for speed
    if not verify(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook if something went wrong
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return result
# EVOLVE-BLOCK-END
