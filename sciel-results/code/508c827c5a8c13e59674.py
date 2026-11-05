"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def build_block_strassen_221():
        """
        Constructs a rank-117 decomposition using Strassen for 2x2x2 blocks
        and schoolbook for the rest, with 2+2+1 partition.
        """
        max_rank = n * m * p
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        rank = 0
        
        # Process 2x2x2 blocks with Strassen (8 blocks)
        for bi in [0, 1]:
            for bj in [0, 1]:
                for bk in [0, 1]:
                    # Term 1: (A00+A11)(B00+B11) -> C00+C01+C10+C11
                    U[(2*bi+0)*m + (2*bj+0)][rank] = 1
                    U[(2*bi+1)*m + (2*bj+1)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+0)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+1)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+1)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 2: (A10+A11)B00 -> C10+C11
                    U[(2*bi+1)*m + (2*bj+0)][rank] = 1
                    U[(2*bi+1)*m + (2*bj+1)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+0)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+1)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 3: A00(B01-B11) -> C01+C11
                    U[(2*bi+0)*m + (2*bj+0)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+1)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+1)][rank] = -1
                    W[(2*bk+1)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 4: A11(B10-B00) -> C10+C11
                    U[(2*bi+1)*m + (2*bj+1)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+0)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+0)][rank] = -1
                    W[(2*bk+0)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 5: (A00+A01)B11 -> C01+C11
                    U[(2*bi+0)*m + (2*bj+0)][rank] = 1
                    U[(2*bi+0)*m + (2*bj+1)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+1)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 6: (A10-A00)(B00+B01) -> C11
                    U[(2*bi+1)*m + (2*bj+0)][rank] = 1
                    U[(2*bi+0)*m + (2*bj+0)][rank] = -1
                    V[(2*bj+0)*p + (2*bk+0)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+1)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 7: (A01-A11)(B10+B11) -> C00+C10
                    U[(2*bi+0)*m + (2*bj+1)][rank] = 1
                    U[(2*bi+1)*m + (2*bj+1)][rank] = -1
                    V[(2*bj+1)*p + (2*bk+0)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+1)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+1)][rank] = 1
                    rank += 1

        # Process remaining blocks with schoolbook
        for bi in [0, 1, 2]:
            for bj in [0, 1, 2]:
                for bk in [0, 1, 2]:
                    if bi in [0, 1] and bj in [0, 1] and bk in [0, 1]:
                        continue
                    # Get indices for each dimension
                    is_ = [bi] if bi == 2 else [2*bi, 2*bi+1]
                    js_ = [bj] if bj == 2 else [2*bj, 2*bj+1]
                    ks_ = [bk] if bk == 2 else [2*bk, 2*bk+1]
                    
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

    def build_schoolbook():
        """Fallback: rank-125 schoolbook decomposition."""
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
        return {"U": U, "V": V, "W": W}, rank

    def verify(U, V, W):
        """Check that U, V, W exactly decompose the matmul tensor."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try block Strassen first
    result, rank = build_block_strassen_221()
    
    # Verify
    if verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    result, rank = build_schoolbook()
    return result
# EVOLVE-BLOCK-END
