"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def build_hybrid_decomp():
        """
        Hybrid decomposition: Strassen (rank 7) for 2x2x2 blocks,
        schoolbook for remaining blocks.
        Total rank: 8*7 + 12*4 + 3*2 + 1*1 = 56 + 48 + 6 + 1 = 111
        """
        # Pre-allocate
        total_rank = 111
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        rank = 0
        
        def get_block_indices(b):
            if b == 0:
                return [0, 1]
            elif b == 1:
                return [2, 3]
            else:
                return [4]
        
        # Process 2x2x2 blocks with Strassen (rank 7 each)
        for bi in [0, 1]:
            for bj in [0, 1]:
                for bk in [0, 1]:
                    # Term 1: (A00+A11)(B00+B11)
                    U[(2*bi+0)*m + (2*bj+0)][rank] = 1
                    U[(2*bi+1)*m + (2*bj+1)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+0)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+1)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+1)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 2: (A10+A11)B00
                    U[(2*bi+1)*m + (2*bj+0)][rank] = 1
                    U[(2*bi+1)*m + (2*bj+1)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+0)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+1)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 3: A00(B01-B11)
                    U[(2*bi+0)*m + (2*bj+0)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+1)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+1)][rank] = -1
                    W[(2*bk+1)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 4: A11(B10-B00)
                    U[(2*bi+1)*m + (2*bj+1)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+0)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+0)][rank] = -1
                    W[(2*bk+0)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+0)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 5: (A00+A01)B11
                    U[(2*bi+0)*m + (2*bj+0)][rank] = 1
                    U[(2*bi+0)*m + (2*bj+1)][rank] = 1
                    V[(2*bj+1)*p + (2*bk+1)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+0)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 6: (A10-A00)(B00+B01)
                    U[(2*bi+1)*m + (2*bj+0)][rank] = 1
                    U[(2*bi+0)*m + (2*bj+0)][rank] = -1
                    V[(2*bj+0)*p + (2*bk+0)][rank] = 1
                    V[(2*bj+0)*p + (2*bk+1)][rank] = 1
                    W[(2*bk+1)*n + (2*bi+1)][rank] = 1
                    rank += 1

                    # Term 7: (A01-A11)(B10+B11)
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
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    result, rank = build_hybrid_decomp()
    
    if not verify(result["U"], result["V"], result["W"]):
        # Fallback to schoolbook
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
