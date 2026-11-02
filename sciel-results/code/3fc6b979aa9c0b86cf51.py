import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def strassen_2x2x2():
        """
        Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (u, v, w) with 4 entries each.
        """
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # (A00+A11)(B00+B11)
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # (A10+A11)B00
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # A00(B10-B11)
            ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 0]), # A11(B01-B00)
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # (A00+A01)B11
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # (A10-A00)(B00+B01)
            ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]), # (A01-A11)(B10+B11)
        ]

    def build_strassen_decomposition():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_terms = strassen_2x2x2()
        
        col = 0
        # Split m=4 into 2 blocks of 2
        # Split p=6 into 3 blocks of 2
        for col_blk in range(3):
            for row_blk in range(2):
                for s_idx in range(7):
                    u_vec, v_vec, w_vec = strassen_terms[s_idx]
                    
                    # Fill U: n*m = 2*4 = 8 entries
                    # Local A is 2x2. Global i in [0,1], j in [row_blk*2, row_blk*2+1]
                    for a in range(2):
                        for b in range(2):
                            local_idx = a * 2 + b
                            global_i = a
                            global_j = row_blk * 2 + b
                            U[global_i * m + global_j][col] += u_vec[local_idx]
                            
                    # Fill V: m*p = 4*6 = 24 entries
                    # Local B is 2x2. Global j in [row_blk*2, row_blk*2+1], k in [col_blk*2, col_blk*2+1]
                    for a in range(2):
                        for b in range(2):
                            local_idx = a * 2 + b
                            # v_vec index corresponds to B's local (row, col)
                            # B's local row is the inner index (j). B's local col is outer (k).
                            # In row-major, local_idx = row * 2 + col.
                            # So local row = a, local col = b.
                            global_j = row_blk * 2 + a
                            global_k = col_blk * 2 + b
                            V[global_j * p + global_k][col] += v_vec[local_idx]
                            
                    # Fill W: p*n = 6*2 = 12 entries
                    # Local C is 2x2. Global k in [col_blk*2, col_blk*2+1], i in [0,1]
                    for a in range(2):
                        for b in range(2):
                            local_idx = a * 2 + b
                            # w_vec index corresponds to C's local (row, col)
                            # C's local row is outer (k). C's local col is inner (i).
                            # In row-major, local_idx = row * 2 + col.
                            # So local row = a, local col = b.
                            global_k = col_blk * 2 + a
                            global_i = b
                            W[global_k * n + global_i][col] += w_vec[local_idx]
                            
                    col += 1
        
        return U, V, W, rank

    def verify(U, V, W, r, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for rr in range(r):
                        val += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(val - 1) > tol:
                        return False
        return True

    U, V, W, r = build_strassen_decomposition()
    
    # Optional: Verify to be safe
    if not verify(U, V, W, r):
        # Fallback to schoolbook if Strassen fails (should not happen)
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
