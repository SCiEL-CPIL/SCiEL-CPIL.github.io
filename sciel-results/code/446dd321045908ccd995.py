"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook_block(si, sj, sk, U, V, W, col, i_off, j_off, k_off):
        """
        Add a schoolbook decomposition for a block of size si x sj x sk.
        i_off, j_off, k_off are the global offsets.
        Returns the new column index.
        """
        for li in range(si):
            for lj in range(sj):
                for lk in range(sk):
                    gi = i_off + li
                    gj = j_off + lj
                    gk = k_off + lk
                    U[gi * m + gj][col] = 1
                    V[gj * p + gk][col] = 1
                    W[gk * n + gi][col] = 1
                    col += 1
        return col

    def strassen_block(si, sj, sk, U, V, W, col, i_off, j_off, k_off):
        """
        Add a Strassen decomposition for a 2x2x2 block.
        Only works if si==2, sj==2, sk==2.
        """
        if si != 2 or sj != 2 or sk != 2:
            return schoolbook_block(si, sj, sk, U, V, W, col, i_off, j_off, k_off)
        
        # Strassen terms for 2x2x2
        terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for u_pat, v_pat, w_pat in terms:
            for (li, lj), val_u in u_pat.items():
                gi = i_off + li
                gj = j_off + lj
                U[gi * m + gj][col] = val_u
            for (lj, lk), val_v in v_pat.items():
                gj = j_off + lj
                gk = k_off + lk
                V[gj * p + gk][col] = val_v
            for (lk, li), val_w in w_pat.items():
                gk = k_off + lk
                gi = i_off + li
                W[gk * n + gi][col] = val_w
            col += 1
        return col

    def build_optimized_decomposition():
        """
        Build a decomposition using a 3x3x3 block and a 3x2x3 block.
        3x3x3 block: Strassen on 2x2x2 corner, Schoolbook on rest.
        3x2x3 block: Strassen on 2x2x2 corner, Schoolbook on rest.
        """
        # Total rank estimate
        # 3x3x3: 7 (Strassen) + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26
        # 3x2x3: 7 (Strassen) + 4 + 4 + 2 = 17
        total_rank = 26 + 17
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        # Block 1: i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
        # Sub-block 1.1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
        col = strassen_block(2, 2, 2, U, V, W, col, 0, 0, 0)
        
        # Sub-block 1.2: i in {0,1}, j in {0,1}, k in {2} -> Schoolbook
        col = schoolbook_block(2, 2, 1, U, V, W, col, 0, 0, 2)
        
        # Sub-block 1.3: i in {0,1}, j in {2}, k in {0,1} -> Schoolbook
        col = schoolbook_block(2, 1, 2, U, V, W, col, 0, 2, 0)
        
        # Sub-block 1.4: i in {0,1}, j in {2}, k in {2} -> Schoolbook
        col = schoolbook_block(2, 1, 1, U, V, W, col, 0, 2, 2)
        
        # Sub-block 1.5: i in {2}, j in {0,1}, k in {0,1} -> Schoolbook
        col = schoolbook_block(1, 2, 2, U, V, W, col, 2, 0, 0)
        
        # Sub-block 1.6: i in {2}, j in {0,1}, k in {2} -> Schoolbook
        col = schoolbook_block(1, 2, 1, U, V, W, col, 2, 0, 2)
        
        # Sub-block 1.7: i in {2}, j in {2}, k in {0,1} -> Schoolbook
        col = schoolbook_block(1, 1, 2, U, V, W, col, 2, 2, 0)
        
        # Sub-block 1.8: i in {2}, j in {2}, k in {2} -> Schoolbook
        col = schoolbook_block(1, 1, 1, U, V, W, col, 2, 2, 2)
        
        # Block 2: i in {0,1,2}, j in {3,4}, k in {3,4,5}
        # Sub-block 2.1: i in {0,1}, j in {3,4}, k in {3,4} -> Strassen
        col = strassen_block(2, 2, 2, U, V, W, col, 0, 3, 3)
        
        # Sub-block 2.2: i in {0,1}, j in {3,4}, k in {5} -> Schoolbook
        col = schoolbook_block(2, 2, 1, U, V, W, col, 0, 3, 5)
        
        # Sub-block 2.3: i in {2}, j in {3,4}, k in {3,4} -> Schoolbook
        col = schoolbook_block(1, 2, 2, U, V, W, col, 2, 3, 3)
        
        # Sub-block 2.4: i in {2}, j in {3,4}, k in {5} -> Schoolbook
        col = schoolbook_block(1, 2, 1, U, V, W, col, 2, 3, 5)
        
        return {"U": U, "V": V, "W": W, "rank": col}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    # Build the optimized decomposition
    result = build_optimized_decomposition()
    rank = result.pop("rank")
    
    # Verify
    if verify(result["U"], result["V"], result["W"], rank):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
