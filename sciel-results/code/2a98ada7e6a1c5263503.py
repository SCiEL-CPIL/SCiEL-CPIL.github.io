import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Target rank is 60
    rank = 60
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen block for M_{2,2,2}
    # i_off: offset in i (0..2)
    # j_off: offset in j (0..3)
    # k_off: offset in k (0..5)
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # U vectors (size 4)
        u = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, 1], # M2
            [1, 0, 0, 0], # M3
            [0, 0, 0, 1], # M4
            [1, 1, 0, 0], # M5
            [-1, 0, 1, 0],# M6
            [0, 1, 0, -1] # M7
        ]
        # V vectors (size 4)
        v = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 0, 1, -1],# M3
            [1, -1, 0, 0],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        # W vectors (size 4)
        # C_{00} = M1 + M4 - M5 + M7
        # C_{01} = M3 + M5
        # C_{10} = M2 + M4
        # C_{11} = M1 - M2 + M3 + M6
        w = [
            [1, 0, 0, 1], # M1
            [0, 0, 1, -1],# M2
            [0, 1, 0, 1], # M3
            [1, 0, 1, 0], # M4
            [-1, 1, 0, 0],# M5
            [0, 0, 0, 1], # M6
            [1, 0, 0, 0]  # M7
        ]
        
        for t in range(7):
            for di in range(4):
                U[(i_off + di // 2) * m + (j_off + di % 2), col] += u[t][di]
            for dj in range(4):
                V[(j_off + dj // 2) * p + (k_off + dj % 2), col] += v[t][dj]
            for dk in range(4):
                W[(k_off + dk // 2) * n + (i_off + dk % 2), col] += w[t][dk]
            col += 1

    # Helper to add a block for M_{1,2,2}
    # i is single index i_off
    # j_off, k_off
    def add_122(i_off, j_off, k_off):
        nonlocal col
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        # Rank 3 decomposition:
        # M1 = A0 (B00 + B10)
        # M2 = A1 (B01 + B11)
        # M3 = (A0 + A1) (B01 - B10)
        # C0 = M1 - 0.5 M3
        # C1 = M2 + 0.5 M3
        
        u = [
            [1, 0], # M1: A0
            [0, 1], # M2: A1
            [1, 1]  # M3: A0+A1
        ]
        v = [
            [1, 1], # M1: B00+B10
            [1, 1], # M2: B01+B11
            [-1, 1] # M3: -B10+B01
        ]
        w = [
            [1, 0], # M1: C0
            [0, 1], # M2: C1
            [-0.5, 0.5] # M3: -0.5 C0 + 0.5 C1
        ]
        
        for t in range(3):
            for di in range(2):
                U[(i_off + 0) * m + (j_off + di), col] += u[t][di]
            for dj in range(2):
                V[(j_off + 0) * p + (k_off + dj), col] += v[t][dj]
            for dk in range(2):
                W[(k_off + dk) * n + (i_off + 0), col] += w[t][dk]
            col += 1

    # Block 1: A_{11} B_{11} (i=0,1; j=0,1; k=0,1) -> M_{2,2,2}
    add_strassen_222(0, 0, 0)
    # Block 2: A_{11} B_{12} (i=0,1; j=0,1; k=2,3) -> M_{2,2,2}
    add_strassen_222(0, 0, 2)
    # Block 3: A_{12} B_{21} (i=0,1; j=2,3; k=0,1) -> M_{2,2,2}
    add_strassen_222(0, 2, 0)
    # Block 4: A_{12} B_{22} (i=0,1; j=2,3; k=2,3) -> M_{2,2,2}
    add_strassen_222(0, 2, 2)
    
    # Block 5: A_{21} B_{11} (i=2; j=0,1; k=0,1) -> M_{1,2,2}
    add_122(2, 0, 0)
    # Block 6: A_{21} B_{12} (i=2; j=0,1; k=2,3) -> M_{1,2,2}
    add_122(2, 0, 2)
    # Block 7: A_{22} B_{21} (i=2; j=2,3; k=0,1) -> M_{1,2,2}
    add_122(2, 2, 0)
    # Block 8: A_{22} B_{22} (i=2; j=2,3; k=2,3) -> M_{1,2,2}
    add_122(2, 2, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
