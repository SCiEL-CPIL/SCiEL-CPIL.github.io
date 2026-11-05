"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115  # Safety margin
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    # Rank 56 decomposition using Strassen for 2x2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 12 x 56
    V = [[0] * rank for _ in range(m * p)]  # 20 x 56
    W = [[0] * rank for _ in range(p * n)]  # 15 x 56
    
    col = 0
    
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        if col >= rank:
            return False
        for idx, val in u_vals:
            U[idx][col] += val
        for idx, val in v_vals:
            V[idx][col] += val
        for idx, val in w_vals:
            W[idx][col] += val
        col += 1
        return True
    
    def strassen_2x2(u_base, v_base, w_base):
        """Add 7 Strassen terms for 2x2x2 multiplication."""
        nonlocal col
        if col + 7 > rank:
            return False
        
        # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
        add_term(
            [(u_base(0,0), 1), (u_base(1,1), 1)],
            [(v_base(0,0), 1), (v_base(1,1), 1)],
            [(w_base(0,0), 1), (w_base(1,1), 1)]
        )
        # M2: (A10+A11)B00 -> C10+1, C11-1
        add_term(
            [(u_base(1,0), 1), (u_base(1,1), 1)],
            [(v_base(0,0), 1)],
            [(w_base(1,0), 1), (w_base(1,1), -1)]
        )
        # M3: A00(B01-B11) -> C01+1, C11+1
        add_term(
            [(u_base(0,0), 1)],
            [(v_base(0,1), 1), (v_base(1,1), -1)],
            [(w_base(0,1), 1), (w_base(1,1), 1)]
        )
        # M4: A11(B10-B00) -> C00+1, C10+1
        add_term(
            [(u_base(1,1), 1)],
            [(v_base(1,0), 1), (v_base(0,0), -1)],
            [(w_base(0,0), 1), (w_base(1,0), 1)]
        )
        # M5: (A00+A01)B11 -> C00-1, C01+1
        add_term(
            [(u_base(0,0), 1), (u_base(0,1), 1)],
            [(v_base(1,1), 1)],
            [(w_base(0,0), -1), (w_base(0,1), 1)]
        )
        # M6: (A10-A00)(B00+B01) -> C11+1
        add_term(
            [(u_base(1,0), 1), (u_base(0,0), -1)],
            [(v_base(0,0), 1), (v_base(0,1), 1)],
            [(w_base(1,1), 1)]
        )
        # M7: (A01-A11)(B10+B11) -> C00+1
        add_term(
            [(u_base(0,1), 1), (u_base(1,1), -1)],
            [(v_base(1,0), 1), (v_base(1,1), 1)],
            [(w_base(0,0), 1)]
        )
        return True
    
    def standard_2x2x1(u_base, v_base, w_base):
        """Standard multiplication for 2x2x1 (rank 4)."""
        nonlocal col
        if col + 4 > rank:
            return False
        for i in range(2):
            for j in range(2):
                add_term(
                    [(u_base(i, j), 1)],
                    [(v_base(j, 0), 1)],
                    [(w_base(i, 0), 1)]
                )
        return True
    
    def standard_1x2x2(u_base, v_base, w_base):
        """Standard multiplication for 1x2x2 (rank 4)."""
        nonlocal col
        if col + 4 > rank:
            return False
        for j in range(2):
            for k in range(2):
                add_term(
                    [(u_base(0, j), 1)],
                    [(v_base(j, k), 1)],
                    [(w_base(0, k), 1)]
                )
        return True
    
    def standard_1x2x3(u_base, v_base, w_base):
        """Standard multiplication for 1x2x3 (rank 6)."""
        nonlocal col
        if col + 6 > rank:
            return False
        for j in range(2):
            for k in range(3):
                add_term(
                    [(u_base(0, j), 1)],
                    [(v_base(j, k), 1)],
                    [(w_base(0, k), 1)]
                )
        return True
    
    def block_2x2x3(u_base, v_base, w_base):
        """2x2x3 using Strassen(2x2x2) + standard(2x2x1) = 7+4 = 11."""
        nonlocal col
        if col + 11 > rank:
            return False
        # Strassen on first 2 columns
        strassen_2x2(
            lambda i, j: u_base(i, j),
            lambda j, k: v_base(j, k),
            lambda i, k: w_base(i, k)
        )
        # Standard on last column
        standard_2x2x1(
            lambda i, j: u_base(i, j),
            lambda j, k: v_base(j, k + 2),
            lambda i, k: w_base(i, k + 2)
        )
        return True
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2, rank 7)
    strassen_2x2(
        lambda i, j: u_idx(i, j),
        lambda j, k: v_idx(j, k),
        lambda i, k: w_idx(i, k)
    )
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2x2, rank 7)
    strassen_2x2(
        lambda i, j: u_idx(i, j + 2),
        lambda j, k: v_idx(j + 2, k),
        lambda i, k: w_idx(i, k)
    )
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5] (2x2x3, rank 11)
    block_2x2x3(
        lambda i, j: u_idx(i, j),
        lambda j, k: v_idx(j, k + 2),
        lambda i, k: w_idx(i, k + 2)
    )
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5] (2x2x3, rank 11)
    block_2x2x3(
        lambda i, j: u_idx(i, j + 2),
        lambda j, k: v_idx(j + 2, k + 2),
        lambda i, k: w_idx(i, k + 2)
    )
    
    # Block 5: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (1x2x2, rank 4)
    standard_1x2x2(
        lambda i, j: u_idx(2, j),
        lambda j, k: v_idx(j, k),
        lambda i, k: w_idx(2, k)
    )
    
    # Block 6: A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2] (1x2x2, rank 4)
    standard_1x2x2(
        lambda i, j: u_idx(2, j + 2),
        lambda j, k: v_idx(j + 2, k),
        lambda i, k: w_idx(2, k)
    )
    
    # Block 7: A[2, 0:2] * B[0:2, 2:5] -> C[2, 2:5] (1x2x3, rank 6)
    standard_1x2x3(
        lambda i, j: u_idx(2, j),
        lambda j, k: v_idx(j, k + 2),
        lambda i, k: w_idx(2, k + 2)
    )
    
    # Block 8: A[2, 2:4] * B[2:4, 2:5] -> C[2, 2:5] (1x2x3, rank 6)
    standard_1x2x3(
        lambda i, j: u_idx(2, j + 2),
        lambda j, k: v_idx(j + 2, k + 2),
        lambda i, k: w_idx(2, k + 2)
    )
    
    if col != rank:
        raise ValueError(f"Expected {rank} columns, used {col}")
    
    # Verify decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        raise ValueError(f"Decomposition error: {max_err}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
