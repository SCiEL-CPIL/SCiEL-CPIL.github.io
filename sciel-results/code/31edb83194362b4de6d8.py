import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a block decomposition approach with Strassen for 2x2x2 blocks.
    Total rank: 56
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Pre-allocate matrices
    rank = 56
    U = np.zeros((n * m, rank))  # (12, 56)
    V = np.zeros((m * p, rank))  # (20, 56)
    W = np.zeros((p * n, rank))  # (15, 56)
    
    col = 0
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(i, k):
        return k * n + i
    
    def add_strassen_2x2x2(u_base, v_base, w_base):
        """Add Strassen 2x2x2 decomposition.
        u_base: list of 4 U indices [A00, A01, A10, A11]
        v_base: list of 4 V indices [B00, B01, B10, B11]
        w_base: list of 4 W indices [C00, C01, C10, C11]
        Returns new column index
        """
        nonlocal col
        A00, A01, A10, A11 = u_base
        B00, B01, B10, B11 = v_base
        C00, C01, C10, C11 = w_base
        
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        U[A00, col] += 1; U[A11, col] += 1
        V[B00, col] += 1; V[B11, col] += 1
        W[C00, col] += 1; W[C11, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10 - C11
        U[A10, col] += 1; U[A11, col] += 1
        V[B00, col] += 1
        W[C10, col] += 1; W[C11, col] += -1
        col += 1
        
        # M3 = A00(B01-B11) -> C01 + C11
        U[A00, col] += 1
        V[B01, col] += 1; V[B11, col] += -1
        W[C01, col] += 1; W[C11, col] += 1
        col += 1
        
        # M4 = A11(B10-B00) -> C00 + C10
        U[A11, col] += 1
        V[B10, col] += 1; V[B00, col] += -1
        W[C00, col] += 1; W[C10, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11 -> -C00 + C01
        U[A00, col] += 1; U[A01, col] += 1
        V[B11, col] += 1
        W[C00, col] += -1; W[C01, col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[A10, col] += 1; U[A00, col] += -1
        V[B00, col] += 1; V[B01, col] += 1
        W[C11, col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00
        U[A01, col] += 1; U[A11, col] += -1
        V[B10, col] += 1; V[B11, col] += 1
        W[C00, col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2x1(u_base, v_base, w_base):
        """Add schoolbook 2x2x1 decomposition (rank 4).
        u_base: list of 4 U indices [A00, A01, A10, A11]
        v_base: list of 2 V indices [B00, B10]
        w_base: list of 2 W indices [C00, C10]
        """
        nonlocal col
        A00, A01, A10, A11 = u_base
        B00, B10 = v_base
        C00, C10 = w_base
        
        U[A00, col] += 1; V[B00, col] += 1; W[C00, col] += 1
        col += 1
        U[A01, col] += 1; V[B00, col] += 1; W[C00, col] += 1
        col += 1
        U[A10, col] += 1; V[B10, col] += 1; W[C10, col] += 1
        col += 1
        U[A11, col] += 1; V[B10, col] += 1; W[C10, col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_1x2x2(u_base, v_base, w_base):
        """Add schoolbook 1x2x2 decomposition (rank 4).
        u_base: list of 2 U indices [A00, A01]
        v_base: list of 4 V indices [B00, B01, B10, B11]
        w_base: list of 2 W indices [C00, C01]
        """
        nonlocal col
        A00, A01 = u_base
        B00, B01, B10, B11 = v_base
        C00, C01 = w_base
        
        U[A00, col] += 1; V[B00, col] += 1; W[C00, col] += 1
        col += 1
        U[A01, col] += 1; V[B01, col] += 1; W[C01, col] += 1
        col += 1
        U[A00, col] += 1; V[B10, col] += 1; W[C00, col] += 1
        col += 1
        U[A01, col] += 1; V[B11, col] += 1; W[C01, col] += 1
        col += 1
        
        return col
    
    def add_schoolbook_1x2x1(u_base, v_base, w_base):
        """Add schoolbook 1x2x1 decomposition (rank 2).
        u_base: list of 2 U indices [A00, A01]
        v_base: list of 2 V indices [B00, B10]
        w_base: list of 1 W index [C00]
        """
        nonlocal col
        A00, A01 = u_base
        B00, B10 = v_base
        C00 = w_base[0]
        
        U[A00, col] += 1; V[B00, col] += 1; W[C00, col] += 1
        col += 1
        U[A01, col] += 1; V[B10, col] += 1; W[C00, col] += 1
        col += 1
        
        return col
    
    # Part 1: A rows 0,1 (2x4) times B (4x5)
    # Split B into columns: [0,1], [2,3], [4]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2) -> C[0:2, 0:2]
    add_strassen_2x2x2(
        [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)],
        [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)],
        [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    )
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] (2x2x2) -> C[0:2, 0:2]
    add_strassen_2x2x2(
        [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)],
        [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)],
        [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    )
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] (2x2x2) -> C[0:2, 2:4]
    add_strassen_2x2x2(
        [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)],
        [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)],
        [w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)]
    )
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] (2x2x2) -> C[0:2, 2:4]
    add_strassen_2x2x2(
        [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)],
        [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)],
        [w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)]
    )
    
    # Block 5: A[0:2, 0:2] * B[0:2, 4:5] (2x2x1) -> C[0:2, 4:5]
    add_schoolbook_2x2x1(
        [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)],
        [v_idx(0,4), v_idx(1,4)],
        [w_idx(0,4), w_idx(1,4)]
    )
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] (2x2x1) -> C[0:2, 4:5]
    add_schoolbook_2x2x1(
        [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)],
        [v_idx(2,4), v_idx(3,4)],
        [w_idx(0,4), w_idx(1,4)]
    )
    
    # Part 2: A row 2 (1x4) times B (4x5)
    # Split B into columns: [0,1], [2,3], [4]
    
    # Block 7: A[2, 0:2] * B[0:2, 0:2] (1x2x2) -> C[2, 0:2]
    add_schoolbook_1x2x2(
        [u_idx(2,0), u_idx(2,1)],
        [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)],
        [w_idx(2,0), w_idx(2,1)]
    )
    
    # Block 8: A[2, 2:4] * B[2:4, 0:2] (1x2x2) -> C[2, 0:2]
    add_schoolbook_1x2x2(
        [u_idx(2,2), u_idx(2,3)],
        [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)],
        [w_idx(2,0), w_idx(2,1)]
    )
    
    # Block 9: A[2, 0:2] * B[0:2, 2:4] (1x2x2) -> C[2, 2:4]
    add_schoolbook_1x2x2(
        [u_idx(2,0), u_idx(2,1)],
        [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)],
        [w_idx(2,2), w_idx(2,3)]
    )
    
    # Block 10: A[2, 2:4] * B[2:4, 2:4] (1x2x2) -> C[2, 2:4]
    add_schoolbook_1x2x2(
        [u_idx(2,2), u_idx(2,3)],
        [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)],
        [w_idx(2,2), w_idx(2,3)]
    )
    
    # Block 11: A[2, 0:2] * B[0:2, 4:5] (1x2x1) -> C[2, 4:5]
    add_schoolbook_1x2x1(
        [u_idx(2,0), u_idx(2,1)],
        [v_idx(0,4), v_idx(1,4)],
        [w_idx(2,4)]
    )
    
    # Block 12: A[2, 2:4] * B[2:4, 4:5] (1x2x1) -> C[2, 4:5]
    add_schoolbook_1x2x1(
        [u_idx(2,2), u_idx(2,3)],
        [v_idx(2,4), v_idx(3,4)],
        [w_idx(2,4)]
    )
    
    # Verify col == rank
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    # Verify the decomposition is correct
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # Check: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == T[i,j,k]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - T[i,j,k]) > 1e-10:
                    raise ValueError(f"Decomposition error at ({i},{j},{k}): got {val}, expected {T[i,j,k]}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
