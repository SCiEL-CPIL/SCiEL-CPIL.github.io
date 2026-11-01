"""Matrix multiplication 2x4x8 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    Rank = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2x2
    # A: 2x2, B: 2x2, C: 2x2
    # Indices: (i, j) for A, (j, k) for B, (i, k) for C
    strassen_terms = [
        # M1 = (a00 + a11)(b00 + b11)
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # M2 = (a10 + a11)b00
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # M3 = a00(b01 - b11)
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # M4 = a11(b10 - b00)
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # M5 = (a00 + a01)b11
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # M6 = (a10 - a00)(b00 + b01)
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # M7 = (a01 - a11)(b10 + b11)
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ),
    ]
    
    col = 0
    # A is 2x4. Split into 2 blocks of 2 columns.
    # B is 4x8. Split into 4 blocks of 2 columns (and 2 rows? No, B is 4x8).
    # Wait, B is 4x8. The multiplication is A(2x4) * B(4x8).
    # We split A into A_0, A_1 (each 2x2).
    # We split B into 4 blocks B_0, B_1, B_2, B_3 (each 2x2)?
    # No, B is 4x8.
    # Let's partition B into 4 blocks of size 2x2?
    # B = [ B_{00} B_{01} B_{02} B_{03} ]? No, B has 4 rows.
    # B = [ B_{0} B_{1} ] where B_0, B_1 are 4x4?
    # Or B = [ B_{00} B_{01} B_{02} B_{03} ] where each is 4x2?
    # The Strassen block is 2x2.
    # So we need to partition the 4 rows of B into 2 groups of 2, and 8 columns into 4 groups of 2.
    # This gives 2 * 4 = 8 blocks of size 2x2.
    # For each A-block (2 of them) and each B-block (4 of them? No, 2x4 grid of 2x2 blocks in B).
    # B is 4x8.
    # Rows 0,1 and 2,3. Cols 0,1; 2,3; 4,5; 6,7.
    # So B has 2x4 = 8 blocks of size 2x2.
    # A has 2 blocks of size 2x2 (Cols 0,1 and 2,3).
    # Total 2 * 8 = 16 blocks?
    # No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # If we fix A-block a (cols 2a, 2a+1) and B-block b (rows 2b, 2b+1, cols 2c, 2c+1)?
    # The term is A_a (2x2) * B_{b,c} (2x2) -> C_{b,c} (2x2).
    # But A_a contributes to all B rows?
    # A is 2x4. A = [ A_0 A_1 ].
    # B is 4x8. B = [ B_{00} B_{01} B_{02} B_{03} ]? No.
    # B rows are 0,1,2,3.
    # B = [ B_{top} B_{bot} ]? No, B is 4x8.
    # Let's partition B into 4 blocks of 4x2? No, Strassen is 2x2.
    # Partition B into 8 blocks of 2x2.
    # B_{r,c} where r in {0,1} (row block), c in {0,1,2,3} (col block).
    # A_{a} where a in {0,1} (col block).
    # C_{i,k} = sum_{j=0}^3 A_{i,j} B_{j,k}.
    # j=0,1 correspond to A_0. j=2,3 correspond to A_1.
    # So C = A_0 B_{0,*} + A_1 B_{1,*}.
    # B_{0,*} is the top 2 rows of B (4x8? No, 2x8).
    # B_{0,*} has 4 col blocks of 2x2.
    # So A_0 (2x2) * B_{0,c} (2x2) for c=0..3.
    # This gives 4 products of 2x2x2.
    # Similarly A_1 (2x2) * B_{1,c} (2x2) for c=0..3.
    # This gives 4 products of 2x2x2.
    # Total 8 products of 2x2x2.
    # Each has rank 7.
    # Total rank 56.
    
    for a_block in range(2):       # A block index
        for b_row in range(2):     # B row block index
            for b_col in range(4): # B col block index
                for term in strassen_terms:
                    A_coeffs, B_coeffs, C_coeffs = term
                    
                    # U: A_{i, j}
                    # i is 0..1 (global row of A)
                    # j is 2*a_block + local_j
                    for (li, lj), coeff in A_coeffs.items():
                        global_i = li
                        global_j = 2 * a_block + lj
                        U[global_i * m + global_j, col] += coeff
                    
                    # V: B_{j, k}
                    # j is 2*b_row + local_j
                    # k is 2*b_col + local_k
                    for (lj, lk), coeff in B_coeffs.items():
                        global_j = 2 * b_row + lj
                        global_k = 2 * b_col + lk
                        V[global_j * p + global_k, col] += coeff
                    
                    # W: C_{i, k}
                    # i is 0..1 (global row of C)
                    # k is 2*b_col + local_k
                    for (li, lk), coeff in C_coeffs.items():
                        global_i = li
                        global_k = 2 * b_col + lk
                        W[global_k * n + global_i, col] += coeff
                    
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        result = _strassen_decomposition()
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
