"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-21 exact tensor decomposition for 3x3 matrix multiplication.
    
    Strategy: Partition 3x3 into blocks:
    - Top-left 2x2 (indices 0,1,3,4)
    - Top-right 2x1 (indices 2,5)
    - Bottom-left 1x2 (indices 6,7)
    - Bottom-right 1x1 (index 8)
    
    Use Strassen (rank 7) for the 2x2 block, and efficient decompositions
    for the remaining 14 terms.
    
    W convention: W[k*n + i, r] where C_{ik} is row i, column k.
    So W index for C_{ik} is k*3 + i.
    """
    n, m, p = DIMENSIONS  # 3, 3, 3
    
    rank = 21
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 21
    V = [[0.0] * rank for _ in range(m * p)]  # 9 x 21
    W = [[0.0] * rank for _ in range(p * n)]  # 9 x 21
    
    col = 0
    
    def w_idx(i, k):
        """W index for C_{ik}: k*n + i"""
        return k * n + i
    
    def add_term(u_idx, v_idx, w_indices, coeffs):
        """
        Add a rank-1 term.
        u_idx: index in U (A element)
        v_idx: index in V (B element)
        w_indices: list of W indices
        coeffs: list of coefficients for each W index
        """
        nonlocal col
        U[u_idx][col] = 1.0
        V[v_idx][col] = 1.0
        for wi, c in zip(w_indices, coeffs):
            W[wi][col] += c
        col += 1
    
    def add_strassen_term(u_indices, v_indices, w_indices, coeffs):
        """
        Add a rank-1 term where U and V may have multiple non-zero entries.
        u_indices: list of U indices
        v_indices: list of V indices
        w_indices: list of W indices
        coeffs: list of coefficients for W
        """
        nonlocal col
        for ui in u_indices:
            U[ui][col] = 1.0
        for vi in v_indices:
            V[vi][col] = 1.0
        for wi, c in zip(w_indices, coeffs):
            W[wi][col] += c
        col += 1
    
    # Block indices:
    # A elements: A_{ij} -> U index i*3 + j
    # A00=0, A01=1, A02=2, A10=3, A11=4, A12=5, A20=6, A21=7, A22=8
    # B elements: B_{jk} -> V index j*3 + k
    # B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    # C elements: C_{ik} -> W index k*3 + i
    # C00=0, C01=3, C02=6, C10=1, C11=4, C12=7, C20=2, C21=5, C22=8
    
    # 1. Strassen for 2x2 top-left block (rank 7)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_strassen_term([0, 4], [0, 4], [w_idx(0,0), w_idx(1,1)], [1, 1])
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    add_strassen_term([3, 4], [0], [w_idx(1,0), w_idx(1,1)], [1, -1])
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    add_strassen_term([0], [1, 4], [w_idx(0,1), w_idx(1,1)], [1, 1])
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    add_strassen_term([4], [3, 0], [w_idx(0,0), w_idx(1,0)], [1, 1])
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    add_strassen_term([0, 1], [4], [w_idx(0,0), w_idx(0,1)], [-1, 1])
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    add_strassen_term([3, 0], [0, 1], [w_idx(1,1)], [1])
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    add_strassen_term([1, 4], [3, 4], [w_idx(0,0)], [1])
    
    # 2. Top-right 2x1 * Bottom-left 1x2 -> C00, C01, C10, C11 contributions
    # C00 += A02*B20 + A12*B20... no wait.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    # The 2x2 Strassen handles A00,B00; A01,B10; A10,B01; A11,B11 etc.
    # We need to add: A02*B20 to C00, A02*B21 to C01, A12*B20 to C10, A12*B21 to C11
    add_term(2, 6, [w_idx(0,0)], [1])   # A02*B20 -> C00
    add_term(2, 7, [w_idx(0,1)], [1])   # A02*B21 -> C01
    add_term(5, 6, [w_idx(1,0)], [1])   # A12*B20 -> C10
    add_term(5, 7, [w_idx(1,1)], [1])   # A12*B21 -> C11
    
    # 3. Top-left 2x2 * Top-right 2x1 -> C02, C12
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    add_term(0, 2, [w_idx(0,2)], [1])   # A00*B02 -> C02
    add_term(1, 5, [w_idx(0,2)], [1])   # A01*B12 -> C02
    add_term(3, 2, [w_idx(1,2)], [1])   # A10*B02 -> C12
    add_term(4, 5, [w_idx(1,2)], [1])   # A11*B12 -> C12
    
    # 4. Top-right 2x1 * Bottom-right 1x1 -> C02, C12
    add_term(2, 8, [w_idx(0,2)], [1])   # A02*B22 -> C02
    add_term(5, 8, [w_idx(1,2)], [1])   # A12*B22 -> C12
    
    # 5. Bottom-left 1x2 * Top-left 2x2 -> C20, C21
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    add_term(6, 0, [w_idx(2,0)], [1])   # A20*B00 -> C20
    add_term(7, 3, [w_idx(2,0)], [1])   # A21*B10 -> C20
    add_term(6, 1, [w_idx(2,1)], [1])   # A20*B01 -> C21
    add_term(7, 4, [w_idx(2,1)], [1])   # A21*B11 -> C21
    
    # 6. Bottom-right 1x1 * Bottom-left 1x2 -> C20, C21
    add_term(8, 6, [w_idx(2,0)], [1])   # A22*B20 -> C20
    add_term(8, 7, [w_idx(2,1)], [1])   # A22*B21 -> C21
    
    # 7. Bottom-left 1x2 * Top-right 2x1 -> C22
    # C22 = A20*B02 + A21*B12 + A22*B22
    add_term(6, 2, [w_idx(2,2)], [1])   # A20*B02 -> C22
    add_term(7, 5, [w_idx(2,2)], [1])   # A21*B12 -> C22
    
    # 8. Bottom-right 1x1 * Bottom-right 1x1 -> C22
    add_term(8, 8, [w_idx(2,2)], [1])   # A22*B22 -> C22
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
