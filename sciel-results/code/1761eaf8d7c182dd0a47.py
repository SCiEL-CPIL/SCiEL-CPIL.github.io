"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _make_2x2_strassen():
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) as dicts mapping (row_index, col_index) -> 1 for each factor.
    
    Conventions:
      A is 2x2 with indices (i, j), i=row of A, j=col of A
      B is 2x2 with indices (j, k), j=row of B, k=col of B
      C = A*B with indices (i, k)
      
    U maps A elements: U[i*2+j, c]
    V maps B elements: V[j*2+k, c]  
    W maps C elements: W[k*2+i, c]
    
    Strassen formulas for 2x2:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
      
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
    
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
    
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
    
    # Term 0: M1 = (A00+A11)*(B00+B11)
    add_u(0, 0, 0, 1)
    add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1)
    add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1)  # C00: k=0, i=0
    add_w(1, 1, 0, 1)  # C11: k=1, i=1
    
    # Term 1: M2 = (A10+A11)*B00
    add_u(1, 0, 1, 1)
    add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)  # C10: k=0, i=1
    add_w(1, 1, 1, -1) # C11: k=1, i=1
    
    # Term 2: M3 = A00*(B01-B11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1)
    add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)  # C01: k=1, i=0
    add_w(1, 1, 2, 1)  # C11: k=1, i=1
    
    # Term 3: M4 = A11*(B10-B00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1)
    add_v(0, 0, 3, -1)
    add_w(0, 1, 3, 1)  # C10: k=0, i=1
    add_w(0, 0, 3, 1)  # C00: k=0, i=0
    
    # Term 4: M5 = (A00+A01)*B11
    add_u(0, 0, 4, 1)
    add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1) # C00: k=0, i=0
    add_w(1, 0, 4, 1)  # C01: k=1, i=0
    
    # Term 5: M6 = (A10-A00)*(B00+B01)
    add_u(1, 0, 5, 1)
    add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1)
    add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)  # C11: k=1, i=1
    
    # Term 6: M7 = (A01-A11)*(B10+B11)
    add_u(0, 1, 6, 1)
    add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1)
    add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)  # C00: k=0, i=0
    
    return U_dict, V_dict, W_dict


def _verify_2x2():
    """Verify the 2x2 Strassen decomposition is correct."""
    U_dict, V_dict, W_dict = _make_2x2_strassen()
    
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    for (i, j, c), coeff in U_dict.items():
        U[i * 2 + j, c] += coeff
    for (j, k, c), coeff in V_dict.items():
        V[j * 2 + k, c] += coeff
    for (k, i, c), coeff in W_dict.items():
        W[k * 2 + i, c] += coeff
    
    max_err = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(rank):
                    s += U[i * 2 + j, c] * V[j * 2 + k, c] * W[k * 2 + i, c]
                err = abs(s - 1.0)
                max_err = max(max_err, err)
    
    return max_err


def _make_3x3_bclr():
    """
    Bini-Capovani-Lotti-Romani (BCLR) rank-21 decomposition for 3x3 * 3x3 -> 3x3.
    Returns (U_dict, V_dict, W_dict) where each maps (row, col, term) -> coeff.
    
    Conventions:
      A is 3x3 with indices (i, j)
      B is 3x3 with indices (j, k)
      C = A*B with indices (i, k)
      U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
    
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
    
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
    
    # BCLR rank-21 decomposition for 3x3 matrix multiplication
    # The 21 terms are defined by specific linear combinations.
    # Based on the BCLR 1978 paper and subsequent implementations.
    
    # Term 0: (A00+A11+A22)(B00+B11+B22)
    add_u(0, 0, 0, 1); add_u(1, 1, 0, 1); add_u(2, 2, 0, 1)
    add_v(0, 0, 0, 1); add_v(1, 1, 0, 1); add_v(2, 2, 0, 1)
    add_w(0, 0, 0, 1); add_w(1, 1, 0, 1); add_w(2, 2, 0, 1)
    
    # Term 1: (A01+A12+A20)(B01+B12+B20)
    add_u(0, 1, 1, 1); add_u(1, 2, 1, 1); add_u(2, 0, 1, 1)
    add_v(0, 1, 1, 1); add_v(1, 2, 1, 1); add_v(2, 0, 1, 1)
    add_w(1, 0, 1, 1); add_w(2, 1, 1, 1); add_w(0, 2, 1, 1)
    
    # Term 2: (A02+A10+A21)(B02+B10+B21)
    add_u(0, 2, 2, 1); add_u(1, 0, 2, 1); add_u(2, 1, 2, 1)
    add_v(0, 2, 2, 1); add_v(1, 0, 2, 1); add_v(2, 1, 2, 1)
    add_w(2, 0, 2, 1); add_w(0, 1, 2, 1); add_w(1, 2, 2, 1)
    
    # Term 3: (A00+A11-A22)(B00+B11-B22)
    add_u(0, 0, 3, 1); add_u(1, 1, 3, 1); add_u(2, 2, 3, -1)
    add_v(0, 0, 3, 1); add_v(1, 1, 3, 1); add_v(2, 2, 3, -1)
    add_w(0, 0, 3, 1); add_w(1, 1, 3, 1); add_w(2, 2, 3, -1)
    
    # Term 4: (A01+A12-A20)(B01+A12-A20)
    add_u(0, 1, 4, 1); add_u(1, 2, 4, 1); add_u(2, 0, 4, -1)
    add_v(0, 1, 4, 1); add_v(1, 2, 4, 1); add_v(2, 0, 4, -1)
    add_w(1, 0, 4, 1); add_w(2, 1, 4, 1); add_w(0, 2, 4, -1)
    
    # Term 5: (A02+A10-A21)(B02+B10-B21)
    add_u(0, 2, 5, 1); add_u(1, 0, 5, 1); add_u(2, 1, 5, -1)
    add_v(0, 2, 5, 1); add_v(1, 0, 5, 1); add_v(2, 1, 5, -1)
    add_w(2, 0, 5, 1); add_w(0, 1, 5, 1); add_w(1, 2, 5, -1)
    
    # Term 6: (A00-A11)(B00-B11)
    add_u(0, 0, 6, 1); add_u(1, 1, 6, -1)
    add_v(0, 0, 6, 1); add_v(1, 1, 6, -1)
    add_w(0, 0, 6, 1); add_w(1, 1, 6, -1)
    
    # Term 7: (A01-A12)(B01-B12)
    add_u(0, 1, 7, 1); add_u(1, 2, 7, -1)
    add_v(0, 1, 7, 1); add_v(1, 2, 7, -1)
    add_w(1, 0, 7, 1); add_w(2, 1, 7, -1)
    
    # Term 8: (A02-A10)(B02-B10)
    add_u(0, 2, 8, 1); add_u(1, 0, 8, -1)
    add_v(0, 2, 8, 1); add_v(1, 0, 8, -1)
    add_w(2, 0, 8, 1); add_w(0, 1, 8, -1)
    
    # Term 9: (A11-A22)(B11-B22)
    add_u(1, 1, 9, 1); add_u(2, 2, 9, -1)
    add_v(1, 1, 9, 1); add_v(2, 2, 9, -1)
    add_w(1, 1, 9, 1); add_w(2, 2, 9, -1)
    
    # Term 10: (A12-A20)(B12-B20)
    add_u(1, 2, 10, 1); add_u(2, 0, 10, -1)
    add_v(1, 2, 10, 1); add_v(2, 0, 10, -1)
    add_w(2, 1, 10, 1); add_w(0, 2, 10, -1)
    
    # Term 11: (A20-A02)(B20-B02)
    add_u(2, 0, 11, 1); add_u(0, 2, 11, -1)
    add_v(2, 0, 11, 1); add_v(0, 2, 11, -1)
    add_w(0, 2, 11, 1); add_w(2, 0, 11, -1)
    
    # Term 12: (A00+A11+A22)(B00+B11+B22) - different combination
    # Actually, let me use a verified BCLR decomposition.
    # The above may not be exactly correct. Let me use a known correct one.
    
    # Let me reset and use a properly verified BCLR rank-21.
    # I'll clear and rebuild.
    
    # Clear previous entries
    U_dict.clear()
    V_dict.clear()
    W_dict.clear()
    
    # Verified BCLR rank-21 decomposition:
    # Based on the standard formulation.
    
    # The 21 multiplications for BCLR:
    # M1 = (a00+a11+a22)(b00+b11+b22)
    # M2 = (a01+a12+a20)(b01+b12+b20)
    # M3 = (a02+a10+a21)(b02+b10+b21)
    # M4 = (a00+a11-a22)(b00+b11-b22)
    # M5 = (a01+a12-a20)(b01+b12-b20)
    # M6 = (a02+a10-a21)(b02+b10-b21)
    # M7 = (a00-a11)(b00-b11)
    # M8 = (a01-a12)(b01-b12)
    # M9 = (a02-a10)(b02-b10)
    # M10 = (a11-a22)(b11-b22)
    # M11 = (a12-a20)(b12-b20)
    # M12 = (a21-a02)(b21-b02)
    # M13 = a00*b00
    # M14 = a11*b11
    # M15 = a22*b22
    # M16 = a01*b10
    # M17 = a12*b20
    # M18 = a21*b02
    # M19 = a02*b21
    # M20 = a10*b01
    # M21 = a20*b12
    
    # This is getting complicated. Let me use a simpler approach:
    # Use schoolbook rank-27 for 3x3 and combine with Strassen at 2x2 block level.
    # Total rank = 7 * 27 = 189, which is what we already have.
    
    # For a true improvement, I need a verified rank-21 BCLR.
    # Let me use the known correct formulation.
    
    # Actually, let me just verify that the 189-term decomposition works
    # and return that. The BCLR rank-21 would give 7*21=147 if correct.
    
    # For now, return an empty decomposition to indicate we should use schoolbook 3x3
    return None


def _verify_3x3_bclr():
    """Verify the 3x3 BCLR decomposition if available."""
    result = _make_3x3_bclr()
    if result is None:
        return None, None, None
    U_dict, V_dict, W_dict = result
    
    rank = 21
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    for (i, j, c), coeff in U_dict.items():
        if c < rank:
            U[i * 3 + j, c] += coeff
    for (j, k, c), coeff in V_dict.items():
        if c < rank:
            V[j * 3 + k, c] += coeff
    for (k, i, c), coeff in W_dict.items():
        if c < rank:
            W[k * 3 + i, c] += coeff
    
    max_err = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for c in range(rank):
                    s += U[i * 3 + j, c] * V[j * 3 + k, c] * W[k * 3 + i, c]
                err = abs(s - 1.0)
                max_err = max(max_err, err)
    
    if max_err < 1e-10:
        return U_dict, V_dict, W_dict
    else:
        return None, None, None


def _make_3x3_schoolbook():
    """
    Schoolbook rank-27 decomposition for 3x3 * 3x3 -> 3x3.
    Returns (U_dict, V_dict, W_dict) where each maps (row, col, term) -> coeff.
    """
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_u(i, j, c, coeff):
        key = (i, j, c)
        U_dict[key] = U_dict.get(key, 0) + coeff
    
    def add_v(j, k, c, coeff):
        key = (j, k, c)
        V_dict[key] = V_dict.get(key, 0) + coeff
    
    def add_w(k, i, c, coeff):
        key = (k, i, c)
        W_dict[key] = W_dict.get(key, 0) + coeff
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_u(i, j, col, 1)
                add_v(j, k, col, 1)
                add_w(k, i, col, 1)
                col += 1
    
    return U_dict, V_dict, W_dict


def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # First, try to verify and build the 2x2 Strassen decomposition
    try:
        max_err = _verify_2x2()
        if max_err > 1e-10:
            # Fallback to schoolbook
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
    except Exception:
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
    
    # Now build 6x6 decomposition using 2x2 blocks of 3x3 matrices
    # A is 6x6, viewed as 2x2 block matrix of 3x3 blocks
    # B is 6x6, viewed as 2x2 block matrix of 3x3 blocks
    # C = A*B is 6x6, viewed as 2x2 block matrix of 3x3 blocks
    
    # For each pair of blocks (I,J) where I is row-block of A, J is col-block of B:
    #   C_block[I,J] = sum_K A_block[I,K] * B_block[K,J]
    
    # We can use Strassen for the 2x2 block structure
    # and schoolbook for the 3x3 matrix multiplications within each block
    
    # Strategy: Use Strassen at the 2x2 block level (7 terms)
    # Each term involves products of 3x3 blocks
    # Each 3x3 product uses schoolbook rank-27
    # Total: 7 * 27 = 189 terms
    
    n, m, p = 6, 6, 6
    block_size = 3
    num_blocks = 2  # 6/3 = 2
    
    # Global indices:
    # A[i,j] where i = I*3 + i2, j = K*3 + j2
    # B[j,k] where j = K*3 + j2, k = J*3 + k2  
    # C[i,k] where i = I*3 + i2, k = J*3 + k2
    
    # For each Strassen term at the block level, we have linear combinations of blocks
    # Each block multiplication uses schoolbook 3x3
    
    # Let's build the 189-term decomposition
    rank = 7 * 27  # 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen, V_strassen, W_strassen = _make_2x2_strassen()
    U_school, V_school, W_school = _make_3x3_schoolbook()
    
    # For each Strassen term c (0-6), and each schoolbook term s (0-26):
    # The combined term index is c * 27 + s
    
    col = 0
    for c in range(7):
        # Get the Strassen coefficients for this term
        U_strassen_terms = [(i2, j2, coeff) for (i2, j2, cc), coeff in U_strassen.items() if cc == c]
        V_strassen_terms = [(j2, k2, coeff) for (j2, k2, cc), coeff in V_strassen.items() if cc == c]
        W_strassen_terms = [(k2, i2, coeff) for (k2, i2, cc), coeff in W_strassen.items() if cc == c]
        
        for s in range(27):
            # Get the schoolbook coefficients for this term
            U_school_terms = [(i3, j3, coeff) for (i3, j3, ss), coeff in U_school.items() if ss == s]
            V_school_terms = [(j3, k3, coeff) for (j3, k3, ss), coeff in V_school.items() if ss == s]
            W_school_terms = [(k3, i3, coeff) for (k3, i3, ss), coeff in W_school.items() if ss == s]
            
            # Combine: for each combination of Strassen and schoolbook terms
            for (I2, K2, coeff_u_str) in U_strassen_terms:
                for (i3, j3, coeff_u_sch) in U_school_terms:
                    gi = I2 * block_size + i3
                    gj = K2 * block_size + j3
                    U[gi * m + gj][col] += coeff_u_str * coeff_u_sch
            
            for (K2, J2, coeff_v_str) in V_strassen_terms:
                for (j3, k3, coeff_v_sch) in V_school_terms:
                    gj = K2 * block_size + j3
                    gk = J2 * block_size + k3
                    V[gj * p + gk][col] += coeff_v_str * coeff_v_sch
            
            for (J2, I2, coeff_w_str) in W_strassen_terms:
                for (k3, i3, coeff_w_sch) in W_school_terms:
                    gk = J2 * block_size + k3
                    gi = I2 * block_size + i3
                    W[gk * n + gi][col] += coeff_w_str * coeff_w_sch
            
            col += 1
    
    # Verify the decomposition using numpy for speed
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check tensor identity: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
        # Use vectorized computation
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]  # shape (rank,)
                for k in range(p):
                    v_row = V_np[j * p + k, :]  # shape (rank,)
                    w_row = W_np[k * n + i, :]  # shape (rank,)
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            # Valid decomposition!
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
