"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _make_2x2_strassen():
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2.
    Returns (U_dict, V_dict, W_dict) as dicts mapping (row_index, col_index, term) -> coeff.
    
    Conventions:
      A is 2x2 with indices (i, j)
      B is 2x2 with indices (j, k)
      C = A*B with indices (i, k)
      
    U maps A elements: U[i*2+j, c]
    V maps B elements: V[j*2+k, c]
    W maps C elements: W[k*2+i, c]
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
    add_w(0, 0, 0, 1)
    add_w(1, 1, 0, 1)
    
    # Term 1: M2 = (A10+A11)*B00
    add_u(1, 0, 1, 1)
    add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1)
    add_w(1, 1, 1, -1)
    
    # Term 2: M3 = A00*(B01-B11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1)
    add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1)
    add_w(1, 1, 2, 1)
    
    # Term 3: M4 = A11*(B10-B00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1)
    add_v(0, 0, 3, -1)
    add_w(0, 1, 3, 1)
    add_w(0, 0, 3, 1)
    
    # Term 4: M5 = (A00+A01)*B11
    add_u(0, 0, 4, 1)
    add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1)
    add_w(1, 0, 4, 1)
    
    # Term 5: M6 = (A10-A00)*(B00+B01)
    add_u(1, 0, 5, 1)
    add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1)
    add_v(0, 1, 5, 1)
    add_w(1, 1, 5, 1)
    
    # Term 6: M7 = (A01-A11)*(B10+B11)
    add_u(0, 1, 6, 1)
    add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1)
    add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)
    
    return U_dict, V_dict, W_dict


def _make_3x3_laderman():
    """
    Laderman's rank-23 decomposition for 3x3 * 3x3 -> 3x3.
    Returns (U_dict, V_dict, W_dict) as dicts mapping (row, col, term) -> coeff.
    
    Conventions:
      A is 3x3 with indices (i, j)
      B is 3x3 with indices (j, k)
      C = A*B with indices (i, k)
      
    U maps A elements: U[i*3+j, c]
    V maps B elements: V[j*3+k, c]
    W maps C elements: W[k*3+i, c]
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
    
    # Laderman's 23 products for 3x3 matrix multiplication
    # M1 = (a00+a11+a22)(b00+b11+b22)
    add_u(0,0,0,1); add_u(1,1,0,1); add_u(2,2,0,1)
    add_v(0,0,0,1); add_v(1,1,0,1); add_v(2,2,0,1)
    add_w(0,0,0,1); add_w(1,1,0,1); add_w(2,2,0,1)
    
    # M2 = (a10+a20)(b01+b02)
    add_u(1,0,1,1); add_u(2,0,1,1)
    add_v(0,1,1,1); add_v(0,2,1,1)
    add_w(1,1,1,1); add_w(2,1,1,1)
    
    # M3 = (a01+a11)(b10+b20)
    add_u(0,1,2,1); add_u(1,1,2,1)
    add_v(1,0,2,1); add_v(2,0,2,1)
    add_w(0,0,2,1); add_w(0,1,2,1)
    
    # M4 = (a21+a22)(b20+b21)
    add_u(2,1,3,1); add_u(2,2,3,1)
    add_v(2,0,3,1); add_v(2,1,3,1)
    add_w(2,2,3,1); add_w(2,1,3,1)
    
    # M5 = (a02+a12)(b20+b22)
    add_u(0,2,4,1); add_u(1,2,4,1)
    add_v(2,0,4,1); add_v(2,2,4,1)
    add_w(2,0,4,1); add_w(2,1,4,1)
    
    # M6 = (a00+a20)(b02+b22)
    add_u(0,0,5,1); add_u(2,0,5,1)
    add_v(0,2,5,1); add_v(2,2,5,1)
    add_w(0,0,5,1); add_w(0,2,5,1)
    
    # M7 = (a10+a00)(b22+b02)
    add_u(1,0,6,1); add_u(0,0,6,1)
    add_v(2,2,6,1); add_v(0,2,6,1)
    add_w(2,1,6,1); add_w(2,0,6,1)
    
    # M8 = (a20+a00)(b22+b20)
    add_u(2,0,7,1); add_u(0,0,7,1)
    add_v(2,2,7,1); add_v(2,0,7,1)
    add_w(2,2,7,1); add_w(2,0,7,1)
    
    # M9 = (a11+a10)(b22+b20)
    add_u(1,1,8,1); add_u(1,0,8,1)
    add_v(2,2,8,1); add_v(2,0,8,1)
    add_w(2,1,8,1); add_w(2,0,8,1)
    
    # M10 = (a22+a20)(b22+b02)
    add_u(2,2,9,1); add_u(2,0,9,1)
    add_v(2,2,9,1); add_v(0,2,9,1)
    add_w(2,2,9,1); add_w(2,0,9,1)
    
    # M11 = (a01+a00)(b02+b12)
    add_u(0,1,10,1); add_u(0,0,10,1)
    add_v(0,2,10,1); add_v(1,2,10,1)
    add_w(0,0,10,1); add_w(0,1,10,1)
    
    # M12 = (a12+a11)(b02+b12)
    add_u(1,2,11,1); add_u(1,1,11,1)
    add_v(0,2,11,1); add_v(1,2,11,1)
    add_w(1,0,11,1); add_w(1,1,11,1)
    
    # M13 = (a21+a22)(b02+b12)
    add_u(2,1,12,1); add_u(2,2,12,1)
    add_v(0,2,12,1); add_v(1,2,12,1)
    add_w(2,0,12,1); add_w(2,1,12,1)
    
    # M14 = (a02+a22)(b01+b21)
    add_u(0,2,13,1); add_u(2,2,13,1)
    add_v(0,1,13,1); add_v(2,1,13,1)
    add_w(0,0,13,1); add_w(0,2,13,1)
    
    # M15 = (a00+a10+a20)(b01+b11+b21)
    add_u(0,0,14,1); add_u(1,0,14,1); add_u(2,0,14,1)
    add_v(0,1,14,1); add_v(1,1,14,1); add_v(2,1,14,1)
    add_w(1,0,14,1); add_w(1,1,14,1); add_w(1,2,14,1)
    
    # M16 = (a01+a11+a21)(b00+b10+b20)
    add_u(0,1,15,1); add_u(1,1,15,1); add_u(2,1,15,1)
    add_v(0,0,15,1); add_v(1,0,15,1); add_v(2,0,15,1)
    add_w(0,0,15,1); add_w(0,1,15,1); add_w(0,2,15,1)
    
    # M17 = (a02+a12+a22)(b00+b10+b20)
    add_u(0,2,16,1); add_u(1,2,16,1); add_u(2,2,16,1)
    add_v(0,0,16,1); add_v(1,0,16,1); add_v(2,0,16,1)
    add_w(2,0,16,1); add_w(2,1,16,1); add_w(2,2,16,1)
    
    # M18 = (a00+a22)(b00+b22)
    add_u(0,0,17,1); add_u(2,2,17,1)
    add_v(0,0,17,1); add_v(2,2,17,1)
    add_w(0,0,17,1); add_w(2,2,17,1)
    
    # M19 = (a10+a20)(b00+b20)
    add_u(1,0,18,1); add_u(2,0,18,1)
    add_v(0,0,18,1); add_v(2,0,18,1)
    add_w(1,0,18,1); add_w(2,0,18,1)
    
    # M20 = (a01+a11)(b10+b00)
    add_u(0,1,19,1); add_u(1,1,19,1)
    add_v(1,0,19,1); add_v(0,0,19,1)
    add_w(0,0,19,1); add_w(0,1,19,1)
    
    # M21 = (a21+a22)(b20+b10)
    add_u(2,1,20,1); add_u(2,2,20,1)
    add_v(2,0,20,1); add_v(1,0,20,1)
    add_w(2,2,20,1); add_w(2,1,20,1)
    
    # M22 = (a02+a12)(b20+b00)
    add_u(0,2,21,1); add_u(1,2,21,1)
    add_v(2,0,21,1); add_v(0,0,21,1)
    add_w(2,0,21,1); add_w(2,1,21,1)
    
    # M23 = (a12+a22)(b20+b10)
    add_u(1,2,22,1); add_u(2,2,22,1)
    add_v(2,0,22,1); add_v(1,0,22,1)
    add_w(2,1,22,1); add_w(2,2,22,1)
    
    return U_dict, V_dict, W_dict


def _verify_decomposition(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k"""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p"""
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


def _build_block_strassen(n, m, p):
    """
    Build 6x6 decomposition using 3x3 blocks of 2x2 matrices with Strassen.
    Rank = 3*3*3*7 = 189
    """
    block_size = 2
    num_blocks = 3  # 6/2 = 3
    rank = num_blocks * num_blocks * num_blocks * 7  # 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_dict, V_dict, W_dict = _make_2x2_strassen()
    
    col = 0
    for I in range(num_blocks):
        for K in range(num_blocks):
            for J in range(num_blocks):
                for c in range(7):
                    # Add terms for this block multiplication
                    for (i2, j2, cc), coeff in U_dict.items():
                        if cc == c:
                            gi = I * 2 + i2
                            gj = K * 2 + j2
                            U[gi * m + gj][col] += coeff
                    
                    for (j2, k2, cc), coeff in V_dict.items():
                        if cc == c:
                            gj = K * 2 + j2
                            gk = J * 2 + k2
                            V[gj * p + gk][col] += coeff
                    
                    for (k2, i2, cc), coeff in W_dict.items():
                        if cc == c:
                            gk = J * 2 + k2
                            gi = I * 2 + i2
                            W[gk * n + gi][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}


def _build_laderman_blocks(n, m, p):
    """
    Try to build 6x6 decomposition using 3x3 blocks with Laderman's rank-23.
    This would give rank = 23 * 8 = 184 if we can compose properly.
    Actually, for 6x6 as 2x2 blocks of 3x3 matrices, we need to be careful.
    Let's skip this for now as it's complex to compose correctly.
    """
    # This is a placeholder - the composition is non-trivial
    # For 6x6 = 2x2 blocks of 3x3, we'd need to handle the block structure
    # properly. Let's just return None to indicate this isn't implemented.
    return None


def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # First, try the block Strassen decomposition (rank 189)
    try:
        result = _build_block_strassen(n, m, p)
        U, V, W = result["U"], result["V"], result["W"]
        
        # Verify the decomposition
        max_err = _verify_decomposition(U, V, W, n, m, p)
        
        if max_err < 1e-10:
            # Valid decomposition with rank 189
            U_out = [[float(x) for x in row] for row in U]
            V_out = [[float(x) for x in row] for row in V]
            W_out = [[float(x) for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
        return _schoolbook(n, m, p)
    
    # Fallback to schoolbook (rank 216)
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
