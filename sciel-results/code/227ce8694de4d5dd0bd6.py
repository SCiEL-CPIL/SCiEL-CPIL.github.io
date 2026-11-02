"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _verify_exact(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check compatible triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s - 1) > 1e-12:
                    return False
    
    # Check all incompatible triples
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if j_a == j_b and k_b == k_c and i_c == i_a:
                    continue
                s = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(s) > 1e-12:
                    return False
    return True


def _get_strassen_vectors():
    """
    Return U, V, W vectors for 2x2x2 Strassen decomposition.
    Local indices: i,j,k in {0,1}.
    U is 2x2 (flattened 4), V is 2x2 (flattened 4), W is 2x2 (flattened 4).
    """
    # Standard Strassen products
    # m1 = (a00+a11)(b00+b11)
    # m2 = (c00+c11)(b00+b01)  -- Note: c is row 1 of A
    # m3 = (a00-a10)(b10+b11)
    # m4 = a11(b10-b00)
    # m5 = a00(b00-b10)
    # m6 = (c00+c11)b11
    # m7 = (a00-a10)b01
    
    # Let's use the verified set from literature
    # U vectors (4 elements: a00, a01, a10, a11)
    # V vectors (4 elements: b00, b01, b10, b11)
    # W vectors (4 elements: c00, c01, c10, c11)
    
    U = [
        [1, 0, 1, 1],  # m1: a00 + a11 (wait, a10 is index 2, a11 is index 3)
        # a00=0, a01=1, a10=2, a11=3
        [1, 0, 0, 1],  # m1: a00 + a11
        [0, 0, 1, 1],  # m2: a10 + a11
        [1, 0, -1, 0], # m3: a00 - a10
        [0, 0, 0, 1],  # m4: a11
        [1, 0, 0, 0],  # m5: a00
        [0, 0, 1, 1],  # m6: a10 + a11
        [1, 0, -1, 0]  # m7: a00 - a10
    ]
    
    V = [
        [1, 0, 0, 1],  # m1: b00 + b11
        [1, 1, 0, 0],  # m2: b00 + b01
        [0, 0, 1, 1],  # m3: b10 + b11
        [0, 0, 1, -1], # m4: b10 - b00
        [1, 0, -1, 0], # m5: b00 - b10
        [0, 0, 0, 1],  # m6: b11
        [0, 1, 0, 0]   # m7: b01
    ]
    
    W = [
        [1, 0, 0, 0],  # c00
        [0, 1, 0, 0],  # c01
        [0, 0, 1, 0],  # c10
        [0, 0, 0, 1]   # c11
    ]
    
    # Combinations for C:
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # We need to construct the rank-7 decomposition.
    # This means we need 7 vectors u_r, v_r, w_r such that
    # sum_r u_r[i] v_r[j] w_r[k] = C_ik?
    # No, the tensor decomposition is:
    # T_ijk = sum_r U_r[i,j] V_r[j,k] W_r[k,i]
    
    # So for each r, we define U_r, V_r, W_r.
    # And the sum over r must yield the identity for the product.
    
    # Let's define the 7 ranks explicitly.
    
    ranks = []
    
    # r=1: m1
    # U1: a00, a11 -> [1, 0, 0, 1]
    # V1: b00, b11 -> [1, 0, 0, 1]
    # W1: contributes to c00, c01, c10, c11
    # c00 += 1, c01 += 0, c10 += 0, c11 += 1?
    # m1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # This contributes to c00 (a00b00), c01 (a00b11? no, a00b11 is part of c01? no, c01 = a00b01 + a01b11)
    # This is tricky. The Strassen products are intermediate.
    # The decomposition T = sum u_r v_r w_r must satisfy the tensor equation.
    
    # I will use the known valid Strassen tensor decomposition vectors.
    
    # Reference: "Tensor Rank of 2x2x2 Matrix Multiplication is 7"
    
    # U (7x4), V (7x4), W (7x4)
    
    U_str = [
        [1, 0, 0, 1],  # 1
        [0, 0, 1, 1],  # 2
        [1, 0, -1, 0], # 3
        [0, 0, 0, 1],  # 4
        [1, 0, 0, 0],  # 5
        [0, 0, 1, 1],  # 6
        [1, 0, -1, 0]  # 7
    ]
    
    V_str = [
        [1, 0, 0, 1],  # 1
        [1, 1, 0, 0],  # 2
        [0, 0, 1, 1],  # 3
        [0, 0, 1, -1], # 4
        [1, 0, -1, 0], # 5
        [0, 0, 0, 1],  # 6
        [0, 1, 0, 0]   # 7
    ]
    
    W_str = [
        [1, 0, 0, 1],  # 1: c00, c11
        [0, 0, 1, -1], # 2: c10, -c11
        [0, 1, 0, 0],  # 3: c01
        [1, 0, 0, 0],  # 4: c00
        [0, -1, 0, 0], # 5: -c01
        [0, 0, 0, 1],  # 6: c11
        [0, 1, 0, 0]   # 7: c01
    ]
    
    return U_str, V_str, W_str


def _get_schoolbook_2x2x1_vectors():
    """
    Return U, V, W vectors for 2x2x1 Schoolbook decomposition.
    Local indices: i in {0,1}, j in {0,1}, k=0.
    U is 2x2 (flattened 4), V is 2x1 (flattened 2), W is 1x2 (flattened 2).
    """
    U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    W = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    U_str, V_str, W_str = _get_strassen_vectors()
    U_sc, V_sc, W_sc = _get_schoolbook_2x2x1_vectors()
    
    total_rank = 36
    U = [[0] * total_rank for _ in range(nm)]
    V = [[0] * total_rank for _ in range(mp)]
    W = [[0] * total_rank for _ in range(pn)]
    
    rank_offset = 0
    
    # Sub-problem 1: A_L B_t1 (2x2x2)
    # A cols 0,1. B rows 0,1. C cols 0,1.
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][rank_offset + r] = U_str[r][i*2 + j]
        for j in range(2):
            for k in range(2):
                V[j*5 + k][rank_offset + r] = V_str[r][j*2 + k]
        for k in range(2):
            for i in range(2):
                W[k*2 + i][rank_offset + r] = W_str[r][k*2 + i]
    rank_offset += 7
    
    # Sub-problem 2: A_L B_t2 (2x2x2)
    # A cols 0,1. B rows 0,1. C cols 2,3.
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][rank_offset + r] = U_str[r][i*2 + j]
        for j in range(2):
            for k in range(2):
                V[j*5 + (k+2)][rank_offset + r] = V_str[r][j*2 + k]
        for k in range(2):
            for i in range(2):
                W[(k+2)*2 + i][rank_offset + r] = W_str[r][k*2 + i]
    rank_offset += 7
    
    # Sub-problem 3: A_L B_t3 (2x2x1)
    # A cols 0,1. B rows 0,1. C col 4.
    for r in range(4):
        for i in range(2):
            for j in range(2):
                U[i*4 + j][rank_offset + r] = U_sc[r][i*2 + j]
        for j in range(2):
            V[j*5 + 4][rank_offset + r] = V_sc[r][j]
        for i in range(2):
            W[4*2 + i][rank_offset + r] = W_sc[r][i]
    rank_offset += 4
    
    # Sub-problem 4: A_R B_b1 (2x2x2)
    # A cols 2,3. B rows 2,3. C cols 0,1.
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2)][rank_offset + r] = U_str[r][i*2 + j]
        for j in range(2):
            for k in range(2):
                V[(j+2)*5 + k][rank_offset + r] = V_str[r][j*2 + k]
        for k in range(2):
            for i in range(2):
                W[k*2 + i][rank_offset + r] = W_str[r][k*2 + i]
    rank_offset += 7
    
    # Sub-problem 5: A_R B_b2 (2x2x2)
    # A cols 2,3. B rows 2,3. C cols 2,3.
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2)][rank_offset + r] = U_str[r][i*2 + j]
        for j in range(2):
            for k in range(2):
                V[(j+2)*5 + (k+2)][rank_offset + r] = V_str[r][j*2 + k]
        for k in range(2):
            for i in range(2):
                W[(k+2)*2 + i][rank_offset + r] = W_str[r][k*2 + i]
    rank_offset += 7
    
    # Sub-problem 6: A_R B_b3 (2x2x1)
    # A cols 2,3. B rows 2,3. C col 4.
    for r in range(4):
        for i in range(2):
            for j in range(2):
                U[i*4 + (j+2)][rank_offset + r] = U_sc[r][i*2 + j]
        for j in range(2):
            V[(j+2)*5 + 4][rank_offset + r] = V_sc[r][j]
        for i in range(2):
            W[4*2 + i][rank_offset + r] = W_sc[r][i]
    rank_offset += 4
    
    # Verify
    if not _verify_exact(U, V, W):
        # Fallback to schoolbook if Strassen vectors are wrong
        return _schoolbook()
        
    return {"U": U, "V": V, "W": W}


def _schoolbook():
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
# EVOLVE-BLOCK-END
