"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    U, V, W are lists of 4 elements each.
    """
    factors = []
    # p1 = (A00+A11)(B00+B11)
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # p2 = (A10+A11) * B00
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # p3 = A00 * (B01-B11)
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # p4 = A11 * (B10-B00)
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    # p5 = (A00+A01) * B11
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # p6 = (A10-A00) * (B00+B01)
    factors.append([[-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    # p7 = (A01-A11) * (B10+B11)
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _try_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    """
    n, m, p = DIMENSIONS
    s2_factors = _strassen_2x2_factors()
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We have two 4x4 blocks in B (since p=8, m=4)
    # So t ranges 0..1
    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r
                
                s2_u_block = s2_factors[s][0]
                s2_v_block = s2_factors[s][1]
                s2_w_block = s2_factors[s][2]
                
                s2_u_scalar = s2_factors[r][0]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]
                
                # U: (n*m, rank) -> (4*4, 98)
                # Index: i*m + j
                # Block index bi, bj (0..1)
                # Element index ei, ej (0..1)
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_u_block[block_idx]
                        if c_block == 0: continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_u_scalar[scalar_idx]
                                if c_scalar == 0: continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += c_block * c_scalar
                                
                # V: (m*p, rank) -> (4*8, 98)
                # Index: j*p + k
                # Block index bi (for j), bj (for k block in 4x4)
                # Note: k is in 0..7. k = 4*t + k_local.
                # k_local is in 0..3.
                # k_local = 2 * bk + ek.
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_v_block[block_idx]
                        if c_block == 0: continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_v_scalar[scalar_idx]
                                if c_scalar == 0: continue
                                j = 2 * bi + ei
                                k_local = 2 * bj + ej
                                k = 4 * t + k_local
                                V[j * p + k][col] += c_block * c_scalar
                                
                # W: (p*n, rank) -> (8*4, 98)
                # Index: k*n + i
                # Block index ci (for k), cj (for i)
                for ci in range(2):
                    for cj in range(2):
                        block_idx = ci * 2 + cj
                        c_block = s2_w_block[block_idx]
                        if c_block == 0: continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_w_scalar[scalar_idx]
                                if c_scalar == 0: continue
                                i = 2 * cj + ej
                                k_local = 2 * ci + ei
                                k = 4 * t + k_local
                                W[k * n + i][col] += c_block * c_scalar
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    if len(U) == 0: return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    try:
        result = _try_strassen_4x4_4x8()
        if time.time() > deadline:
            return _schoolbook()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    return _schoolbook()
# EVOLVE-BLOCK-END
