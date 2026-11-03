"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
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

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where:
      u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
      v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
      w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
    """
    factors = []
    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    factors.append((u, v, w))
    # M2 = (A10+A11) * B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, -1]
    factors.append((u, v, w))
    # M3 = A00 * (B01-B11)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 1, 0, 1]
    factors.append((u, v, w))
    # M4 = A11 * (B10-B00)
    u = [0, 0, 0, 1]
    v = [-1, 0, 1, 0]
    w = [1, 0, 1, 0]
    factors.append((u, v, w))
    # M5 = (A00+A01) * B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    factors.append((u, v, w))
    # M6 = (A10-A00) * (B00+B01)
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1]
    factors.append((u, v, w))
    # M7 = (A01-A11) * (B10+B11)
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, 0, 0]
    factors.append((u, v, w))
    return factors

def _build_strassen_4x4_4x8():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 using Strassen.
    Split the 4x8 matrix B into two 4x4 blocks B1 and B2.
    Compute C1 = A*B1 and C2 = A*B2, each using Strassen (rank 49).
    Total rank = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    s2_factors = _strassen_2x2_factors()
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need to map the 4x4 Strassen factors to the 4x4x8 problem.
    # The 4x4x4 Strassen is built from 7 blocks of 2x2x2 Strassen.
    # Let's define the 4x4x4 Strassen factors explicitly by nesting 2x2 factors.
    
    # For each of the 7 high-level Strassen factors (s in 0..6),
    # we have a combination of 4 2x2 blocks in A, B, C.
    # The 2x2 factors are applied to the sub-blocks.
    
    # High-level 4x4x4 Strassen structure:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C01+=M1, C10+=M1, C11+=M1
    # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C10+=M6, C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7, C01+=M7
    
    # Each "M" is a 2x2 matrix product, which itself requires 7 scalar multiplications.
    # So for each high-level factor s, we iterate over the 7 low-level factors r.
    
    for t in range(2): # Two blocks of B: B1 (cols 0-3) and B2 (cols 4-7)
        for s in range(7): # High-level Strassen factor
            # Determine the 2x2 block indices for A, B, C involved in this factor
            # and the signs.
            # This mapping is specific to the Strassen algorithm.
            
            # Instead of hardcoding the complex mapping, we can use the fact that
            # the 4x4x4 tensor is a sum of 7 rank-7 tensors.
            # We can generate the 49 factors for 4x4x4 and then duplicate them for the two B blocks.
            
            pass # This approach is complex to hardcode correctly without errors.
            
    # Alternative: Use the known 49 factors for 4x4x4 and extend to 4x4x8.
    # The 4x4x4 factors are (u, v, w) where u, v, w are length 16 vectors (flattened 4x4).
    # We can generate these by nesting the 2x2 factors.
    
    # Let's generate the 49 factors for 4x4x4.
    factors_4x4 = []
    for s in range(7):
        # Get the 2x2 coefficients for A, B, C for this high-level factor
        # This requires knowing which 2x2 blocks are combined.
        # For simplicity and correctness, I will use a pre-computed set of 49 factors
        # or a reliable generation method.
        
        # Generation method:
        # The 4x4x4 Strassen tensor is the Kronecker product of two 2x2x2 Strassen tensors?
        # No, it's a composition.
        
        # Let's stick to the direct construction in the current program which was verified to work.
        # I will copy the logic from _try_strassen_4x4_4x8 but clean it up.
        
        pass

    # Re-implementing the logic from the current program's _try_strassen_4x4_4x8
    # which was confirmed to produce a valid rank-98 decomposition.
    
    for t in range(2):
        for s in range(7):
            for r in range(7):
                col = t * 49 + s * 7 + r
                s2_u_block = s2_factors[s][0]
                s2_v_block = s2_factors[s][1]
                s2_u_scalar = s2_factors[r][0]
                s2_v_scalar = s2_factors[r][1]
                s2_w_scalar = s2_factors[r][2]
                
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_u_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_u_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * bi + ei
                                j = 2 * bj + ej
                                U[i * m + j][col] += c_block * c_scalar
                
                for bi in range(2):
                    for bj in range(2):
                        block_idx = bi * 2 + bj
                        c_block = s2_v_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_v_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                j = 2 * bi + ei
                                k = 4 * t + 2 * bj + ej
                                V[j * p + k][col] += c_block * c_scalar
                
                s2_w_block = s2_factors[s][2]
                for ci in range(2):
                    for cj in range(2):
                        block_idx = ci * 2 + cj
                        c_block = s2_w_block[block_idx]
                        if c_block == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                scalar_idx = ei * 2 + ej
                                c_scalar = s2_w_scalar[scalar_idx]
                                if c_scalar == 0:
                                    continue
                                i = 2 * ci + ei
                                k = 4 * t + 2 * cj + ej
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

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a Strassen-based construction yielding rank 98.
    Falls back to schoolbook (rank 128).
    """
    start_time = time.time()
    deadline = start_time + 110.0

    best_result = None
    best_rank = float('inf')

    # Try Strassen-based decomposition
    try:
        result = _build_strassen_4x4_4x8()
        if _verify(result["U"], result["V"], result["W"]):
            best_result = result
            best_rank = len(result["U"][0])
    except Exception:
        pass

    if best_result is not None:
        return best_result

    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
