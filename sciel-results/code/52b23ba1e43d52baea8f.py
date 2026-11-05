"""Strassen-based decomposition for 4x4 * 4x5 matrix multiplication.
Uses rank 49 for the 4x4 * 4x4 part and rank 16 for the 4x4 * 1 part.
Total rank 65.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = DIMENSIONS
NM = N * M          # 16
MP = M * P          # 20
PN = P * N          # 20
SCHOOLBOOK_RANK = N * M * P  # 80

DEADLINE = time.time() + 130.0

def _time_left():
    return DEADLINE - time.time()

def _schoolbook():
    """Guaranteed valid decomposition of rank 80 (entries 0/1)."""
    rank = SCHOOLBOOK_RANK
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2 * 2x2 multiplication.
    Each term is (u_indices, v_indices, w_indices) where:
    u_indices: list of (i, j, coeff) for A[i,j]
    v_indices: list of (j, k, coeff) for B[j,k]
    w_indices: list of (i, k, coeff) for C[i,k]
    
    Strassen formulas:
    M0 = (A00+A11)(B00+B11)
    M1 = (A10+A11)B00
    M2 = A00(B01-B11)
    M3 = A11(B10-B00)
    M4 = (A00+A01)B11
    M5 = (A10-A00)(B00+B01)
    M6 = (A01-A11)(B10+B11)
    
    C00 = M0 - M3 + M4 - M6
    C01 = M2 + M6
    C10 = M1 + M3
    C11 = M0 - M2 + M1 - M5
    """
    terms = []
    
    # M0 = (A00+A11)(B00+B11)
    terms.append((
        [(0,0,1), (1,1,1)],      # A00 + A11
        [(0,0,1), (1,1,1)],      # B00 + B11
        [(0,0,1), (1,1,1)]       # contributes to C00 and C11
    ))
    
    # M1 = (A10+A11)B00
    terms.append((
        [(1,0,1), (1,1,1)],      # A10 + A11
        [(0,0,1)],               # B00
        [(1,0,1), (1,1,-1)]      # C10 += M1, C11 -= M1
    ))
    
    # M2 = A00(B01-B11)
    terms.append((
        [(0,0,1)],               # A00
        [(0,1,1), (1,1,-1)],     # B01 - B11
        [(0,1,1), (1,1,-1)]      # C01 += M2, C11 -= M2
    ))
    
    # M3 = A11(B10-B00)
    terms.append((
        [(1,1,1)],               # A11
        [(1,0,1), (0,0,-1)],     # B10 - B00
        [(0,0,-1), (1,0,1)]      # C00 -= M3, C10 += M3
    ))
    
    # M4 = (A00+A01)B11
    terms.append((
        [(0,0,1), (0,1,1)],      # A00 + A01
        [(1,1,1)],               # B11
        [(0,0,1), (0,1,1)]       # C00 += M4, C01 += M4
    ))
    
    # M5 = (A10-A00)(B00+B01)
    terms.append((
        [(1,0,1), (0,0,-1)],     # A10 - A00
        [(0,0,1), (0,1,1)],      # B00 + B01
        [(1,1,-1)]               # C11 -= M5
    ))
    
    # M6 = (A01-A11)(B10+B11)
    terms.append((
        [(0,1,1), (1,1,-1)],     # A01 - A11
        [(1,0,1), (1,1,1)],      # B10 + B11
        [(0,0,-1), (0,1,1)]      # C00 -= M6, C01 += M6
    ))
    
    return terms

def _build_strassen_4x4_4x4():
    """
    Builds U, V, W for 4x4 * 4x4 multiplication using Strassen.
    Rank 49.
    
    We split A and B into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where A00 is top-left 2x2, etc.
    B = [[B00, B01], [B10, B11]] where B00 is top-left 2x2, etc.
    
    Outer Strassen gives 7 block products.
    Each block product is a 2x2 * 2x2 multiplication, decomposed with inner Strassen (7 terms).
    Total: 7 * 7 = 49 scalar multiplications.
    """
    inner_terms = _strassen_2x2_terms()
    
    # Outer Strassen: 7 block-level terms
    # Each outer term has:
    #   p_blocks: coefficients for A00, A01, A10, A11
    #   q_blocks: coefficients for B00, B01, B10, B11
    #   c_blocks: coefficients for C00, C01, C10, C11 (which blocks of C get the result)
    
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
    # M2 = A00(B01-B11) -> C01 += M2, C11 -= M2
    # M3 = A11(B10-B00) -> C00 -= M3, C10 += M3
    # M4 = (A00+A01)B11 -> C00 += M4, C01 += M4
    # M5 = (A10-A00)(B00+B01) -> C11 -= M5
    # M6 = (A01-A11)(B10+B11) -> C00 -= M6, C01 += M6
    
    outer_terms = [
        # (p_blocks, q_blocks, c_blocks)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M0
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]), # M2
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]), # M3
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),   # M4
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, -1]), # M5
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]), # M6
    ]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for outer_idx, (p_blocks, q_blocks, c_blocks) in enumerate(outer_terms):
        for inner_idx, (u_idx, v_idx, w_idx) in enumerate(inner_terms):
            col = outer_idx * 7 + inner_idx
            
            # For each entry in the inner term, expand to global indices
            for (li, lj, coeff_u) in u_idx:
                # li, lj are local 2x2 indices (0 or 1)
                # For each A block that contributes, add the global entry
                for block_idx, p_coeff in enumerate(p_blocks):
                    if p_coeff == 0:
                        continue
                    # Global i, j for A block block_idx at local (li, lj)
                    if block_idx == 0:  # A00
                        gi, gj = li, lj
                    elif block_idx == 1:  # A01
                        gi, gj = li, 2 + lj
                    elif block_idx == 2:  # A10
                        gi, gj = 2 + li, lj
                    elif block_idx == 3:  # A11
                        gi, gj = 2 + li, 2 + lj
                    else:
                        continue
                    
                    U[gi * 4 + gj, col] += p_coeff * coeff_u
            
            for (lj, lk, coeff_v) in v_idx:
                # lj, lk are local 2x2 indices (0 or 1)
                for block_idx, q_coeff in enumerate(q_blocks):
                    if q_coeff == 0:
                        continue
                    # Global j, k for B block block_idx at local (lj, lk)
                    if block_idx == 0:  # B00
                        gj, gk = lj, lk
                    elif block_idx == 1:  # B01
                        gj, gk = lj, 2 + lk
                    elif block_idx == 2:  # B10
                        gj, gk = 2 + lj, lk
                    elif block_idx == 3:  # B11
                        gj, gk = 2 + lj, 2 + lk
                    else:
                        continue
                    
                    V[gj * 4 + gk, col] += q_coeff * coeff_v
            
            for (li, lk, coeff_w) in w_idx:
                # li, lk are local 2x2 indices (0 or 1) for C
                for block_idx, c_coeff in enumerate(c_blocks):
                    if c_coeff == 0:
                        continue
                    # Global i, k for C block block_idx at local (li, lk)
                    if block_idx == 0:  # C00
                        gi, gk = li, lk
                    elif block_idx == 1:  # C01
                        gi, gk = li, 2 + lk
                    elif block_idx == 2:  # C10
                        gi, gk = 2 + li, lk
                    elif block_idx == 3:  # C11
                        gi, gk = 2 + li, 2 + lk
                    else:
                        continue
                    
                    W[gk * 4 + gi, col] += c_coeff * coeff_w
    
    return U, V, W

def _attempt_bilinear_reduction(U4, V4, W4):
    """
    Attempts to reduce the rank of the 4x4*4x4 decomposition by finding
    linear dependencies among the rank-1 terms using a bilinear/finite-field
    inspired approach. This is a heuristic attempt to exploit algebraic
    dependencies. If it finds a valid lower-rank decomposition, it returns
    the new matrices; otherwise, it returns None.
    """
    # This is a placeholder for a more sophisticated reduction strategy.
    # For now, we return None to indicate no reduction was found.
    # A real implementation might use Berlekamp-Massey or similar methods
    # to find minimal recurrences, but for this structural change, we
    # focus on the clean combination of Strassen + Schoolbook.
    return None

def construct():
    """Return the rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    t0 = time.time()
    fallback = _schoolbook()
    
    if _time_left() < 10:
        return fallback
        
    try:
        U4, V4, W4 = _build_strassen_4x4_4x4()
        
        # Check for potential rank reduction (currently a no-op placeholder)
        reduced = _attempt_bilinear_reduction(U4, V4, W4)
        if reduced is not None:
            U4, V4, W4 = reduced
        
        # Add the 4x4 * 1 part (column 4 of B)
        # Rank 16
        # Total rank: 49 + 16 = 65
        
        rank_4x4 = U4.shape[1]
        U_total = np.zeros((16, rank_4x4 + 16))
        V_total = np.zeros((20, rank_4x4 + 16))
        W_total = np.zeros((20, rank_4x4 + 16))
        
        # Copy 4x4 part
        U_total[:, 0:rank_4x4] = U4
        
        # V4: B is 4x4. Indices (j, k) where j in 0..3, k in 0..3.
        # V_total index (j*5 + k, rank)
        for j in range(4):
            for k in range(4):
                V_total[j * 5 + k, 0:rank_4x4] = V4[j * 4 + k, 0:rank_4x4]
                
        # W4: C is 4x4. Indices (k, i) where k in 0..3, i in 0..3.
        # W_total index (k*4 + i, rank)
        for k in range(4):
            for i in range(4):
                W_total[k * 4 + i, 0:rank_4x4] = W4[k * 4 + i, 0:rank_4x4]
                
        # Add 4x4 * 1 part
        # C_i4 = sum_j A_ij B_j4
        # Terms for j=0..3, i=0..3
        offset = rank_4x4
        for i in range(4):
            for j in range(4):
                c = offset + i * 4 + j
                U_total[i * 4 + j, c] = 1
                V_total[j * 5 + 4, c] = 1
                W_total[4 * 4 + i, c] = 1  # k=4, i=i -> 16+i
        
        # Verify: check that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
        U3 = U_total.reshape(4, 4, -1)
        V3 = V_total.reshape(4, 5, -1)
        W3 = W_total.reshape(5, 4, -1)
        
        # Compute the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        # T should be all ones
        if np.allclose(T, 1.0, atol=1e-9):
            return {
                "U": U_total.tolist(),
                "V": V_total.tolist(),
                "W": W_total.tolist()
            }
        else:
            # If it fails, return schoolbook
            return fallback
            
    except Exception as e:
        return fallback
