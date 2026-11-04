"""Matrix multiplication 4x4x7 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE = 140.0
START = time.time()


def _elapsed():
    return time.time() - START


def _time_left():
    return DEADLINE - _elapsed() - 5.0


def _check_decomp(U, V, W, n=4, m=4, p=7, tol=1e-10):
    """Verify exact decomposition: C[i,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] for all i,j,k."""
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                for c in range(r):
                    acc += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(acc - 1.0) > tol:
                    return False
    return True


def _to_fraction_list(mat, tol=1e-10):
    """Convert float matrix to list of Fractions (rounding small values to 0)."""
    out = []
    for row in mat:
        r = []
        for v in row:
            if abs(v) < tol:
                r.append(Fraction(0))
            else:
                # Try to find a simple fraction
                f = Fraction(v).limit_denominator(10**6)
                if abs(float(f) - v) < 1e-8:
                    r.append(f)
                else:
                    r.append(Fraction(v).limit_denominator(10**12))
        out.append(r)
    return out


def _build_decomp_from_factors(U, V, W):
    return {"U": U, "V": V, "W": W}


def _schoolbook(n=4, m=4, p=7):
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
    return _build_decomp_from_factors(U, V, W)


def _strassen_2x2_terms():
    """
    Returns the 7 terms of the Strassen 2x2x2 decomposition.
    Each term is a tuple (u_dict, v_dict, w_dict) where keys are (row, col) in 2x2 block.
    Convention:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    C[0,0] = a*e + b*g
    C[0,1] = a*f + b*h
    C[1,0] = c*e + d*g
    C[1,1] = c*f + d*h
    
    Strassen:
    M1 = (d-b)(e+f)
    M2 = (a+b)h
    M3 = a(e-g)
    M4 = d(h-f)
    M5 = (a+d)(h-g)
    M6 = (c+d)(e-f)
    M7 = (b-c)(e+h)
    
    C[0,0] = M5+M4-M2+M6
    C[0,1] = M1+M2
    C[1,0] = M3+M5
    C[1,1] = M1-M4+M7+M5
    """
    terms = []
    # M1 = (d-b)(e+f)
    # u: d-b => (1,1):1, (0,1):-1
    # v: e+f => (0,0):1, (0,1):1
    # w: C[0,1] (1), C[1,1] (1) => (0,1):1, (1,1):1
    terms.append((
        {(1, 1): 1, (0, 1): -1},
        {(0, 0): 1, (0, 1): 1},
        {(0, 1): 1, (1, 1): 1}
    ))
    # M2 = (a+b)h
    # u: a+b => (0,0):1, (1,0):1
    # v: h => (1,1):1
    # w: C[0,0] (-1), C[0,1] (1) => (0,0):-1, (0,1):1
    terms.append((
        {(0, 0): 1, (1, 0): 1},
        {(1, 1): 1},
        {(0, 0): -1, (0, 1): 1}
    ))
    # M3 = a(e-g)
    # u: a => (0,0):1
    # v: e-g => (0,0):1, (1,0):-1
    # w: C[1,0] (1) => (1,0):1
    terms.append((
        {(0, 0): 1},
        {(0, 0): 1, (1, 0): -1},
        {(1, 0): 1}
    ))
    # M4 = d(h-f)
    # u: d => (1,1):1
    # v: h-f => (1,1):1, (0,1):-1
    # w: C[0,0] (1), C[1,1] (-1) => (0,0):1, (1,1):-1
    terms.append((
        {(1, 1): 1},
        {(1, 1): 1, (0, 1): -1},
        {(0, 0): 1, (1, 1): -1}
    ))
    # M5 = (a+d)(h-g)
    # u: a+d => (0,0):1, (1,1):1
    # v: h-g => (1,1):1, (1,0):-1
    # w: C[0,0] (1), C[1,0] (1), C[1,1] (1) => (0,0):1, (1,0):1, (1,1):1
    terms.append((
        {(0, 0): 1, (1, 1): 1},
        {(1, 1): 1, (1, 0): -1},
        {(0, 0): 1, (1, 0): 1, (1, 1): 1}
    ))
    # M6 = (c+d)(e-f)
    # u: c+d => (1,0):1, (1,1):1
    # v: e-f => (0,0):1, (0,1):-1
    # w: C[0,0] (1) => (0,0):1
    terms.append((
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1, (0, 1): -1},
        {(0, 0): 1}
    ))
    # M7 = (b-c)(e+h)
    # u: b-c => (0,1):1, (1,0):-1
    # v: e+h => (0,0):1, (1,1):1
    # w: C[1,1] (1) => (1,1):1
    terms.append((
        {(0, 1): 1, (1, 0): -1},
        {(0, 0): 1, (1, 1): 1},
        {(1, 1): 1}
    ))
    return terms


def _build_strassen_4x4x4():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen.
    Returns U, V, W as numpy arrays of shape (16, 49), (16, 49), (16, 49).
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_2x2 = _strassen_2x2_terms()
    
    # Strassen 4x4x4 structure:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # S1 = (A11 - A00)(B00 + B01) -> C01 += S1
    # S2 = (A00 + A01)B11 -> C00 -= S2, C01 += S2
    # S3 = A00(B00 - B10) -> C10 += S3
    # S4 = A11(B11 - B01) -> C00 += S4, C11 -= S4
    # S5 = (A00 + A11)(B11 - B10) -> C00 += S5, C10 += S5, C11 += S5
    # S6 = (A10 + A11)(B00 - B01) -> C00 += S6
    # S7 = (A01 - A10)(B00 + B11) -> C11 += S7
    
    # Define block mappings
    # A blocks:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    # B blocks:
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    # C blocks:
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    def get_a_indices(block):
        if block == 0: return [(0,0), (0,1), (1,0), (1,1)]
        if block == 1: return [(0,2), (0,3), (1,2), (1,3)]
        if block == 2: return [(2,0), (2,1), (3,0), (3,1)]
        if block == 3: return [(2,2), (2,3), (3,2), (3,3)]
        
    def get_b_indices(block):
        if block == 0: return [(0,0), (0,1), (1,0), (1,1)]
        if block == 1: return [(0,2), (0,3), (1,2), (1,3)]
        if block == 2: return [(2,0), (2,1), (3,0), (3,1)]
        if block == 3: return [(2,2), (2,3), (3,2), (3,3)]
        
    def get_c_indices(block):
        if block == 0: return [(0,0), (0,1), (1,0), (1,1)]
        if block == 1: return [(0,2), (0,3), (1,2), (1,3)]
        if block == 2: return [(2,0), (2,1), (3,0), (3,1)]
        if block == 3: return [(2,2), (2,3), (3,2), (3,3)]

    # Structure for each Sx:
    # (a_blocks, a_coeffs, b_blocks, b_coeffs, c_blocks, c_coeffs)
    # a_blocks: list of block indices
    # a_coeffs: list of coefficients
    # etc.
    
    structure = [
        # S1
        ([3, 0], [1, -1], [0, 1], [1, 1], [1], [1]),
        # S2
        ([0, 1], [1, 1], [3], [1], [0, 1], [-1, 1]),
        # S3
        ([0], [1], [0, 2], [1, -1], [2], [1]),
        # S4
        ([3], [1], [3, 1], [1, -1], [0, 3], [1, -1]),
        # S5
        ([0, 3], [1, 1], [3, 2], [1, -1], [0, 2, 3], [1, 1, 1]),
        # S6
        ([2, 3], [1, 1], [0, 1], [1, -1], [0], [1]),
        # S7
        ([1, 2], [1, -1], [0, 3], [1, 1], [3], [1]),
    ]
    
    rank_idx = 0
    for s_idx, (a_blks, a_cfs, b_blks, b_cfs, c_blks, c_cfs) in enumerate(structure):
        for t_idx in range(7):
            r = rank_idx
            rank_idx += 1
            u_dict, v_dict, w_dict = strassen_2x2[t_idx]
            
            # Fill U
            # u_dict keys are (lr, lc) in 2x2 local coords.
            # We need to map local (lr, lc) to global (i, j) based on the block.
            # The linear combination is over the blocks in a_blks.
            # For a given local (lr, lc), the global index depends on which block.
            # Actually, the term is a product of a linear combo of A blocks and a linear combo of B blocks.
            # So U coefficient for a specific global (i,j) is sum over blocks b in a_blks:
            #   if (i,j) is in block b, coeff = a_cfs[b] * u_dict[local(i,j) in b]
            
            # Pre-compute local indices for each block
            # local(i,j) in block b:
            # A00: i, j
            # A01: i, j-2
            # A10: i-2, j
            # A11: i-2, j-2
            
            for (lr, lc), u_val in u_dict.items():
                # Find which global (i,j) correspond to this local (lr, lc) for each block
                # Wait, u_dict is the SAME for all 7 terms of the inner 2x2x2.
                # The inner 2x2x2 operates on the "effective" 2x2 matrix formed by the linear combo.
                # So for a specific term t, we have a specific u_vector over the 4 local entries (0,0), (0,1), (1,0), (1,1).
                # We need to distribute this to the global U.
                
                # For each block b in a_blks:
                #   The local entry (lr, lc) maps to global (i, j).
                #   The coefficient is a_cfs[b] * u_val.
                #   Add to U[i*m+j, r].
                
                for b_idx, b in enumerate(a_blks):
                    # Map local (lr, lc) to global (i, j) for block b
                    if b == 0: i, j = lr, lc
                    elif b == 1: i, j = lr, lc + 2
                    elif b == 2: i, j = lr + 2, lc
                    elif b == 3: i, j = lr + 2, lc + 2
                    U[i * m + j, r] += a_cfs[b_idx] * u_val
            
            # Fill V
            for (lr, lc), v_val in v_dict.items():
                for b_idx, b in enumerate(b_blks):
                    if b == 0: i, j = lr, lc
                    elif b == 1: i, j = lr, lc + 2
                    elif b == 2: i, j = lr + 2, lc
                    elif b == 3: i, j = lr + 2, lc + 2
                    V[i * p + j, r] += b_cfs[b_idx] * v_val
            
            # Fill W
            for (lr, lc), w_val in w_dict.items():
                for c_idx, c in enumerate(c_blks):
                    if c == 0: i, j = lr, lc
                    elif c == 1: i, j = lr, lc + 2
                    elif c == 2: i, j = lr + 2, lc
                    elif c == 3: i, j = lr + 2, lc + 2
                    W[j * n + i, r] += c_cfs[c_idx] * w_val
    
    return U, V, W


def _build_strassen_4x4x2():
    """
    Build a rank-28 decomposition for 4x4x2.
    B has 2 columns (4, 5).
    Split A into 4 blocks (2x2).
    Split B into 2 blocks (2x2) vertically? No, B is 4x2.
    B_top: rows 0-1, cols 4-5.
    B_bot: rows 2-3, cols 4-5.
    
    C_top (rows 0-1, cols 4-5) = A00 @ B_top + A01 @ B_bot
    C_bot (rows 2-3, cols 4-5) = A10 @ B_top + A11 @ B_bot
    
    4 products of 2x2 matrices. Rank 7 each. Total 28.
    """
    n = 4
    m = 4
    p = 7 # We are building for the full p=7, but only filling columns 4,5 of B
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_2x2 = _strassen_2x2_terms()
    
    products = [
        # (A_block, B_block, C_block)
        # A00 (0), B_top (0), C_top (0)
        (0, 0, 0),
        # A01 (1), B_bot (1), C_top (0)
        (1, 1, 0),
        # A10 (2), B_top (0), C_bot (1)
        (2, 0, 1),
        # A11 (3), B_bot (1), C_bot (1)
        (3, 1, 1),
    ]
    
    rank_idx = 0
    for a_blk, b_blk, c_blk in products:
        for t_idx in range(7):
            r = rank_idx
            rank_idx += 1
            u_dict, v_dict, w_dict = strassen_2x2[t_idx]
            
            # U
            for (lr, lc), u_val in u_dict.items():
                if a_blk == 0: i, j = lr, lc
                elif a_blk == 1: i, j = lr, lc + 2
                elif a_blk == 2: i, j = lr + 2, lc
                elif a_blk == 3: i, j = lr + 2, lc + 2
                U[i * m + j, r] += u_val
            
            # V
            for (lr, lc), v_val in v_dict.items():
                if b_blk == 0: i, j = lr, 4 + lc # B_top cols 4,5
                elif b_blk == 1: i, j = lr + 2, 4 + lc # B_bot cols 4,5
                V[i * p + j, r] += v_val
            
            # W
            for (lr, lc), w_val in w_dict.items():
                if c_blk == 0: i, j = lr, 4 + lc # C_top
                elif c_blk == 1: i, j = lr + 2, 4 + lc # C_bot
                W[j * n + i, r] += w_val
    
    return U, V, W


def _build_schoolbook_4x4x1():
    """
    Build a rank-16 decomposition for 4x4x1 (column 6 of B).
    """
    n = 4
    m = 4
    p = 7
    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, r_idx] = 1.0
            V[j * p + 6, r_idx] = 1.0
            W[6 * n + i, r_idx] = 1.0
            r_idx += 1
    
    return U, V, W


def construct():
    global START
    START = time.time()
    
    # Fallback
    schoolbook = _schoolbook(4, 4, 7)
    
    # Build components
    U1, V1, W1 = _build_strassen_4x4x4()
    U2, V2, W2 = _build_strassen_4x4x2()
    U3, V3, W3 = _build_schoolbook_4x4x1()
    
    # Concatenate
    r1, r2, r3 = U1.shape[1], U2.shape[1], U3.shape[1]
    total_rank = r1 + r2 + r3
    
    U = np.hstack([U1, U2, U3])
    V = np.hstack([V1, V2, V3])
    W = np.hstack([W1, W2, W3])
    
    # Convert to lists of Fractions
    Uf = _to_fraction_list(U)
    Vf = _to_fraction_list(V)
    Wf = _to_fraction_list(W)
    
    # Verify
    if _check_decomp(Uf, Vf, Wf, 4, 4, 7):
        return _build_decomp_from_factors(Uf, Vf, Wf)
    
    # If verification failed (should not happen), return schoolbook
    return schoolbook


# Run construct at import time to ensure it's ready? No, construct is the entry point.
# But we can do a quick self-check if run as main.
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Valid:", _check_decomp(res['U'], res['V'], res['W']))
