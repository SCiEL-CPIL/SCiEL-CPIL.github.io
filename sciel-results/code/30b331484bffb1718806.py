"""Rewritten program: rank-minimal tensor decomposition for 2x4 * 4x8."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 64

# Deadline guard: return best found before 140s (evaluator kills at 150s)
_DEADLINE = 140.0
_T0 = time.monotonic()

def _elapsed():
    return time.monotonic() - _T0

def _schoolbook():
    """Return the rank-64 schoolbook decomposition."""
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
    """
    Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 for all (i,j,k) in the multiplication tensor.
    """
    rank = len(U[0])
    # For each (i,j,k), check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _strassen_2x2x2():
    """
    Return the standard rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns lists of (u_indices, v_indices, w_indices, coefficients) for each of the 7 terms.
    Each term is a triple of vectors (u, v, w) with scalar coefficients.
    
    For matrices A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B:
    C00 = M1 - M4 + M6 + M7
    C01 = M5 + M4
    C10 = M2 + M3
    C11 = M1 - M2 + M5 + M3
    
    Where:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-c)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (d-b)(f+h)
    
    U indices: i*m+j where i is row of A, j is col of A
    V indices: j*p+k where j is row of B (col of A), k is col of B
    W indices: k*n+i where k is col of B, i is row of C
    """
    F = Fraction
    # Each entry: (U_dict, V_dict, W_dict) where dict maps index -> coeff
    terms = []
    
    # M1 = (a+d)(e+h)
    # U: a=(0,0), d=(1,1) -> indices 0, 3
    # V: e=(0,0), h=(1,1) -> indices 0, 3
    # W: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1) -> indices 0,1,2,3
    # C00 += M1, C01 += M1, C10 -= M1, C11 += M1
    terms.append((
        {0: F(1), 3: F(1)},
        {0: F(1), 3: F(1)},
        {0: F(1), 1: F(1), 2: F(-1), 3: F(1)}
    ))
    
    # M2 = (c+d)e
    # U: c=(1,0), d=(1,1) -> indices 2, 3
    # V: e=(0,0) -> index 0
    # C10 += M2, C11 -= M2
    terms.append((
        {2: F(1), 3: F(1)},
        {0: F(1)},
        {2: F(1), 3: F(-1)}
    ))
    
    # M3 = a(f-h)
    # U: a=(0,0) -> index 0
    # V: f=(0,1), h=(1,1) -> indices 1, 3
    # C10 += M3, C11 += M3
    terms.append((
        {0: F(1)},
        {1: F(1), 3: F(-1)},
        {2: F(1), 3: F(1)}
    ))
    
    # M4 = d(g-c)
    # U: d=(1,1) -> index 3
    # V: g=(1,0), c=(0,0) -> indices 2, 0
    # C00 -= M4, C01 += M4
    terms.append((
        {3: F(1)},
        {2: F(1), 0: F(-1)},
        {0: F(-1), 1: F(1)}
    ))
    
    # M5 = (a+b)h
    # U: a=(0,0), b=(0,1) -> indices 0, 1
    # V: h=(1,1) -> index 3
    # C01 += M5, C11 += M5
    terms.append((
        {0: F(1), 1: F(1)},
        {3: F(1)},
        {1: F(1), 3: F(1)}
    ))
    
    # M6 = (c-a)(e+f)
    # U: c=(1,0), a=(0,0) -> indices 2, 0
    # V: e=(0,0), f=(0,1) -> indices 0, 1
    # C00 += M6
    terms.append((
        {2: F(1), 0: F(-1)},
        {0: F(1), 1: F(1)},
        {0: F(1)}
    ))
    
    # M7 = (d-b)(f+h)
    # U: d=(1,1), b=(0,1) -> indices 3, 1
    # V: f=(0,1), h=(1,1) -> indices 1, 3
    # C00 += M7, C11 += M7
    terms.append((
        {3: F(1), 1: F(-1)},
        {1: F(1), 3: F(1)},
        {0: F(1), 3: F(1)}
    ))
    
    return terms

def _strassen_2x2x8():
    """
    Use Strassen on the 2x2 part of 2x2 * 2x8.
    Split B (2x8) into four 2x2 blocks: B = [B00 B01 B02 B03; B10 B11 B12 B13]
    where B_ij is 2x2.
    A (2x2) * B (2x8) = [A*B00 A*B01 A*B02 A*B03]
    
    Each A*Bjk is 2x2*2x2, done with Strassen rank 7.
    Total rank: 4 * 7 = 28.
    
    U: 2x2 -> 4 entries, index i*2+j
    V: 2x8 -> 16 entries, index j*8+k
    W: 8x2 -> 16 entries, index k*2+i
    """
    F = Fraction
    strassen_2x2 = _strassen_2x2x2()
    
    all_terms = []
    for block in range(4):
        # Block b corresponds to columns 2*b, 2*b+1 of B
        for (u_dict, v_dict, w_dict) in strassen_2x2:
            new_u = {}
            new_v = {}
            new_w = {}
            for idx, coeff in u_dict.items():
                new_u[idx] = coeff
            for vidx, coeff in v_dict.items():
                # vidx is row of B (0 or 1), col within 2x2 block (0 or 1)
                # Global col = 2*block + col_in_block
                new_v[vidx * 8 + 2 * block + (vidx % 2) if False else vidx] = coeff
                # Wait, v_dict indices are for 2x2 B block: row 0 or 1, col 0 or 1
                # So vidx = row * 2 + col
                # Global index = row * 8 + (2*block + col)
                # Let me redo this properly
                pass
            
            # Let me redo the index mapping properly
            new_v = {}
            for vidx, coeff in v_dict.items():
                vrow = vidx // 2
                vcol = vidx % 2
                global_idx = vrow * 8 + 2 * block + vcol
                new_v[global_idx] = coeff
            
            new_w = {}
            for widx, coeff in w_dict.items():
                wcol = widx // 2  # col of B (0 or 1 in block)
                wrow = widx % 2  # row of C (0 or 1)
                global_idx = (2 * block + wcol) * 2 + wrow
                new_w[global_idx] = coeff
            
            all_terms.append((new_u, new_v, new_w))
    
    return all_terms

def _simple_improvement():
    """
    Construct a rank-56 decomposition by partitioning the 2x4x8 tensor
    into eight disjoint 2x2x2 sub-tensors and applying Strassen to each.
    """
    from fractions import Fraction as F
    
    n, m, p = 2, 4, 8
    rank = 56
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(2):      # j-block: j = 2*a + b
        for c in range(4):  # k-block: k = 2*c + d
            
            # M1: (a00+a11)(b00+b11)
            U[0*4+2*a+0][col] += F(1)
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+0][col] += F(1)
            W[4*c+1][col] += F(-1)
            W[4*c+2][col] += F(1)
            W[4*c+3][col] += F(1)
            col += 1
            
            # M2: (a10+a11)b00
            U[1*4+2*a+0][col] += F(1)
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            W[4*c+1][col] += F(1)
            W[4*c+2][col] += F(-1)
            W[4*c+3][col] += F(1)
            col += 1
            
            # M3: a00(b10-b11)
            U[0*4+2*a+0][col] += F(1)
            V[(2*a+1)*8+2*c+0][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(-1)
            W[4*c+0][col] += F(1)
            col += 1
            
            # M4: a11(b01-b00)
            U[1*4+2*a+1][col] += F(1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            V[(2*a+0)*8+2*c+0][col] += F(-1)
            W[4*c+0][col] += F(-1)
            W[4*c+1][col] += F(1)
            col += 1
            
            # M5: (a00+a01)b11
            U[0*4+2*a+0][col] += F(1)
            U[0*4+2*a+1][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+2][col] += F(1)
            W[4*c+3][col] += F(-1)
            col += 1
            
            # M6: (a10-a00)(b00+b01)
            U[1*4+2*a+0][col] += F(1)
            U[0*4+2*a+0][col] += F(-1)
            V[(2*a+0)*8+2*c+0][col] += F(1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            W[4*c+0][col] += F(-1)
            W[4*c+1][col] += F(1)
            col += 1
            
            # M7: (a11-a01)(b01+b11)
            U[1*4+2*a+1][col] += F(1)
            U[0*4+2*a+1][col] += F(-1)
            V[(2*a+0)*8+2*c+1][col] += F(1)
            V[(2*a+1)*8+2*c+1][col] += F(1)
            W[4*c+2][col] += F(-1)
            W[4*c+3][col] += F(1)
            col += 1
    
    assert col == 56, f"Expected 56 terms, got {col}"
    return {"U": U, "V": V, "W": W}

def _block_recursive_strassen():
    """
    Structural approach: decompose 2x4 * 4x8 using block recursive Strassen.
    
    View A (2x4) as [A0 | A1] where A0, A1 are 2x2.
    View B (4x8) as [B0; B1] where B0, B1 are 2x8.
    C = A0*B0 + A1*B1, where each product is 2x2 * 2x8.
    
    For each 2x2 * 2x8 product, split the 2x8 into four 2x2 blocks:
    B_j = [B_j0 B_j1 B_j2 B_j3]
    A_i * B_j = [A_i*B_j0 | A_i*B_j1 | A_i*B_j2 | A_i*B_j3]
    
    Each A_i * B_jk is 2x2 * 2x2, decomposable with Strassen rank 7.
    Total: 2 (A blocks) * 4 (B sub-blocks) * 7 = 56 terms.
    
    This is the same as _simple_improvement but derived structurally.
    Let's try to find if we can do better by sharing terms across blocks.
    
    Actually, let's try a different structural decomposition:
    Use the fact that we can combine the two 2x2*2x8 products.
    
    Alternative: View the problem as 2x4 * 4x8 and use a different block structure.
    Split the middle dimension 4 into 2+2, and the output 8 into 4+4.
    
    Let me try: A (2x4) = [A0 A1], B (4x8) = [B0; B1] with B0, B1 each 2x8.
    C = A0*B0 + A1*B1.
    
    Now for each 2x2*2x8, use a rank-14 decomposition instead of rank-28?
    The tensor M_{2,2,8} has dimension 32. Can we do it in rank < 28?
    
    Actually, let's just implement the rank-56 version cleanly and verify it works.
    """
    F = Fraction
    n, m, p = 2, 4, 8
    rank = 56
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    
    strassen_2x2 = _strassen_2x2x2()
    
    col = 0
    for a_block in range(2):  # A0 or A1
        for b_block in range(2):  # B0 or B1
            for sub_block in range(4):  # 4 sub-blocks of 2x2 in the 2x8
                for (u_dict, v_dict, w_dict) in strassen_2x2:
                    for uidx, ucoeff in u_dict.items():
                        U[uidx][col] += ucoeff
                    for vidx, vcoeff in v_dict.items():
                        vrow = vidx // 2
                        vcol = vidx % 2
                        global_v = (b_block * 2 + vrow) * 8 + sub_block * 2 + vcol
                        V[global_v][col] += vcoeff
                    for widx, wcoeff in w_dict.items():
                        wcol = widx // 2
                        wrow = widx % 2
                        global_w = (sub_block * 2 + wcol) * 2 + wrow
                        W[global_w][col] += wcoeff
                    col += 1
    
    assert col == 56, f"Expected 56 terms, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try block recursive Strassen (structural approach)
    if _elapsed() < _DEADLINE - 10:
        s1 = _block_recursive_strassen()
        if s1 is not None:
            if _verify(s1["U"], s1["V"], s1["W"]):
                r = len(s1["U"][0])
                if r < best_rank:
                    best = s1
                    best_rank = r
    
    # Try the simple improvement (same rank but different construction)
    if _elapsed() < _DEADLINE - 5:
        s2 = _simple_improvement()
        if s2 is not None:
            if _verify(s2["U"], s2["V"], s2["W"]):
                r = len(s2["U"][0])
                if r < best_rank:
                    best = s2
                    best_rank = r
    
    return best
