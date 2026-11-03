"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _standard_decomp(n, m, p):
    """Standard (schoolbook) decomposition."""
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Strassen decomposition for 2x2 matrix multiplication. Rank 7."""
    # Indices: U is (4,7), V is (4,7), W is (4,7)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+cg, af+bh], [ce+dg, cf+bh]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 - m4 + m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m2 - m6
    
    # Map to U (A indices), V (B indices), W (C indices)
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=C11, 1=C12, 2=C21, 3=C22
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1: (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2: (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # m3: a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # m4: d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # m5: (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # m6: (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = -1
    
    # m7: (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _combine_decomps(decomp_list, n, m, p):
    """
    Combine a list of decompositions for sub-blocks into one for the full matrix.
    decomp_list: list of (U, V, W, i_offset, j_offset, k_offset, n_sub, m_sub, p_sub)
    """
    total_rank = sum(len(d[0]) for d in decomp_list) # Wait, len(U) is n*m. Rank is len(U[0])
    total_rank = sum(len(d[0][0]) for d in decomp_list)
    
    U_full = [[0]*total_rank for _ in range(n*m)]
    V_full = [[0]*total_rank for _ in range(m*p)]
    W_full = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for U_sub, V_sub, W_sub, i_off, j_off, k_off, n_sub, m_sub, p_sub in decomp_list:
        r_sub = len(U_sub[0])
        
        # Copy U
        for i in range(n_sub):
            for j in range(m_sub):
                U_idx = i*m_sub + j
                U_full_idx = (i_off+i)*m + (j_off+j)
                for c in range(r_sub):
                    U_full[U_full_idx][col_offset+c] = U_sub[U_idx][c]
        
        # Copy V
        for j in range(m_sub):
            for k in range(p_sub):
                V_idx = j*p_sub + k
                V_full_idx = (j_off+j)*p + (k_off+k)
                for c in range(r_sub):
                    V_full[V_full_idx][col_offset+c] = V_sub[V_idx][c]
        
        # Copy W
        for k in range(p_sub):
            for i in range(n_sub):
                W_idx = k*n_sub + i
                W_full_idx = (k_off+k)*n + (i_off+i)
                for c in range(r_sub):
                    W_full[W_full_idx][col_offset+c] = W_sub[W_idx][c]
        
        col_offset += r_sub
        
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 145.0
    
    # 1. Block Decomposition
    # Split A (3x5) into A_L (3x2) and A_R (3x3)
    # Split B (5x5) into B_LL (2x2), B_LR (2x3), B_RL (3x2), B_RR (3x3)
    
    decomp_list = []
    
    # Block 1: A_L (3x2) * B_LL (2x2) -> C_LL (3x2)
    # Use M_3,2,2.
    # M_3,2,2 = M_2,2,2 (for first 2 rows of A) + M_1,2,2 (for 3rd row)
    # M_2,2,2: Rank 7
    U_222, V_222, W_222 = _strassen_2x2()
    # Shift indices for A rows 0,1; B cols 0,1; C cols 0,1
    decomp_list.append((U_222, V_222, W_222, 0, 0, 0, 2, 2, 2))
    
    # M_1,2,2: Rank 4 (Standard)
    U_122, V_122, W_122 = _standard_decomp(1, 2, 2)
    # Shift indices for A row 2; B cols 0,1; C cols 0,1
    decomp_list.append((U_122, V_122, W_122, 2, 0, 0, 1, 2, 2))
    
    # Block 2: A_L (3x2) * B_LR (2x3) -> C_LR (3x3)
    # M_3,2,3.
    # M_3,2,3 = M_2,2,3 (first 2 rows) + M_1,2,3 (3rd row)
    # M_2,2,3 = M_2,2,2 (cols 0,1) + M_2,2,1 (col 2)
    U_222_2, V_222_2, W_222_2 = _strassen_2x2()
    # Shift: A rows 0,1; B cols 0,1; C cols 0,1
    decomp_list.append((U_222_2, V_222_2, W_222_2, 0, 0, 2, 2, 2, 2))
    
    U_221, V_221, W_221 = _standard_decomp(2, 2, 1)
    # Shift: A rows 0,1; B col 2; C col 2
    decomp_list.append((U_221, V_221, W_221, 0, 2, 4, 2, 2, 1))
    
    # M_1,2,3 = M_1,2,2 (cols 0,1) + M_1,2,1 (col 2)
    U_122_2, V_122_2, W_122_2 = _standard_decomp(1, 2, 2)
    # Shift: A row 2; B cols 0,1; C cols 0,1
    decomp_list.append((U_122_2, V_122_2, W_122_2, 2, 0, 2, 1, 2, 2))
    
    U_121, V_121, W_121 = _standard_decomp(1, 2, 1)
    # Shift: A row 2; B col 2; C col 2
    decomp_list.append((U_121, V_121, W_121, 2, 2, 4, 1, 2, 1))
    
    # Block 3: A_R (3x3) * B_RL (3x2) -> C_RL (3x2)
    # M_3,3,2.
    # M_3,3,2 = M_2,3,2 (first 2 rows) + M_1,3,2 (3rd row)
    # M_2,3,2 = M_2,2,2 (cols 0,1) + M_2,1,2 (col 2)
    U_222_3, V_222_3, W_222_3 = _strassen_2x2()
    # Shift: A rows 0,1; B cols 0,1; C cols 0,1
    decomp_list.append((U_222_3, V_222_3, W_222_3, 0, 2, 0, 2, 2, 2))
    
    U_212, V_212, W_212 = _standard_decomp(2, 1, 2)
    # Shift: A rows 0,1; B col 2; C cols 0,1
    decomp_list.append((U_212, V_212, W_212, 0, 4, 0, 2, 1, 2))
    
    # M_1,3,2 = M_1,2,2 (cols 0,1) + M_1,1,2 (col 2)
    U_122_3, V_122_3, W_122_3 = _standard_decomp(1, 2, 2)
    # Shift: A row 2; B cols 0,1; C cols 0,1
    decomp_list.append((U_122_3, V_122_3, W_122_3, 2, 2, 0, 1, 2, 2))
    
    U_112, V_112, W_112 = _standard_decomp(1, 1, 2)
    # Shift: A row 2; B col 2; C cols 0,1
    decomp_list.append((U_112, V_112, W_112, 2, 4, 0, 1, 1, 2))
    
    # Block 4: A_R (3x3) * B_RR (3x3) -> C_RR (3x3)
    # M_3,3,3.
    # Use standard for now (Rank 27).
    U_333, V_333, W_333 = _standard_decomp(3, 3, 3)
    # Shift: A rows 0,1,2; B cols 0,1,2; C cols 0,1,2
    decomp_list.append((U_333, V_333, W_333, 0, 2, 2, 3, 3, 3))
    
    U, V, W = _combine_decomps(decomp_list, n, m, p)
    
    # Verify
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to standard
        U0, V0, W0 = _standard_decomp(n, m, p)
        return {"U": U0, "V": V0, "W": W0}

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    r = len(U[0])
    Uf = [[Fraction(U[i][c]) for c in range(r)] for i in range(n*m)]
    Vf = [[Fraction(V[j][c]) for c in range(r)] for j in range(m*p)]
    Wf = [[Fraction(W[k][c]) for c in range(r)] for k in range(p*n)]
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = Fraction(0)
                    for c in range(r):
                        s += Wf[k*n+i][c] * Uf[i*m+j1][c] * Vf[j2*p+k][c]
                    target = 1 if j1 == j2 else 0
                    if s != target:
                        return False
    return True

# EVOLVE-BLOCK-END
