import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 terms of the rank-7 decomposition for 2x2x2 matrix multiplication.
    """
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 1])  # M7
    ]
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(rank):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_4x4_decomp():
    """
    Construct a rank-49 decomposition for 4x4x4 matrix multiplication using Strassen's 2x2 algorithm.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    base_terms = _strassen_2x2_terms()
    
    block_terms = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 0]]),
        ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [[-1, 0], [1, 0]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        ([[0, 0], [1, -1]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        ([[0, 0], [1, -1]], [[0, 0], [1, 1]], [[-1, 0], [0, 1]])
    ]
    
    col = 0
    for t_block in range(7):
        A_coeffs, B_coeffs, C_coeffs = block_terms[t_block]
        for t_inner in range(7):
            U_inner, V_inner, W_inner = base_terms[t_inner]
            
            for r in range(2):
                for s in range(2):
                    if A_coeffs[r][s] != 0:
                        for dr in range(2):
                            for dc in range(2):
                                u_idx = (2*r + dr) * 4 + (2*s + dc)
                                u_inner_idx = dr * 2 + dc
                                U[u_idx][col] += A_coeffs[r][s] * U_inner[u_inner_idx]
            
            for s in range(2):
                for t in range(2):
                    if B_coeffs[s][t] != 0:
                        for dr in range(2):
                            for dc in range(2):
                                v_idx = (2*s + dr) * 4 + (2*t + dc)
                                v_inner_idx = dr * 2 + dc
                                V[v_idx][col] += B_coeffs[s][t] * V_inner[v_inner_idx]
            
            for t in range(2):
                for r in range(2):
                    if C_coeffs[t][r] != 0:
                        for dr in range(2):
                            for dc in range(2):
                                w_idx = (2*t + dc) * 4 + (2*r + dr)
                                w_inner_idx = dr * 2 + dc
                                W[w_idx][col] += C_coeffs[t][r] * W_inner[w_inner_idx]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _decomp_423():
    """
    Construct a rank-22 decomposition for 4x2x3 matrix multiplication.
    A: 4x2, B: 2x3, C: 4x3.
    Split A into A1 (2x2), A2 (2x2).
    Split B into B1 (2x2), B2 (2x1).
    C = [A1 B1, A1 B2; A2 B1, A2 B2].
    """
    n, m, p = 4, 2, 3
    rank = 22
    U = [[0] * rank for _ in range(n * m)] # 8 x 22
    V = [[0] * rank for _ in range(m * p)] # 6 x 22
    W = [[0] * rank for _ in range(p * n)] # 12 x 22
    
    base_terms = _strassen_2x2_terms()
    
    # Helper to add a 2x2x2 Strassen term
    def add_strassen_222(offset_u, offset_v, offset_w, scale=1):
        nonlocal col
        for t_inner in range(7):
            U_inner, V_inner, W_inner = base_terms[t_inner]
            for dr in range(2):
                for dc in range(2):
                    # U
                    u_idx = offset_u + dr * 2 + dc
                    U[u_idx][col] += scale * U_inner[dr * 2 + dc]
                    # V
                    v_idx = offset_v + dr * 2 + dc
                    V[v_idx][col] += scale * V_inner[dr * 2 + dc]
                    # W
                    w_idx = offset_w + dc * 2 + dr # W[k*n+i] -> W[dc*2+dr] for 2x2 block?
                    # Wait, W index for 2x2 block at (row_block, col_block) in C.
                    # C is 4x3.
                    # This helper is for a specific 2x2 block of C.
                    # The W index in the full 4x3 matrix is k*4+i.
                    # For a 2x2 block at rows r0..r0+1, cols c0..c0+1.
                    # W index = (c0+dc)*4 + (r0+dr).
                    W[w_idx][col] += scale * W_inner[dr * 2 + dc]
            col += 1

    # Helper to add a 2x2x1 schoolbook term
    def add_schoolbook_221(offset_u, offset_v, offset_w, scale=1):
        nonlocal col
        # 2x2x1: C[0] = A[0,0]B[0] + A[0,1]B[1]
        # C[1] = A[1,0]B[0] + A[1,1]B[1]
        # Terms:
        # 1: A[0,0] * B[0] -> C[0]
        # 2: A[0,1] * B[1] -> C[0]
        # 3: A[1,0] * B[0] -> C[1]
        # 4: A[1,1] * B[1] -> C[1]
        
        # U indices for A (2x2): offset_u + 0, 1, 2, 3
        # V indices for B (2x1): offset_v + 0, 1
        # W indices for C (2x1): offset_w + 0, 1 (k*4+i, but here k is fixed? No, this is a block)
        # Actually, 2x2x1 means A is 2x2, B is 2x1, C is 2x1.
        # W index in full matrix: k*4+i.
        # If this block is at C rows r0..r0+1, col c.
        # W index = c*4 + r.
        
        # Term 1: A[0,0] B[0] -> C[0]
        U[offset_u + 0][col] += scale * 1
        V[offset_v + 0][col] += scale * 1
        W[offset_w + 0][col] += scale * 1
        col += 1
        
        # Term 2: A[0,1] B[1] -> C[0]
        U[offset_u + 1][col] += scale * 1
        V[offset_v + 1][col] += scale * 1
        W[offset_w + 0][col] += scale * 1
        col += 1
        
        # Term 3: A[1,0] B[0] -> C[1]
        U[offset_u + 2][col] += scale * 1
        V[offset_v + 0][col] += scale * 1
        W[offset_w + 1][col] += scale * 1
        col += 1
        
        # Term 4: A[1,1] B[1] -> C[1]
        U[offset_u + 3][col] += scale * 1
        V[offset_v + 1][col] += scale * 1
        W[offset_w + 1][col] += scale * 1
        col += 1

    col = 0
    
    # A1 (rows 0,1), B1 (cols 0,1) -> C11 (rows 0,1, cols 0,1)
    # U offset: 0 (A rows 0,1, cols 0,1)
    # V offset: 0 (B rows 0,1, cols 0,1)
    # W offset: 0*4 + 0 = 0 (C col 0, rows 0,1) -> indices 0, 1
    # Wait, W index is k*4+i.
    # C block at col 0, rows 0,1.
    # W indices: 0*4+0=0, 0*4+1=1.
    add_strassen_222(0, 0, 0)
    
    # A1 (rows 0,1), B2 (col 2) -> C12 (rows 0,1, col 2)
    # U offset: 0
    # V offset: 2*3 + 0 = 6? No, V is 2x3. V[j*3+k].
    # B2 is row 0,1? No, B is 2x3. B1 is 2x2 (cols 0,1), B2 is 2x1 (col 2).
    # V offset for B2: j*3 + 2. j=0,1.
    # So V indices: 2, 5.
    # W offset: 2*4 + 0 = 8 (C col 2, rows 0,1). Indices 8, 9.
    add_schoolbook_221(0, 2, 8)
    
    # A2 (rows 2,3), B1 (cols 0,1) -> C21 (rows 2,3, cols 0,1)
    # U offset: 2*2 + 0 = 4 (A rows 2,3, cols 0,1).
    # V offset: 0.
    # W offset: 0*4 + 2 = 2 (C col 0, rows 2,3). Indices 2, 3.
    add_strassen_222(4, 0, 2)
    
    # A2 (rows 2,3), B2 (col 2) -> C22 (rows 2,3, col 2)
    # U offset: 4.
    # V offset: 2.
    # W offset: 2*4 + 2 = 10 (C col 2, rows 2,3). Indices 10, 11.
    add_schoolbook_221(4, 2, 10)
    
    return {"U": U, "V": V, "W": W}

def _decomp_443_optimized():
    """
    Construct a rank-44 decomposition for 4x4x3 matrix multiplication.
    Split A into AL (4x2) and AR (4x2).
    Split B into BT (2x3) and BB (2x3).
    C = AL BT + AR BB.
    """
    n, m, p = 4, 4, 3
    rank = 44
    U = [[0] * rank for _ in range(n * m)] # 16 x 44
    V = [[0] * rank for _ in range(m * p)] # 12 x 44
    W = [[0] * rank for _ in range(p * n)] # 12 x 44
    
    d1 = _decomp_423()
    d2 = _decomp_423()
    
    r1 = 22
    r2 = 22
    
    # Copy d1 (AL BT)
    # U: AL is cols 0,1 of A. U1 is 8x22 (rows 0..7 of A).
    # A rows 0..3, cols 0..1.
    # U1[i*2+j] for i=0..3, j=0..1.
    # This matches U[0..7] in d1.
    for i in range(8):
        for c in range(r1):
            U[i][c] = d1["U"][i][c]
            
    # V: BT is rows 0,1 of B. V1 is 6x22 (rows 0..5 of B).
    # B rows 0..1, cols 0..2.
    # V1[j*3+k] for j=0..1, k=0..2.
    # This matches V[0..5] in d1.
    for i in range(6):
        for c in range(r1):
            V[i][c] = d1["V"][i][c]
            
    # W: C1 is rows 0..3, cols 0..2.
    # W1 is 12x22.
    for i in range(12):
        for c in range(r1):
            W[i][c] = d1["W"][i][c]
            
    # Copy d2 (AR BB)
    # U: AR is cols 2,3 of A. U2 is 8x22.
    # A rows 0..3, cols 2..3.
    # U2[i*2+j] for i=0..3, j=0..1.
    # In global U, this is U[i*4 + (2+j)].
    for i in range(4):
        for j in range(2):
            for c in range(r2):
                U[i*4 + 2 + j][r1 + c] = d2["U"][i*2 + j][c]
                
    # V: BB is rows 2,3 of B. V2 is 6x22.
    # B rows 2..3, cols 0..2.
    # V2[j*3+k] for j=0..1, k=0..2.
    # In global V, this is V[(2+j)*3 + k].
    for j in range(2):
        for k in range(3):
            for c in range(r2):
                V[(2+j)*3 + k][r1 + c] = d2["V"][j*3 + k][c]
                
    # W: C2 is rows 0..3, cols 0..2.
    # W2 is 12x22.
    for i in range(12):
        for c in range(r2):
            W[i][r1 + c] = d2["W"][i][c]
            
    return {"U": U, "V": V, "W": W}

def _combine_decomps(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions:
    decomp1: for n x m * m x p1
    decomp2: for n x m * m x p2
    Result: for n x m * m x (p1+p2)
    """
    r1 = len(decomp1["U"][0])
    r2 = len(decomp2["U"][0])
    r = r1 + r2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * (p1 + p2))]
    W = [[0] * r for _ in range((p1 + p2) * n)]
    
    for i in range(n * m):
        for c in range(r1):
            U[i][c] = decomp1["U"][i][c]
        for c in range(r2):
            U[i][r1 + c] = decomp2["U"][i][c]
            
    for i in range(m * p1):
        for c in range(r1):
            V[i][c] = decomp1["V"][i][c]
    for i in range(m * p2):
        for c in range(r2):
            V[m * p1 + i][r1 + c] = decomp2["V"][i][c]
            
    for i in range(p1 * n):
        for c in range(r1):
            W[i][c] = decomp1["W"][i][c]
    for i in range(p2 * n):
        for c in range(r2):
            W[p1 * n + i][r1 + c] = decomp2["W"][i][c]
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Start with schoolbook (rank 112)
    result = _schoolbook_447()
    
    # Try hybrid: Strassen for 4x4x4 (rank 49) + Optimized 4x4x3 (rank 44) = rank 93
    try:
        decomp_444 = _strassen_4x4_decomp()
        decomp_443 = _decomp_443_optimized()
        hybrid = _combine_decomps(decomp_444, decomp_443, n, m, 4, 3)
        
        # Verify
        if _verify_decomposition(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
            result = hybrid
    except Exception:
        pass
    
    return result

def _schoolbook_447():
    """
    Schoolbook decomposition for 4x4 * 4x7 matrix multiplication.
    Rank = 4*4*7 = 112.
    """
    n, m, p = 4, 4, 7
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
