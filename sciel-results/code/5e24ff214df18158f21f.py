import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _build_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    Rank 7.
    Indices:
    U: 2x2 (4 rows)
    V: 2x2 (4 rows)
    W: 2x2 (4 rows)
    """
    # Strassen's 7 terms
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Term 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    W[1][0] = -1; W[2][0] = -1 # Wait, C11 gets M1, C22 gets M1.
    # C11 (0) = M1 ...
    # C22 (3) = M1 ...
    # So W[0][0]=1, W[3][0]=1.
    # What about the other signs?
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Reset W
    W = [[0]*7 for _ in range(4)]
    
    # T1
    W[0][0] = 1
    W[3][0] = 1
    
    # T2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    W[3][1] = -1
    
    # T3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # T4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # T5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # T6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # T7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook_221():
    """
    Returns U, V, W for 2x2x1 matrix multiplication.
    Rank 4.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)] # 2x1 B matrix has 2 entries
    W = [[0]*4 for _ in range(2)] # 2x1 C matrix has 2 entries
    
    # C1 = A11 B11 + A12 B21
    # C2 = A21 B11 + A22 B21
    
    # T1: A11 B11 -> C1
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1
    
    # T2: A12 B21 -> C1
    U[1][1] = 1
    V[1][1] = 1
    W[0][1] = 1
    
    # T3: A21 B11 -> C2
    U[2][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # T4: A22 B21 -> C2
    U[3][3] = 1
    V[1][3] = 1
    W[1][3] = 1
    
    return U, V, W

def _build_223():
    """
    Builds 2x2x3 decomposition from 222 (Strassen) and 221 (Schoolbook).
    Rank 11.
    """
    U1, V1, W1 = _build_strassen_222()
    U2, V2, W2 = _build_schoolbook_221()
    
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)] # 2x3 B has 6 entries
    W = [[0]*11 for _ in range(6)] # 2x3 C has 6 entries
    
    # Map 222 to k=0,1
    for a in range(4):
        for r in range(7):
            U[a][r] = U1[a][r]
            
    v_map = [0, 1, 3, 4]
    for b in range(4):
        for r in range(7):
            V[v_map[b]][r] = V1[b][r]
            
    w_map = [0, 1, 3, 4]
    for c in range(4):
        for r in range(7):
            W[w_map[c]][r] = W1[c][r]
            
    # Map 221 to k=2
    for a in range(4):
        for r in range(4):
            U[a][r + 7] = U2[a][r]
            
    v_map2 = [2, 5]
    for b in range(2):
        for r in range(4):
            V[v_map2[b]][r + 7] = V2[b][r]
            
    w_map2 = [2, 5]
    for c in range(2):
        for r in range(4):
            W[w_map2[c]][r + 7] = W2[c][r]
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    def make_vec(dim, indices_vals):
        vec = [0] * dim
        for idx, val in indices_vals:
            vec[idx] = val
        return vec

    # Pre-compute 222 decomposition (Strassen)
    U222, V222, W222 = _build_strassen_222()
    
    # 1. Row 0 (1x4 x 4x6): schoolbook, 24 terms
    for j in range(4):
        for k in range(6):
            u = make_vec(U_dim, [(j, 1)])  # A[0,j] is at index 0*4+j = j
            v = make_vec(V_dim, [(j*6+k, 1)])  # B[j,k] is at index j*6+k
            w = make_vec(W_dim, [(k*3+0, 1)])  # C[0,k] is at index k*3+0
            add_term(u, v, w)
            
    # 2. Rows 1-2 (2x4 x 4x6): Use Strassen for each 2x4 x 4x2 block
    # Split columns into 3 blocks of 2: cols 0-1, 2-3, 4-5
    # For each 2x4 x 4x2 block, split inner dim into 2+2
    # Each 2x2 x 2x2 uses Strassen (rank 7)
    # Total: 3 col blocks * 2 inner blocks * 7 = 42 terms
    
    for col_block in range(3):
        col_offset = col_block * 2
        for inner_block in range(2):
            inner_offset = inner_block * 2
            
            for r in range(7):
                u_vec = make_vec(U_dim, [])
                v_vec = make_vec(V_dim, [])
                w_vec = make_vec(W_dim, [])
                
                # U mapping: A rows 1-2, cols inner_offset to inner_offset+1
                # A[1, inner_offset] -> index 1*4 + inner_offset = 4 + inner_offset
                # A[1, inner_offset+1] -> index 1*4 + inner_offset + 1 = 5 + inner_offset
                # A[2, inner_offset] -> index 2*4 + inner_offset = 8 + inner_offset
                # A[2, inner_offset+1] -> index 2*4 + inner_offset + 1 = 9 + inner_offset
                u_map = [
                    4 + inner_offset,      # A[1, inner_offset]
                    5 + inner_offset,      # A[1, inner_offset+1]
                    8 + inner_offset,      # A[2, inner_offset]
                    9 + inner_offset       # A[2, inner_offset+1]
                ]
                for a in range(4):
                    if U222[a][r] != 0:
                        u_vec[u_map[a]] = U222[a][r]
                        
                # V mapping: B rows inner_offset to inner_offset+1, cols col_offset to col_offset+1
                # B[inner_offset, col_offset] -> index inner_offset*6 + col_offset
                # B[inner_offset, col_offset+1] -> index inner_offset*6 + col_offset + 1
                # B[inner_offset+1, col_offset] -> index (inner_offset+1)*6 + col_offset
                # B[inner_offset+1, col_offset+1] -> index (inner_offset+1)*6 + col_offset + 1
                v_map = [
                    inner_offset * 6 + col_offset,
                    inner_offset * 6 + col_offset + 1,
                    (inner_offset + 1) * 6 + col_offset,
                    (inner_offset + 1) * 6 + col_offset + 1
                ]
                for b in range(4):
                    if V222[b][r] != 0:
                        v_vec[v_map[b]] = V222[b][r]
                        
                # W mapping: C rows 1-2, cols col_offset to col_offset+1
                # C[1, col_offset] -> index col_offset*3 + 1
                # C[1, col_offset+1] -> index (col_offset+1)*3 + 1
                # C[2, col_offset] -> index col_offset*3 + 2
                # C[2, col_offset+1] -> index (col_offset+1)*3 + 2
                w_map = [
                    col_offset * 3 + 1,
                    (col_offset + 1) * 3 + 1,
                    col_offset * 3 + 2,
                    (col_offset + 1) * 3 + 2
                ]
                for c in range(4):
                    if W222[c][r] != 0:
                        w_vec[w_map[c]] = W222[c][r]
                        
                add_term(u_vec, v_vec, w_vec)
                
    rank = len(terms)
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    
    for r, (u, v, w) in enumerate(terms):
        for a in range(U_dim):
            if u[a] != 0:
                U[a][r] = u[a]
        for b in range(V_dim):
            if v[b] != 0:
                V[b][r] = v[b]
        for c in range(W_dim):
            if w[c] != 0:
                W[c][r] = w[c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
