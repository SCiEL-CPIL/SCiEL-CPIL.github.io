import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # T[a,b,c] = 1 iff a//m == c%n (i match), a%m == b//p (j match), b%p == c//n (k match)
    # a = i*m + j, b = j*p + k, c = k*n + i
    for a in range(n * m):
        i_a, j_a = divmod(a, m)
        for b in range(m * p):
            j_b, k_b = divmod(b, p)
            for c in range(p * n):
                k_c, i_c = divmod(c, n)
                expected = 1 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if s != expected:
                    return False
    return True

def _schoolbook(n, m, p):
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = DIMENSIONS # 3, 5, 5
    
    # We construct a rank-72 decomposition by partitioning the problem into blocks.
    # Block 1: i in 0..2, j in 0..2, k in 0..2 -> 3x3x3. Rank 27 (schoolbook)
    # Block 2: i in 0..2, j in 3..4, k in 0..2 -> 3x2x3. Rank 17 (Coppersmith-Sommese)
    # Block 3: i in 0..2, j in 0..2, k in 3..4 -> 3x3x2. Rank 17 (Coppersmith-Sommese)
    # Block 4: i in 0..2, j in 3..4, k in 3..4 -> 3x2x2. Rank 11 (Split 1x2x2 (Rank 4) + 2x2x2 (Rank 7))
    # Total: 27 + 17 + 17 + 11 = 72.
    
    U = {}
    V = {}
    W = {}
    
    def add_term(u_dict, v_dict, w_dict):
        # u_dict: {a_idx: coeff}
        # v_dict: {b_idx: coeff}
        # w_dict: {c_idx: coeff}
        col = len(U)
        for idx, val in u_dict.items():
            if idx not in U: U[idx] = {}
            U[idx][col] = val
        for idx, val in v_dict.items():
            if idx not in V: V[idx] = {}
            V[idx][col] = val
        for idx, val in w_dict.items():
            if idx not in W: W[idx] = {}
            W[idx][col] = val
            
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(k, i): return k * n + i
    
    # Block 1: 3x3x3 (Schoolbook)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_term({a_idx(i,j): 1}, {b_idx(j,k): 1}, {c_idx(k,i): 1})
                
    # Block 2: 3x2x3 (Coppersmith-Sommese Rank 17)
    # A is 3x2 (i:0-2, j:0-1 local), B is 2x3 (j:0-1 local, k:0-2 local)
    # Global: i:0-2, j:3-4, k:0-2
    # Local j=0 -> Global j=3, Local j=1 -> Global j=4
    # Coppersmith-Sommese for 3x2x3:
    # Let A = [A1 A2] (3x1, 3x1). B = [B1 B2] (1x3, 1x3).
    # C = A1 B1 + A2 B2.
    # The 17 terms are derived from the decomposition of the 2x2 block structure or similar.
    # Actually, for 3x2x3, we can split A into two 3x1 vectors and B into two 1x3 vectors.
    # A simpler way: Split A into A_top (rows 0,1) and A_bot (row 2).
    # A_top is 2x2, B is 2x3. Rank(2x2x3) = 11 (via Coppersmith-Sommese or similar).
    # A_bot is 1x2, B is 2x3. Rank(1x2x3) = 6 (Schoolbook 1*2*3=6).
    # Total 17.
    
    # Rank 11 for 2x2x3:
    # A (2x2), B (2x3).
    # Use Strassen-like or known optimal.
    # 2x2x3 can be done in 11.
    # Let's use the explicit 11-term decomposition for 2x2x3.
    # Or simpler: Split B into B_left (2x2) and B_right (2x1).
    # A (2x2) * B_left (2x2) -> Rank 7 (Strassen).
    # A (2x2) * B_right (2x1) -> Rank 4 (Schoolbook 2*2*1=4).
    # Total 11.
    
    # So for Block 2 (Global i:0-2, j:3-4, k:0-2):
    # Split i: i_top (0,1) and i_bot (2).
    # Sub-block 2a: i:0-1, j:3-4, k:0-2 (2x2x3). Rank 11.
    # Sub-block 2b: i:2, j:3-4, k:0-2 (1x2x3). Rank 6.
    
    # Sub-block 2a: 2x2x3 (i:0-1, j:3-4, k:0-2)
    # Split k: k_left (0,1) and k_right (2).
    # Sub-sub-block 2a-i: 2x2x2 (i:0-1, j:3-4, k:0-1). Rank 7 (Strassen).
    # Sub-sub-block 2a-ii: 2x2x1 (i:0-1, j:3-4, k:2). Rank 4.
    
    # Strassen for 2x2x2 (Local i:0-1, Local j:0-1, Local k:0-1)
    # Global i:0-1, Global j:3-4, Global k:0-1
    # A11=A[0,3], A12=A[0,4], A21=A[1,3], A22=A[1,4]
    # B11=B[3,0], B12=B[3,1], B21=B[4,0], B22=B[4,1]
    # C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
    
    def strassen_block(i_off, j_off, k_off):
        # i_off, j_off, k_off are global offsets for the 2x2x2 block
        # Local i=0,1; Local j=0,1; Local k=0,1
        A11 = a_idx(i_off+0, j_off+0)
        A12 = a_idx(i_off+0, j_off+1)
        A21 = a_idx(i_off+1, j_off+0)
        A22 = a_idx(i_off+1, j_off+1)
        B11 = b_idx(j_off+0, k_off+0)
        B12 = b_idx(j_off+0, k_off+1)
        B21 = b_idx(j_off+1, k_off+0)
        B22 = b_idx(j_off+1, k_off+1)
        C11 = c_idx(k_off+0, i_off+0)
        C12 = c_idx(k_off+1, i_off+0)
        C21 = c_idx(k_off+0, i_off+1)
        C22 = c_idx(k_off+1, i_off+1)
        
        # M1 = (A11+A22)B11
        add_term({A11: 1, A22: 1}, {B11: 1}, {C11: 1, C21: 1})
        # M2 = (A11+A22)B22
        add_term({A11: 1, A22: 1}, {B22: 1}, {C12: 1, C22: 1})
        # M3 = (A11+A12)B22
        add_term({A11: 1, A12: 1}, {B22: 1}, {C12: 1})
        # M4 = (A21-A11)B11
        add_term({A21: 1, A11: -1}, {B11: 1}, {C21: 1})
        # M5 = (A11+A21)B12
        add_term({A11: 1, A21: 1}, {B12: 1}, {C12: 1})
        # M6 = (A12-A22)(B11+B22)
        add_term({A12: 1, A22: -1}, {B11: 1, B22: 1}, {C11: 1})
        # M7 = (A22-A12)(B12+B21)
        add_term({A22: 1, A12: -1}, {B12: 1, B21: 1}, {C22: 1})
        
    strassen_block(0, 3, 0) # i:0-1, j:3-4, k:0-1
    
    # 2x2x1 (i:0-1, j:3-4, k:2) Rank 4
    # C[i, 2] = A[i,3]B[3,2] + A[i,4]B[4,2]
    for i in [0, 1]:
        for j in [3, 4]:
            add_term({a_idx(i, j): 1}, {b_idx(j, 2): 1}, {c_idx(2, i): 1})
            
    # Sub-block 2b: 1x2x3 (i:2, j:3-4, k:0-2) Rank 6 (Schoolbook)
    for j in [3, 4]:
        for k in range(3):
            add_term({a_idx(2, j): 1}, {b_idx(j, k): 1}, {c_idx(k, 2): 1})
            
    # Block 3: 3x3x2 (i:0-2, j:0-2, k:3-4) Rank 17
    # Symmetric to Block 2.
    # Split i: i_top (0,1) and i_bot (2).
    # Sub-block 3a: i:0-1, j:0-2, k:3-4 (2x3x2). Rank 11.
    # Sub-block 3b: i:2, j:0-2, k:3-4 (1x3x2). Rank 6.
    
    # Sub-block 3a: 2x3x2 (i:0-1, j:0-2, k:3-4)
    # Split j: j_left (0,1) and j_right (2).
    # Sub-sub-block 3a-i: 2x2x2 (i:0-1, j:0-1, k:3-4). Rank 7 (Strassen).
    # Sub-sub-block 3a-ii: 2x1x2 (i:0-1, j:2, k:3-4). Rank 4.
    
    strassen_block(0, 0, 3) # i:0-1, j:0-1, k:3-4
    
    # 2x1x2 (i:0-1, j:2, k:3-4) Rank 4
    # C[i, k] = A[i, 2] B[2, k]
    for i in [0, 1]:
        for k in [3, 4]:
            add_term({a_idx(i, 2): 1}, {b_idx(2, k): 1}, {c_idx(k, i): 1})
            
    # Sub-block 3b: 1x3x2 (i:2, j:0-2, k:3-4) Rank 6
    for j in range(3):
        for k in [3, 4]:
            add_term({a_idx(2, j): 1}, {b_idx(j, k): 1}, {c_idx(k, 2): 1})
            
    # Block 4: 3x2x2 (i:0-2, j:3-4, k:3-4) Rank 11
    # Split i: i_top (0) and i_bot (1,2).
    # Sub-block 4a: i:0, j:3-4, k:3-4 (1x2x2). Rank 4.
    # Sub-block 4b: i:1-2, j:3-4, k:3-4 (2x2x2). Rank 7 (Strassen).
    
    # Sub-block 4a: 1x2x2 (i:0, j:3-4, k:3-4) Rank 4
    for j in [3, 4]:
        for k in [3, 4]:
            add_term({a_idx(0, j): 1}, {b_idx(j, k): 1}, {c_idx(k, 0): 1})
            
    # Sub-block 4b: 2x2x2 (i:1-2, j:3-4, k:3-4) Rank 7
    strassen_block(1, 3, 3)
    
    # Convert sparse dicts to dense lists of lists
    num_U = n * m
    num_V = m * p
    num_W = p * n
    rank = len(U)
    
    U_list = [[0] * rank for _ in range(num_U)]
    V_list = [[0] * rank for _ in range(num_V)]
    W_list = [[0] * rank for _ in range(num_W)]
    
    for idx, terms in U.items():
        for col, val in terms.items():
            U_list[idx][col] = val
    for idx, terms in V.items():
        for col, val in terms.items():
            V_list[idx][col] = val
    for idx, terms in W.items():
        for col, val in terms.items():
            W_list[idx][col] = val
            
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
