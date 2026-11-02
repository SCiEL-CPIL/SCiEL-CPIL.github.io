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
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # T1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
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

def _build_schoolbook_122():
    """
    Returns U, V, W for 1x2x2 matrix multiplication.
    Rank 4.
    A is 1x2, B is 2x2, C is 1x2.
    U: 2 rows
    V: 4 rows
    W: 2 rows
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(4)]
    W = [[0]*4 for _ in range(2)]
    
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # T1: A1 B11 -> C1
    U[0][0] = 1
    V[0][0] = 1 # B11
    W[0][0] = 1 # C1
    
    # T2: A2 B21 -> C1
    U[1][1] = 1
    V[2][1] = 1 # B21
    W[0][1] = 1 # C1
    
    # T3: A1 B12 -> C2
    U[0][2] = 1
    V[1][2] = 1 # B12
    W[1][2] = 1 # C2
    
    # T4: A2 B22 -> C2
    U[1][3] = 1
    V[3][3] = 1 # B22
    W[1][3] = 1 # C2
    
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

    U222, V222, W222 = _build_strassen_222()
    U122, V122, W122 = _build_schoolbook_122()
    
    # Part 1: i=0,1 (Top 2 rows)
    # Use Strassen for 2x2x2 blocks
    # A is 2x4, B is 4x6
    # Split A cols into 2 blocks of 2
    # Split B cols into 3 blocks of 2
    
    for k_block in range(2):
        k_offset = k_block * 2
        for j_block in range(3):
            j_offset = j_block * 2
            
            for r in range(7):
                u_vec = make_vec(U_dim, [])
                v_vec = make_vec(V_dim, [])
                w_vec = make_vec(W_dim, [])
                
                # A indices: i in {0,1}, k in {k_offset, k_offset+1}
                # U index: i*m + k
                u_map = [
                    0*4 + k_offset,
                    0*4 + k_offset + 1,
                    1*4 + k_offset,
                    1*4 + k_offset + 1
                ]
                for a in range(4):
                    if U222[a][r] != 0:
                        u_vec[u_map[a]] = U222[a][r]
                        
                # B indices: k in {k_offset, k_offset+1}, j in {j_offset, j_offset+1}
                # V index: k*p + j
                v_map = [
                    k_offset * 6 + j_offset,
                    k_offset * 6 + j_offset + 1,
                    (k_offset+1) * 6 + j_offset,
                    (k_offset+1) * 6 + j_offset + 1
                ]
                for b in range(4):
                    if V222[b][r] != 0:
                        v_vec[v_map[b]] = V222[b][r]
                        
                # C indices: i in {0,1}, j in {j_offset, j_offset+1}
                # W index: j*n + i
                w_map = [
                    j_offset * 3 + 0,
                    j_offset * 3 + 1,
                    (j_offset+1) * 3 + 0,
                    (j_offset+1) * 3 + 1
                ]
                for c in range(4):
                    if W222[c][r] != 0:
                        w_vec[w_map[c]] = W222[c][r]
                        
                add_term(u_vec, v_vec, w_vec)
                
    # Part 2: i=2 (Bottom row)
    # Use Schoolbook for 1x2x2 blocks
    # A is 1x4, B is 4x6
    # Split A cols into 2 blocks of 2
    # Split B cols into 3 blocks of 2
    
    for k_block in range(2):
        k_offset = k_block * 2
        for j_block in range(3):
            j_offset = j_block * 2
            
            for r in range(4):
                u_vec = make_vec(U_dim, [])
                v_vec = make_vec(V_dim, [])
                w_vec = make_vec(W_dim, [])
                
                # A indices: i=2, k in {k_offset, k_offset+1}
                # U index: 2*4 + k
                u_map = [
                    2*4 + k_offset,
                    2*4 + k_offset + 1
                ]
                for a in range(2):
                    if U122[a][r] != 0:
                        u_vec[u_map[a]] = U122[a][r]
                        
                # B indices: k in {k_offset, k_offset+1}, j in {j_offset, j_offset+1}
                # V index: k*p + j
                v_map = [
                    k_offset * 6 + j_offset,
                    k_offset * 6 + j_offset + 1,
                    (k_offset+1) * 6 + j_offset,
                    (k_offset+1) * 6 + j_offset + 1
                ]
                for b in range(4):
                    if V122[b][r] != 0:
                        v_vec[v_map[b]] = V122[b][r]
                        
                # C indices: i=2, j in {j_offset, j_offset+1}
                # W index: j*n + 2
                w_map = [
                    j_offset * 3 + 2,
                    (j_offset+1) * 3 + 2
                ]
                for c in range(2):
                    if W122[c][r] != 0:
                        w_vec[w_map[c]] = W122[c][r]
                        
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
