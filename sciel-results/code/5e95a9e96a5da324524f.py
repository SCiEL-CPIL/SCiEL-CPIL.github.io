import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target rank 24
    rank = 24
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    def add_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal col
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] = val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] = val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] = val
        col += 1
        
    # Trivial 2x2 multiplication
    # A: i_start, i_start+1; j_start, j_start+1
    # B: k_start, k_start+1; l_start, l_start+1
    # C: c_i_start, c_i_start+1; c_k_start, c_k_start+1
    def add_2x2_trivial(i_start, j_start, k_start, l_start, c_i_start, c_k_start):
        nonlocal col
        # C00 = A00 B00 + A01 B10
        add_term([i_start*m + j_start], [1], [k_start*p + l_start], [1], [c_k_start*n + c_i_start], [1])
        add_term([i_start*m + j_start+1], [1], [(k_start+1)*p + l_start], [1], [c_k_start*n + c_i_start], [1])
        
        # C01 = A00 B01 + A01 B11
        add_term([i_start*m + j_start], [1], [k_start*p + l_start+1], [1], [(c_k_start+1)*n + c_i_start], [1])
        add_term([i_start*m + j_start+1], [1], [(k_start+1)*p + l_start+1], [1], [(c_k_start+1)*n + c_i_start], [1])
        
        # C10 = A10 B00 + A11 B10
        add_term([(i_start+1)*m + j_start], [1], [k_start*p + l_start], [1], [c_k_start*n + c_i_start+1], [1])
        add_term([(i_start+1)*m + j_start+1], [1], [(k_start+1)*p + l_start], [1], [c_k_start*n + c_i_start+1], [1])
        
        # C11 = A10 B01 + A11 B11
        add_term([(i_start+1)*m + j_start], [1], [k_start*p + l_start+1], [1], [(c_k_start+1)*n + c_i_start+1], [1])
        add_term([(i_start+1)*m + j_start+1], [1], [(k_start+1)*p + l_start+1], [1], [(c_k_start+1)*n + c_i_start+1], [1])

    # Trivial 1x2 multiplication
    # A: i_start; j_start, j_start+1
    # B: k_start, k_start+1; l_start, l_start+1
    # C: c_i_start; c_k_start, c_k_start+1
    def add_1x2_trivial(i_start, j_start, k_start, l_start, c_i_start, c_k_start):
        nonlocal col
        # C0 = A0 B00 + A1 B10
        add_term([i_start*m + j_start], [1], [k_start*p + l_start], [1], [c_k_start*n + c_i_start], [1])
        add_term([i_start*m + j_start+1], [1], [(k_start+1)*p + l_start], [1], [c_k_start*n + c_i_start], [1])
        
        # C1 = A0 B01 + A1 B11
        add_term([i_start*m + j_start], [1], [k_start*p + l_start+1], [1], [(c_k_start+1)*n + c_i_start], [1])
        add_term([i_start*m + j_start+1], [1], [(k_start+1)*p + l_start+1], [1], [(c_k_start+1)*n + c_i_start], [1])

    # Block 1: A11 (rows 0,1; cols 0,1) * B11 (rows 0,1; cols 0,1) -> C11 (rows 0,1; cols 0,1)
    add_2x2_trivial(0, 0, 0, 0, 0, 0)
    
    # Block 2: A12 (rows 0,1; cols 2,3) * B21 (rows 2,3; cols 0,1) -> C12 (rows 0,1; cols 0,1)
    # Note: The output of A12*B21 is added to C11?
    # C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For i in {0,1}, k in {0,1}:
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    # = (A11 B11)[i,k] + (A12 B21)[i,k]
    # So we add the results of A12*B21 to the same C indices.
    add_2x2_trivial(0, 2, 2, 0, 0, 0)
    
    # Block 3: A21 (row 2; cols 0,1) * B11 (rows 0,1; cols 0,1) -> C21 (row 2; cols 0,1)
    # C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k] + A[2,3]B[3,k]
    # = (A21 B11)[k] + (A22 B21)[k]
    add_1x2_trivial(2, 0, 0, 0, 2, 0)
    
    # Block 4: A22 (row 2; cols 2,3) * B21 (rows 2,3; cols 0,1) -> C22 (row 2; cols 0,1)
    add_1x2_trivial(2, 2, 2, 0, 2, 0)
    
    return {"U": U, "V": V, "W": W}
