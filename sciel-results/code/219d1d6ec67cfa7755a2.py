"""Rewritten program for the 2x4x7 matrix multiplication tensor decomposition task."""
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add Strassen block
    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col_offset
        # Local U indices: i*2 + j for i in 0,1 and j in 0,1
        # Global U index: (a_row_start + i)*m + (a_col_start + j)
        # Local V indices: k*2 + l for k in 0,1 and l in 0,1
        # Global V index: (b_row_start + k)*p + (b_col_start + l)
        # Local W indices: l*2 + i for l in 0,1 and i in 0,1
        # Global W index: (c_col_start + l)*n + (c_row_start + i)
        
        # Strassen factors
        # M1: (a+d)(e+h)
        # U: a(0), d(3)
        # V: e(0), h(3)
        # W: c11(0), c22(3)
        u_idx = [a_row_start*m + a_col_start, (a_row_start+1)*m + (a_col_start+1)]
        v_idx = [b_row_start*p + b_col_start, (b_row_start+1)*p + (b_col_start+1)]
        w_idx = [(c_col_start)*n + c_row_start, (c_col_start+1)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[1]][col_offset] += 1
        col_offset += 1
        
        # M2: (c+d)e
        # U: c(2), d(3)
        # V: e(0)
        # W: c21(2), c22(3) -> 1, -1
        u_idx = [(a_row_start+1)*m + a_col_start, (a_row_start+1)*m + (a_col_start+1)]
        v_idx = [b_row_start*p + b_col_start]
        w_idx = [(c_col_start)*n + (c_row_start+1), (c_col_start+1)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[1]][col_offset] -= 1
        col_offset += 1
        
        # M3: a(f-h)
        # U: a(0)
        # V: f(1), -h(3)
        # W: c12(1), c22(3)
        u_idx = [a_row_start*m + a_col_start]
        v_idx = [b_row_start*p + (b_col_start+1), (b_row_start+1)*p + (b_col_start+1)]
        w_idx = [(c_col_start+1)*n + c_row_start, (c_col_start+1)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] -= 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[1]][col_offset] += 1
        col_offset += 1
        
        # M4: d(g-e)
        # U: d(3)
        # V: g(2), -e(0)
        # W: c11(0), c21(2)
        u_idx = [(a_row_start+1)*m + (a_col_start+1)]
        v_idx = [(b_row_start+1)*p + b_col_start, b_row_start*p + b_col_start]
        w_idx = [(c_col_start)*n + c_row_start, (c_col_start)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] -= 1
        W[w_idx[0]][col_offset] += 1
        W[w_idx[1]][col_offset] += 1
        col_offset += 1
        
        # M5: (a+b)h
        # U: a(0), b(1)
        # V: h(3)
        # W: -c11(0), c12(1)
        u_idx = [a_row_start*m + a_col_start, a_row_start*m + (a_col_start+1)]
        v_idx = [(b_row_start+1)*p + (b_col_start+1)]
        w_idx = [(c_col_start)*n + c_row_start, (c_col_start+1)*n + c_row_start]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] += 1
        V[v_idx[0]][col_offset] += 1
        W[w_idx[0]][col_offset] -= 1
        W[w_idx[1]][col_offset] += 1
        col_offset += 1
        
        # M6: (c-a)(e+f)
        # U: c(2), -a(0)
        # V: e(0), f(1)
        # W: c22(3)
        u_idx = [(a_row_start+1)*m + a_col_start, a_row_start*m + a_col_start]
        v_idx = [b_row_start*p + b_col_start, b_row_start*p + (b_col_start+1)]
        w_idx = [(c_col_start+1)*n + (c_row_start+1)]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] -= 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        col_offset += 1
        
        # M7: (b-d)(g+h)
        # U: b(1), -d(3)
        # V: g(2), h(3)
        # W: c11(0)
        u_idx = [a_row_start*m + (a_col_start+1), (a_row_start+1)*m + (a_col_start+1)]
        v_idx = [(b_row_start+1)*p + b_col_start, (b_row_start+1)*p + (b_col_start+1)]
        w_idx = [(c_col_start)*n + c_row_start]
        U[u_idx[0]][col_offset] += 1
        U[u_idx[1]][col_offset] -= 1
        V[v_idx[0]][col_offset] += 1
        V[v_idx[1]][col_offset] += 1
        W[w_idx[0]][col_offset] += 1
        col_offset += 1

    # Helper to add Dot Product block M_{2,2,1}
    def add_dot(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col_offset
        # C_i = sum A_ik B_k1
        # C_0 = A_00 B_00 + A_01 B_10
        # C_1 = A_10 B_00 + A_11 B_10
        
        # Term 1: A_00 B_00 -> C_0
        U[a_row_start*m + a_col_start][col_offset] += 1
        V[b_row_start*p + b_col_start][col_offset] += 1
        W[c_col_start*n + c_row_start][col_offset] += 1
        col_offset += 1
        
        # Term 2: A_01 B_10 -> C_0
        U[a_row_start*m + (a_col_start+1)][col_offset] += 1
        V[(b_row_start+1)*p + b_col_start][col_offset] += 1
        W[c_col_start*n + c_row_start][col_offset] += 1
        col_offset += 1
        
        # Term 3: A_10 B_00 -> C_1
        U[(a_row_start+1)*m + a_col_start][col_offset] += 1
        V[b_row_start*p + b_col_start][col_offset] += 1
        W[c_col_start*n + (c_row_start+1)][col_offset] += 1
        col_offset += 1
        
        # Term 4: A_11 B_10 -> C_1
        U[(a_row_start+1)*m + (a_col_start+1)][col_offset] += 1
        V[(b_row_start+1)*p + b_col_start][col_offset] += 1
        W[c_col_start*n + (c_row_start+1)][col_offset] += 1
        col_offset += 1

    # Block 1: T1a
    add_strassen(0, 0, 0, 0, 0, 0)
    # Block 2: T1b
    add_strassen(0, 0, 0, 2, 0, 2)
    # Block 3: T2a
    add_strassen(0, 2, 2, 0, 0, 0)
    # Block 4: T2b
    add_strassen(0, 2, 2, 2, 0, 2)
    # Block 5: T3a
    add_strassen(0, 0, 0, 4, 0, 4)
    # Block 6: T4a
    add_strassen(0, 2, 2, 4, 0, 4)
    # Block 7: T3b
    add_dot(0, 0, 0, 6, 0, 6)
    # Block 8: T4b
    add_dot(0, 2, 2, 6, 0, 6)

    return {"U": U, "V": V, "W": W}
