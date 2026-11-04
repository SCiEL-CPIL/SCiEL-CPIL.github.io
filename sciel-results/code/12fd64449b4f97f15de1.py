"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    
    # Build the exact multiplication tensor T
    # T[k, i, j] is the coefficient of A[i,j] * B[j,k] in C[i,k]
    # Flattened indices:
    # A: i*m + j
    # B: j*p + k
    # C: k*n + i
    
    # We want U, V, W such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == T[k, i, j]
    
    # Schoolbook decomposition (Rank 24)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, idx] = 1
                V_school[j * p + k, idx] = 1
                W_school[k * n + i, idx] = 1
                idx += 1
    
    # Try rank 19 with direct construction
    # Strategy: Use Strassen for 2x2x2 blocks and optimize the rest
    
    # The 2x3x4 multiplication:
    # A is 2x3, B is 3x4, C is 2x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Decomposition approach:
    # Split by j dimension:
    # C = A[:,0:2] * B[0:2,:] + A[:,2:3] * B[2:3,:]
    # First term: 2x2 * 2x4 = two 2x2x2 blocks (Strassen rank 7 each = 14)
    # Second term: 2x1 * 1x4 = rank 8
    # Total: 22
    
    # For rank 19, try a more efficient decomposition:
    # Use a rank-12 decomposition for 2x2 * 2x4 and rank 7 for 2x1 * 1x4
    
    # Actually, let's try a direct approach with rank 19
    # We'll construct the factor matrices with specific support patterns
    
    rank_target = 19
    U = np.zeros((n * m, rank_target))
    V = np.zeros((m * p, rank_target))
    W = np.zeros((p * n, rank_target))
    
    # Strassen for first 2x2x2 block (A[0:2,0:2] * B[0:2,0:2])
    # Indices in U: A[i,j] -> i*m+j
    # Indices in V: B[j,k] -> j*p+k  
    # Indices in W: C[k,i] -> k*n+i
    
    # m1: (a00+a11)(b00+b11)
    U[0*3+0, 0] += 1; U[1*3+1, 0] += 1
    V[0*4+0, 0] += 1; V[1*4+1, 0] += 1
    W[0*2+0, 0] += 1; W[1*2+1, 0] += 1
    
    # m2: (a10+a11)b00
    U[1*3+0, 1] += 1; U[1*3+1, 1] += 1
    V[0*4+0, 1] += 1
    W[0*2+1, 1] += 1; W[1*2+1, 1] -= 1
    
    # m3: a00(b01-b11)
    U[0*3+0, 2] += 1
    V[0*4+1, 2] += 1; V[1*4+1, 2] -= 1
    W[1*2+0, 2] += 1; W[1*2+1, 2] += 1
    
    # m4: a11(b10-b00)
    U[1*3+1, 3] += 1
    V[1*4+0, 3] += 1; V[0*4+0, 3] -= 1
    W[0*2+0, 3] += 1; W[0*2+1, 3] += 1
    
    # m5: (a00+a01)b11
    U[0*3+0, 4] += 1; U[0*3+1, 4] += 1
    V[1*4+1, 4] += 1
    W[0*2+0, 4] -= 1; W[1*2+0, 4] += 1
    
    # m6: (a10-a00)(b00+b10)
    U[1*3+0, 5] += 1; U[0*3+0, 5] -= 1
    V[0*4+0, 5] += 1; V[1*4+0, 5] += 1
    W[1*2+1, 5] += 1
    
    # m7: (a01-a11)(b10+b11)
    U[0*3+1, 6] += 1; U[1*3+1, 6] -= 1
    V[1*4+0, 6] += 1; V[1*4+1, 6] += 1
    W[0*2+0, 6] += 1
    
    # Second 2x2x2 block (A[0:2,0:2] * B[0:2,2:4])
    # B indices: B[j,k] where j in {0,1}, k in {2,3}
    # V index: j*p+k = j*4+k
    
    # m1: (a00+a11)(b02+b13)
    U[0*3+0, 7] += 1; U[1*3+1, 7] += 1
    V[0*4+2, 7] += 1; V[1*4+3, 7] += 1
    W[2*2+0, 7] += 1; W[3*2+1, 7] += 1
    
    # m2: (a10+a11)b02
    U[1*3+0, 8] += 1; U[1*3+1, 8] += 1
    V[0*4+2, 8] += 1
    W[2*2+1, 8] += 1; W[3*2+1, 8] -= 1
    
    # m3: a00(b03-b13)
    U[0*3+0, 9] += 1
    V[0*4+3, 9] += 1; V[1*4+3, 9] -= 1
    W[3*2+0, 9] += 1; W[3*2+1, 9] += 1
    
    # m4: a11(b12-b02)
    U[1*3+1, 10] += 1
    V[1*4+2, 10] += 1; V[0*4+2, 10] -= 1
    W[2*2+0, 10] += 1; W[2*2+1, 10] += 1
    
    # m5: (a00+a01)b13
    U[0*3+0, 11] += 1; U[0*3+1, 11] += 1
    V[1*4+3, 11] += 1
    W[2*2+0, 11] -= 1; W[3*2+0, 11] += 1
    
    # m6: (a10-a00)(b02+b12)
    U[1*3+0, 12] += 1; U[0*3+0, 12] -= 1
    V[0*4+2, 12] += 1; V[1*4+2, 12] += 1
    W[3*2+1, 12] += 1
    
    # m7: (a01-a11)(b12+b13)
    U[0*3+1, 13] += 1; U[1*3+1, 13] -= 1
    V[1*4+2, 13] += 1; V[1*4+3, 13] += 1
    W[2*2+0, 13] += 1
    
    # Third part: A[0:2,2:3] * B[2:3,0:4]
    # A indices: A[i,2] -> i*3+2
    # B indices: B[2,k] -> 2*4+k = 8+k
    # C indices: C[k,i] -> k*2+i
    
    # This is a 2x1 * 1x4 multiplication, rank 8
    # But we only have 5 more slots (14-18)
    # Let's try to use rank 5 for this part
    
    # C[i,k] = A[i,2] * B[2,k]
    # We need: sum_r U[i,2,r] * V[2,k,r] * W[k,i,r] = 1 for all i,k
    
    # Try: use 5 terms with specific patterns
    # Term 14: i=0,k=0
    U[0*3+2, 14] += 1
    V[2*4+0, 14] += 1
    W[0*2+0, 14] += 1
    
    # Term 15: i=0,k=1
    U[0*3+2, 15] += 1
    V[2*4+1, 15] += 1
    W[1*2+0, 15] += 1
    
    # Term 16: i=0,k=2
    U[0*3+2, 16] += 1
    V[2*4+2, 16] += 1
    W[2*2+0, 16] += 1
    
    # Term 17: i=0,k=3
    U[0*3+2, 17] += 1
    V[2*4+3, 17] += 1
    W[3*2+0, 17] += 1
    
    # Term 18: i=1, all k
    # This won't work with just 1 term for i=1
    
    # Let's try a different approach: use the 5 remaining terms more cleverly
    # Reset and try a different decomposition
    
    # Actually, let's verify if the current construction works
    T = np.zeros((p, n, m))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k, i, j] = 1
                
    valid = True
    for k in range(p):
        for i in range(n):
            for j in range(m):
                total = 0
                for r in range(rank_target):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(total, T[k, i, j]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 22
    rank_22 = 22
    U22 = np.zeros((n * m, rank_22))
    V22 = np.zeros((m * p, rank_22))
    W22 = np.zeros((p * n, rank_22))
    
    def add_u(i, j, c, val):
        U22[i * m + j, c] += val
    def add_v(j, k, c, val):
        V22[j * p + k, c] += val
    def add_w(k, i, c, val):
        W22[k * n + i, c] += val
        
    # Strassen for 2x2
    def add_strassen_block(a_rows, b_rows, c_cols, offset):
        # m1: (a00+a11)(b00+b11)
        add_u(a_rows[0][0], a_rows[0][1], offset + 0, 1)
        add_u(a_rows[1][0], a_rows[1][1], offset + 0, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 0, 1)
        add_v(b_rows[1][0], b_rows[1][1], offset + 0, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 0, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 0, 1)
        
        # m2: (a10+a11)b00
        add_u(a_rows[2][0], a_rows[2][1], offset + 1, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 1, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 1, 1)
        add_w(c_cols[2][0], c_cols[2][1], offset + 1, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 1, -1)
        
        # m3: a00(b01-b11)
        add_u(a_rows[0][0], a_rows[0][1], offset + 2, 1)
        add_v(b_rows[1][0], b_rows[1][1], offset + 2, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 2, -1)
        add_w(c_cols[3][0], c_cols[3][1], offset + 2, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 2, 1)
        
        # m4: a11(b10-b00)
        add_u(a_rows[3][0], a_rows[3][1], offset + 3, 1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 3, 1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 3, -1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 3, 1)
        add_w(c_cols[2][0], c_cols[2][1], offset + 3, 1)
        
        # m5: (a00+a01)b11
        add_u(a_rows[0][0], a_rows[0][1], offset + 4, 1)
        add_u(a_rows[1][0], a_rows[1][1], offset + 4, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 4, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 4, -1)
        add_w(c_cols[3][0], c_cols[3][1], offset + 4, 1)
        
        # m6: (a10-a00)(b00+b10)
        add_u(a_rows[2][0], a_rows[2][1], offset + 5, 1)
        add_u(a_rows[0][0], a_rows[0][1], offset + 5, -1)
        add_v(b_rows[0][0], b_rows[0][1], offset + 5, 1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 5, 1)
        add_w(c_cols[1][0], c_cols[1][1], offset + 5, 1)
        
        # m7: (a01-a11)(b10+b11)
        add_u(a_rows[1][0], a_rows[1][1], offset + 6, 1)
        add_u(a_rows[3][0], a_rows[3][1], offset + 6, -1)
        add_v(b_rows[2][0], b_rows[2][1], offset + 6, 1)
        add_v(b_rows[3][0], b_rows[3][1], offset + 6, 1)
        add_w(c_cols[0][0], c_cols[0][1], offset + 6, 1)

    a_rows_1 = [(0,0), (0,1), (1,0), (1,1)]
    b_rows_1 = [(0,0), (0,1), (1,0), (1,1)]
    c_cols_1 = [(0,0), (1,1), (0,1), (1,0)]
    add_strassen_block(a_rows_1, b_rows_1, c_cols_1, 0)
    
    b_rows_2 = [(0,2), (0,3), (1,2), (1,3)]
    c_cols_2 = [(2,0), (3,1), (2,1), (3,0)]
    add_strassen_block(a_rows_1, b_rows_2, c_cols_2, 7)
    
    offset = 14
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            add_u(i, 2, offset, 1)
            add_v(2, k, offset, 1)
            add_w(k, i, offset, 1)
            offset += 1
            
    valid = True
    for k in range(p):
        for i in range(n):
            for j in range(m):
                total = 0
                for r in range(rank_22):
                    total += U22[i*m+j, r] * V22[j*p+k, r] * W22[k*n+i, r]
                if not np.isclose(total, T[k, i, j]):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if valid:
        return {"U": U22.tolist(), "V": V22.tolist(), "W": W22.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
