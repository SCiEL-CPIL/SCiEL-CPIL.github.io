"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS

    def schoolbook():
        """Rank 36 schoolbook decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify tensor identity: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2x2 (rank 7).
        Returns U (4x7), V (4x7), W (4x7) where:
        U[i*m+j, c] with m=2: indices 0=a00, 1=a01, 2=a10, 3=a11
        V[j*p+k, c] with p=2: indices 0=b00, 1=b01, 2=b10, 3=b11
        W[k*n+i, c] with n=2: indices 0=C00, 1=C01, 2=C10, 3=C11
        
        Strassen formulas:
        M1 = (a00+a11)(b00+b11)
        M2 = (a10+a11)b00
        M3 = a00(b10-b11)
        M4 = a11(b01-b00)
        M5 = (a00+a01)b11
        M6 = (a10-a00)(b00+b10)
        M7 = (a01-a11)(b01+b11)
        
        Output:
        C00 = M1 - M4 + M5 + M7
        C01 = M3 + M4
        C10 = M2 + M6
        C11 = M1 - M3 + M6 + M7
        """
        # U factors (for A matrix entries)
        # U[0,c] = coeff of a00, U[1,c] = coeff of a01, U[2,c] = coeff of a10, U[3,c] = coeff of a11
        U = [
            [1, 0, 1, 1, 1, -1, 1],  # a00 coefficients for M1-M7
            [0, 0, 0, 0, 1, 0, -1],  # a01 coefficients
            [0, 1, 0, 0, 0, 1, 0],   # a10 coefficients
            [1, 1, 0, 1, 0, 0, 0]    # a11 coefficients
        ]
        
        # V factors (for B matrix entries)
        # V[0,c] = coeff of b00, V[1,c] = coeff of b01, V[2,c] = coeff of b10, V[3,c] = coeff of b11
        V = [
            [1, 1, 0, -1, 0, 1, 0],  # b00 coefficients
            [0, 0, 0, 1, 0, 0, 1],   # b01 coefficients
            [0, 0, 1, 0, 0, 1, 0],   # b10 coefficients
            [1, 0, -1, 0, 1, 0, 1]   # b11 coefficients
        ]
        
        # W factors (for C matrix entries)
        # W[0,c] = coeff for C00, W[1,c] = coeff for C01, W[2,c] = coeff for C10, W[3,c] = coeff for C11
        W = [
            [1, 0, 0, 0, 1, 0, 1],   # C00 = M1 - M4 + M5 + M7
            [0, 0, 1, 1, 0, 0, 0],   # C01 = M3 + M4
            [0, 1, 0, 0, 0, 1, 0],   # C10 = M2 + M6
            [1, 0, -1, 0, 0, 1, 1]   # C11 = M1 - M3 + M6 + M7
        ]
        
        return U, V, W

    def combine_decompositions(blocks):
        """
        Combine multiple block decompositions into one.
        Each block is a tuple (U_block, V_block, W_block, i_offset, j_offset, k_offset, n_block, m_block, p_block)
        where the block handles a sub-problem of size n_block x m_block x p_block
        at position (i_offset, j_offset, k_offset) in the full problem.
        """
        total_rank = sum(len(U_b[0]) for U_b, V_b, W_b, *_ in blocks)
        
        # Full matrices
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        for U_b, V_b, W_b, i_off, j_off, k_off, n_b, m_b, p_b in blocks:
            rank_b = len(U_b[0])
            for i_local in range(n_b):
                for j_local in range(m_b):
                    i_global = i_off + i_local
                    j_global = j_off + j_local
                    u_idx = i_global * m + j_global
                    for c in range(rank_b):
                        U[u_idx][col_offset + c] = U_b[i_local * m_b + j_local][c]
            
            for j_local in range(m_b):
                for k_local in range(p_b):
                    j_global = j_off + j_local
                    k_global = k_off + k_local
                    v_idx = j_global * p + k_global
                    for c in range(rank_b):
                        V[v_idx][col_offset + c] = V_b[j_local * p_b + k_local][c]
            
            for k_local in range(p_b):
                for i_local in range(n_b):
                    k_global = k_off + k_local
                    i_global = i_off + i_local
                    w_idx = k_global * n + i_global
                    for c in range(rank_b):
                        W[w_idx][col_offset + c] = W_b[k_local * n_b + i_local][c]
            
            col_offset += rank_b
        
        return {"U": U, "V": V, "W": W}

    def check_time():
        return time.time() - start_time

    # First, verify Strassen works for 2x2x2
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Verify Strassen decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for c in range(7):
                    total += U_s[i*2+j][c] * V_s[j*2+k][c] * W_s[k*2+i][c]
                if abs(total - 1) > 1e-9:
                    # Strassen verification failed, fall back to schoolbook
                    return schoolbook()
    
    # Build rank-32 decomposition: 4 Strassen blocks (2x2x2) + 1 schoolbook block (2x2x1)
    # Split p=9 into: [0,1], [2,3], [4,5], [6,7], [8]
    
    blocks = []
    
    # Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    for pair_idx in range(4):
        k_start = pair_idx * 2
        blocks.append((U_s, V_s, W_s, 0, 0, k_start, 2, 2, 2))
    
    # Schoolbook block for column 8 (2x2x1)
    U_sb = [[1, 0], [0, 1], [0, 0], [0, 0]]  # Wait, this is wrong
    # For 2x2x1: U has shape (4, 4), V has shape (2, 4), W has shape (2, 4)
    # Actually, let me construct it properly
    U_sb = [[0]*4 for _ in range(4)]
    V_sb = [[0]*4 for _ in range(2)]
    W_sb = [[0]*4 for _ in range(2)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            U_sb[i*2+j][col] = 1
            V_sb[j][col] = 1  # j*p+k where p=1, so just j
            W_sb[i][col] = 1  # k*n+i where n=2, k=0, so just i
            col += 1
    
    blocks.append((U_sb, V_sb, W_sb, 0, 0, 8, 2, 2, 1))
    
    result = combine_decompositions(blocks)
    
    # Verify the combined decomposition
    if verify(result["U"], result["V"], result["W"]):
        # Check time
        if check_time() < deadline - 5:
            return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
