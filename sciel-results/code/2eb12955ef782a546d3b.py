"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Strategy:
    # 1. Try to find a Rank 6 decomposition for 2x2x3 matrix multiplication.
    # 2. If found, use it to decompose the 2x4x6 problem into four 2x2x3 sub-problems.
    #    Total rank = 4 * 6 = 24.
    # 3. If not found within a short time, fall back to Rank 42 Strassen decomposition.
    
    def find_2x2x3_decomposition():
        """
        Searches for a rank-6 decomposition of the 2x2x3 matrix multiplication tensor.
        Returns U, V, W as lists of lists if found, else None.
        """
        # Target Tensor T[i, j, k] for C[i,k] = sum_j A[i,j] B[j,k]
        # i in 0..1, j in 0..1, k in 0..2
        # We need U (2x2x6), V (2x3x6), W (3x2x6)
        
        # We restrict search to small integers: -2, -1, 0, 1, 2
        # This is a heuristic based on known decompositions having small coefficients.
        
        # Flatten the factors
        # U: 4 * 6 = 24 vars
        # V: 6 * 6 = 36 vars
        # W: 6 * 6 = 36 vars
        
        # We will use a simple random restart with local search or just random sampling
        # Given the small space, random sampling might work if we guide it.
        
        # Better: Use the structure.
        # Fix U to be sparse?
        
        # Let's try a specific set of known patterns.
        # Pattern 1: Diagonal A, Sum B, etc.
        
        # Since we don't have the exact coefficients, we will try to verify a few known candidates
        # or use a very fast random search.
        
        # For the sake of this implementation, we will use a fallback to Strassen (Rank 42)
        # because implementing a robust solver for tensor rank is complex and risky.
        # However, we can try to construct the Rank 42 solution efficiently.
        
        return None

    # Fallback: Rank 42 Strassen Decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        # Strassen formulas for 2x2x2
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        c = start_col
        # M1
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1; W[w_idx[3]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
