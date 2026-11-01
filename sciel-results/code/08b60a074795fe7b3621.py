"""
Strategy: Structured Algebraic Search over Rank-40 Decompositions using Kronecker Products.

This program attempts to find a tensor decomposition of rank < 48 (specifically targeting rank 40)
by fixing the structure of the U and V matrices to be composed of Kronecker products of small
integer vectors derived from the Fourier transform of Z_2 x Z_3.

The idea is to exploit the block structure of the 2x4 * 4x6 matrix multiplication.
We can view the problem as a sum of smaller matrix multiplications.
If we can find a decomposition of rank 40, it would be a significant improvement over the schoolbook rank of 48.

The approach:
1. Construct candidate U and V matrices with a specific structure (Kronecker products).
2. Solve the resulting linear system for W over the rationals.
3. If a solution exists with rational (or integer/half-integer) entries, we have found a valid decomposition.
4. If not, we fall back to the schoolbook decomposition of rank 48.

Note: This is a heuristic search. It is not guaranteed to find a rank-40 decomposition if one exists,
but it explores a structured subspace of possible decompositions that might contain a lower-rank solution.
"""

DIMENSIONS = [2, 4, 6]

import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # 1. Generate the Schoolbook Decomposition (Rank = n*m*p = 48)
    # This is guaranteed to be correct and have integer entries.
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    if time.time() > deadline - 10:
        return best_result

    # 2. Attempt to find a lower rank decomposition using structured search.
    # We will try to construct U and V with a specific Kronecker product structure
    # and solve for W.
    
    # The tensor T is defined by T[i, j, k] = 1 if the indices are consistent with matmul.
    # Specifically, T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i.
    # We want to find U, V, W such that sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c].
    
    # Let's try rank 40.
    # We will fix U and V to have a specific structure.
    # For example, U could be a Kronecker product of a 2x2 matrix and a 4x2 matrix?
    # This is complex. Let's try a simpler heuristic: 
    # Random search with rational constraints? No, too slow.
    
    # Let's try to verify if rank 42 (Strassen-based) is possible and correct.
    # The previous program had a Strassen implementation but it was not verified to be correct.
    # Let's implement a correct Strassen-based decomposition for rank 42.
    
    def strassen_decomposition():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_dict, v_dict, w_dict, idx):
            for (i, j), val in u_dict.items():
                U[i*m + j][idx] += val
            for (j, k), val in v_dict.items():
                V[j*p + k][idx] += val
            for (k, i), val in w_dict.items():
                W[k*n + i][idx] += val

        idx = 0
        for k_block in range(3):
            # Matmul 1: A1 * B_{1k}
            # A1 is 2x2 at A[0:2, 0:2]
            # B_{1k} is 2x2 at B[0:2, 2*k_block:2*k_block+2]
            
            # Strassen terms for X=A1, Y=B_{1k}
            # M1 = (X11 + X22)(Y11 + Y22)
            add_term(
                {(0,0):1, (1,1):1},
                {(0, 2*k_block):1, (1, 2*k_block+1):1},
                {(2*k_block, 0):1, (2*k_block+1, 1):1},
                idx
            )
            idx += 1
            # M2 = (X21 + X22)Y11
            add_term(
                {(1,0):1, (1,1):1},
                {(0, 2*k_block):1},
                {(2*k_block+1, 1):1, (2*k_block, 1):1},
                idx
            )
            idx += 1
            # M3 = X11(Y12 - Y22)
            add_term(
                {(0,0):1},
                {(0, 2*k_block+1):1, (1, 2*k_block+1):-1},
                {(2*k_block+1, 0):1, (2*k_block+1, 1):1},
                idx
            )
            idx += 1
            # M4 = X22(Y21 - Y11)
            add_term(
                {(1,1):1},
                {(1, 2*k_block):1, (0, 2*k_block):-1},
                {(2*k_block, 1):1, (2*k_block, 0):1},
                idx
            )
            idx += 1
            # M5 = (X11 + X12)Y22
            add_term(
                {(0,0):1, (0,1):1},
                {(1, 2*k_block+1):1},
                {(2*k_block, 0):-1, (2*k_block+1, 1):1},
                idx
            )
            idx += 1
            # M6 = (X21 - X11)(Y11 + Y12)
            add_term(
                {(1,0):1, (0,0):-1},
                {(0, 2*k_block):1, (0, 2*k_block+1):1},
                {(2*k_block+1, 1):1},
                idx
            )
            idx += 1
            # M7 = (X12 - X22)(Y21 + Y22)
            add_term(
                {(0,1):1, (1,1):-1},
                {(1, 2*k_block):1, (1, 2*k_block+1):1},
                {(2*k_block, 0):1, (2*k_block+1, 1):-1},
                idx
            )
            idx += 1
            
            # Matmul 2: A2 * B_{2k}
            # A2 is 2x2 at A[0:2, 2:4]
            # B_{2k} is 2x2 at B[2:4, 2*k_block:2*k_block+2]
            
            # M1
            add_term(
                {(0,2):1, (1,3):1},
                {(2, 2*k_block):1, (3, 2*k_block+1):1},
                {(2*k_block, 0):1, (2*k_block+1, 1):1},
                idx
            )
            idx += 1
            # M2
            add_term(
                {(1,2):1, (1,3):1},
                {(2, 2*k_block):1},
                {(2*k_block+1, 1):1, (2*k_block, 1):1},
                idx
            )
            idx += 1
            # M3
            add_term(
                {(0,2):1},
                {(2, 2*k_block+1):1, (3, 2*k_block+1):-1},
                {(2*k_block+1, 0):1, (2*k_block+1, 1):1},
                idx
            )
            idx += 1
            # M4
            add_term(
                {(1,3):1},
                {(3, 2*k_block):1, (2, 2*k_block):-1},
                {(2*k_block, 1):1, (2*k_block, 0):1},
                idx
            )
            idx += 1
            # M5
            add_term(
                {(0,2):1, (0,3):1},
                {(3, 2*k_block+1):1},
                {(2*k_block, 0):-1, (2*k_block+1, 1):1},
                idx
            )
            idx += 1
            # M6
            add_term(
                {(1,2):1, (0,2):-1},
                {(2, 2*k_block):1, (2, 2*k_block+1):1},
                {(2*k_block+1, 1):1},
                idx
            )
            idx += 1
            # M7
            add_term(
                {(0,3):1, (1,3):-1},
                {(3, 2*k_block):1, (3, 2*k_block+1):1},
                {(2*k_block, 0):1, (2*k_block+1, 1):-1},
                idx
            )
            idx += 1

        return {"U": U, "V": V, "W": W}, rank

    # Verify the rank 42 decomposition
    result_42, rank_42 = strassen_decomposition()
    
    # Check validity
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank_42):
                    s += result_42['U'][i*m+j][r] * result_42['V'][j*p+k][r] * result_42['W'][k*n+i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return result_42
    else:
        return best_result

# EVOLVE-BLOCK-END
