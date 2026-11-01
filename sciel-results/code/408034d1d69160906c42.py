import random
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0  # Safety margin against 150s kill
    
    # Trivial solution (Rank 16)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial

    def verify_solution(U, V, W, R):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 8 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 8:
                        return False
        return True

    def solve_for_V(U_cand, W_cand, R):
        """
        For fixed U and W, solve for V.
        For each (j,k), solve M_{j,k} * V_{j,k} = 8*ones(n)
        where M_{j,k}[i, r] = U[i*m+j, r] * W[k*n+i, r]
        """
        V_cand = [[0] * R for _ in range(m * p)]
        
        for j in range(m):
            for k in range(p):
                # Construct M (n x R)
                M = []
                for i in range(n):
                    row = []
                    for r in range(R):
                        row.append(U_cand[i * m + j][r] * W_cand[k * n + i][r])
                    M.append(row)
                
                # Solve M * x = 8 * ones(n)
                # Use Fraction for exact arithmetic
                M_frac = [[Fraction(M[i][r]) for r in range(R)] for i in range(n)]
                b_frac = [Fraction(8)] * n
                
                # Gaussian Elimination with RREF
                A = [M_frac[i] + [b_frac[i]] for i in range(n)]
                
                rank = 0
                pivot_cols = []
                for col in range(R):
                    # Find pivot
                    pivot_row = -1
                    for row in range(rank, n):
                        if A[row][col] != 0:
                            pivot_row = row
                            break
                    if pivot_row == -1:
                        continue
                    
                    # Swap
                    A[rank], A[pivot_row] = A[pivot_row], A[rank]
                    
                    # Normalize pivot row
                    pivot_val = A[rank][col]
                    if pivot_val != 1:
                        A[rank] = [x / pivot_val for x in A[rank]]
                    
                    # Eliminate other rows
                    for row in range(n):
                        if row != rank and A[row][col] != 0:
                            factor = A[row][col]
                            A[row] = [A[row][c] - factor * A[rank][c] for c in range(R + 1)]
                    
                    rank += 1
                    pivot_cols.append(col)
                    if rank == n:
                        break
                
                # Check consistency
                for i in range(rank, n):
                    if A[i][R] != 0:
                        return None
                
                # Extract solution (set free variables to 0)
                x = [Fraction(0)] * R
                for i in range(rank):
                    p_col = pivot_cols[i]
                    x[p_col] = A[i][R]
                
                # Check if x is integer
                for val in x:
                    if val.denominator != 1:
                        return None
                
                V_cand[j * p + k] = [int(val) for val in x]
        
        return V_cand

    def generate_and_solve(R, max_iters=50000):
        """Generate random U, W and try to solve for V"""
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 10:
                return None
            
            # Generate U: (4, R) with entries in {-1, 0, 1}
            U_cand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)]
            # Generate W: (8, R) with entries in {-1, 0, 1}
            W_cand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(p * n)]
            
            # Try to solve for V
            V_cand = solve_for_V(U_cand, W_cand, R)
            if V_cand is None:
                continue
            
            # Verify the solution
            if verify_solution(U_cand, V_cand, W_cand, R):
                # Convert to half-integers
                U_res = [[x / 2.0 for x in row] for row in U_cand]
                V_res = [[x / 2.0 for x in row] for row in V_cand]
                W_res = [[x / 2.0 for x in row] for row in W_cand]
                return (U_res, V_res, W_res)
        
        return None

    # Try ranks from 14 down to 10
    for target_rank in range(14, 9, -1):
        if time.time() - start_time > time_limit - 15:
            break
        
        result = generate_and_solve(target_rank)
        if result is not None:
            U_res, V_res, W_res = result
            best_rank = target_rank
            best_U = U_res
            best_V = V_res
            best_W = W_res
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
