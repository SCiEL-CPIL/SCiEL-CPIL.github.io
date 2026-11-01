import time
import random
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target T[i, j, k] = 1 for all i, j, k.
    """
    # Use Fraction for exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _als_step(X, Y, Z, n, m, p, rank, target_val=1):
    """
    Perform one ALS step. X, Y, Z are lists of lists of Fractions.
    Returns updated X, Y, Z.
    X: (n*m, rank), Y: (m*p, rank), Z: (p*n, rank)
    """
    # Precompute matrices for faster fraction arithmetic
    # We want to minimize sum_{i,j,k} (X_ij . (Y_jk * Z_ki)) - 1)^2 ? 
    # Actually, ALS for tensor decomposition usually solves linear systems.
    # Given Y, Z, solve for X:
    # For each i, j:
    # sum_k target(i,j,k) * Z[k,i] = sum_c X[i,j,c] * (sum_k Y[j,k,c] * Z[k,i,c])
    # Let M[i,j,c] = sum_k Y[j,k,c] * Z[k,i,c]
    # Let b[i,j] = sum_k target(i,j,k) * Z[k,i,c] ... wait, target is constant 1.
    # The equation is: sum_c X[i,j,c] * M[i,j,c] = sum_k Z[k,i,c] (since T=1)
    # Wait, the term is X[i,j,c] * Y[j,k,c] * Z[k,i,c].
    # Sum over c: sum_c X[i,j,c] * (sum_k Y[j,k,c] * Z[k,i,c]) = sum_k 1 * Z[k,i,c]
    
    new_X = [[Fraction(0) for _ in range(rank)] for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            # Compute M[i,j,c]
            M = [Fraction(0) for _ in range(rank)]
            b = Fraction(0)
            for k in range(p):
                z_val = Z[k*n+i]
                b += z_val # Sum over k of Z[k,i] (since T=1)
                for c in range(rank):
                    M[c] += Y[j*p+k][c] * z_val[c]
            
            # Solve M . X_i,j = b
            # If M is zero vector, X is indeterminate (set to 0)
            # If M has one non-zero, X = b/M
            # If M has multiple non-zeros, we need a least squares solution or a specific constraint.
            # For simplicity and to keep integers/half-integers, we use a heuristic:
            # If rank is small, we can try to find a simple solution.
            # However, standard ALS uses SVD or pseudo-inverse.
            # With Fractions, we can solve if the system is square and invertible, 
            # but here we have 1 equation for 'rank' unknowns. It's underdetermined.
            # We need to minimize norm(X) subject to M.X = b.
            # The minimum norm solution is X = M^T (M M^T)^{-1} b.
            
            # Compute MMt (scalar since M is 1D vector of length rank? No, M is a vector of length rank.
            # The equation is M . X = b.
            # Min norm solution: X_c = M_c * b / sum(M_d^2)
            
            sum_sq = sum(mc * mc for mc in M)
            if sum_sq != 0:
                factor = b / sum_sq
                for c in range(rank):
                    new_X[i*m+j][c] = M[c] * factor
            # else leave as 0
            
    new_Y = [[Fraction(0) for _ in range(rank)] for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            M = [Fraction(0) for _ in range(rank)]
            b = Fraction(0)
            for i in range(n):
                x_val = X[i*m+j]
                b += x_val # Sum over i of X[i,j] (since T=1)
                for c in range(rank):
                    M[c] += x_val[c] * Z[k*n+i][c]
            
            sum_sq = sum(mc * mc for mc in M)
            if sum_sq != 0:
                factor = b / sum_sq
                for c in range(rank):
                    new_Y[j*p+k][c] = M[c] * factor

    new_Z = [[Fraction(0) for _ in range(rank)] for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            M = [Fraction(0) for _ in range(rank)]
            b = Fraction(0)
            for j in range(m):
                y_val = Y[j*p+k]
                b += y_val # Sum over j of Y[j,k] (since T=1)
                for c in range(rank):
                    M[c] += y_val[c] * X[i*m+j][c]
            
            sum_sq = sum(mc * mc for mc in M)
            if sum_sq != 0:
                factor = b / sum_sq
                for c in range(rank):
                    new_Z[k*n+i][c] = M[c] * factor

    return new_X, new_Y, new_Z

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    Strategy: 
    1. Start with the known rank-10 solution (Strassen + Identity).
    2. Attempt to reduce to rank 9 using randomized ALS over Fractions.
    3. If rank 9 is found and verified, return it.
    4. Otherwise, return the rank-10 solution.
    """
    n, m, p = 2, 2, 5
    start_time = time.time()
    deadline = 140.0  # Safety margin

    # --- Build Rank 10 Solution (Fallback) ---
    U10 = [[0]*10 for _ in range(4)]
    V10 = [[0]*10 for _ in range(10)]
    W10 = [[0]*10 for _ in range(10)]
    
    # Strassen part (r=0..6) for k=0,1
    # P1 = (a+d)(e+h)
    U10[0][0] = 1; U10[3][0] = 1
    V10[0][0] = 1; V10[6][0] = 1
    W10[0][0] = 1; W10[3][0] = 1
    
    # P2 = (c+d)e
    U10[2][1] = 1; U10[3][1] = 1
    V10[0][1] = 1
    W10[2][1] = 1; W10[3][1] = -1
    
    # P3 = a(f-h)
    U10[0][2] = 1
    V10[1][2] = 1; V10[7][2] = -1
    W10[1][2] = 1; W10[3][2] = 1
    
    # P4 = d(g-e)
    U10[3][3] = 1
    V10[0][3] = -1; V10[5][3] = 1
    W10[0][3] = 1; W10[2][3] = 1
    
    # P5 = (a+b)h
    U10[0][4] = 1; U10[1][4] = 1
    V10[6][4] = 1
    W10[0][4] = -1; W10[1][4] = 1
    
    # P6 = (c-a)(e+f)
    U10[2][5] = 1; U10[0][5] = -1
    V10[0][5] = 1; V10[1][5] = 1
    W10[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U10[1][6] = 1; U10[3][6] = -1
    V10[5][6] = 1; V10[6][6] = 1
    W10[0][6] = 1
    
    # Identity part for k=2,3,4 (r=7,8,9)
    # For k=2 (r=7):
    for i in range(2):
        for j in range(2):
            U10[i*2+j][7] = 1
    for j in range(2):
        V10[j*5+2][7] = 1
    for i in range(2):
        W10[2*2+i][7] = 1
        
    # For k=3 (r=8):
    for i in range(2):
        for j in range(2):
            U10[i*2+j][8] = 1
    for j in range(2):
        V10[j*5+3][8] = 1
    for i in range(2):
        W10[3*2+i][8] = 1
        
    # For k=4 (r=9):
    for i in range(2):
        for j in range(2):
            U10[i*2+j][9] = 1
    for j in range(2):
        V10[j*5+4][9] = 1
    for i in range(2):
        W10[4*2+i][9] = 1

    # Verify rank 10
    if _verify(U10, V10, W10, n, m, p, 10):
        best_U, best_V, best_W = U10, V10, W10
        best_rank = 10
    else:
        # Fallback to schoolbook rank 20 if rank 10 fails (should not happen)
        best_rank = 20
        best_U = [[0]*20 for _ in range(4)]
        best_V = [[0]*20 for _ in range(10)]
        best_W = [[0]*20 for _ in range(10)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j][col] = 1
                    best_V[j*p+k][col] = 1
                    best_W[k*n+i][col] = 1
                    col += 1
        return {"U": best_U, "V": best_V, "W": best_W}

    # --- Attempt Rank 9 Search ---
    # We try to find a rank 9 solution.
    # Initialize with a random perturbation of a rank 9 subset of the rank 10 solution?
    # Or just random integer initialization.
    # Given the difficulty, we will try a few random restarts.
    
    found_rank9 = False
    best_U9, best_V9, best_W9 = None, None, None
    
    # Limit number of restarts to keep within time
    max_restarts = 20
    max_iters = 100
    
    for r in range(max_restarts):
        if time.time() > deadline - 20: # Leave 20s for verification and return
            break
            
        # Initialize U, V, W with small random integers
        # Scale: entries in {-2, -1, 0, 1, 2}
        U = [[random.randint(-2, 2) for _ in range(9)] for _ in range(4)]
        V = [[random.randint(-2, 2) for _ in range(9)] for _ in range(10)]
        W = [[random.randint(-2, 2) for _ in range(9)] for _ in range(10)]
        
        # Ensure some non-zero structure to avoid collapse to 0
        # Force first few entries to be 1
        U[0][0] = 1
        V[0][0] = 1
        W[0][0] = 1
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
                
            U, V, W = _als_step(U, V, W, n, m, p, 9)
            
            # Check for convergence/validity periodically
            if it % 10 == 0:
                if _verify(U, V, W, n, m, p, 9):
                    found_rank9 = True
                    best_U9, best_V9, best_W9 = U, V, W
                    break
        
        if found_rank9:
            break

    if found_rank9:
        return {"U": best_U9, "V": best_V9, "W": best_W9}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
