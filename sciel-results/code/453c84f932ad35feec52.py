"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
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
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U ⊗ V ⊗ W equals the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                    if total != 1:
                        return False
        return True
    
    def try_rank_r(n, m, p, target_rank, num_trials=100):
        """
        Try to find a rank-target_rank decomposition.
        
        For fixed U (nm x r) and V (mp x r), the equations for W decouple by c:
        For each c in range(pn), the equations involving W[c][*] are linear.
        """
        random.seed(42)
        
        for trial in range(num_trials):
            # Randomly initialize U and V with small integers
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
            
            U_frac = [[Fraction(U[i][t]) for t in range(target_rank)] for i in range(nm)]
            V_frac = [[Fraction(V[i][t]) for t in range(target_rank)] for i in range(mp)]
            
            # For each c, collect the linear equations for W[c][0..target_rank-1]
            # Equation (i,j,k) with c=k*n+i:
            # sum_t U_frac[i*m+j][t] * V_frac[j*p+k][t] * W[c][t] = 1
            
            W = [[Fraction(0) for _ in range(target_rank)] for _ in range(pn)]
            
            success = True
            for c in range(pn):
                # Collect all (i,j,k) such that k*n+i = c
                eqs = []
                for i in range(n):
                    for k in range(p):
                        if k * n + i == c:
                            for j in range(m):
                                a = i * m + j
                                b = j * p + k
                                # Equation: sum_t U_frac[a][t] * V_frac[b][t] * W[c][t] = 1
                                coeffs = [U_frac[a][t] * V_frac[b][t] for t in range(target_rank)]
                                eqs.append(coeffs)
                
                # Now solve the linear system: eqs * W[c] = 1
                # eqs is a list of (target_rank,) vectors, we need W[c] such that
                # for each eq in eqs: dot(eq, W[c]) = 1
                
                # This is an underdetermined system (3 equations, 41 unknowns for r=41)
                # Find one solution.
                
                # If no equations, W[c] can be anything (set to 0)
                if not eqs:
                    continue
                
                # Use Gaussian elimination to find a particular solution
                # Number of equations: len(eqs), unknowns: target_rank
                num_eqs = len(eqs)
                
                # Build augmented matrix
                A = [eq[:] for eq in eqs]
                b = [Fraction(1) for _ in range(num_eqs)]
                
                # Gaussian elimination
                pivot_rows = []
                row = 0
                for col in range(target_rank):
                    if row >= num_eqs:
                        break
                    # Find pivot
                    pivot = -1
                    for r in range(row, num_eqs):
                        if A[r][col] != 0:
                            pivot = r
                            break
                    if pivot == -1:
                        continue
                    # Swap
                    A[row], A[pivot] = A[pivot], A[row]
                    b[row], b[pivot] = b[pivot], b[row]
                    # Normalize
                    pv = A[row][col]
                    for cc in range(col, target_rank):
                        A[row][cc] = A[row][cc] / pv
                    b[row] = b[row] / pv
                    # Eliminate
                    for r in range(num_eqs):
                        if r != row and A[r][col] != 0:
                            factor = A[r][col]
                            for cc in range(col, target_rank):
                                A[r][cc] -= factor * A[row][cc]
                            b[r] -= factor * b[row]
                    pivot_rows.append((row, col))
                    row += 1
                
                # Extract solution: set free vars to 0, back-substitute
                w_c = [Fraction(0) for _ in range(target_rank)]
                for (r, col) in pivot_rows:
                    w_c[col] = b[r]
                
                W[c] = w_c
            
            # Verify
            result = {
                "U": U,
                "V": V,
                "W": [[int(W[c][t]) if W[c][t].denominator == 1 else W[c][t] for t in range(target_rank)] for c in range(pn)]
            }
            
            if verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
        
        return None
    
    # Try rank 41
    result = try_rank_r(n, m, p, 41, num_trials=50)
    if result is not None:
        return result
    
    # Try rank 40
    result = try_rank_r(n, m, p, 40, num_trials=50)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
