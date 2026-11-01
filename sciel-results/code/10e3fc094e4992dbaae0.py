import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank = n*m*p = 24)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school

    start_time = time.time()
    deadline = 140.0  # 10s margin

    # Target tensor: T_ijk = 1 for all i, j, k
    # Equation: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    
    # We search for rank r < 24
    # We fix U and V randomly, then solve for W exactly using rationals.
    # Variables: W[k*n+i][c] for k in [0,p), i in [0,n), c in [0,r)
    # Total variables: p * n * r
    
    def solve_for_W(U, V, r):
        # Build linear system A * w = b
        # w is a vector of size p*n*r
        # Index mapping for w: idx = (k * n + i) * r + c
        num_vars = p * n * r
        num_eqs = n * m * p
        
        # A is num_eqs x num_vars
        # We store rows as lists of (col, val) for sparsity, but here we can use dense Fraction if small
        # 24 x (24*23) is ~ 24 x 552. Dense is fine.
        A = [[Fraction(0) for _ in range(num_vars)] for _ in range(num_eqs)]
        b = [Fraction(1) for _ in range(num_eqs)]
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Equation eq_idx: sum_c U[i*m+j][c] V[j*p+k][c] W[k*n+i][c] = 1
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_base = (k * n + i) * r
                    for c in range(r):
                        coeff = U[u_idx][c] * V[v_idx][c]
                        if coeff != 0:
                            A[eq_idx][w_base + c] = Fraction(coeff)
                    eq_idx += 1
        
        # Solve A * w = b using Gaussian Elimination with Fractions
        # w = [0] * num_vars
        # We augment A with b
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        # Gaussian Elimination
        rank = 0
        for col in range(num_vars):
            # Find pivot
            pivot_row = -1
            for row in range(rank, num_eqs):
                if M[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row == -1:
                continue
            
            # Swap
            M[rank], M[pivot_row] = M[pivot_row], M[rank]
            
            # Normalize pivot row (optional, but good for fractions to keep small? No, just eliminate)
            pivot_val = M[rank][col]
            
            # Eliminate below
            for row in range(rank + 1, num_eqs):
                factor = M[row][col] / pivot_val
                if factor != 0:
                    for c in range(col, num_vars + 1):
                        M[row][c] -= factor * M[rank][c]
            
            rank += 1
            if rank == num_vars:
                break
        
        # Check for inconsistency (0 = non-zero)
        for row in range(rank, num_eqs):
            # If A part is 0, b part must be 0
            if M[row][num_vars] != 0:
                return None
        
        # Back substitution
        w = [Fraction(0) for _ in range(num_vars)]
        for row in range(rank - 1, -1, -1):
            # Find the leading column in this row
            lead_col = -1
            for c in range(num_vars):
                if M[row][c] != 0:
                    lead_col = c
                    break
            if lead_col == -1:
                continue
            
            # Sum of known terms
            s = M[row][num_vars]
            for c in range(lead_col + 1, num_vars):
                s -= M[row][c] * w[c]
            
            w[lead_col] = s / M[row][lead_col]
        
        # Reconstruct W matrix
        W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                base = (k * n + i) * r
                for c in range(r):
                    W[k * n + i][c] = w[base + c]
        
        return W

    def is_valid(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def to_int_or_half(W, r):
        # Check if all entries are integers or half-integers
        # And convert to int if possible? The problem says "integer/half-integer entries".
        # Usually this means values like 1, 0.5, -1.5.
        # We can return them as floats if they are .0 or .5, or keep as Fractions?
        # The interface says "factor matrices ... with integer/half-integer entries".
        # It's safer to return floats if they are exact, or ints.
        # Let's return a list of floats where value * 2 is an integer.
        out = []
        for row in W:
            new_row = []
            for val in row:
                # val is Fraction
                if (val * 2).denominator == 1:
                    # It's an integer or half-integer
                    new_row.append(float(val))
                else:
                    return None
            out.append(new_row)
        return out

    # Search for lower rank
    for r in range(15, 24):
        if time.time() > deadline:
            break
            
        max_trials = 50
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Random sparse U, V
            # Bias towards 0 to keep numbers small
            U = [[random.choice([0, 0, 0, 1, -1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([0, 0, 0, 1, -1]) for _ in range(r)] for _ in range(m * p)]
            
            # Ensure U and V are not all zeros for any row (to avoid trivial 0=1 failure)
            # Actually, if a row of U is 0, then for that i,j, the sum is 0 for all k, so 0=1 fails.
            # So we must ensure every row of U is non-zero?
            # If U[i*m+j] is all 0, then LHS is 0 for all k. RHS is 1. Failure.
            # So we need at least one non-zero in each row of U and each row of V?
            # Not necessarily each row, but the combination must allow solution.
            
            W = solve_for_W(U, V, r)
            if W is not None:
                # Check validity (should be true by construction if consistent)
                if is_valid(U, V, W, r):
                    # Check if entries are half-integers
                    W_float = to_int_or_half(W, r)
                    if W_float is not None:
                        # We found a valid solution!
                        # Convert U, V to float as well (they are already ints)
                        U_float = [[float(x) for x in row] for row in U]
                        V_float = [[float(x) for x in row] for row in V]
                        
                        return {"U": U_float, "V": V_float, "W": W_float}
                        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
