import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s safety margin

    # Schoolbook decomposition (Rank 42)
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

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def is_halfint(x):
        """Check if x is an integer or half-integer (multiple of 0.5)"""
        if isinstance(x, Fraction):
            return x.denominator in (1, 2)
        return x * 2 == int(x * 2)

    def verify_decomposition(U, V, W, R):
        """Verify the tensor decomposition exactly using Fractions."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    def solve_for_W(U, V, R):
        """
        Given U (nm x R) and V (mp x R) as lists of lists of Fractions,
        find W (pn x R) such that the tensor equation holds.
        Returns W as list of lists of Fractions, or None if no half-integer solution found.
        """
        W = [[0] * R for _ in range(p * n)]
        
        for k in range(p):
            for i in range(n):
                # Build the system: for all j in [0, m):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * w[r] = 1
                # A is m x R, b is m
                A = []
                b = []
                for j in range(m):
                    row = []
                    for r in range(R):
                        row.append(U[i*m+j][r] * V[j*p+k][r])
                    A.append(row)
                    b.append(Fraction(1))
                
                # Gaussian elimination with partial pivoting (using Fractions)
                aug = []
                for j in range(m):
                    row = A[j] + [b[j]]
                    aug.append(row)
                
                num_rows = len(aug)
                num_cols = R + 1
                rank_A = 0
                pivot_cols = []
                
                for col in range(R):
                    # Find pivot
                    pivot_row = None
                    for row in range(rank_A, num_rows):
                        if aug[row][col] != 0:
                            pivot_row = row
                            break
                    
                    if pivot_row is None:
                        continue
                    
                    # Swap rows
                    if pivot_row != rank_A:
                        aug[rank_A], aug[pivot_row] = aug[pivot_row], aug[rank_A]
                    
                    # Eliminate
                    pivot_val = aug[rank_A][col]
                    for row in range(num_rows):
                        if row != rank_A and aug[row][col] != 0:
                            factor = aug[row][col] / pivot_val
                            for c in range(col, num_cols):
                                aug[row][c] -= factor * aug[rank_A][c]
                    
                    pivot_cols.append(col)
                    rank_A += 1
                
                # Check consistency
                for row in range(num_rows):
                    all_zero = all(aug[row][c] == 0 for c in range(R))
                    if all_zero and aug[row][R] != 0:
                        return None  # Inconsistent
                
                # Find particular solution and nullspace
                pivot_set = set(pivot_cols)
                free_cols = [c for c in range(R) if c not in pivot_set]
                
                # Particular solution: set free variables to 0
                w_particular = [Fraction(0)] * R
                for row_idx in range(rank_A):
                    col = pivot_cols[row_idx]
                    w_particular[col] = aug[row_idx][R]
                
                # If no free variables, check if particular solution is half-integer
                if not free_cols:
                    if all(is_halfint(x) for x in w_particular):
                        W[k*n+i] = w_particular
                    else:
                        return None
                    continue
                
                # Check if particular solution is already half-integer
                if all(is_halfint(x) for x in w_particular):
                    W[k*n+i] = w_particular
                    continue
                
                # For small number of free variables, enumerate
                num_free = len(free_cols)
                if num_free <= 4:
                    candidate_vals = [Fraction(0), Fraction(1,2), Fraction(1), Fraction(-1,2), Fraction(-1)]
                    for combo in product(candidate_vals, repeat=num_free):
                        w_test = [Fraction(0)] * R
                        for r in range(R):
                            w_test[r] = w_particular[r]
                            for idx in range(num_free):
                                w_test[r] += combo[idx] * 0  # Simplified: just check particular
                        # Actually need to use nullspace basis
                        # Let's reconstruct nullspace
                        nullspace_basis = []
                        for f in free_cols:
                            v = [Fraction(0)] * R
                            v[f] = Fraction(1)
                            for row_idx in range(rank_A):
                                col = pivot_cols[row_idx]
                                v[col] = -aug[row_idx][f]
                            nullspace_basis.append(v)
                        
                        w_test = [Fraction(0)] * R
                        for r in range(R):
                            w_test[r] = w_particular[r]
                            for idx in range(num_free):
                                w_test[r] += combo[idx] * nullspace_basis[idx][r]
                        if all(is_halfint(x) for x in w_test):
                            W[k*n+i] = w_test
                            break
                    else:
                        return None
                    continue
                
                # For larger nullspace, try a few simple adjustments
                # Try setting each free variable to 0, 1/2, 1, -1/2, -1 independently
                # This is a heuristic
                nullspace_basis = []
                for f in free_cols:
                    v = [Fraction(0)] * R
                    v[f] = Fraction(1)
                    for row_idx in range(rank_A):
                        col = pivot_cols[row_idx]
                        v[col] = -aug[row_idx][f]
                    nullspace_basis.append(v)
                
                # Try small integer/half-integer values for free variables
                found = False
                for trial in range(50):
                    w_test = [Fraction(0)] * R
                    for r in range(R):
                        w_test[r] = w_particular[r]
                    for idx in range(num_free):
                        val = random.choice([Fraction(0), Fraction(1,2), Fraction(1), Fraction(-1,2), Fraction(-1)])
                        for r in range(R):
                            w_test[r] += val * nullspace_basis[idx][r]
                    if all(is_halfint(x) for x in w_test):
                        W[k*n+i] = w_test
                        found = True
                        break
                if not found:
                    return None
        
        return W

    # Try to find a lower rank decomposition
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Try ranks from 21 upward, prioritizing values where solutions might exist.
    
    entries_frac = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]
    
    # Prioritize ranks that are known to be achievable or close to lower bound
    ranks_to_try = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 40]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try several random initializations
        num_attempts = 150 if R <= 25 else 80
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Random U and V with small entries
            U = [[random.choice(entries_frac) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice(entries_frac) for _ in range(R)] for _ in range(m * p)]
            
            # Solve for W
            W = solve_for_W(U, V, R)
            
            if W is not None:
                # Verify the decomposition
                if verify_decomposition(U, V, W, R):
                    # Convert to the required format (int or half-integer)
                    U_out = [[float(x) for x in row] for row in U]
                    V_out = [[float(x) for x in row] for row in V]
                    W_out = [[float(x) for x in row] for row in W]
                    
                    best_result = {"U": U_out, "V": V_out, "W": W_out}
                    best_rank = R
                    break  # Found a valid decomposition for this rank, move to next rank
        
        if best_rank == R:
            # Found the minimum rank (since we're going from low to high)
            break
    
    return best_result
# EVOLVE-BLOCK-END
