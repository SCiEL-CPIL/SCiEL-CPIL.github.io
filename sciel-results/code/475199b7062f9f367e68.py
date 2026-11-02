"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_exact(U, V, W):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Convert to Fractions for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if s != 1:
                    return False
    return True

def _is_simple_number(x):
    """Check if a Fraction is an integer or half-integer."""
    if x.denominator == 1:
        return True
    if x.denominator == 2:
        return True
    return False

def _search_structured(max_rank, deadline):
    """
    Search for a decomposition by fixing U and V, and solving for W via linear systems.
    For each (i,k), we have m equations in r unknowns (entries of W[k*n+i, :]).
    We require the solution to be consistent and composed of simple numbers.
    """
    n, m, p = DIMENSIONS
    best_result = None
    
    # We search for ranks lower than schoolbook
    # Theoretical lower bound is not 1, but let's try from max_rank down to a reasonable limit
    # Start from max_rank down to 30 (arbitrary, can adjust)
    
    for rank in range(max_rank, 29, -1):
        if time.time() > deadline:
            break
            
        attempts = 0
        max_attempts = 20000 # Adjust based on time
        
        while attempts < max_attempts:
            attempts += 1
            if time.time() > deadline:
                break
            
            # Generate random sparse U and V
            # Sparsity is crucial. If U and V are dense, the systems are likely over-determined or have complex solutions.
            # Let's try sparsity 0.2 - 0.4
            sparsity = 0.3
            val_range = 2 # -2, -1, 0, 1, 2
            
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            
            for i in range(n * m):
                for c in range(rank):
                    if random.random() < sparsity:
                        U[i][c] = random.randint(-val_range, val_range)
                        
            for j in range(m * p):
                for c in range(rank):
                    if random.random() < sparsity:
                        V[j][c] = random.randint(-val_range, val_range)
                        
            # Now solve for W
            W = [[0] * rank for _ in range(p * n)]
            valid = True
            
            # For each i in [0, n), k in [0, p)
            for i in range(n):
                for k in range(p):
                    # Construct the system M * w = 1
                    # M is m x rank
                    # M[j, c] = U[i*m + j, c] * V[j*p + k, c]
                    M = np.zeros((m, rank), dtype=float)
                    for j in range(m):
                        for c in range(rank):
                            M[j, c] = U[i * m + j][c] * V[j * p + k][c]
                            
                    # Solve M w = 1
                    # Use least squares or exact rational solve
                    # Since we need exact simple numbers, let's use Fraction-based Gaussian elimination
                    # or check if the system has a solution with small integers.
                    
                    # First, check rank of M. If rank < m, we might have infinite solutions or no solution.
                    # We want a specific solution.
                    
                    # Let's try to find a solution using sympy or manual fraction elimination.
                    # Given small size (m=3, rank<=42), manual is fine.
                    
                    # Copy M to a list of lists of Fractions
                    M_frac = [[Fraction(int(M[j][c])) for c in range(rank)] for j in range(m)]
                    b_frac = [Fraction(1)] * m
                    
                    # Gaussian Elimination
                    # We want to solve for w.
                    # If the system is underdetermined (rank < m), we have free variables.
                    # We can set free variables to 0 and solve for the rest.
                    # If the solution contains fractions with large denominators, reject.
                    
                    # Perform elimination
                    A_aug = [row + [b_frac[i]] for i, row in enumerate(M_frac)]
                    num_rows = m
                    num_cols = rank
                    pivot_row = 0
                    pivots = []
                    
                    for col in range(num_cols):
                        if pivot_row >= num_rows:
                            break
                        # Find pivot
                        pivot_idx = -1
                        for r in range(pivot_row, num_rows):
                            if A_aug[r][col] != 0:
                                pivot_idx = r
                                break
                        if pivot_idx == -1:
                            continue
                            
                        # Swap rows
                        A_aug[pivot_row], A_aug[pivot_idx] = A_aug[pivot_idx], A_aug[pivot_row]
                        pivots.append(col)
                        
                        # Normalize pivot row (optional, but helps with fractions)
                        pivot_val = A_aug[pivot_row][col]
                        if pivot_val != 1:
                            A_aug[pivot_row] = [x / pivot_val for x in A_aug[pivot_row]]
                            
                        # Eliminate other rows
                        for r in range(num_rows):
                            if r != pivot_row and A_aug[r][col] != 0:
                                factor = A_aug[r][col]
                                A_aug[r] = [A_aug[r][c] - factor * A_aug[pivot_row][c] for c in range(num_cols + 1)]
                                
                        pivot_row += 1
                        
                    # Check for inconsistency (0 = 1)
                    inconsistent = False
                    for r in range(pivot_row, num_rows):
                        # Check if all coefficients are 0 but RHS is not
                        if all(A_aug[r][c] == 0 for c in range(num_cols)) and A_aug[r][num_cols] != 0:
                            inconsistent = True
                            break
                            
                    if inconsistent:
                        valid = False
                        break
                        
                    # Now we have a solution.
                    # Free variables are columns not in pivots. Set them to 0.
                    w_sol = [Fraction(0)] * rank
                    for c in pivots:
                        # The variable corresponding to pivot column c is in the row where it was the pivot.
                        # Actually, after elimination, the matrix is in RREF.
                        # The value of the variable x_c is the RHS of the row where the pivot is in column c.
                        # We need to map pivot columns to rows.
                        pass
                    
                    # Re-do RREF logic to extract solution properly
                    A_rref = [row[:] for row in A_aug] # Deep copy
                    pivot_cols = []
                    r = 0
                    for c in range(num_cols):
                        if r >= num_rows: break
                        # Find non-zero in column c from row r downwards
                        pivot_idx = -1
                        for rr in range(r, num_rows):
                            if A_rref[rr][c] != 0:
                                pivot_idx = rr
                                break
                        if pivot_idx == -1:
                            continue
                        A_rref[r], A_rref[pivot_idx] = A_rref[pivot_idx], A_rref[r]
                        pivot_val = A_rref[r][c]
                        if pivot_val != 1:
                            A_rref[r] = [x / pivot_val for x in A_rref[r]]
                        for rr in range(num_rows):
                            if rr != r and A_rref[rr][c] != 0:
                                factor = A_rref[rr][c]
                                A_rref[rr] = [A_rref[rr][cc] - factor * A_rref[r][cc] for cc in range(num_cols + 1)]
                        pivot_cols.append(c)
                        r += 1
                        
                    # Extract solution
                    w_sol = [Fraction(0)] * rank
                    for idx, pc in enumerate(pivot_cols):
                        w_sol[pc] = A_rref[idx][num_cols]
                        
                    # Check if solution is simple
                    simple = True
                    for val in w_sol:
                        if not _is_simple_number(val):
                            simple = False
                            break
                            
                    if not simple:
                        valid = False
                        break
                    else:
                        # Store solution in W
                        w_row_idx = k * n + i
                        W[w_row_idx] = [val.numerator / val.denominator if val.denominator != 1 else int(val) for val in w_sol]
                        # Wait, the interface requires integer/half-integer entries.
                        # If val is 1/2, we should store 0.5 or 1/2?
                        # "integer/half-integer entries" usually means numbers like 1, 0.5, -0.5, 2, etc.
                        # Python lists can hold floats. 0.5 is fine.
                        # But _verify_exact uses Fraction. 0.5 becomes Fraction(1,2).
                        # Let's store as float for simplicity, but ensure it's exact.
                        # Actually, storing as Fraction is safer for verification, but the output format
                        # usually expects numbers. Let's convert to float if it's .5 or int.
                        W[w_row_idx] = [float(val) for val in w_sol]
                        
            if valid:
                # Final verification with exact arithmetic to be sure
                if _verify_exact(U, V, W):
                    return {"U": U, "V": V, "W": W}
                    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Get schoolbook as fallback
    schoolbook = _schoolbook()
    best_rank = len(schoolbook["U"][0])  # 42
    
    # Try structured search for lower ranks
    # We start from rank 41 down to 30
    result = _search_structured(max_rank=best_rank - 1, deadline=deadline)
    
    if result is not None:
        return result
        
    # If no better found, return schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
