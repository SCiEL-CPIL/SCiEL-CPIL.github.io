"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook decomposition
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
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    # Verify initial decomposition
    def verify(U_f, V_f, W_f, n, m, p):
        """Verify that the decomposition is exact."""
        # For each output element (i,k), check: sum_r U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 if (i,k) is in tensor
        # Actually, the tensor T[i,j,k] = 1 for all (i,j,k), and we need:
        # T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # For matrix multiplication, T[i,j,k] should be 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_f[i*m+j][r] * V_f[j*p+k][r] * W_f[k*n+i][r] for r in range(len(U_f[0])))
                    if val != 1:
                        return False
        return True
    
    if not verify(U_frac, V_frac, W_frac, n, m, p):
        # Return the schoolbook if verification fails
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: For each pair of columns, check if one can be expressed as a linear combination of others
    # If so, eliminate it
    
    current_U = U_frac
    current_V = V_frac
    current_W = W_frac
    current_rank = rank
    
    # Build the tensor from the decomposition
    def build_tensor(U_f, V_f, W_f, n, m, p, r):
        """Build the n*m*p tensor from the decomposition."""
        T = [[[Fraction(0) for _ in range(p)] for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i][j][k] = sum(U_f[i*m+j][col] * V_f[j*p+k][col] * W_f[k*n+i][col] for col in range(r))
        return T
    
    # Flatten tensor to vector
    def flatten_tensor(T, n, m, p):
        vec = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec.append(T[i][j][k])
        return vec
    
    # For each column index, check if it's linearly dependent on others
    # We can do this by checking if the tensor slice for that column is in the span of others
    
    # More efficient approach: Use the fact that each rank-1 term contributes to the tensor
    # We can try to merge terms that are linearly dependent
    
    # Try eliminating columns one by one
    for attempt in range(min(20, current_rank)):  # Limit attempts for time
        if time.time() > deadline - 10:
            break
        
        found_removal = False
        for col_to_try in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            # Check if column col_to_try can be removed
            # This means: the tensor contribution from this column must be representable by others
            # i.e., there exist coefficients c_r for r != col_to_try such that:
            # U[:,col_to_try] * V[:,col_to_try] * W[:,col_to_try] = sum_{r != col_to_try} c_r * U[:,r] * V[:,r] * W[:,r]
            
            # This is a system of n*m*p equations in (current_rank-1) unknowns
            # Let's set up the system
            
            # Vectorize the rank-1 terms
            def vec_term(U_f, V_f, W_f, col, n, m, p):
                vec = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            vec.append(U_f[i*m+j][col] * V_f[j*p+k][col] * W_f[k*n+i][col])
                return vec
            
            target = vec_term(current_U, current_V, current_W, col_to_try, n, m, p)
            
            # Build matrix A where A[:,r] = vec_term(r) for r != col_to_try
            num_vars = current_rank - 1
            A = []
            var_idx = 0
            for r in range(current_rank):
                if r == col_to_try:
                    continue
                A.append(vec_term(current_U, current_V, current_W, r, n, m, p))
                var_idx += 1
            
            # Now solve A * c = target using Gaussian elimination over Fractions
            # A is (n*m*p) x (current_rank-1), target is (n*m*p)
            
            num_eq = n * m * p
            # Create augmented matrix [A | target]
            aug = []
            for eq in range(num_eq):
                row = []
                for var in range(num_vars):
                    row.append(A[var][eq])
                row.append(target[eq])
                aug.append(row)
            
            # Gaussian elimination
            try:
                pivot_rows = []
                for var in range(num_vars):
                    # Find pivot
                    pivot = -1
                    for eq in range(var, num_eq):
                        if aug[eq][var] != 0:
                            pivot = eq
                            break
                    if pivot == -1:
                        continue
                    
                    # Swap
                    aug[var], aug[pivot] = aug[pivot], aug[var]
                    pivot_rows.append(var)
                    
                    # Eliminate
                    pivot_val = aug[var][var]
                    for eq in range(num_eq):
                        if eq != var and aug[eq][var] != 0:
                            factor = aug[eq][var] / pivot_val
                            for col in range(var, num_vars + 1):
                                aug[eq][col] -= factor * aug[var][col]
                
                # Check if consistent: for rows with all zeros in A part, RHS must be 0
                consistent = True
                for eq in range(num_eq):
                    all_zero = True
                    for var in range(num_vars):
                        if aug[eq][var] != 0:
                            all_zero = False
                            break
                    if all_zero and aug[eq][num_vars] != 0:
                        consistent = False
                        break
                
                if consistent:
                    # We found a valid elimination!
                    # Remove column col_to_try
                    new_U = [row[:col_to_try] + row[col_to_try+1:] for row in current_U]
                    new_V = [row[:col_to_try] + row[col_to_try+1:] for row in current_V]
                    new_W = [row[:col_to_try] + row[col_to_try+1:] for row in current_W]
                    
                    # Verify
                    if verify(new_U, new_V, new_W, n, m, p):
                        current_U = new_U
                        current_V = new_V
                        current_W = new_W
                        current_rank -= 1
                        found_removal = True
                        break
            except Exception:
                continue
        
        if not found_removal:
            break
    
    # Convert back to float/int if possible
    def convert_back(mat_frac):
        mat = []
        for row in mat_frac:
            new_row = []
            for x in row:
                if x.denominator == 1:
                    new_row.append(int(x.numerator))
                else:
                    new_row.append(float(x))
            mat.append(new_row)
        return mat
    
    U_final = convert_back(current_U)
    V_final = convert_back(current_V)
    W_final = convert_back(current_W)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
