"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition with rank = n*m*p."""
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
    return U, V, W

def _berlekamp_massey(sequence, field_mod):
    """
    Berlekamp-Massey algorithm to find the shortest linear recurrence
    that generates the given sequence over a finite field.
    
    Args:
        sequence: list of elements in F_p
        field_mod: the prime modulus p
    
    Returns:
        (C, L) where C is the connection polynomial coefficients [c_0, c_1, ..., c_L]
        with c_0 = 1, and L is the linear complexity.
    """
    n = len(sequence)
    if n == 0:
        return [1], 0
    
    # Initialize
    C = [0] * n  # Connection polynomial
    B = [0] * n  # Previous connection polynomial
    C[0] = 1
    B[0] = 1
    L = 0
    m = 1  # Number of iterations since last update
    b = 1  # Last discrepancy
    
    for N in range(n):
        # Compute discrepancy
        d = sequence[N]
        for i in range(1, L + 1):
            if i < n:
                d = (d + C[i] * sequence[N - i]) % field_mod
        
        if d == 0:
            m += 1
        else:
            # Update C
            T = C[:]
            coef = (d * pow(b, field_mod - 2, field_mod)) % field_mod
            for i in range(m, n):
                C[i] = (C[i] - coef * B[i - m]) % field_mod
            
            if 2 * L <= N:
                L = N + 1 - L
                B = T
                b = d
                m = 1
            else:
                m += 1
    
    # Extract the actual connection polynomial of length L+1
    C_final = C[:L + 1]
    return C_final, L

def _find_linear_dependencies(matrix, field_mod):
    """
    Find linear dependencies among columns of a matrix over F_p.
    Uses Berlekamp-Massey on each row to find recurrences.
    
    Args:
        matrix: 2D list, shape (rows, cols)
        field_mod: prime modulus
    
    Returns:
        List of dependency vectors (as lists) where each vector v satisfies
        sum_j v[j] * matrix[:, j] = 0
    """
    rows, cols = len(matrix), len(matrix[0])
    if cols <= 1:
        return []
    
    # For each row, apply Berlekamp-Massey to find recurrence
    # Then verify which recurrences work across all rows
    
    # Collect candidate recurrences from the first few rows
    candidate_recs = set()
    sample_rows = min(rows, 5)
    
    for r in range(sample_rows):
        row = matrix[r]
        C, L = _berlekamp_massey(row, field_mod)
        if L > 0 and L <= cols:
            # C = [c_0, c_1, ..., c_L] with c_0 = 1
            # Recurrence: sum_{i=0}^{L} c_i * x_{n-i} = 0
            # This gives a dependency vector of length L+1
            # But we need to map this to column indices
            # For a row of length cols, if L < cols, we can pad
            dep = [0] * cols
            # The recurrence says: x_n = -sum_{i=1}^{L} c_i * x_{n-i}
            # So for position n (0-indexed from the end):
            # x_{cols-1} = -sum_{i=1}^{L} c_i * x_{cols-1-i}
            # Dependency: x_{cols-1} + sum_{i=1}^{L} c_i * x_{cols-1-i} = 0
            # Vector: [0, ..., 0, c_L, c_{L-1}, ..., c_1, 1] (length cols)
            if L < cols:
                for i in range(L + 1):
                    idx = cols - 1 - i
                    dep[idx] = C[i] % field_mod
            candidate_recs.add(tuple(dep))
    
    # Verify candidates against all rows
    valid_deps = []
    for dep in candidate_recs:
        valid = True
        for r in range(rows):
            total = 0
            for j in range(cols):
                total = (total + dep[j] * matrix[r][j]) % field_mod
            if total != 0:
                valid = False
                break
        if valid:
            valid_deps.append(list(dep))
    
    return valid_deps

def _build_dependency_matrix(U, V, W, n, m, p, rank, field_mod):
    """
    Build the tensor identity matrix over F_p.
    For each (i, j, k) triple, we need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    We form a matrix where each row corresponds to an (i,j,k) constraint,
    and columns correspond to the rank-1 terms.
    """
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = [0] * rank
                for r in range(rank):
                    val = (U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]) % field_mod
                    row[r] = val
                constraints.append(row)
    
    return constraints

def _try_rank_reduction_bm(n, m, p, current_rank, target_rank, time_limit=10.0):
    """
    Try to reduce rank using Berlekamp-Massey based dependency detection over F_5.
    
    Args:
        n, m, p: dimensions
        current_rank: starting rank (e.g., 48)
        target_rank: desired rank
        time_limit: max time in seconds
    
    Returns:
        (U, V, W) if successful, else None
    """
    start_time = time.time()
    deadline = start_time + time_limit
    field_mod = 5
    
    # Start from schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    current_rank = n * m * p
    
    # Try to reduce by eliminating one column at a time
    for reduction in range(current_rank - target_rank + 1):
        if time.time() > deadline:
            break
        
        target_r = current_rank - reduction
        
        # Build the constraint matrix
        # Each constraint (i,j,k) requires sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # For rank reduction, we look for a linear combination of columns
        # that can be eliminated while preserving the identity
        
        # Approach: Try to find a column r such that we can express it as
        # a linear combination of other columns in a way that preserves all constraints
        
        # Simpler approach: Use BM to find dependencies in the "flattened" tensor
        # Create a 2D matrix where rows are (i,j,k) and columns are r
        # But the relationship is multiplicative, not additive
        
        # Alternative: Randomized approach with BM verification
        # Generate random perturbations and check if BM finds valid dependencies
        
        best_U, best_V, best_W = None, None, None
        
        # Try several random strategies
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Start from schoolbook and try to zero out one column
            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]
            
            # Pick a column to potentially eliminate
            col_to_eliminate = random.randint(0, target_r - 1)
            
            # Try to redistribute its contribution to other columns
            success = True
            
            # For each constraint, check if we can adjust other columns
            # to compensate for zeroing out column col_to_eliminate
            for i in range(n):
                if time.time() > deadline:
                    success = False
                    break
                for j in range(m):
                    if not success:
                        break
                    for k in range(p):
                        if time.time() > deadline:
                            success = False
                            break
                        
                        # Current contribution from col_to_eliminate
                        contrib = (U_try[i * m + j][col_to_eliminate] * 
                                  V_try[j * p + k][col_to_eliminate] * 
                                  W_try[k * n + i][col_to_eliminate]) % field_mod
                        
                        if contrib == 0:
                            continue
                        
                        # We need to add this contribution to other columns
                        # Find a column r' where we can adjust
                        found_compensation = False
                        for r_prime in range(target_r):
                            if r_prime == col_to_eliminate:
                                continue
                            
                            # Check if we can adjust U, V, or W at r_prime
                            # to add 'contrib' to this constraint
                            # Simple heuristic: try adding to U
                            if V_try[j * p + k][r_prime] != 0 and W_try[k * n + i][r_prime] != 0:
                                # We can adjust U[i*m+j][r_prime]
                                # New value: U[i*m+j][r_prime] + contrib / (V*W)
                                vw = (V_try[j * p + k][r_prime] * W_try[k * n + i][r_prime]) % field_mod
                                if vw != 0:
                                    inv_vw = pow(vw, field_mod - 2, field_mod)
                                    adjustment = (contrib * inv_vw) % field_mod
                                    U_try[i * m + j][r_prime] = (U_try[i * m + j][r_prime] + adjustment) % field_mod
                                    found_compensation = True
                                    break
                        
                        if not found_compensation:
                            success = False
                            break
                    
                    if not success:
                        break
                
                if not success:
                    break
            
            if success:
                # Zero out the column
                for i in range(n * m):
                    U_try[i][col_to_eliminate] = 0
                for j in range(m * p):
                    V_try[j][col_to_eliminate] = 0
                for k in range(p * n):
                    W_try[k][col_to_eliminate] = 0
                
                # Verify over integers (lift from F_p)
                # Convert to small integers
                U_int = [[x if x <= field_mod // 2 else x - field_mod for x in row] for row in U_try]
                V_int = [[x if x <= field_mod // 2 else x - field_mod for x in row] for row in V_try]
                W_int = [[x if x <= field_mod // 2 else x - field_mod for x in row] for row in W_try]
                
                # Remove the zero column
                U_final = [row[:col_to_eliminate] + row[col_to_eliminate + 1:] for row in U_int]
                V_final = [row[:col_to_eliminate] + row[col_to_eliminate + 1:] for row in V_int]
                W_final = [row[:col_to_eliminate] + row[col_to_eliminate + 1:] for row in W_int]
                
                # Verify
                if _verify_decomposition(U_final, V_final, W_final, n, m, p, target_r - 1):
                    return U_final, V_final, W_final
        
        # If we couldn't reduce by 1, try a different approach
        # Use BM to find actual dependencies
    
    return None

def _try_low_rank_decomposition(n, m, p, target_rank, time_limit=30.0):
    """
    Try to find a rank-reduced decomposition using structured random search.
    Uses a hybrid approach: start from schoolbook-like structure and perturb.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # First try BM-based reduction
    result = _try_rank_reduction_bm(n, m, p, n * m * p, target_rank, time_limit=min(15.0, time_limit / 2))
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W, n, m, p, len(U[0])):
            return U, V, W
    
    # Fall back to random search
    remaining_time = deadline - time.time()
    if remaining_time < 1:
        return None
    
    max_restarts = 300
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Generate random initial decomposition
        U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Local search: try to fix violations
        for iteration in range(150):
            if time.time() > deadline:
                break
            
            # Check if valid
            if _verify_decomposition(U, V, W, n, m, p, target_rank):
                return U, V, W
            
            # Find a violation and try to fix it
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            
            # Compute current value
            current = 0
            for r in range(target_rank):
                current += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
            
            if current == 1:
                continue
            
            # Try modifying one of the three matrices at a random rank index
            r = random.randint(0, target_rank - 1)
            choice = random.choice(['U', 'V', 'W'])
            
            if choice == 'U':
                old_val = U[i * m + j][r]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    U[i * m + j][r] = new_val
                    new_current = 0
                    for rr in range(target_rank):
                        new_current += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(new_current - 1) < abs(current - 1):
                        break
                    else:
                        U[i * m + j][r] = old_val
                        break
            elif choice == 'V':
                old_val = V[j * p + k][r]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    V[j * p + k][r] = new_val
                    new_current = 0
                    for rr in range(target_rank):
                        new_current += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(new_current - 1) < abs(current - 1):
                        break
                    else:
                        V[j * p + k][r] = old_val
                        break
            else:
                old_val = W[k * n + i][r]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    W[k * n + i][r] = new_val
                    new_current = 0
                    for rr in range(target_rank):
                        new_current += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(new_current - 1) < abs(current - 1):
                        break
                    else:
                        W[k * n + i][r] = old_val
                        break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    overall_start = time.time()
    overall_deadline = overall_start + 140  # Leave 10s margin for 150s limit
    
    # Try ranks from 47 down to a reasonable minimum
    for target_rank in range(47, 34, -1):
        if time.time() > overall_deadline - 5:
            break
        
        # Allocate more time for lower ranks (harder to find)
        time_per_rank = min(15.0, (overall_deadline - time.time()) / (47 - target_rank + 1))
        
        result = _try_low_rank_decomposition(n, m, p, target_rank, time_limit=time_per_rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If no reduction found, return schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
