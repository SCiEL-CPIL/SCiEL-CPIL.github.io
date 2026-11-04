"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe (150s wall-clock)
    
    n, m, p = DIMENSIONS
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Build the baseline Strassen+Schoolbook decomposition (rank 99)
    rank_baseline = 99
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    # Part 1: j in {0, 1}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 0, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [0, 1], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [0, 1], range(7), col)
    
    # Part 2: j in {2, 3, 4}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 2, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [2, 3], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [4], range(7), col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [2, 3, 4], range(7), col)
    
    # Verify baseline correctness
    def verify_decomposition(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(len(U[0])):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    # Now attempt to reduce rank via exact rational linear algebra
    def build_constraint_matrix(U, V, W, n, m, p):
        """Build the n*m*p x rank matrix"""
        r = len(U[0])
        M = [[Fraction(0)] * r for _ in range(n * m * p)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        M[idx][r_idx] = Fraction(U[i * m + j][r_idx]) * Fraction(V[j * p + k][r_idx]) * Fraction(W[k * n + i][r_idx])
                    idx += 1
        return M
    
    def find_redundant_columns_with_coefficients(M):
        """
        Perform Gaussian elimination and track which columns are redundant.
        For each redundant column, compute its expression as a linear combination of pivot columns.
        Returns (pivot_cols, redundant_info) where redundant_info maps col_idx -> {pivot_idx: coeff}
        """
        if not M or not M[0]:
            return [], {}
        rows = len(M)
        cols = len(M[0])
        A = [row[:] for row in M]
        pivot_cols = []
        redundant_info = {}
        
        current_row = 0
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                coeff_dict = {}
                for r_idx in range(current_row):
                    if A[r_idx][col_idx] != 0:
                        coeff_dict[pivot_cols[r_idx]] = A[r_idx][col_idx]
                redundant_info[col_idx] = coeff_dict
                continue
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                coeff_dict = {}
                for r_idx in range(current_row):
                    if A[r_idx][col_idx] != 0:
                        coeff_dict[pivot_cols[r_idx]] = A[r_idx][col_idx]
                redundant_info[col_idx] = coeff_dict
                continue
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1
        
        return pivot_cols, redundant_info
    
    # Build the constraint matrix for the baseline solution
    M_base = build_constraint_matrix(U_base, V_base, W_base, n, m, p)
    
    # Find redundant columns
    pivot_cols, redundant_info = find_redundant_columns_with_coefficients(M_base)
    
    # Strategy: Use dependencies to eliminate terms.
    # If T^r = sum_{s in pivot} c_{r,s} T^s, then the dependency is:
    # T^r - sum_{s in pivot} c_{r,s} T^s = 0
    # This means: sum_r d_r T^r = 0 where d_r = 1, d_s = -c_{r,s}
    # We can modify coefficients: new_coeff_r = 1 + d_r for each r
    # If new_coeff_r = 0 for some r, we can drop that term.
    
    # For each redundant column r, we get a dependency.
    # Let's try to use these to eliminate terms greedily.
    
    # Start with all terms having coefficient 1
    coeffs = [Fraction(1)] * rank_baseline
    active = [True] * rank_baseline
    
    # For each redundant column, apply the dependency to try to zero out a term
    # Process redundant columns in order of their column index
    redundant_cols = sorted(redundant_info.keys())
    
    for r in redundant_cols:
        if time.time() > deadline:
            break
        if not active[r]:
            continue
        # Get the dependency: T^r = sum_{s in pivot} c_{r,s} T^s
        # Dependency vector: d_r = 1, d_s = -c_{r,s}
        # Apply: new_coeff_x = coeff_x + d_x for all x
        # But we want to zero out r if possible.
        # After applying: new_coeff_r = 1 + 1 = 2 (not zero)
        # Instead, let's think differently.
        
        # Actually, the dependency says: T^r - sum c_{r,s} T^s = 0
        # So: sum c_{r,s} T^s = T^r
        # The total sum is: sum_all T^x = sum_pivot T^s + sum_redundant T^r
        # = sum_pivot T^s + sum_redundant sum_{s in pivot} c_{r,s} T^s
        # = sum_pivot (1 + sum_redundant c_{r,s}) T^s
        # This should equal the all-ones tensor.
        
        # Alternative approach: just try to find a subset of the 99 terms
        # whose sum equals the all-ones tensor, using as few terms as possible.
        # This is a set cover / subset sum problem in a 105-dim space.
        
        # Let's try a different strategy: use the null space to find
        # dependencies that can zero out specific terms.
        pass
    
    # Let's try a more direct approach: 
    # The rank of the 105x99 matrix tells us how many independent tensors we have.
    # If rank < 99, we have 99 - rank dependencies.
    # We can use these to reduce the number of terms.
    
    # Try to find the minimum number of terms by checking subsets
    # This is NP-hard in general, but for small reductions we can be smart.
    
    # First, let's see what the actual rank is
    actual_rank = len(pivot_cols)
    
    # If we have dependencies, we can potentially reduce the rank.
    # The key question: can we express the all-ones tensor as a sum of
    # fewer than 99 rank-1 tensors from our set?
    
    # Greedy approach: try to remove one term at a time.
    # For each term r, check if the remaining 98 terms can still sum to all-ones.
    # This requires solving a linear system: find coeffs c_s for s != r
    # such that sum_{s != r} c_s T^s = all_ones.
    
    # This is a 105 x 98 linear system (more equations than unknowns),
    # so it's generally overdetermined. We need to check if a solution exists.
    
    # Let's build the system and check solvability using Gaussian elimination.
    
    def can_remove_term(U, V, W, n, m, p, remove_idx):
        """Check if we can express all-ones as a sum of the other terms."""
        r = len(U[0])
        # Build matrix without column remove_idx
        cols_to_use = [i for i in range(r) if i != remove_idx]
        num_cols = len(cols_to_use)
        
        # Build 105 x num_cols matrix
        A = [[Fraction(0)] * num_cols for _ in range(n * m * p)]
        for idx_i in range(n * m * p):
            i = idx_i // (m * p)
            rem = idx_i % (m * p)
            j = rem // p
            k = rem % p
            for col_j, orig_idx in enumerate(cols_to_use):
                A[idx_i][col_j] = Fraction(U[i*m+j][orig_idx]) * Fraction(V[j*p+k][orig_idx]) * Fraction(W[k*n+i][orig_idx])
        
        # RHS is all ones
        # Check if system A * x = 1 is solvable
        # Use Gaussian elimination with augmented matrix
        
        if time.time() > deadline:
            return False
        
        # Augmented matrix
        aug = [row[:] + [Fraction(1)] for row in A]
        rows = len(aug)
        cols = num_cols
        
        current_row = 0
        for col_idx in range(cols):
            if time.time() > deadline:
                return False
            if current_row >= rows:
                break
            # Find pivot
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if aug[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                continue
            aug[current_row], aug[pivot_row] = aug[pivot_row], aug[current_row]
            pivot_val = aug[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if aug[row_idx][col_idx] != 0:
                    factor = aug[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols + 1):
                        aug[row_idx][c] -= factor * aug[current_row][c]
            current_row += 1
        
        # Check for contradictions
        for row_idx in range(current_row, rows):
            # Check if this row is [0, 0, ..., 0, non-zero]
            all_zero = all(aug[row_idx][c] == 0 for c in range(cols))
            if all_zero and aug[row_idx][cols] != 0:
                return False
        
        return True
    
    # Try to remove terms one by one
    U_current = [row[:] for row in U_base]
    V_current = [row[:] for row in V_base]
    W_current = [row[:] for row in W_base]
    active_terms = list(range(rank_baseline))
    
    # Greedy: try to remove each term, keeping the solution if it works
    removed = []
    for r in range(rank_baseline - 1, -1, -1):
        if time.time() > deadline:
            break
        if r not in active_terms:
            continue
        # Check if we can remove term r
        # Temporarily mark as inactive
        test_active = [i for i in active_terms if i != r]
        
        # Check solvability
        # Build the system for the remaining terms
        num_remaining = len(test_active)
        if num_remaining < 1:
            continue
            
        # Quick check: do the remaining terms span the all-ones vector?
        # Build matrix and check rank
        A_check = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for orig_idx in test_active:
                        row.append(Fraction(U_base[i*m+j][orig_idx]) * 
                                  Fraction(V_base[j*p+k][orig_idx]) * 
                                  Fraction(W_base[k*n+i][orig_idx]))
                    A_check.append(row)
        
        # Check if all-ones is in the column space
        # Augmented matrix [A | 1]
        aug_check = [row[:] + [Fraction(1)] for row in A_check]
        rows_c = len(aug_check)
        cols_c = num_remaining
        
        current_row_c = 0
        for col_idx in range(cols_c):
            if time.time() > deadline:
                break
            if current_row_c >= rows_c:
                break
            pivot_row_c = -1
            for row_idx in range(current_row_c, rows_c):
                if aug_check[row_idx][col_idx] != 0:
                    pivot_row_c = row_idx
                    break
            if pivot_row_c == -1:
                continue
            aug_check[current_row_c], aug_check[pivot_row_c] = aug_check[pivot_row_c], aug_check[current_row_c]
            pivot_val_c = aug_check[current_row_c][col_idx]
            for row_idx in range(current_row_c + 1, rows_c):
                if aug_check[row_idx][col_idx] != 0:
                    factor_c = aug_check[row_idx][col_idx] / pivot_val_c
                    for c in range(col_idx, cols_c + 1):
                        aug_check[row_idx][c] -= factor_c * aug_check[current_row_c][c]
            current_row_c += 1
        
        # Check for contradiction
        solvable = True
        for row_idx in range(current_row_c, rows_c):
            all_zero_c = all(aug_check[row_idx][c] == 0 for c in range(cols_c))
            if all_zero_c and aug_check[row_idx][cols_c] != 0:
                solvable = False
                break
        
        if solvable:
            # We can remove term r
            active_terms.remove(r)
            removed.append(r)
    
    # Build the reduced decomposition
    if removed:
        final_rank = len(active_terms)
        U_final = [[0] * final_rank for _ in range(n * m)]
        V_final = [[0] * final_rank for _ in range(m * p)]
        W_final = [[0] * final_rank for _ in range(p * n)]
        
        for new_idx, old_idx in enumerate(active_terms):
            for i in range(n * m):
                U_final[i][new_idx] = U_base[i][old_idx]
            for i in range(m * p):
                V_final[i][new_idx] = V_base[i][old_idx]
            for i in range(p * n):
                W_final[i][new_idx] = W_base[i][old_idx]
        
        # Verify
        if verify_decomposition(U_final, V_final, W_final, n, m, p):
            return {"U": U_final, "V": V_final, "W": W_final}
    
    # If reduction didn't work or verification failed, return baseline
    return {"U": U_base, "V": V_base, "W": W_base}
# EVOLVE-BLOCK-END
