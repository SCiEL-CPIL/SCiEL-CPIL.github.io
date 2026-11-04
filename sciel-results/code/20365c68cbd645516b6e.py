import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val
    
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # Attempt rank reduction using exact rational Gaussian elimination
    # We check if any rank-1 term is redundant by solving a linear system
    
    if time.time() < deadline - 15:
        def check_redundant_term(t, U_mat, V_mat, W_mat, n, m, p, rank):
            """Check if term t is a linear combination of all other terms."""
            num_unknowns = rank - 1
            num_eqs = n * m * p
            
            unknown_to_term = [s for s in range(rank) if s != t]
            
            if time.time() > deadline - 5:
                return False
            
            # Build the system
            A = [[Fraction(0) for _ in range(num_unknowns)] for _ in range(num_eqs)]
            b = [Fraction(0) for _ in range(num_eqs)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        b[eq_idx] = Fraction(U_mat[i*m+j][t]) * Fraction(V_mat[j*p+k][t]) * Fraction(W_mat[k*n+i][t])
                        for u_idx, s in enumerate(unknown_to_term):
                            A[eq_idx][u_idx] = Fraction(U_mat[i*m+j][s]) * Fraction(V_mat[j*p+k][s]) * Fraction(W_mat[k*n+i][s])
            
            # Gaussian elimination
            A_copy = [row[:] for row in A]
            b_copy = b[:]
            
            num_rows = num_eqs
            num_cols = num_unknowns
            
            rank_found = 0
            for col in range(num_cols):
                pivot_row = -1
                for row in range(rank_found, num_rows):
                    if A_copy[row][col] != 0:
                        pivot_row = row
                        break
                
                if pivot_row == -1:
                    continue
                
                if pivot_row != rank_found:
                    A_copy[pivot_row], A_copy[rank_found] = A_copy[rank_found], A_copy[pivot_row]
                    b_copy[pivot_row], b_copy[rank_found] = b_copy[rank_found], b_copy[pivot_row]
                
                pivot_val = A_copy[rank_found][col]
                for row in range(rank_found + 1, num_rows):
                    if A_copy[row][col] != 0:
                        factor = A_copy[row][col] / pivot_val
                        for c in range(col, num_cols):
                            A_copy[row][c] -= factor * A_copy[rank_found][c]
                        b_copy[row] -= factor * A_copy[rank_found]
                
                rank_found += 1
            
            for row in range(rank_found, num_rows):
                all_zero = all(A_copy[row][c] == 0 for c in range(num_cols))
                if all_zero and b_copy[row] != 0:
                    return False
            
            return True
        
        removed_terms = set()
        current_rank = rank
        
        for iteration in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            found_redundant = -1
            for t in range(current_rank):
                if t in removed_terms:
                    continue
                if check_redundant_term(t, U, V, W, n, m, p, current_rank):
                    found_redundant = t
                    break
            
            if found_redundant == -1:
                break
            
            removed_terms.add(found_redundant)
            current_rank -= 1
        
        if removed_terms:
            remaining_terms = [t for t in range(rank) if t not in removed_terms]
            new_rank = len(remaining_terms)
            
            U_new = [[0] * new_rank for _ in range(n * m)]
            V_new = [[0] * new_rank for _ in range(m * p)]
            W_new = [[0] * new_rank for _ in range(p * n)]
            
            for new_idx, old_idx in enumerate(remaining_terms):
                for i in range(n * m):
                    U_new[i][new_idx] = U[i][old_idx]
                for j in range(m * p):
                    V_new[j][new_idx] = V[j][old_idx]
                for k in range(p * n):
                    W_new[k][new_idx] = W[k][old_idx]
            
            U, V, W = U_new, V_new, W_new
    
    # Additional structured compression: try to find global linear dependencies
    # by checking if the tensor can be represented with fewer terms using
    # a different basis. This is equivalent to checking if the rank-1 terms
    # span a subspace of dimension < current_rank in the tensor space.
    
    if time.time() < deadline - 10 and len(U[0]) > 40:
        # Try a more aggressive reduction: check if pairs of terms can be combined
        # This is a heuristic search for near-linear dependencies
        current_rank = len(U[0])
        
        # For each pair of terms, check if their sum is proportional to another term
        # or if there's a 2-term linear combination that's redundant
        # This is expensive, so we only try a few iterations
        
        improved = True
        max_improvement_iterations = 3
        iteration_count = 0
        
        while improved and iteration_count < max_improvement_iterations and time.time() < deadline - 5:
            improved = False
            iteration_count += 1
            current_rank = len(U[0])
            
            # Check all terms for redundancy again after any removals
            removed_in_pass = set()
            
            for t in range(current_rank):
                if t in removed_in_pass:
                    continue
                if check_redundant_term(t, U, V, W, n, m, p, current_rank):
                    removed_in_pass.add(t)
                    improved = True
            
            if removed_in_pass:
                remaining_terms = [t for t in range(current_rank) if t not in removed_in_pass]
                new_rank = len(remaining_terms)
                
                U_new = [[0] * new_rank for _ in range(n * m)]
                V_new = [[0] * new_rank for _ in range(m * p)]
                W_new = [[0] * new_rank for _ in range(p * n)]
                
                for new_idx, old_idx in enumerate(remaining_terms):
                    for i in range(n * m):
                        U_new[i][new_idx] = U[i][old_idx]
                    for j in range(m * p):
                        V_new[j][new_idx] = V[j][old_idx]
                    for k in range(p * n):
                        W_new[k][new_idx] = W[k][old_idx]
                
                U, V, W = U_new, V_new, W_new
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
