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
    # The idea: each rank-1 term t has a "signature" in the tensor T[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t]
    # If term t is linearly dependent on others, we can remove it.
    # We check this by seeing if the column vectors (U[:,t], V[:,t], W[:,t]) are in the span of others.
    # More precisely, we flatten the tensor and check rank.
    
    if time.time() < deadline - 15:
        # Build the tensor as a 3D array using rational arithmetic
        # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # But we want to check if any rank-1 term is redundant.
        
        # Alternative approach: For each term t, check if there exist coefficients c_s (s != t)
        # such that for all i,j,k: U[i,j,t]*V[j,k,t]*W[k,i,t] = sum_{s!=t} c_s * U[i,j,s]*V[j,k,s]*W[k,i,s]
        
        # This is a system of nonlinear equations, hard to solve directly.
        
        # Simpler approach: Check if the "mode-1 flattening" of the tensor has rank < 56.
        # The mode-1 flattening is a matrix of size (n*m) x (m*p*n) where
        # T_flat[i, (j,k)] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # Wait, that's not right. The tensor is T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        
        # Actually, let me reconsider. The tensor for matrix multiplication (2x4)*(4x8) is:
        # T[i, j, k] where i in [0,2), j in [0,4), k in [0,8)
        # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        
        # The mode-1 flattening is a matrix of size (n) x (m*p) where
        # T_mode1[i, (j,k)] = T[i,j,k]
        
        # If the rank of this flattening is < 56, then we might be able to reduce the rank.
        # But actually, the rank of any mode flattening is an upper bound on the tensor rank.
        
        # Let me compute the mode-1 flattening and check its rank over the rationals.
        
        # Actually, a better approach: directly check if any rank-1 term is in the span of others.
        # For term t, define vectors:
        # u_t in R^(n*m) with u_t[i*m+j] = U[i,j,t]
        # v_t in R^(m*p) with v_t[j*p+k] = V[j,k,t]  
        # w_t in R^(p*n) with w_t[k*n+i] = W[k,i,t]
        
        # The tensor contribution of term t is the outer product u_t ⊗ v_t ⊗ w_t (in a suitable flattening)
        # If this outer product is in the span of {u_s ⊗ v_s ⊗ w_s : s != t}, then t is redundant.
        
        # This is equivalent to checking if the tensor T - T_t (where T_t is the contribution of term t)
        # has the same "support" in some sense.
        
        # A practical approach: try to find a linear combination of all other terms that equals term t.
        # This means: for all (i,j,k), sum_{s!=t} c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,t]*V[j,k,t]*W[k,i,t]
        
        # This is a system of n*m*p = 2*4*8 = 64 equations in (rank-1) = 55 unknowns.
        # But it's nonlinear (products of three unknowns... wait, no, the U,V,W are known).
        
        # Actually, U,V,W are FIXED (from the Strassen decomposition). We're looking for c_s such that
        # sum_{s!=t} c_s * (U[i,j,s]*V[j,k,s]*W[k,i,s]) = U[i,j,t]*V[j,k,t]*W[k,i,t] for all i,j,k.
        
        # This IS a linear system in the c_s! Let's set it up.
        
        # For each (i,j,k), we have:
        # sum_{s!=t} c_s * A_{(i,j,k),s} = b_{(i,j,k)}
        # where A_{(i,j,k),s} = U[i,j,s]*V[j,k,s]*W[k,i,s] and b_{(i,j,k)} = U[i,j,t]*V[j,k,t]*W[k,i,t]
        
        # We have 64 equations and 55 unknowns. If the system is consistent, term t is redundant.
        
        # Let's try this for each t and see if any can be removed.
        
        def check_redundant_term(t, U_mat, V_mat, W_mat, n, m, p, rank):
            """Check if term t is a linear combination of all other terms."""
            # Build the linear system
            # Unknowns: c_s for s in {0,...,rank-1} \ {t}
            # Equations: for each (i,j,k), sum_{s!=t} c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,t]*V[j,k,t]*W[k,i,t]
            
            # Number of unknowns
            num_unknowns = rank - 1
            # Number of equations
            num_eqs = n * m * p
            
            # Map unknown index to term index
            unknown_to_term = [s for s in range(rank) if s != t]
            
            # Build augmented matrix using Fraction for exact arithmetic
            # Only proceed if we have enough time
            if time.time() > deadline - 5:
                return False
            
            # Build the system
            # A is num_eqs x num_unknowns, b is num_eqs
            A = [[Fraction(0) for _ in range(num_unknowns)] for _ in range(num_eqs)]
            b = [Fraction(0) for _ in range(num_eqs)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        b[eq_idx] = Fraction(U_mat[i*m+j][t]) * Fraction(V_mat[j*p+k][t]) * Fraction(W_mat[k*n+i][t])
                        for u_idx, s in enumerate(unknown_to_term):
                            A[eq_idx][u_idx] = Fraction(U_mat[i*m+j][s]) * Fraction(V_mat[j*p+k][s]) * Fraction(W_mat[k*n+i][s])
            
            # Solve using Gaussian elimination over rationals
            # Check if the system is consistent
            # We'll do row reduction
            
            # Copy A and b for elimination
            A_copy = [row[:] for row in A]
            b_copy = b[:]
            
            num_rows = num_eqs
            num_cols = num_unknowns
            
            # Gaussian elimination with partial pivoting
            rank_found = 0
            for col in range(num_cols):
                # Find pivot
                pivot_row = -1
                for row in range(rank_found, num_rows):
                    if A_copy[row][col] != 0:
                        pivot_row = row
                        break
                
                if pivot_row == -1:
                    continue
                
                # Swap rows
                if pivot_row != rank_found:
                    A_copy[pivot_row], A_copy[rank_found] = A_copy[rank_found], A_copy[pivot_row]
                    b_copy[pivot_row], b_copy[rank_found] = b_copy[rank_found], b_copy[pivot_row]
                
                # Eliminate below
                pivot_val = A_copy[rank_found][col]
                for row in range(rank_found + 1, num_rows):
                    if A_copy[row][col] != 0:
                        factor = A_copy[row][col] / pivot_val
                        for c in range(col, num_cols):
                            A_copy[row][c] -= factor * A_copy[rank_found][c]
                        b_copy[row] -= factor * b_copy[rank_found]
                
                rank_found += 1
            
            # Check consistency: for all rows, if all A coefficients are 0, then b must be 0
            for row in range(rank_found, num_rows):
                all_zero = all(A_copy[row][c] == 0 for c in range(num_cols))
                if all_zero and b_copy[row] != 0:
                    return False  # Inconsistent system, term t is NOT redundant
            
            return True  # System is consistent, term t IS redundant
        
        # Try to remove terms one by one
        # We need to be careful: removing one term might make another term redundant
        # Let's iteratively check and remove
        
        removed_terms = set()
        current_rank = rank
        
        # Iterate: check if any remaining term is redundant given the others
        # We do this in a loop, checking each term
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
                break  # No more redundant terms
            
            removed_terms.add(found_redundant)
            current_rank -= 1
        
        # If we removed any terms, rebuild the matrices with only the non-removed terms
        if removed_terms:
            # Create new matrices with only the remaining terms
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
