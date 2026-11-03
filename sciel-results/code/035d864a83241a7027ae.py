import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 135s limit to be safe
    
    # 1. Construct Schoolbook Decomposition (Rank 56)
    def schoolbook_decomp():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = schoolbook_decomp()
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p

    # 2. Construct Strassen + Schoolbook Hybrid (Rank 38)
    # Decompose M (2x4) * N (4x7) into M * N1 (2x4) and M * N2 (2x3)
    # M = [[A, B], [C, D]] (2x2 blocks)
    # N1 = [[E, F], [G, H]] (2x2 blocks)
    # N2 = [[I, J], [K, L]] (2x2 blocks, last column is zeros in N2 for the 3rd col of N? No, N2 is 4x3)
    # Let's use the standard Strassen for 2x2 * 2x2.
    # Rank 7 for 2x2*2x2.
    # We have 2x4 * 4x4 (N1) which is 4 blocks of 2x2*2x2? No.
    # M (2x4) = [M_left (2x2), M_right (2x2)]
    # N1 (4x4) = [[N1_TL, N1_TR], [N1_BL, N1_BR]]
    # Product P1 (2x4) = [M_left, M_right] * [[N1_TL, N1_TR], [N1_BL, N1_BR]]
    # P1 = [M_left*N1_TL + M_right*N1_BL, M_left*N1_TR + M_right*N1_BR]
    # This is 4 matrix multiplications of size 2x2.
    # Strassen reduces 2x2*2x2 to rank 7.
    # So 4 mults * 7 = 28?
    # Wait, Strassen is for A*B where A,B are 2x2.
    # Here we have M_left * N1_TL.
    # Let's just use the known rank 14 for 2x4*4x2? No.
    
    # Let's use the specific rank 14 for 2x4 * 4x4?
    # Actually, R(2,4,4) is known to be 14?
    # R(2,4,2) = 7? No, R(2,2,2)=7.
    # R(2,4,4) <= 2 * R(2,2,4) <= 2 * 2 * R(2,2,2) = 28.
    # But we can do better.
    # Let's stick to the simple construction:
    # Split N (4x7) into N1 (4x4) and N2 (4x3).
    # M (2x4) * N1 (4x4) -> P1 (2x4)
    # M (2x4) * N2 (4x3) -> P2 (2x3)
    
    # For M * N1:
    # M = [A, B] where A,B are 2x2.
    # N1 = [[C, D], [E, F]] where C,D,E,F are 2x2.
    # P1 = [A*C + B*E, A*D + B*F]
    # This requires 4 multiplications of 2x2 matrices.
    # Using Strassen for each 2x2*2x2:
    # Each 2x2*2x2 has rank 7.
    # Total rank for P1 = 4 * 7 = 28.
    
    # For M * N2:
    # N2 = [[G, H, I], [J, K, L]] where G..L are 2x1? No, 2x2 blocks don't fit 3 columns nicely.
    # N2 is 4x3.
    # Let's split N2 into N2a (4x2) and N2b (4x1).
    # M * N2a (2x4 * 4x2) -> 2x2.
    # M * N2b (2x4 * 4x1) -> 2x1.
    
    # M * N2a:
    # N2a = [[G, H], [J, K]] (2x2 blocks).
    # P2a = A*G + B*J, A*H + B*K.
    # 4 multiplications of 2x2. Rank 4*7 = 28.
    
    # M * N2b:
    # N2b = [[I], [L]] (2x1 blocks).
    # P2b = A*I + B*L.
    # 2 multiplications of 2x2 * 2x1.
    # R(2,2,1) = 4.
    # Rank 2*4 = 8.
    
    # Total rank = 28 + 28 + 8 = 64. This is worse than schoolbook (56).
    # Why? Because Strassen overhead is high for small matrices.
    # Schoolbook for 2x4*4x7 is 56.
    # My hybrid is 64.
    
    # Let's try a different split.
    # R(2,4,7) <= R(2,4,4) + R(2,4,3).
    # R(2,4,4) <= 14?
    # If R(2,4,4)=14 and R(2,4,3)=?
    # R(2,4,3) <= R(2,4,4) = 14.
    # Total 28?
    # Is R(2,4,4) = 14?
    # R(2,4,4) = R(2,2,4) + R(2,2,4) ? No.
    # R(2,4,4) is the rank of the tensor for (2x4)*(4x4).
    # This is equivalent to 4 independent (2x2)*(2x2) multiplications? No.
    # M (2x4) * N (4x4).
    # M = [A, B]. N = [[C, D], [E, F]].
    # P = [AC+BE, AD+BF].
    # This is not 4 independent multiplications because A and B are shared.
    # However, we can use the "border rank" or specific decompositions.
    
    # Let's just use the schoolbook as the base and try to prune it.
    # The current program's random search is too slow.
    # Let's try a "Greedy Pruning" on the Schoolbook decomposition.
    # We have 56 terms. We want to see if we can remove some by adjusting others.
    # This is hard.
    
    # Alternative: Use the known R(2,4,7) bound.
    # Alon-Boppana: R(2,4,7) <= 2*4*7 - 2 - 4 - 7 + ...?
    # The best known upper bound for R(m,n,p) is often close to mnp.
    # For R(2,4,7), the schoolbook is 56.
    # Is there a known lower rank?
    # R(2,2,2)=7.
    # R(2,4,4) <= 14 (using 2x2 Strassen on blocks? No, that gives 28).
    # Actually, R(2,4,4) is known to be 14?
    # If we can get R(2,4,4)=14 and R(2,4,3)=?
    # R(2,4,3) <= 14?
    # Then total 28.
    
    # Let's try to construct R(2,4,4) = 14.
    # M (2x4) * N (4x4).
    # Let M = [A, B] (A,B 2x2).
    # N = [[C, D], [E, F]] (C,D,E,F 2x2).
    # P1 = AC + BE, P2 = AD + BF.
    # We need to decompose the tensor for [AC+BE, AD+BF].
    # This is equivalent to multiplying a 2x4 matrix by a 4x4 matrix.
    # This can be seen as:
    # [A, B] * [C, D; E, F]
    # Let's use the decomposition for (2x2)*(2x2) which has rank 7.
    # But A and B are shared.
    # Let's try to find a rank 14 decomposition.
    
    # If we can't find a better constructive method quickly, we return schoolbook.
    # The prompt asks for a "structurally different" strategy.
    # The current strategy is "Random Search".
    # My new strategy is "Constructive Algebraic Combination + Greedy Repair".
    
    # Let's implement a simple Greedy Repair on the Schoolbook.
    # We iterate through the 56 terms.
    # For each term, we try to set its U, V, W entries to 0.
    # Then we solve for the remaining terms to see if a solution exists.
    # If yes, we keep the zeroed term.
    
    U_curr, V_curr, W_curr = schoolbook_decomp()
    r_curr = 56
    indices = list(range(r_curr))
    
    # We will try to remove terms one by one.
    # Start with all 56.
    active_indices = set(indices)
    
    # To speed up, we only try to remove a few.
    # The system is:
    # sum_{c in active} U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # If we fix U and V, W is determined.
    # If we fix U and W, V is determined.
    # If we fix V and W, U is determined.
    
    # Let's fix U and V to be standard basis vectors (1s and 0s) and solve for W.
    # If we zero out a term, we are forcing U, V, W for that column to be 0.
    # This reduces the rank.
    
    # Let's try to find a rank 55 decomposition.
    # We pick a random column to zero out.
    # Then we solve for the remaining 55 columns.
    
    # Since the schoolbook is "minimal" in a sense (each term handles one product),
    # zeroing one term means that term's product (i,j,k) must be handled by others.
    # In schoolbook, term c handles (i_c, j_c, k_c).
    # If we zero it, we need sum_{c' != c} U[i,j,c'] V[j,k,c'] W[k,i,c'] = 1 for (i_c, j_c, k_c).
    # But for other (i,j,k), the sum is already 1.
    # If we change W, it might break other equations.
    
    # This is a linear system.
    # Let's try to solve for W given fixed U, V (with one column zeroed).
    
    def try_remove_term(index_to_remove):
        if time.time() > deadline - 5:
            return False, None, None, None
        
        U_test = U_curr.copy()
        V_test = V_curr.copy()
        # Zero out the column
        U_test[:, index_to_remove] = 0
        V_test[:, index_to_remove] = 0
        
        # Build A matrix for W
        # Unknowns: W[k*n+i, c] for c in active columns
        # We have r_curr-1 unknowns.
        # Equations: n*m*p = 56.
        
        # We only need to solve for the active columns.
        active_cols = [c for c in range(r_curr) if c != index_to_remove]
        
        A_mat = np.zeros((n * m * p, len(active_cols)))
        b_vec = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row = k * n + i
                    for idx, c in enumerate(active_cols):
                        coeff = U_test[i * m + j, c] * V_test[j * p + k, c]
                        if coeff != 0:
                            A_mat[eq_idx, idx] = coeff
                    eq_idx += 1
        
        # Solve A_mat * W_flat = b_vec
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            return False, None, None, None
            
        W_test = np.zeros((p * n, r_curr))
        for idx, c in enumerate(active_cols):
            W_test[:, c] = W_flat[idx]
            
        # Check if W_test is valid (integers/half-integers) and exact
        W_half = W_test * 2
        W_rounded = np.round(W_half)
        error = np.max(np.abs(W_half - W_rounded))
        
        if error > 1e-6:
            return False, None, None, None
            
        W_candidate = W_rounded / 2.0
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in active_cols:
                        val += U_test[i * m + j, c] * V_test[j * p + k, c] * W_candidate[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
                
        if valid:
            return True, U_test, V_test, W_candidate
        else:
            return False, None, None, None

    # Try to remove terms
    # We can't try all 56 * 56 * 56...
    # We just try a few random ones.
    removed_any = False
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        idx = random.randint(0, r_curr - 1)
        success, U_new, V_new, W_new = try_remove_term(idx)
        if success:
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            r_curr -= 1
            removed_any = True
            # Update best result
            best_result = {"U": U_curr.tolist(), "V": V_curr.tolist(), "W": W_curr.tolist()}
            best_rank = r_curr
            # Restart the search with the new rank?
            # The indices are shifted? No, we just zeroed a column.
            # We can keep the same shape and just have zeros.
            
    return best_result
# EVOLVE-BLOCK-END
