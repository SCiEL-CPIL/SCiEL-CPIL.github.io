import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c0 + 1
        v10 = (b_r0 + 1) * p + b_c0
        v11 = (b_r0 + 1) * p + b_c0 + 1
        
        w00 = c_c0 * n + c_r0
        w01 = (c_c0 + 1) * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        w11 = (c_c0 + 1) * n + c_r0 + 1
        
        # M1
        U[u00, col] += 1; U[u11, col] += 1
        V[v00, col] += 1; V[v11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # M2
        U[u10, col] += 1; U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        
        # M3
        U[u00, col] += 1
        V[v01, col] += 1; V[v11, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # M4
        U[u11, col] += 1
        V[v10, col] += 1; V[v00, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # M5
        U[u00, col] += 1; U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        
        # M6
        U[u10, col] += 1; U[u00, col] -= 1
        V[v00, col] += 1; V[v01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7
        U[u01, col] += 1; U[u11, col] -= 1
        V[v10, col] += 1; V[v11, col] += 1
        W[w00, col] += 1
        col += 1

    def add_schoolbook_2x4_1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        w0 = c_c0 * n + c_r0
        w1 = c_c0 * n + c_r0 + 1
        
        for k in range(4):
            u_idx = a_r0 * m + a_c0 + k
            v_idx = (b_r0 + k) * p + b_c0
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w0, col] = 1
            col += 1
            
            u_idx = (a_r0 + 1) * m + a_c0 + k
            v_idx = (b_r0 + k) * p + b_c0
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w1, col] = 1
            col += 1

    # Build the base rank-36 solution
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x4_1(0, 0, 0, 4, 0, 4)
    
    # Verification and Pruning Search
    # We try to remove each term r and see if the remaining terms can satisfy the tensor equation
    # by adjusting W (keeping U, V fixed).
    
    # Precompute the "basis" vectors for the linear system in W
    # Equation: sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1
    # For a fixed set of active terms R_active, we solve for W.
    
    def try_remove_term(r_remove):
        if time.time() > deadline:
            return None
            
        active_terms = [r for r in range(rank) if r != r_remove]
        
        # Build the linear system A_w * w_vec = b
        # Variables are W[k*n+i, r'] for r' in active_terms
        # There are p*n*len(active_terms) variables.
        # There are n*m*p equations.
        
        num_vars = p * n * len(active_terms)
        num_eqs = n * m * p
        
        # Map variable index: idx = (k*n + i) * len(active_terms) + pos_in_active
        # This is inefficient. Better to build the matrix directly.
        
        # A matrix shape: (num_eqs, num_vars)
        # b vector shape: (num_eqs,)
        
        # We can construct A sparsely or densely. 40 x 350 is small.
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        # Iterate over all equations (i, j, k)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Coefficients for this equation
                    # For each active term r', the coefficient of W[k*n+i, r'] is U[i*m+j, r'] * V[j*p+k, r']
                    for pos, r_prime in enumerate(active_terms):
                        coeff = U[i * m + j, r_prime] * V[j * p + k, r_prime]
                        if coeff != 0:
                            # Variable index for W[k*n+i, r_prime]
                            # We need a consistent indexing for the solution vector.
                            # Let's map (k, i, r_prime) to a linear index.
                            # Since W is stored as (p*n, rank), and we are solving for specific r's,
                            # let's just use the original W index and keep track.
                            # Actually, it's easier to solve for the full W vector of size p*n*rank, 
                            # with constraints that W[:, r_remove] = 0.
                            # But we want to minimize the number of variables.
                            
                            # Let's stick to the reduced system.
                            # Var index: We need a unique ID for each (k, i, r_prime).
                            # ID = (k*n + i) * len(active_terms) + pos
                            var_idx = (k * n + i) * len(active_terms) + pos
                            A[eq_idx, var_idx] = coeff
                    eq_idx += 1
                    
        # Solve A * w = b
        try:
            # Use least squares
            w_sol, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if the solution is "good" (close to integers/half-integers)
            # We map w_sol back to W matrix
            W_new = np.zeros((p * n, rank))
            for pos, r_prime in enumerate(active_terms):
                for k in range(p):
                    for i in range(n):
                        var_idx = (k * n + i) * len(active_terms) + pos
                        val = w_sol[var_idx]
                        # Round to nearest half-integer
                        rounded_val = np.round(val * 2) / 2
                        if abs(val - rounded_val) < 1e-5:
                            W_new[k * n + i, r_prime] = rounded_val
                        else:
                            # If not close, it's not a valid integer/half-integer solution
                            return None
                            
            # Verify the solution exactly
            # Check if sum_r U V W == 1
            # We can do a quick check on a few random points or all points.
            # All points is 40, so it's fast.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r_prime in active_terms:
                            val += U[i * m + j, r_prime] * V[j * p + k, r_prime] * W_new[k * n + i, r_prime]
                        if abs(val - 1) > 1e-5:
                            return None
                            
            return W_new
        except:
            return None

    best_W = W
    best_rank = rank
    
    # Try removing each term
    for r in range(rank):
        if time.time() > deadline:
            break
        W_candidate = try_remove_term(r)
        if W_candidate is not None:
            best_W = W_candidate
            best_rank = rank - 1
            break  # Found a rank-35 solution, stop searching for rank-35
            
    # If we found a rank-35 solution, we could try to remove another term to get 34.
    # But given time constraints, 35 is a significant jump.
    
    if best_rank < rank:
        # We need to trim U and V as well?
        # No, U and V are the same, just W changed.
        # But we should zero out the column in U and V for the removed term?
        # The rank is defined by the number of non-zero terms.
        # If W[:, r_remove] is all zeros, the term is effectively removed.
        # So we just return U, V, W with the new W.
        return {"U": U.tolist(), "V": V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
