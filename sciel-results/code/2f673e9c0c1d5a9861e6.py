"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill
    
    # Strategy: Use Strassen for the 2x4 * 4x6 multiplication.
    # Rank 42 decomposition.
    # A (2x4) = [A_left | A_right] (each 2x2)
    # B (4x6) = [B_top; B_bot] (each 2x6)
    # C = A_left * B_top + A_right * B_bot
    # B_top and B_bot are split into 3 column blocks of 2x2.
    # Each block contribution is sum of two 2x2 multiplications.
    # Rank of 2x2 multiplication is 7 (Strassen).
    # Total rank = 3 blocks * (7 + 7) = 42.
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2*2x2:
    # M1 = (P00+P11)*(Q00+Q11)
    # M2 = (P10+P11)*Q00
    # M3 = P00*(Q01-Q11)
    # M4 = P11*(Q10-Q00)
    # M5 = (P00+P01)*Q11
    # M6 = (P10-P00)*(Q00+Q01)
    # M7 = (P01-P11)*(Q10+Q11)
    #
    # P, Q, R are 2x2 matrices.
    # P indices: (0,0), (0,1), (1,0), (1,1)
    # Q indices: (0,0), (0,1), (1,0), (1,1)
    # R indices: (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        # (U_factors, V_factors, W_factors)
        # U_factors: list of (p_idx, coeff)
        # V_factors: list of (q_idx, coeff)
        # W_factors: list of (r_idx, coeff)
        # M1
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # M2
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # M3
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # M4
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # M5
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # M6
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
        # M7
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    col_idx = 0
    for source in range(2):  # 0=top (A_left, B_top), 1=bot (A_right, B_bot)
        for c in range(3):  # column block 0,1,2
            for t in range(7):  # 7 Strassen terms
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                # Map P indices to U indices:
                # P is 2x2, P[pi, pj]
                # If source=0: P[pi,pj] = A[pi, pj] -> U[pi*4 + pj, col_idx]
                # If source=1: P[pi,pj] = A[pi, pj+2] -> U[pi*4 + (pj+2), col_idx]
                for (pi, pj), coeff in u_factors:
                    u_row = pi * 4 + (pj + 2*source)
                    U[u_row, col_idx] += coeff
                
                # Map Q indices to V indices:
                # Q is 2x2, Q[qi, qj]
                # Q[qi,qj] = B[qi + 2*source, 2*c + qj]
                # V index: (qi + 2*source) * 6 + (2*c + qj)
                for (qi, qj), coeff in v_factors:
                    v_row = (qi + 2*source) * 6 + (2*c + qj)
                    V[v_row, col_idx] += coeff
                
                # Map R indices to W indices:
                # R is 2x2, R[ri, rj]
                # R[ri,rj] contributes to C[ri, 2*c + rj]
                # W index: (2*c + rj)*2 + ri
                for (ri, rj), coeff in w_factors:
                    w_row = (2*c + rj) * 2 + ri
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
    
    # Verify the decomposition is correct
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U_mat)
        V_arr = np.array(V_mat)
        W_arr = np.array(W_mat)
        
        rank_val = U_arr.shape[1]
        
        # Compute T from U, V, W
        # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
        T_computed = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
        
        # Build target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i_A in range(n):
            for j in range(m):
                for k_B in range(p):
                    i = i_A * m + j
                    j_idx = j * p + k_B
                    k = k_B * n + i_A
                    T_target[i, j_idx, k] = 1.0
        
        # Check if they match
        diff = np.abs(T_computed - T_target)
        max_diff = np.max(diff)
        return max_diff < tol
    
    # Verify our decomposition
    if not verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
        raise ValueError("Decomposition verification failed")
    
    # Check if we have time to attempt rank reduction
    elapsed = time.time() - start_time
    if elapsed < 60:  # We have at least 60 seconds left
        # Try to reduce rank by finding linear dependencies among the columns
        # This is a heuristic approach: try to find a smaller rank decomposition
        # by exploiting the structure of the Strassen algorithm
        
        # The rank-42 decomposition may have redundant terms.
        # We can try to find the actual rank of the tensor by computing
        # the rank of the associated matrices.
        
        # For a tensor decomposition T = sum_r U_r * V_r * W_r,
        # the rank is at most the number of terms, but may be less
        # if there are linear dependencies.
        
        # One approach: try to find a smaller set of terms that still
        # spans the same tensor.
        
        # For now, we'll keep the rank-42 decomposition since it's verified
        # and correct. Further optimization would require more sophisticated
        # tensor rank reduction algorithms.
        
        pass
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
