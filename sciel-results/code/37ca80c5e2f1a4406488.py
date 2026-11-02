"""
Program to find a rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Uses correct W[k*n+i] convention with two Strassen 2x2x2 blocks for rank 14.
Includes a structured search for potential rank 13 by attempting to merge terms.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=2, p=4):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W index convention: W[k*n+i] where k=output column, i=output row.
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for valid (i,j,k)
    and 0 for invalid triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid multiplication terms equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    
    # Check all invalid triples equal 0
    for i_idx in range(n * m):
        for j_idx in range(m * p):
            for k_idx in range(p * n):
                # Decode indices
                a = i_idx // m
                b = i_idx % m
                b_prime = j_idx // p
                c = j_idx % p
                c_prime = k_idx // n
                a_prime = k_idx % n
                
                # Valid if a == a_prime and b == b_prime
                if a == a_prime and b == b_prime:
                    continue
                
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i_idx, r] * V[j_idx, r] * W[k_idx, r]
                if abs(t_val) > 1e-9:
                    return False
    
    return True

def _build_rank14():
    """
    Construct the rank-14 decomposition using two Strassen blocks.
    W index convention: W[k*n+i] where k=output column, i=output row.
    C[0,0]->W[0], C[1,0]->W[1], C[0,1]->W[2], C[1,1]->W[3],
    C[0,2]->W[4], C[1,2]->W[5], C[0,3]->W[6], C[1,3]->W[7]
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Left Block (Columns 0, 1) - Strassen 2x2x2
    r = 0
    U[0][r] = 1; U[3][r] = 1
    V[0][r] = 1; V[5][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    r = 1
    U[2][r] = 1; U[3][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    
    r = 2
    U[0][r] = 1
    V[1][r] = 1; V[5][r] = -1
    W[2][r] = 1; W[3][r] = 1
    
    r = 3
    U[3][r] = 1
    V[4][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[5][r] = 1
    W[0][r] = -1; W[2][r] = 1
    
    r = 5
    U[2][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    
    r = 6
    U[1][r] = 1; U[3][r] = -1
    V[4][r] = 1; V[5][r] = 1
    W[0][r] = 1
    
    # Right Block (Columns 2, 3) - Strassen 2x2x2
    r = 7
    U[0][r] = 1; U[3][r] = 1
    V[2][r] = 1; V[7][r] = 1
    W[4][r] = 1; W[7][r] = 1
    
    r = 8
    U[2][r] = 1; U[3][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    
    r = 9
    U[0][r] = 1
    V[3][r] = 1; V[7][r] = -1
    W[6][r] = 1; W[7][r] = 1
    
    r = 10
    U[3][r] = 1
    V[6][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    
    r = 11
    U[0][r] = 1; U[1][r] = 1
    V[7][r] = 1
    W[4][r] = -1; W[6][r] = 1
    
    r = 12
    U[2][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    
    r = 13
    U[1][r] = 1; U[3][r] = -1
    V[6][r] = 1; V[7][r] = 1
    W[4][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_merge_rank14_to_13(U14, V14, W14, deadline):
    """
    Try to find a rank-13 decomposition by attempting to eliminate one rank term
    from the rank-14 solution. For each pair of terms, check if their combined
    contribution can be absorbed into the remaining terms.
    This is a heuristic: we try removing each term and see if the residual
    can be expressed as a combination of the others with half-integer coefficients.
    """
    n, m, p = 2, 2, 4
    rank14 = 14
    target_rank = 13
    
    # For each term to remove, check if the remaining 13 terms still span the tensor
    # Strategy: for each term r_remove, check if the vector (U_col, V_col, W_col)
    # is in the span of the other terms' contributions to the tensor.
    
    # Build the tensor T from the rank-14 decomposition
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We want to check if removing term r_remove still gives correct T.
    
    # For each candidate removal, compute the residual tensor that term r_remove contributes
    # and check if it can be expressed as a linear combination of the other 13 terms.
    
    U = np.array(U14, dtype=float)
    V = np.array(V14, dtype=float)
    W = np.array(W14, dtype=float)
    
    for r_remove in range(rank14):
        if time.time() > deadline - 15:
            break
        
        # Compute the contribution of term r_remove to the full tensor
        # and check if the remaining terms can compensate.
        
        # Build the 13-term decomposition (remove column r_remove)
        cols = [c for c in range(rank14) if c != r_remove]
        U13 = U[:, cols]
        V13 = V[:, cols]
        W13 = W[:, cols]
        
        # Check if U13, V13, W13 still decompose the tensor
        if _verify(U13.tolist(), V13.tolist(), W13.tolist(), n, m, p):
            return {"U": U13.tolist(), "V": V13.tolist(), "W": W13.tolist()}
    
    return None

def _try_structured_rank13(deadline):
    """
    Structured search for rank-13: try small perturbations of the rank-14 solution.
    For each pair of terms, try to merge them by adjusting coefficients.
    Also try adding a new term with half-integer coefficients that replaces two terms.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    
    # Strategy 1: Try to find if any single term in the rank-14 solution is redundant
    # (i.e., removing it still gives a valid decomposition)
    base = _build_rank14()
    result = _try_merge_rank14_to_13(base["U"], base["V"], base["W"], deadline)
    if result is not None:
        return result
    
    # Strategy 2: Try replacing two terms with one new term
    # For each pair (r1, r2), check if their combined contribution can be
    # expressed as a single rank-1 term with half-integer coefficients.
    U = np.array(base["U"], dtype=float)
    V = np.array(base["V"], dtype=float)
    W = np.array(base["W"], dtype=float)
    rank14 = 14
    
    for r1 in range(rank14):
        if time.time() > deadline - 10:
            break
        for r2 in range(r1 + 1, rank14):
            # The combined contribution of terms r1 and r2 is:
            # sum over i,j,k: (U[i,r1]*V[j,r1]*W[k,r1] + U[i,r2]*V[j,r2]*W[k,r2])
            # We want to express this as U'[i]*V'[j]*W'[k] for some U', V', W'
            # This is generally not possible, but we can try simple cases.
            
            # Try: U'[i] = U[i,r1] + U[i,r2], V'[j] = V[j,r1] + V[j,r2], W'[k] = W[k,r1] + W[k,r2]
            # This won't work in general, but check if it does.
            U_new = U[:, r1] + U[:, r2]
            V_new = V[:, r1] + V[:, r2]
            W_new = W[:, r1] + W[:, r2]
            
            # Build the 13-term decomposition with the new term
            cols = [c for c in range(rank14) if c not in (r1, r2)]
            U13 = np.column_stack([U[:, cols], U_new])
            V13 = np.column_stack([V[:, cols], V_new])
            W13 = np.column_stack([W[:, cols], W_new])
            
            if _verify(U13.tolist(), V13.tolist(), W13.tolist(), n, m, p):
                return {"U": U13.tolist(), "V": V13.tolist(), "W": W13.tolist()}
            
            # Try with subtraction: U'[i] = U[i,r1] - U[i,r2], etc.
            U_new = U[:, r1] - U[:, r2]
            V_new = V[:, r1] - V[:, r2]
            W_new = W[:, r1] - W[:, r2]
            
            U13 = np.column_stack([U[:, cols], U_new])
            V13 = np.column_stack([V[:, cols], V_new])
            W13 = np.column_stack([W[:, cols], W_new])
            
            if _verify(U13.tolist(), V13.tolist(), W13.tolist(), n, m, p):
                return {"U": U13.tolist(), "V": V13.tolist(), "W": W13.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Build the known rank-14 solution
    result_14 = _build_rank14()
    
    # Verify the rank-14 solution
    if _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        # Try to find a rank-13 solution with structured search
        result_13 = _try_structured_rank13(deadline)
        if result_13 is not None:
            return result_13
        return result_14
    
    # Fallback to schoolbook rank 16
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
