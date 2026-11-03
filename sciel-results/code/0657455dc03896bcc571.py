"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
1. Construct the known rank-55 decomposition using Strassen's algorithm 
   on the 2x2 sub-blocks (j=1,2) and naive multiplication for j=0.
2. Verify the decomposition.
3. Attempt to find a rank-54 decomposition by checking if any term in the 
   rank-55 solution is a linear combination of the others, allowing it 
   to be dropped.
"""

import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _fast_verify(U, V, W, n, m, p):
    """Fast verification using tensor flattening and matrix multiply."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build target tensor
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Reconstruct
    T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    
    return np.allclose(T, target, atol=1e-9)


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce rank from len(U[0]) to target_rank by finding linear dependencies.
    Specifically, checks if any term is a linear combination of the others.
    """
    current_rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    iterations_needed = current_rank - target_rank
    for iter_num in range(iterations_needed):
        if time.time() > deadline:
            return None
        
        found_reduction = False
        # Try to eliminate each term
        for r_elim in range(len(U[0])):
            if time.time() > deadline:
                return None
            
            # Build the system A x = b
            # We want to express term r_elim as sum_{s != r_elim} c_s * term_s
            # For each (a,b,c) where T[a,b,c] != 0:
            # U[a,r_elim]*V[b,r_elim]*W[c,r_elim] = sum_{s != r_elim} c_s * U[a,s]*V[b,s]*W[c,s]
            
            # This is a linear system in c_s.
            # Number of equations: n*m*p (only non-zero ones matter, but all are 1 in target)
            # Actually, the target tensor T is fixed.
            # The current decomposition sums to T.
            # So T = term_r_elim + sum_{s != r_elim} term_s.
            # We want to find c_s such that term_r_elim = sum_{s != r_elim} c_s term_s.
            # Then T = sum_{s != r_elim} (1+c_s) term_s.
            
            # Let's solve for c_s.
            # Equations: For all a,b,c:
            # U[a,r_elim]*V[b,r_elim]*W[c,r_elim] = sum_{s != r_elim} c_s U[a,s]V[b,s]W[c,s]
            
            # This is an overdetermined system (60 eqs, 54 unknowns).
            # We use least squares.
            
            # Construct matrix M where M[eq_idx, s_idx] = U[a,s]*V[b,s]*W[c,s]
            # and vector b where b[eq_idx] = U[a,r_elim]*V[b,r_elim]*W[c,r_elim]
            
            # We only need to consider (a,b,c) where the target is non-zero?
            # Actually, the equation must hold for all (a,b,c).
            
            # To save time, we can subsample or use all.
            # 60 equations is small.
            
            eqs = []
            rhs = []
            
            # Indices for a, b, c
            # a in 0..5, b in 0..29, c in 0..19
            # But only 60 are non-zero in the target tensor?
            # No, the tensor T has 1s at specific positions.
            # The equation must hold for ALL positions?
            # Yes, because T is defined everywhere.
            # But T is zero elsewhere.
            # So we must satisfy:
            # If T[a,b,c] = 1: U[a,r]V[b,r]W[c,r] + sum c_s U[a,s]V[b,s]W[c,s] = 1
            # If T[a,b,c] = 0: U[a,r]V[b,r]W[c,r] + sum c_s U[a,s]V[b,s]W[c,s] = 0
            
            # Since we know T = sum all terms, we can just solve for the non-zero part?
            # No, we must maintain the zero parts too.
            # However, if we find c_s that satisfies the non-zero parts, does it satisfy the zero parts?
            # Not necessarily.
            
            # Let's build the full system.
            # 60 * 30 * 20 = 36000 entries? No.
            # a: 6, b: 30, c: 20. Total 3600.
            # This is too large for a dense linear solver in Python quickly?
            # 3600 x 54 matrix.
            
            # Let's just use the 60 non-zero entries first as a heuristic.
            # If it works there, we verify the full tensor.
            
            non_zero_indices = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        non_zero_indices.append((a,b,c))
            
            A_mat = np.zeros((len(non_zero_indices), current_rank - 1))
            b_vec = np.zeros(len(non_zero_indices))
            
            s_idx = 0
            for s in range(current_rank):
                if s == r_elim:
                    continue
                for eq_idx, (a,b,c) in enumerate(non_zero_indices):
                    A_mat[eq_idx, s_idx] = U[a][s] * V[b][s] * W[c][s]
                s_idx += 1
            
            for eq_idx, (a,b,c) in enumerate(non_zero_indices):
                b_vec[eq_idx] = U[a][r_elim] * V[b][r_elim] * W[c][r_elim]
                
            # Solve A_mat x = b_vec
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except Exception:
                continue
                
            # Check residual
            if residuals and residuals[0] < 1e-9:
                # It's a solution!
                # Update the matrices
                # New term s has coefficient (1 + x[s_idx])
                # We drop term r_elim.
                
                new_U = []
                new_V = []
                new_W = []
                
                s_idx = 0
                for s in range(current_rank):
                    if s == r_elim:
                        continue
                    scale = 1.0 + x[s_idx]
                    # Check if scale is close to 0? If so, we can drop it too?
                    # But we just want to reduce rank by 1.
                    
                    # We need to ensure entries are integer/half-integer.
                    # x is likely irrational.
                    # If x is not "nice", we can't use it directly.
                    # However, if the system is exact, x might be rational.
                    
                    new_U.append([U[a][s] * scale for a in range(len(U))])
                    new_V.append([V[b][s] * scale for b in range(len(V))])
                    new_W.append([W[c][s] * scale for c in range(len(W))])
                    s_idx += 1
                    
                # Verify if new entries are valid (integer/half-integer)
                if _is_valid_entries(new_U, new_V, new_W):
                    if _fast_verify(new_U, new_V, new_W, n, m, p):
                        U, V, W = new_U, new_V, new_W
                        found_reduction = True
                        break
                        
        if not found_reduction:
            return None
            
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin: 150s limit, keep 20s buffer

    # Start with rank-55
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_fast_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try to find rank 54 using null-space search
    result = _try_reduce_rank(U_best, V_best, W_best, n, m, p, 54, deadline)
    if result is not None:
        U_test, V_test, W_test = result
        if _fast_verify(U_test, V_test, W_test, n, m, p) and _is_valid_entries(U_test, V_test, W_test):
            U_best, V_best, W_best = U_test, V_test, W_test
            best_rank = 54

    return {"U": U_best, "V": V_best, "W": W_best}
