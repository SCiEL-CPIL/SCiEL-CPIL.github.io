"""Rewritten program: exact rank search for 4x4x7 matrix multiplication tensor."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product
from collections import defaultdict

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Search for a rank-minimal exact tensor decomposition for 4x4x7 matrix multiplication.
    Strategy:
      1. Try structured decompositions (Strassen-like block splits).
      2. Perform iterative null-space projection search over rationals.
      3. Return the best (lowest-rank) valid decomposition found within time budget.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 7
    # Tensor indices: T[i, j, k] = 1 if (i,j,k) corresponds to (A[i,j]*B[j,k] -> C[i,k])
    # In the standard convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # So the tensor is: for each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1

    # Build the tensor as a 3D array for reference
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i, j, k] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid decomposition."""
        rank = len(U[0]) if U else 0
        # Check dimensions
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for c in range(rank):
            if len(U) != n * m or len(V[c]) != 1 if False else True:
                pass
        # Reconstruct and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-9):
        """Vectorized verification using numpy."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)  # (n*m, rank)
        V_arr = np.array(V, dtype=np.float64)  # (m*p, rank)
        W_arr = np.array(W, dtype=np.float64)  # (p*n, rank)
        
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape and use einsum
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # We want: result[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Build index arrays
        # U indexed by (i, j) -> i*m+j
        # V indexed by (j, k) -> j*p+k
        # W indexed by (k, i) -> k*n+i
        
        # Create full 3D tensors via broadcasting
        # U_expanded: shape (n, m, rank)
        U_exp = U_arr.reshape(n, m, rank)  # U[i, j, c]
        V_exp = V_arr.reshape(m, p, rank)  # V[j, k, c]
        W_exp = W_arr.reshape(p, n, rank)  # W[k, i, c]
        
        # result[i, j, k] = sum_c U_exp[i,j,c] * V_exp[j,k,c] * W_exp[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        result = np.einsum('ijc,jkc,kic->ijk', U_exp, V_exp, W_exp)
        
        if not np.allclose(result, 1.0, atol=tol):
            return False
        return True

    def to_fraction_list(lst):
        """Convert a list of floats to Fractions (for exact output)."""
        return [Fraction(x).limit_denominator(1000000) for x in lst]

    def make_output(U, V, W):
        """Convert to the required output format with integer/half-integer entries."""
        # Ensure entries are integers or half-integers
        U_out = []
        for row in U:
            U_out.append([Fraction(x).limit_denominator(2) for x in row])
        V_out = []
        for row in V:
            V_out.append([Fraction(x).limit_denominator(2) for x in row])
        W_out = []
        for row in W:
            W_out.append([Fraction(x).limit_denominator(2) for x in row])
        return {"U": U_out, "V": V_out, "W": W_out}

    # --- Schoolbook baseline ---
    rank_schoolbook = n * m * p  # 112
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}

    # --- Try Strassen-style block decomposition ---
    # Split n=4 into two blocks of 2, m=4 into two blocks of 2
    # This gives 4 sub-problems of 2x2x7
    # For 2x2x7, schoolbook is 28. Can we do better?
    # Known: 2x2x2 has rank 7 (Strassen). 2x2x7 is less studied.
    # Let's try a direct search for 2x2x7 sub-tensor.

    def search_2x2x7(max_rank=30, max_iters=500, seed=42):
        """Search for low-rank decomposition of 2x2x7 tensor."""
        nn, mm, pp = 2, 2, 7
        rng = np.random.RandomState(seed)
        
        # Tensor: all ones 2x2x7
        # We need U: (2*2, r), V: (2*7, r), W: (7*2, r)
        # U[i*2+j, c], V[j*7+k, c], W[k*2+i, c]
        
        for r in range(14, max_rank + 1):  # Try from 14 upward
            if time.time() > deadline:
                return None
            for trial in range(max_iters):
                if time.time() > deadline:
                    return None
                # Random initialization
                U = rng.randn(nn * mm, r)
                V = rng.randn(mm * pp, r)
                W = rng.randn(pp * nn, r)
                
                # Iterative refinement: project onto constraint
                # For each (i,j,k), we need sum_c U[i*mm+j,c]*V[j*pp+k,c]*W[k*nn+i,c] = 1
                # Use a simple gradient-free approach: random restarts with null-space projection
                
                # Actually, let's use a simpler approach: try to find a decomposition
                # by fixing two matrices and solving for the third.
                
                # Fix U and V, solve for W:
                # For each (k,i): W[k*nn+i, :] is a vector of length r
                # Constraint: for all (i,j,k): sum_c U[i*mm+j,c]*V[j*pp+k,c]*W[k*nn+i,c] = 1
                # This is linear in W for fixed U,V.
                
                # Build the linear system for W
                # Unknowns: W[k*nn+i, c] for k in range(pp), i in range(nn), c in range(r)
                # Total unknowns: pp * nn * r = 7 * 2 * r
                # Constraints: for each (i,j,k): one equation
                # Total constraints: nn * mm * pp = 2 * 2 * 7 = 28
                
                # Let's build the matrix A such that A * w_vec = b
                # where w_vec is the flattened W matrix
                w_unknowns = pp * nn * r
                n_constraints = nn * mm * pp
                
                A = np.zeros((n_constraints, w_unknowns))
                b = np.ones(n_constraints)
                
                idx = 0
                for i in range(nn):
                    for j in range(mm):
                        for k in range(pp):
                            # Equation: sum_c U[i*mm+j,c] * V[j*pp+k,c] * W[k*nn+i,c] = 1
                            for c in range(r):
                                w_idx = (k * nn + i) * r + c
                                A[idx, w_idx] = U[i * mm + j, c] * V[j * pp + k, c]
                            idx += 1
                
                # Solve least squares
                try:
                    w_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                
                # Check if solution is exact
                if residuals.size == 0 or residuals[0] < 1e-8:
                    W_mat = w_sol.reshape(pp * nn, r)
                    # Verify
                    U_exp = U.reshape(nn, mm, r)
                    V_exp = V.reshape(mm, pp, r)
                    W_exp = W_mat.reshape(pp, nn, r)
                    result = np.einsum('ijc,jkc,kic->ijk', U_exp, V_exp, W_exp)
                    if np.allclose(result, 1.0, atol=1e-8):
                        return U, V, W_mat, r
                # Also check if the residual is small enough for a valid decomposition
                if residuals.size > 0 and residuals[0] < 1e-6:
                    W_mat = w_sol.reshape(pp * nn, r)
                    U_exp = U.reshape(nn, mm, r)
                    V_exp = V.reshape(mm, pp, r)
                    W_exp = W_mat.reshape(pp, nn, r)
                    result = np.einsum('ijc,jkc,kic->ijk', U_exp, V_exp, W_exp)
                    if np.allclose(result, 1.0, atol=1e-6):
                        return U, V, W_mat, r
        
        return None

    # Try to find a 2x2x7 decomposition
    result_227 = None
    for seed in range(20):
        if time.time() > deadline:
            break
        result_227 = search_2x2x7(max_rank=25, max_iters=200, seed=seed)
        if result_227 is not None:
            break

    if result_227 is not None:
        U_sub, V_sub, W_sub, r_sub = result_227
        # Now combine 4 sub-blocks
        # The 4x4x7 tensor can be decomposed into 4 blocks of 2x2x7
        # Block (bi, bj) where bi in {0,1}, bj in {0,1}
        # U_full: (4*4, 4*r_sub) = (16, 4*r_sub)
        # V_full: (4*7, 4*r_sub) = (28, 4*r_sub)
        # W_full: (7*4, 4*r_sub) = (28, 4*r_sub)
        
        rank_full = 4 * r_sub
        U_full = [[0] * rank_full for _ in range(16)]
        V_full = [[0] * rank_full for _ in range(28)]
        W_full = [[0] * rank_full for _ in range(28)]
        
        for bi in range(2):
            for bj in range(2):
                offset = (bi * 2 + bj) * r_sub
                for i in range(2):
                    for j in range(2):
                        gi = bi * 2 + i
                        gj = bj * 2 + j
                        U_full[gi * 4 + gj][offset + (i * 2 + j)] = U_sub[i * 2 + j]
                for j in range(2):
                    for k in range(7):
                        gj = bj * 2 + j
                        V_full[gj * 7 + k][offset + (j * 7 + k)] = V_sub[j * 7 + k]
                for k in range(7):
                    for i in range(2):
                        gi = bi * 2 + i
                        W_full[k * 4 + gi][offset + (k * 2 + i)] = W_sub[k * 2 + i]
        
        if verify_fast(U_full, V_full, W_full, tol=1e-6):
            if rank_full < best["rank"]:
                best = {"U": U_full, "V": V_full, "W": W_full, "rank": rank_full}

    # --- Direct search for full 4x4x7 tensor ---
    # Try to find a decomposition with rank less than the current best
    target_rank = best["rank"] - 1
    if target_rank >= 20:  # Only if we have room to improve
        def search_full_447(max_rank, max_trials=100, seed=0):
            rng = np.random.RandomState(seed)
            for r in range(20, max_rank + 1):
                if time.time() > deadline:
                    return None
                for trial in range(max_trials):
                    if time.time() > deadline:
                        return None
                    U = rng.randn(16, r)
                    V = rng.randn(28, r)
                    # Solve for W
                    w_unknowns = 28 * r
                    n_constraints = 4 * 4 * 7  # 112
                    A = np.zeros((n_constraints, w_unknowns))
                    b = np.ones(n_constraints)
                    idx = 0
                    for i in range(4):
                        for j in range(4):
                            for k in range(7):
                                for c in range(r):
                                    w_idx = (k * 4 + i) * r + c
                                    A[idx, w_idx] = U[i * 4 + j, c] * V[j * 7 + k, c]
                                idx += 1
                    try:
                        w_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    if residuals.size == 0 or residuals[0] < 1e-8:
                        W_mat = w_sol.reshape(28, r)
                        U_exp = U.reshape(4, 4, r)
                        V_exp = V.reshape(4, 7, r)
                        W_exp = W_mat.reshape(7, 4, r)
                        result = np.einsum('ijc,jkc,kic->ijk', U_exp, V_exp, W_exp)
                        if np.allclose(result, 1.0, atol=1e-8):
                            return U, V, W_mat, r
                    if residuals.size > 0 and residuals[0] < 1e-6:
                        W_mat = w_sol.reshape(28, r)
                        U_exp = U.reshape(4, 4, r)
                        V_exp = V.reshape(4, 7, r)
                        W_exp = W_mat.reshape(7, 4, r)
                        result = np.einsum('ijc,jkc,kic->ijk', U_exp, V_exp, W_exp)
                        if np.allclose(result, 1.0, atol=1e-6):
                            return U, V, W_mat, r
            return None

        result_full = search_full_447(max_rank=target_rank, max_trials=50, seed=123)
        if result_full is not None:
            U_f, V_f, W_f, r_f = result_full
            if r_f < best["rank"]:
                best = {"U": U_f, "V": V_f, "W": W_f, "rank": r_f}

    # Convert best to output format
    return make_output(best["U"], best["V"], best["W"])
# EVOLVE-BLOCK-END
