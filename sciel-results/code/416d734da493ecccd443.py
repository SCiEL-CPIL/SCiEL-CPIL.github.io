"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Construct the rank-35 Strassen baseline, then attempt to reduce rank
by exploiting cross-block linear dependencies via structured search and
Gaussian elimination on the bilinear map coefficient system.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0


def _build_strassen_baseline():
    """Build the safe rank-35 Strassen baseline."""
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-multiplication decomposition for 2x2x2
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    strassen_A = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: (e+h)
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: (f-h)
        [-1, 0, 1, 0], # P4: (g-e)
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: (e+f)
        [0, 0, 1, 1]   # P7: (g+h)
    ]
    
    # W contributions for each P_r to [C00, C01, C10, C11]
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    w_contrib = [
        [1, 0, 0, 0],   # P1 -> C00
        [0, 0, 1, -1],  # P2 -> C10, -C11
        [0, 1, 0, 1],   # P3 -> C01, C11
        [1, 0, 1, 0],   # P4 -> C00, C10
        [-1, 1, 0, 0],  # P5 -> -C00, C01
        [0, 0, 0, 1],   # P6 -> C11
        [1, 0, 0, 0]    # P7 -> C00
    ]
    
    for b in range(num_blocks):
        base_col = b * 7
        for r in range(7):
            col = base_col + r
            
            # Fill U
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]
            
            # Fill V
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]
            
            # Fill W
            W[(2 * b) * 2 + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * 2 + 0][col] = w_contrib[r][1]
            W[(2 * b) * 2 + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * 2 + 1][col] = w_contrib[r][3]
    
    return U, V, W, rank


def _verify_decomposition(U, V, W, rank, n, m, p):
    """Verify that the decomposition exactly performs matrix multiplication."""
    rng = np.random.default_rng(12345)
    A_test = rng.integers(-5, 6, size=(n, m))
    B_test = rng.integers(-5, 6, size=(m, p))
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    C_computed = np.zeros(p * n)
    for r in range(rank):
        uA = np.dot(U_arr[:, r], A_flat)
        vB = np.dot(V_arr[:, r], B_flat)
        C_computed += uA * vB * W_arr[:, r]
    
    C_expected_reorder = np.zeros(p * n)
    for i in range(n):
        for k in range(p):
            C_expected_reorder[k * n + i] = C_expected[i, k]
    
    return np.allclose(C_computed, C_expected_reorder, atol=1e-9)


def _try_cross_block_reduction(U, V, W, rank, n, m, p, deadline):
    """Try to reduce rank by finding cross-block linear dependencies.
    
    The 5 Strassen blocks are independent. We check if any subset of rank-1
    terms from different blocks can be merged into fewer terms via linear
    combinations, exploiting the fact that the U-factors are shared across
    blocks (same A input).
    
    Key insight: All blocks use the same U structure (4-dim input from A).
    If two blocks have U-factors that are linearly dependent, we might merge.
    """
    nm = n * m  # 4
    mp = m * p  # 20
    pn = p * n  # 20
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The U matrix is 4 x rank. For rank-35 Strassen, columns 0-6 are block 0,
    # 7-13 are block 1, etc. The U columns for different blocks are identical
    # (same Strassen pattern), so U[:, b*7+t] = U[:, t] for all b.
    # This means the 35 U-columns span only a 7-dimensional space!
    
    # Let's verify: check rank of U
    u_rank = np.linalg.matrix_rank(U_arr, tol=1e-10)
    
    if u_rank < rank:
        # There's redundancy in U. We can potentially reduce the total rank.
        # The key question: can we re-express the decomposition with fewer
        # terms by exploiting the low-rank structure of U?
        
        # Approach: Since U has rank u_rank, we can write U = U_low @ Q
        # where U_low is 4 x u_rank and Q is u_rank x rank.
        # Then T = sum_r U[:,r] V[:,r] W[:,r] = sum_r (U_low @ Q)[:,r] V[:,r] W[:,r]
        # = sum_{a=1}^{u_rank} U_low[:,a] * (sum_{r: Q[a,r]!=0} Q[a,r] V[:,r] W[:,r])
        # But this doesn't directly reduce rank unless we can restructure.
        
        # Better approach: Use the fact that the bilinear map is linear in each
        # factor. Set up a linear system to find a lower-rank decomposition.
        
        # For target_rank < rank, try to find U', V', W' with target_rank columns
        # such that the tensor matches.
        
        for target_rank in range(max(u_rank, rank - 5), rank):
            if time.time() > deadline - 20:
                break
            
            success = _try_als_decomposition_structured(
                n, m, p, target_rank, U_arr, V_arr, W_arr, deadline
            )
            if success:
                # Verify
                if _verify_decomposition(U_arr, V_arr, W_arr, target_rank, n, m, p):
                    return U_arr, V_arr, W_arr, target_rank
    
    return None, None, None, None


def _try_als_decomposition_structured(n, m, p, target_rank, U_init, V_init, W_init, deadline):
    """ALS with structured initialization exploiting block symmetry."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_vec_u = T.reshape(nm, -1)
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, -1)
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, -1)
    
    best_err = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 3
    max_iters = 80
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        rng = np.random.default_rng(777 + restart * 137 + target_rank)
        
        # Initialize with structure: U has repeated block patterns
        if restart == 0:
            # Use the Strassen pattern with merged blocks
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Assign columns to blocks
            cols_per_block = target_rank // 5
            remainder = target_rank % 5
            
            for b in range(5):
                start = sum(cols_per_block + (1 if i < remainder else 0) for i in range(b))
                end = start + cols_per_block + (1 if b < remainder else 0)
                num_cols = end - start
                
                if num_cols >= 7:
                    # Use full Strassen for this block
                    for t in range(7):
                        col = start + t
                        U[:, col] = U_init[:, t]  # Strassen U pattern
                        V[:, col] = V_init[:, b * 7 + t]
                        W[:, col] = W_init[:, b * 7 + t]
                elif num_cols > 0:
                    # Use partial Strassen + random
                    for t in range(min(num_cols, 7)):
                        col = start + t
                        U[:, col] = U_init[:, t]
                        V[:, col] = V_init[:, b * 7 + t]
                        W[:, col] = W_init[:, b * 7 + t]
                    for t in range(7, num_cols):
                        col = start + t
                        U[:, col] = rng.normal(size=nm)
                        V[:, col] = rng.normal(size=mp)
                        W[:, col] = rng.normal(size=pn)
        else:
            U = rng.normal(size=(nm, target_rank))
            V = rng.normal(size=(mp, target_rank))
            W = rng.normal(size=(pn, target_rank))
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Update U
            C_mat = np.einsum('jr,kr->rjk', V, W)
            C_flat = C_mat.reshape(target_rank, -1)
            try:
                A_mat = C_flat @ C_flat.T
                U = T_vec_u @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            # Update V
            C_mat = np.einsum('ir,kr->rik', U, W)
            C_flat = C_mat.reshape(target_rank, -1)
            try:
                A_mat = C_flat @ C_flat.T
                V = T_vec_v @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            # Update W
            C_mat = np.einsum('ir,jr->rij', U, V)
            C_flat = C_mat.reshape(target_rank, -1)
            try:
                A_mat = C_flat @ C_flat.T
                W = T_vec_w @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W, True
    
    return None, None, None, False


def _try_als_decomposition(T, n, m, p, target_rank, max_iters=200, tol=1e-10):
    """Try to find a rank-`target_rank` decomposition using ALS."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    T_vec_u = T.reshape(nm, -1)
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, -1)
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, -1)
    
    best_err = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    for restart in range(5):
        rng = np.random.default_rng(42 + restart * 1000 + target_rank)
        U = rng.normal(size=(nm, target_rank))
        V = rng.normal(size=(mp, target_rank))
        W = rng.normal(size=(pn, target_rank))
        
        for iteration in range(max_iters):
            C_mat = np.einsum('jr,kr->rjk', V, W)
            C_flat = C_mat.reshape(target_rank, -1)
            try:
                A_mat = C_flat @ C_flat.T
                U = T_vec_u @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            C_mat = np.einsum('ir,kr->rik', U, W)
            C_flat = C_mat.reshape(target_rank, -1)
            try:
                A_mat = C_flat @ C_flat.T
                V = T_vec_v @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            C_mat = np.einsum('ir,jr->rij', U, V)
            C_flat = C_mat.reshape(target_rank, -1)
            try:
                A_mat = C_flat @ C_flat.T
                W = T_vec_w @ C_flat.T @ np.linalg.pinv(A_mat)
            except:
                pass
            
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < tol:
                break
        
        if best_err < tol:
            break
    
    if best_err < tol:
        return best_U, best_V, best_W, True
    
    return None, None, None, False


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _DEADLINE - 10  # 10s safety margin
    
    # Start with the safe baseline
    U, V, W, rank = _build_strassen_baseline()
    
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = 1
    nm = n * m
    mp = m * p
    pn = p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try cross-block reduction first (exploits structure)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_red, V_red, W_red, rank_red = _try_cross_block_reduction(
        U, V, W, rank, n, m, p, deadline
    )
    if U_red is not None:
        U, V, W, rank = U_red, V_red, W_red, rank_red
    
    # Try general ALS for remaining ranks
    if time.time() < deadline - 30:
        for target_rank in range(rank - 1, 29, -1):
            if time.time() > deadline - 25:
                break
            
            U_try, V_try, W_try, success = _try_als_decomposition(
                T, n, m, p, target_rank, max_iters=60, tol=1e-10
            )
            
            if success:
                if _verify_decomposition(U_try.tolist(), V_try.tolist(), W_try.tolist(),
                                         target_rank, n, m, p):
                    rank = target_rank
                    U = U_try.tolist()
                    V = V_try.tolist()
                    W = W_try.tolist()
                    break
    
    # Final verification
    if not _verify_decomposition(U, V, W, rank, n, m, p):
        U, V, W, rank = _build_strassen_baseline()
    
    return {"U": U, "V": V, "W": W}
