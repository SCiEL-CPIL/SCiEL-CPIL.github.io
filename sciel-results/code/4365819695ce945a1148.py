"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return (U, V, W) for Strassen 2x2x2 multiplication, rank 7."""
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # U row (i,j):
    U[0][0] = 1.0; U[0][2] = 1.0; U[0][4] = 1.0; U[0][5] = -1.0; U[0][6] = -1.0
    U[1][2] = 1.0; U[1][4] = 1.0; U[1][5] = -1.0; U[1][6] = -1.0
    U[2][0] = 1.0; U[2][1] = 1.0; U[2][5] = 1.0
    U[3][0] = 1.0; U[3][1] = 1.0; U[3][3] = 1.0; U[3][6] = 1.0
    
    # V row (j,k):
    V[0][0] = 1.0; V[0][1] = 1.0; V[0][5] = 1.0
    V[1][0] = 1.0; V[1][3] = -1.0; V[1][5] = 1.0
    V[2][2] = 1.0; V[2][4] = 1.0; V[2][6] = 1.0
    V[3][0] = 1.0; V[3][3] = 1.0; V[3][4] = 1.0; V[3][6] = 1.0
    
    # W row (k,i):
    W[0][0] = 1.0; W[0][3] = 1.0; W[0][4] = -1.0; W[0][6] = -1.0
    W[1][2] = 1.0; W[1][4] = 1.0
    W[2][1] = 1.0; W[2][3] = 1.0
    W[3][0] = 1.0; W[3][1] = -1.0; W[3][2] = 1.0; W[3][6] = 1.0
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta(i,j,k) for all i,j,k."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if nm > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (i, j, k) == (0, 0, 0) else 0.0
                u_vals = U_arr[i * m + j, :]
                v_vals = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                total = float(np.dot(np.multiply(u_vals, v_vals), w_val))
                if abs(total - expected) > 1e-6:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline, max_iters=5000):
    """
    Try to reduce the rank of a valid decomposition to target_rank using
    a rank-reduction approach: find linear combinations of columns that
    preserve the tensor identity.
    
    Returns (U', V', W') with rank target_rank, or None if not successful.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    current_rank = len(U[0])
    
    if target_rank >= current_rank:
        return None
    
    # We'll try to find a rank-reduction by looking for dependencies.
    # The approach: for each pair of columns, check if we can replace them
    # with a single column (or reduce the rank by 1) while maintaining validity.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] = delta(i,j,k)
    
    # Strategy: try to find a column c that can be expressed as a linear
    # combination of other columns in a way that preserves the tensor.
    # This is equivalent to finding a null vector in the "constraint" space.
    
    # A simpler approach: try random rank-reducing transformations.
    # If we have rank r, we can try to find a 1-dim subspace to eliminate.
    
    # Let's try: for each pair of columns (a, b), try to see if we can
    # replace them with a single column by finding alpha, beta such that
    # the tensor contribution is preserved.
    
    # Actually, let's use a more systematic approach:
    # The tensor is T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is the c-th column of U, etc.
    # We want to find if T can be written with fewer terms.
    
    # Check if the current rank is already minimal by trying to find
    # a linear dependency among the "flattened" tensors.
    
    # For each column c, define T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # The sum over c should equal delta(i,j,k).
    
    # To reduce rank by 1, we need to find coefficients alpha_c such that
    # sum_c alpha_c * T_c = 0, and then we can eliminate one column.
    
    # This is a linear system. Let's set it up.
    # Number of equations: n*m*p = 75
    # Number of unknowns: current_rank
    
    # Build matrix A where A[(i,j,k), c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want to find null space of A.
    
    A = np.zeros((n * m * p, current_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for c in range(current_rank):
                    A[idx, c] = U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
    
    # Find null space
    try:
        _, S, Vt = np.linalg.svd(A)
        # Null space vectors correspond to small singular values
        tol = 1e-10
        null_rank = np.sum(S > tol)
        null_dim = current_rank - null_rank
    except:
        return None
    
    if null_dim <= 0:
        return None
    
    # We found null space vectors. Use them to reduce rank.
    # For each null vector v, we can modify the decomposition.
    
    # Actually, the null space tells us that the current representation is not unique.
    # We can use this to try to simplify the representation.
    
    # Let's try a different approach: iterative rank reduction.
    # Start with the current decomposition and try to reduce rank step by step.
    
    # For rank reduction, we can try to find a column that is a linear combination
    # of others in a specific way.
    
    # Let's try: for each column c, check if it can be removed by adjusting others.
    # This requires solving a system.
    
    # Simpler: try to find if there's a column that's all zeros (already reduced).
    for c in range(current_rank):
        if np.allclose(U_arr[:, c], 0) or np.allclose(V_arr[:, c], 0) or np.allclose(W_arr[:, c], 0):
            # This column doesn't contribute, remove it
            U_new = U_arr[:, [c2 for c2 in range(current_rank) if c2 != c]]
            V_new = V_arr[:, [c2 for c2 in range(current_rank) if c2 != c]]
            W_new = W_arr[:, [c2 for c2 in range(current_rank) if c2 != c]]
            if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                if U_new.shape[1] == target_rank:
                    return U_new.tolist(), V_new.tolist(), W_new.tolist()
                # Recurse
                result = _try_rank_reduction(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, target_rank, deadline, max_iters)
                if result is not None:
                    return result
            return None
    
    # Try to find a linear dependency to reduce rank
    # Use the null space to create a new representation
    
    if null_dim > 0:
        # Get null space basis
        null_basis = Vt[null_rank:]  # shape (null_dim, current_rank)
        
        # Try to use the first null vector to reduce rank
        # If v is in null space, then sum_c v[c] * T_c = 0
        # This means we can add v[c] * T_c to any column's contribution
        
        # Let's try to find a column that can be eliminated.
        # If there exists c with v[c] != 0, we can try to express T_c in terms of others.
        
        for v in null_basis:
            # Find the largest component
            max_idx = np.argmax(np.abs(v))
            if abs(v[max_idx]) < 1e-10:
                continue
            
            # Try to eliminate column max_idx
            # We need to find new U', V', W' such that the sum is preserved
            # without column max_idx
            
            # This is complex. Let's try a simpler heuristic:
            # If the null space has dimension >= 1, the rank might not be minimal,
            # but we need a specific algorithm to reduce it.
            
            pass
    
    # If we couldn't reduce, return None
    return None

def construct():
    t0 = time.time()
    DEADLINE = 120.0
    MARGIN = 10.0
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Define block splits
    # i: 2, 1
    # j: 2, 2, 1
    # k: 2, 2, 1
    
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3], [4]]
    k_blocks = [[0, 1], [2, 3], [4]]
    
    # Pre-compute Strassen factors
    sU, sV, sW = _strassen_2x2x2()
    
    # Calculate total rank
    total_rank = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                ni = len(ib)
                nj = len(jb)
                nk = len(kb)
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    # Initialize matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                ni = len(ib)
                nj = len(jb)
                nk = len(kb)
                
                if ni == 2 and nj == 2 and nk == 2:
                    # Use Strassen
                    for c in range(7):
                        for li in range(2):
                            for lj in range(2):
                                gi = ib[li]
                                gj = jb[lj]
                                U[gi * m + gj][col + c] += sU[li * 2 + lj][c]
                        for lj in range(2):
                            for lk in range(2):
                                gj = jb[lj]
                                gk = kb[lk]
                                V[gj * p + gk][col + c] += sV[lj * 2 + lk][c]
                        for lk in range(2):
                            for li in range(2):
                                gk = kb[lk]
                                gi = ib[li]
                                W[gk * n + gi][col + c] += sW[lk * 2 + li][c]
                    col += 7
                else:
                    # Schoolbook
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                gi = ib[li]
                                gj = jb[lj]
                                gk = kb[lk]
                                U[gi * m + gj][col] = 1.0
                                V[gj * p + gk][col] = 1.0
                                W[gk * n + gi][col] = 1.0
                                col += 1
    
    assert col == total_rank
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce the rank
    current_rank = total_rank
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    # Try to reduce rank step by step
    for target_rank in range(current_rank - 1, max(current_rank - 10, 50), -1):
        if time.time() - t0 > DEADLINE - MARGIN:
            break
        
        result = _try_rank_reduction(best_U, best_V, best_W, n, m, p, target_rank, DEADLINE)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
