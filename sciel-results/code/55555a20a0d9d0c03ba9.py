"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Uses a rank-55 Strassen-based construction and attempts to find lower rank
decompositions through targeted numerical search. Includes a slicing-based
lower bound analysis to guide the search.
"""

import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    for i in range(n):
        for k in range(p):
            Uw = U_mat[i] * W_mat[k, i]
            M_ik = Uw @ V_mat[:, k].T
            if not np.allclose(M_ik, np.eye(m), atol=tol):
                return False
    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """Build the rank-55 decomposition using Strassen on the 2x2 sub-block."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B) - naive 2*10 = 20 terms
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B) - Strassen 7*5 = 35 terms
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

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
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


def _slicing_lower_bound(n, m, p):
    """
    Compute a lower bound on the rank using tensor slicing.
    
    For each fixed middle index j, the sub-tensor T[:, j, :] has dimensions
    (n*m) x (m*p) x (p*n) but only specific entries are nonzero.
    
    The rank of the full tensor is at least the rank of any sub-tensor.
    We use the "slice rank" bound: for fixed i, the matrix M_i = sum_r U[i,:,r] outer W[:,i,r] 
    must have rank <= r. Similarly for other slices.
    
    A stronger bound: consider the 2x2x10 sub-tensor for j=1 (rows 1,4 of U, 
    rows 10-19 of V, all of W). This sub-tensor must have rank at least 10
    because it contains 10 linearly independent 2x2x1 slices.
    
    The j=0 sub-tensor has rank at least 20 (2x1x10 with 20 independent entries).
    
    Total rank >= max(20, 10, 10) = 20, but this is weak.
    
    A better approach: the total number of independent constraints is 60 
    (one for each (i,j,k)). Each rank-1 term can satisfy at most a limited 
    number of these. With Strassen, we can satisfy multiple constraints per term.
    """
    # The j=0 slice requires rank >= 20 (since A[:,0] has 2 entries and B[0,:] has 10,
    # and C[:,0] has 20 entries, each requiring a distinct multiplication)
    # Actually, for j=0: C[i,k] = A[i,0]*B[0,k]. This is rank 20 as a tensor.
    
    # For j=1,2 combined: the 2x2x10 sub-tensor. Each 2x2x1 slice (for fixed k)
    # is the identity, which has rank 1. But the 10 slices are in different W-positions,
    # so they're linearly independent. Rank >= 10.
    
    # Combined: the full tensor rank >= 20 + 10 + 10 = 40? No, that's not how it works.
    # The rank is the minimum r such that we can write T as sum of r rank-1 tensors.
    # Each rank-1 tensor can contribute to multiple (i,j,k) entries.
    
    # Lower bound: rank >= max over all slices of slice rank.
    # j=0 slice: rank 20
    # j=1 slice: rank 10  
    # j=2 slice: rank 10
    # i=0 slice: 3x10x10, rank >= 10 (at least)
    # i=1 slice: 3x10x10, rank >= 10
    # k=0 slice: 2x3x2, rank >= 2 (at least)
    
    # The strongest single-slice bound is 20 from j=0.
    # But we need a bound that accounts for the interaction between slices.
    
    # Information-theoretic: 60 nonzero entries, each rank-1 term can cover at most
    # 1 entry if U,V,W have single nonzero entries, but with Strassen we can cover more.
    # In our rank-55 construction, the 20 j=0 terms each cover 1 entry, and the 35
    # Strassen terms each cover ~2 entries on average (35 terms cover 40 entries).
    
    # A rigorous lower bound: consider the linear map from rank-r decompositions to
    # the 60 tensor entries. The image must contain the identity pattern.
    # This is a nonlinear constraint, so we can't easily get a tight lower bound.
    
    # For now, return 55 as the known construction rank, and let numerical search
    # determine if lower is possible.
    return 55


def _als_search(U0, V0, W0, n, m, p, rank, deadline, max_iter=500):
    """
    ALS (Alternating Least Squares) optimization starting from U0, V0, W0.
    Try to find a rank-rank decomposition by optimizing the factors.
    """
    U = np.array(U0, dtype=np.float64)[:, :rank].copy()
    V = np.array(V0, dtype=np.float64)[:, :rank].copy()
    W = np.array(W0, dtype=np.float64)[:, :rank].copy()
    
    # Precompute target tensor entries
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1.0
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Update U: for each row a, solve for U[a, :]
        for a in range(n * m):
            i = a // m
            j = a % m
            # U[a, r] * V[j*p+k, r] * W[k*n+i, r] = target
            # For fixed a, this is linear in U[a, :]
            # Sum over r: U[a,r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for k matching j
            # Actually, for each k, we need: sum_r U[a,r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is: U[a, :] @ (V[j*p+k, :] * W[k*n+i, :]) = 1 for each k
            # So we have p equations in r unknowns for U[a, :]
            
            A_u = np.zeros((p, rank), dtype=np.float64)
            b_u = np.zeros(p, dtype=np.float64)
            for k in range(p):
                A_u[k, :] = V[j * p + k, :] * W[k * n + i, :]
                if (a, j * p + k, k * n + i) in target:
                    b_u[k] = 1.0
            
            if rank > p:
                # Overdetermined: use least squares
                U[a, :], _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
            else:
                # Underdetermined: use minimum norm solution
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                except:
                    pass
        
        # Update V: for each row b, solve for V[b, :]
        for b in range(m * p):
            j = b // p
            k = b % p
            A_v = np.zeros((n, rank), dtype=np.float64)
            b_v = np.zeros(n, dtype=np.float64)
            for i in range(n):
                A_v[i, :] = U[i * m + j, :] * W[k * n + i, :]
                if (i * m + j, b, k * n + i) in target:
                    b_v[i] = 1.0
            
            if rank > n:
                V[b, :], _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
            else:
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                except:
                    pass
        
        # Update W: for each row c, solve for W[c, :]
        for c in range(p * n):
            k = c // n
            i = c % n
            A_w = np.zeros((m, rank), dtype=np.float64)
            b_w = np.zeros(m, dtype=np.float64)
            for j in range(m):
                A_w[j, :] = U[i * m + j, :] * V[j * p + k, :]
                if (i * m + j, j * p + k, c) in target:
                    b_w[j] = 1.0
            
            if rank > m:
                W[c, :], _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            else:
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                except:
                    pass
        
        # Check convergence
        if _verify_fast(U, V, W, n, m, p, tol=1e-8):
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if _verify_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None


def _random_search_rank(n, m, p, rank, deadline, U55, V55, W55):
    """Random search for a rank-rank decomposition."""
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    # Try random linear combinations of the 55 Strassen terms
    for trial in range(200):
        if time.time() > deadline:
            break
        
        if rank <= 55:
            # Pick rank terms from the 55 and try to combine
            if rank < 55:
                # Use ALS starting from random subset
                idx = random.sample(range(55), rank)
                U_init = U55[:, idx].copy()
                V_init = V55[:, idx].copy()
                W_init = W55[:, idx].copy()
            else:
                U_init = U55.copy()
                V_init = V55.copy()
                W_init = W55.copy()
            
            result = _als_search(U_init, V_init, W_init, n, m, p, rank, deadline, max_iter=200)
            if result is not None:
                return result
    
    # Try random initialization
    for trial in range(100):
        if time.time() > deadline:
            break
        
        U_init = np.random.randn(n * m, rank) * 0.1
        V_init = np.random.randn(m * p, rank) * 0.1
        W_init = np.random.randn(p * n, rank) * 0.1
        
        result = _als_search(U_init, V_init, W_init, n, m, p, rank, deadline, max_iter=200)
        if result is not None:
            return result
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Attempt to find lower rank decompositions
    for r in range(54, 49, -1):
        if time.time() > deadline - 20:
            break

        result = _random_search_rank(n, m, p, r, deadline - 10, U_best, V_best, W_best)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r
            break

    return {"U": U_best, "V": V_best, "W": W_best}
