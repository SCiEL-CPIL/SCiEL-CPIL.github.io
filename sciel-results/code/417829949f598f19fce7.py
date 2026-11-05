import time
import sys
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
DEADLINE = 120.0

def _time_left():
    return DEADLINE - time.time()

def construct():
    start = time.time()
    deadline = start + DEADLINE - 1.0  # safety margin

    n, m, p = DIMENSIONS  # 3, 3, 5
    rank = n * m * p  # 45

    # Build the schoolbook decomposition
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

    base_result = {"U": U, "V": V, "W": W}

    if time.time() > deadline - 10:
        return base_result

    # Try to find a lower-rank decomposition
    # The matrix multiplication tensor T for (n,m,p) = (3,3,5)
    # has structure: T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k'
    #
    # Let's try to find a rank-r decomposition using the following approach:
    # We look for U (9 x r), V (15 x r), W (15 x r) such that
    # sum_c U[a,c] * V[b,c] * W[c2,c] = T[a,b,c2]
    #
    # For small r, we can try to solve this via:
    # 1. Random initialization + iterative refinement
    # 2. Try specific structured decompositions

    # First, let's try known lower bounds.
    # For 3x3x3, the Laderman algorithm gives rank 7 (vs schoolbook 27).
    # For 3x3x5, the minimum is at least the max of the matricizations.
    # Let's compute the matricization ranks to get a lower bound.

    # Build the full tensor as a 3D array: T[i,j,k] where
    # i indexes A (row i, col j_A), j indexes B (row j_B, col k), k indexes C
    # Actually, let's use the standard formulation:
    # T[a, b, c] where a = (i, j) in [0,9), b = (j', k') in [0,15), c = (i', k'') in [0,15)
    # T[a, b, c] = 1 if i==i' and j==j' and k==k''

    # Matricization 1: fix c, get 9x15 matrix M_c... no, that's not right.
    # Matricization: reshape T as (9*15) x 15 or 9 x (15*15) etc.

    # Let's build the tensor T as a 9x15x15 array
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Matricization along axis 2: 9*15 x 15
    M1 = T.reshape(n * m * m * p, p * n)
    # Compute rank using SVD
    try:
        s1 = np.linalg.svd(M1, compute_uv=False)
        rank_lb1 = int(np.sum(s1 > 1e-9))
    except:
        rank_lb1 = n * m * p

    # Matricization along axis 0: 9 x 15*15
    M2 = T.reshape(n * m, m * p * p * n)
    try:
        s2 = np.linalg.svd(M2, compute_uv=False)
        rank_lb2 = int(np.sum(s2 > 1e-9))
    except:
        rank_lb2 = n * m * p

    # Matricization along axis 1: 15 x 9*15
    M3 = T.reshape(m * p, n * m * p * n)
    try:
        s3 = np.linalg.svd(M3, compute_uv=False)
        rank_lb3 = int(np.sum(s3 > 1e-9))
    except:
        rank_lb3 = n * m * p

    lower_bound = max(rank_lb1, rank_lb2, rank_lb3)
    # The tensor rank is at least max of matricization ranks
    # But also at least the max dimension
    lower_bound = max(lower_bound, n * m, m * p, p * n)

    if time.time() > deadline - 5:
        return base_result

    # Try to find a decomposition with rank = lower_bound or slightly above
    # Use a greedy/iterative approach

    best_rank = rank
    best_decomp = base_result

    # Try ranks from lower_bound up to rank-1
    for r in range(lower_bound, rank):
        if time.time() > deadline - 5:
            break

        # Try to find U, V, W of shape (9, r), (15, r), (15, r)
        # such that sum_c U[a,c]*V[b,c]*W[c2,c] = T[a,b,c2]
        #
        # This is a system of equations. Let's try a simple approach:
        # Fix W to be some random/structured matrix, then solve for U, V.
        # Or use an alternating least squares approach.

        # Approach: Initialize U, V, W randomly, then iteratively update.
        # For each c, we want the outer product u_c ⊗ v_c ⊗ w_c to approximate
        # the "residual" tensor.

        # Simpler approach: try to solve as a linear system.
        # Flatten: for each (a, b, c2), sum_c U[a,c]*V[b,c]*W[c2,c] = T[a,b,c2]
        # This is nonlinear in U, V, W.

        # Let's try: fix U and V, solve for W.
        # For fixed U, V: W[c2, c] = ?
        # T[a, b, c2] = sum_c U[a,c] * V[b,c] * W[c2, c]
        # For each c2: T[:, :, c2] = sum_c (U[:, c] * V[:, c]) * W[c2, c]
        # So T[:, :, c2] = sum_c M_c * W[c2, c] where M_c = U[:, c] ⊗ V[:, c]
        # This gives a system: for each c2, T[:,:,c2] = sum_c W[c2,c] * (U[:,c] ⊗ V[:,c])
        # Flatten: T_flat[:, c2] = W_flat[c2, :] @ M_flat
        # where M_flat has rows indexed by (a,b) and columns by c
        # M_flat[(a,b), c] = U[a,c] * V[b,c]

        # So for fixed U, V:
        # T_flat (135 x 15) = W_flat (15 x r) @ M_flat (135 x r)^T? No.
        # T[a,b,c2] = sum_c U[a,c] * V[b,c] * W[c2,c]
        # Let's define for each c: M_c[a,b] = U[a,c] * V[b,c]
        # Then T[:,:,c2] = sum_c W[c2,c] * M_c
        # So if we stack M_c: M_stack (135, r) where M_stack[(a,b), c] = U[a,c]*V[b,c]
        # And T_flat (135, 15) where T_flat[(a,b), c2] = T[a,b,c2]
        # Then T_flat = M_stack @ W (where W is (r, 15))
        # Wait: T_flat[(a,b), c2] = sum_c M_stack[(a,b), c] * W[c, c2]
        # So T_flat = M_stack @ W where W is (r, 15). Yes!

        # Similarly, for fixed U, W:
        # T[a,b,c2] = sum_c U[a,c] * V[b,c] * W[c2,c]
        # T[:, :, c2] = sum_c U[:, c] * (V[:, c] * W[c2, c])... not quite.
        # T[a,b,c2] = sum_c (V[b,c] * W[c2,c]) * U[a,c]
        # For fixed b, c2: T[a,b,c2] = sum_c U[a,c] * (V[b,c]*W[c2,c])
        # So T[:, b, c2] = U @ d where d[c] = V[b,c]*W[c2,c]
        # T[:, :, c2] = U @ (V * W[c2,:]) where * is elementwise along c
        # This is not a simple linear system.

        # Let's just try the approach: fix U, V, solve for W.
        # Initialize U, V randomly, iterate.

        max_iter = 200
        np.random.seed(42 + r)

        # Random initialization
        U_arr = np.random.randn(n * m, r)
        V_arr = np.random.randn(m * p, r)
        W_arr = np.random.randn(p * n, r)

        # T_flat: (135, 15)
        T_flat = T.reshape(n * m * m * p, p * n).astype(float)

        converged = False
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Update W: T_flat = M_stack @ W
            M_stack = np.einsum('ac,bc->abc', U_arr, V_arr).reshape(n * m * m * p, r)
            # Solve: T_flat (135, 15) = M_stack (135, r) @ W (r, 15)
            # Use least squares
            try:
                W_arr, _, _, _ = np.linalg.lstsq(M_stack, T_flat, rcond=None)
            except:
                W_arr = np.random.randn(r, p * n) * 0.1
            W_arr = W_arr.T  # Now W_arr is (15, r)

            # Update U: For fixed V, W
            # T[a,b,c2] = sum_c U[a,c] * V[b,c] * W[c2,c]
            # For fixed b, c2: T[a,b,c2] = sum_c U[a,c] * (V[b,c]*W[c2,c])
            # T[:, b, c2] = U @ d_{b,c2} where d_{b,c2}[c] = V[b,c]*W[c2,c]
            # So for each (b, c2): U = lstsq(d_{b,c2}, T[:, b, c2])
            # Combine: T_all (15*15, 9) = U (9, r) @ D (r, 15*15) where D[c, b*15+c2] = V[b,c]*W[c2,c]
            D = np.einsum('bc,c2c->cb2', V_arr, W_arr).reshape(r, m * p * p * n)
            T_all = T.reshape(p * n * m * p, n * m).T.astype(float)  # (15*15, 9)
            try:
                U_arr, _, _, _ = np.linalg.lstsq(D, T_all, rcond=None)
                U_arr = U_arr.T  # (9, r)
            except:
                U_arr = np.random.randn(n * m, r) * 0.1

            # Update V: For fixed U, W
            # T[a,b,c2] = sum_c U[a,c] * V[b,c] * W[c2,c]
            # For fixed a, c2: T[a,b,c2] = sum_c V[b,c] * (U[a,c]*W[c2,c])
            # T[a, :, c2] = V @ e_{a,c2} where e_{a,c2}[c] = U[a,c]*W[c2,c]
            E = np.einsum('ac,c2c->ac2', U_arr, W_arr).reshape(r, n * m * p * n)
            T_v = T.reshape(n * m, m * p * p * n).astype(float)  # (9, 15*15)
            try:
                V_arr, _, _, _ = np.linalg.lstsq(E, T_v, rcond=None)
                V_arr = V_arr.T  # (15, r)
            except:
                V_arr = np.random.randn(m * p, r) * 0.1

            # Check convergence
            if it > 10:
                # Compute residual
                U_np = U_arr.astype(float)
                V_np = V_arr.astype(float)
                W_np = W_arr.astype(float)
                T_recon = np.einsum('ac,bc,c2c->abc', U_np, V_np, W_np)
                residual = np.linalg.norm(T_recon - T)
                if residual < 1e-6:
                    converged = True
                    break

        if converged:
            # Verify the decomposition exactly
            U_list = [[int(round(U_arr[a, c])) for c in range(r)] for a in range(n * m)]
            V_list = [[int(round(V_arr[b, c])) for c in range(r)] for b in range(m * p)]
            W_list = [[int(round(W_arr[c2, c])) for c in range(r)] for c2 in range(p * n)]

            # Check if all entries are integers
            U_int = np.round(U_arr).astype(int)
            V_int = np.round(V_arr).astype(int)
            W_int = np.round(W_arr).astype(int)

            if np.allclose(U_arr, U_int, atol=1e-9) and \
               np.allclose(V_arr, V_int, atol=1e-9) and \
               np.allclose(W_arr, W_int, atol=1e-9):
                # Verify
                T_check = np.einsum('ac,bc,c2c->abc', U_int.astype(float), V_int.astype(float), W_int.astype(float))
                if np.allclose(T_check, T, atol=1e-9):
                    best_rank = r
                    best_decomp = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    break
        if time.time() > deadline - 5:
            break

    return best_decomp
