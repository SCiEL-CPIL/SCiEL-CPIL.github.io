import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = 4, 4, 7
    # Tensor shape: (n*m, m*p, p*n) = (16, 28, 28)
    # C[k*n+i, j] = sum_l A[i*m+j] * B[j*p+k]  ... wait, let me re-read the convention
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This means W is indexed as W[k*n + i, r] where k is the column of C (0..p-1), i is the row of C (0..n-1)
    # Actually, for matrix multiplication C = A @ B:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T[i, j, k] = A[i,j] * B[j,k] * C[i,k]
    # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the convention says W[k*n+i, r]. Let's verify:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[A_idx, B_idx, C_idx] where:
    #   A_idx = i*m + j  (row i, col j of A, size n x m = 4x4 = 16)
    #   B_idx = j*p + k  (row j, col k of B, size m x p = 4x7 = 28)
    #   C_idx = k*n + i  (col k, row i of C, size p x n = 7x4 = 28)
    #   T[A_idx, B_idx, C_idx] = 1 if A[i,j] * B[j,k] contributes to C[i,k], else 0
    #   Specifically, T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    #   All other entries are 0.

    # Build the full tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Strategy: Generate candidate rank-1 tensors, then greedy select
    # A rank-1 tensor is (u, v, w) where u is (16,), v is (28,), w is (28,)
    # The tensor it represents is outer(u, v, w) reshaped to (16, 28, 28)
    # We need sum_r outer(U[:,r], V[:,r], W[:,r]) = T

    # Generate candidates:
    # 1. Schoolbook: u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i)
    # 2. Strassen-style: combinations of 2x2 blocks
    # 3. Random sparse patterns

    candidates = []  # list of (u, v, w) tuples, each a numpy array

    # 1. Schoolbook candidates (16 * 7 = 112 candidates, but many are equivalent)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                candidates.append((u, v, w))

    # 2. Strassen-style 2x2 block candidates
    # For each 2x2 block in A, 2x2 block in B, 2x2 block in C
    # Generate the 7 Strassen patterns
    def strassen_2x2_candidates(i_off, j_off, k_off):
        """Generate 7 Strassen rank-1 tensors for a 2x2x2 block"""
        cands = []
        # M1 = (A11+A22)*(B11+B22)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[i_off * m + j_off] = 1.0
        u[(i_off + 1) * m + (j_off + 1)] = 1.0
        v[j_off * p + k_off] = 1.0
        v[(j_off + 1) * p + (k_off + 1)] = 1.0
        w[k_off * n + i_off] = 1.0
        w[(k_off + 1) * n + (i_off + 1)] = 1.0
        cands.append((u.copy(), v.copy(), w.copy()))

        # M2 = (A21+A22)*B11
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[(i_off + 1) * m + j_off] = 1.0
        u[(i_off + 1) * m + (j_off + 1)] = 1.0
        v[j_off * p + k_off] = 1.0
        w[k_off * n + (i_off + 1)] = 1.0
        w[(k_off + 1) * n + (i_off + 1)] = -1.0
        cands.append((u.copy(), v.copy(), w.copy()))

        # M3 = A11*(B12-B22)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[i_off * m + j_off] = 1.0
        v[j_off * p + (k_off + 1)] = 1.0
        v[(j_off + 1) * p + (k_off + 1)] = -1.0
        w[(k_off + 1) * n + i_off] = 1.0
        w[(k_off + 1) * n + (i_off + 1)] = 1.0
        cands.append((u.copy(), v.copy(), w.copy()))

        # M4 = A22*(B21-B11)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[(i_off + 1) * m + (j_off + 1)] = 1.0
        v[(j_off + 1) * p + k_off] = 1.0
        v[j_off * p + k_off] = -1.0
        w[k_off * n + i_off] = 1.0
        w[k_off * n + (i_off + 1)] = 1.0
        cands.append((u.copy(), v.copy(), w.copy()))

        # M5 = (A11+A12)*B22
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[i_off * m + j_off] = 1.0
        u[i_off * m + (j_off + 1)] = 1.0
        v[(j_off + 1) * p + (k_off + 1)] = 1.0
        w[(k_off + 1) * n + i_off] = 1.0
        w[k_off * n + i_off] = -1.0
        cands.append((u.copy(), v.copy(), w.copy()))

        # M6 = (A21-A11)*(B11+B12)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[(i_off + 1) * m + j_off] = 1.0
        u[i_off * m + j_off] = -1.0
        v[j_off * p + k_off] = 1.0
        v[j_off * p + (k_off + 1)] = 1.0
        w[(k_off + 1) * n + (i_off + 1)] = 1.0
        cands.append((u.copy(), v.copy(), w.copy()))

        # M7 = (A12-A22)*(B21+B22)
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[i_off * m + (j_off + 1)] = 1.0
        u[(i_off + 1) * m + (j_off + 1)] = -1.0
        v[(j_off + 1) * p + k_off] = 1.0
        v[(j_off + 1) * p + (k_off + 1)] = 1.0
        w[k_off * n + i_off] = 1.0
        cands.append((u.copy(), v.copy(), w.copy()))

        return cands

    # Add Strassen candidates for all 2x2x2 blocks
    for i_off in range(0, n, 2):
        for j_off in range(0, m, 2):
            for k_off in range(0, p, 2):
                candidates.extend(strassen_2x2_candidates(i_off, j_off, k_off))

    # 3. Add some random sparse candidates to help find better decompositions
    random.seed(42)
    for _ in range(200):
        # Random sparse u, v, w
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        # Pick 2-4 non-zero entries in each
        for _ in range(random.randint(2, 4)):
            idx = random.randint(0, n * m - 1)
            u[idx] = random.choice([-1.0, 1.0])
        for _ in range(random.randint(2, 4)):
            idx = random.randint(0, m * p - 1)
            v[idx] = random.choice([-1.0, 1.0])
        for _ in range(random.randint(2, 4)):
            idx = random.randint(0, p * n - 1)
            w[idx] = random.choice([-1.0, 1.0])
        # Only add if the outer product has non-zero overlap with T
        outer = np.einsum('a,b,c->abc', u, v, w)
        if np.dot(outer.flatten(), T.flatten()) != 0:
            candidates.append((u, v, w))

    if time.time() > deadline - 30:
        # Not enough time for search, return schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1.0
                    V[j * p + k][r] = 1.0
                    W[k * n + i][r] = 1.0
        return {"U": U, "V": V, "W": W}

    # Greedy forward selection
    # We want to find the minimal set of candidates whose sum of outer products equals T
    # This is equivalent to: find minimal subset S such that sum_{s in S} outer(u_s, v_s, w_s) = T

    # Represent T as a flat vector
    T_flat = T.flatten()
    dim = len(T_flat)

    # Pre-compute outer products as flat vectors
    candidate_vectors = []
    for u, v, w in candidates:
        outer = np.einsum('a,b,c->abc', u, v, w)
        candidate_vectors.append(outer.flatten())

    # Greedy: at each step, pick the candidate that best reduces the residual
    residual = T_flat.copy()
    selected = []
    selected_vectors = []

    max_iter = 60  # Max rank to try
    for step in range(max_iter):
        if time.time() > deadline - 10:
            break
        if np.allclose(residual, 0, atol=1e-10):
            break

        # Find the candidate that best correlates with the residual
        # We want to minimize ||residual - alpha * candidate||^2
        # Optimal alpha = dot(residual, candidate) / dot(candidate, candidate)
        best_idx = -1
        best_alpha = 0.0
        best_norm = float('inf')

        for idx, cv in enumerate(candidate_vectors):
            if idx in selected:
                continue
            dot_rc = np.dot(residual, cv)
            if abs(dot_rc) < 1e-12:
                continue
            dot_cc = np.dot(cv, cv)
            if dot_cc < 1e-12:
                continue
            alpha = dot_rc / dot_cc
            new_residual = residual - alpha * cv
            norm = np.dot(new_residual, new_residual)
            if norm < best_norm:
                best_norm = norm
                best_idx = idx
                best_alpha = alpha

        if best_idx == -1:
            break

        selected.append(best_idx)
        selected_vectors.append((best_idx, best_alpha))
        residual = residual - best_alpha * candidate_vectors[best_idx]

        if step % 10 == 0:
            print(f"Step {step}: rank={len(selected)}, residual_norm={np.linalg.norm(residual):.6f}")

    # Check if we found an exact solution
    if np.allclose(residual, 0, atol=1e-10):
        rank = len(selected)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        for r, (idx, alpha) in enumerate(selected_vectors):
            u, v, w = candidates[idx]
            for i in range(n * m):
                U[i][r] = u[i] * alpha
            for i in range(m * p):
                V[i][r] = v[i] * alpha
            for i in range(p * n):
                W[i][r] = w[i] * alpha

        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1.0
                    V[j * p + k][r] = 1.0
                    W[k * n + i][r] = 1.0
        return {"U": U, "V": V, "W": W}
