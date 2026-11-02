"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 108.0

    # Build the base rank-54 decomposition via Strassen blocks + standard terms
    def build_base():
        R = 54
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        c = 0

        def put(u_pairs, v_pairs, w_pairs):
            nonlocal c
            for idx, val in u_pairs:
                U[idx, c] += val
            for idx, val in v_pairs:
                V[idx, c] += val
            for idx, val in w_pairs:
                W[idx, c] += val
            c += 1

        def strassen(ja, jb, ka, kb):
            ua0 = 0 * m + ja
            ub0 = 0 * m + jb
            ua1 = 1 * m + ja
            ub1 = 1 * m + jb
            va_ka = ja * p + ka
            va_kb = ja * p + kb
            vb_ka = jb * p + ka
            vb_kb = jb * p + kb
            wk0_0 = ka * n + 0
            wk0_1 = ka * n + 1
            wk1_0 = kb * n + 0
            wk1_1 = kb * n + 1

            put([(ua0, 1), (ub1, 1)], [(va_ka, 1), (vb_kb, 1)], [(wk0_0, 1), (wk1_1, 1)])
            put([(ua1, 1), (ub1, 1)], [(va_ka, 1)], [(wk0_1, 1), (wk1_1, -1)])
            put([(ua0, 1)], [(va_kb, 1), (vb_kb, -1)], [(wk1_0, 1), (wk1_1, 1)])
            put([(ub1, 1)], [(vb_ka, 1), (va_ka, -1)], [(wk0_0, 1), (wk0_1, 1)])
            put([(ua0, 1), (ub0, 1)], [(vb_kb, 1)], [(wk0_0, -1), (wk1_0, 1)])
            put([(ua1, 1), (ua0, -1)], [(va_ka, 1), (va_kb, 1)], [(wk1_1, 1)])
            put([(ub0, 1), (ub1, -1)], [(vb_ka, 1), (vb_kb, 1)], [(wk0_0, 1)])

        for (ja, jb) in [(0, 1), (2, 3)]:
            for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
                strassen(ja, jb, ka, kb)

        for k in range(p):
            for i in range(n):
                U[i * m + 4, c] = 1.0
                V[4 * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1

        return U, V, W, c

    U, V, W, R = build_base()

    # Verify base decomposition
    def verify(Um, Vm, Wm, Rm):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Um[i * m + j, :Rm] * Vm[j * p + k, :Rm], Wm[k * n + i, :Rm]))
                    max_err = max(max_err, abs(s - 1.0))
        return max_err < 1e-10

    if not verify(U, V, W, R):
        # Fallback: trivial rank 60
        Uf = np.zeros((n * m, n * m * p), dtype=float)
        Vf = np.zeros((m * p, n * m * p), dtype=float)
        Wf = np.zeros((p * n, n * m * p), dtype=float)
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, cc] = 1.0
                    Vf[j * p + k, cc] = 1.0
                    Wf[k * n + i, cc] = 1.0
                    cc += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # Attempt rank reduction via random linear combinations (greedy subspace search)
    # The idea: if the rank-54 decomposition has linear dependencies in its tensor
    # space, we can find a smaller spanning set.
    # Strategy: try to express one rank-1 tensor as a combination of others.
    
    best_rank = R
    best_U, best_V, best_W = U, V, W

    # Try removing each column and checking if the remaining still spans
    # This is O(R * verification) which is manageable for R=54
    rng = np.random.RandomState(42)
    
    # First, try simple column elimination
    for attempt in range(R):
        if time.time() > deadline - 15.0:
            break
        col_to_try = attempt
        # Check if column col_to_try is in the span of the others
        # Build the tensor for this column
        u_col = U[:, col_to_try]
        v_col = V[:, col_to_try]
        w_col = W[:, col_to_try]
        
        # The tensor T_col = u_col ⊗ v_col ⊗ w_col
        # Check if it's in the span of other tensors
        # This is a linear algebra problem: is T_col in span{T_0, ..., T_{col-1}, T_{col+1}, ...}?
        
        # Flatten approach: the multiplication tensor has a specific structure.
        # A simpler check: try to find coefficients alpha_r such that
        # sum_r alpha_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = u_col ⊗ v_col ⊗ w_col
        # This is a system of n*m*p = 60 equations in R-1 = 53 unknowns.
        # If a solution exists, we can eliminate this column.
        
        if time.time() > deadline - 10.0:
            break
        
        # Build the linear system
        # Each equation corresponds to (i,j,k): sum_r alpha_r * U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = u_col[i*m+j]*v_col[j*p+k]*w_col[k*n+i]
        # This is a bilinear/quadratic system, not linear in alpha_r alone.
        # So simple elimination doesn't work directly.
        
        # Alternative: try random linear combinations to see if we can reduce rank
        # by finding a linear dependency among the U, V, W columns simultaneously.
        
        # Try: pick a random subset of columns and see if we can adjust them
        # to maintain the tensor while using fewer effective dimensions.
        pass

    # Try a different approach: use the fact that the standard blocks for j=4
    # might be partially absorbable into Strassen blocks via clever linear combinations.
    # Specifically, try to merge one standard term with a Strassen term.
    
    # For now, return the base rank-54 result if no improvement found
    if best_rank < R:
        return {"U": best_U[:, :best_rank].tolist(), "V": best_V[:, :best_rank].tolist(), "W": best_W[:, :best_rank].tolist()}
    
    return {"U": U[:, :R].tolist(), "V": V[:, :R].tolist(), "W": W[:, :R].tolist()}
# EVOLVE-BLOCK-END
