"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin below 150s

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1

    def verify_decomposition(U, V, W):
        """Check if U, V, W (as arrays) exactly decompose T."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        # T_reconstructed[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # This is the tensor product along the rank dimension
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)

    def round_to_half_int(A):
        """Round entries to nearest half-integer and check if exact."""
        B = np.round(A * 2) / 2.0
        # Check if B is close to A (i.e., rounding didn't change the decomposition)
        if not np.allclose(B, A, atol=1e-9):
            return None
        return B

    # Start with schoolbook
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Try random low-rank decompositions
    # Strategy: pick random rank r, initialize U, V, W randomly, then use
    # gradient-free local search / perturbation to minimize ||T - U⊗V⊗W||
    # Since exact integer/half-integer is required, we do a coarse search
    # and then snap to half-integers.

    rng = np.random.RandomState(42)

    # Try ranks from a small value up to schoolbook
    for r in range(12, rank_school + 1):
        if time.time() > deadline - 20:
            break
        max_iters = 200
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            # Initialize
            U = rng.randn(nm, r) * 0.5
            V = rng.randn(mp, r) * 0.5
            W = rng.randn(pn, r) * 0.5

            # Simple local search: perturb random columns, accept if objective decreases
            obj_prev = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                # Pick a random matrix and column to perturb
                which = rng.randint(0, 3)
                col_idx = rng.randint(0, r)
                if which == 0:
                    delta = rng.randn(nm) * 0.1
                    U_new = U.copy()
                    U_new[:, col_idx] += delta
                    cand = (U_new, V, W)
                elif which == 1:
                    delta = rng.randn(mp) * 0.1
                    V_new = V.copy()
                    V_new[:, col_idx] += delta
                    cand = (U, V_new, W)
                else:
                    delta = rng.randn(pn) * 0.1
                    W_new = W.copy()
                    W_new[:, col_idx] += delta
                    cand = (U, V, W_new)

                obj_new = np.linalg.norm(np.einsum('ar,br,cr->abc', *cand) - T)
                if obj_new < obj_prev:
                    U, V, W = cand
                    obj_prev = obj_new

            # Try to snap to half-integers
            U_h = round_to_half_int(U)
            V_h = round_to_half_int(V)
            W_h = round_to_half_int(W)
            if U_h is not None and V_h is not None and W_h is not None:
                if verify_decomposition(U_h, V_h, W_h):
                    if r < best_rank:
                        best_rank = r
                        best_U, best_V, best_W = U_h, V_h, W_h
                    break  # found exact at this rank, no need for more attempts

    # Convert best to list format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    # Ensure entries are exact half-integers (0.0, 0.5, 1.0, 1.5, ...)
    def clean_half_int(lst):
        out = []
        for row in lst:
            new_row = []
            for v in row:
                v2 = round(v * 2) / 2.0
                if v2 == int(v2):
                    new_row.append(int(v2))
                else:
                    new_row.append(v2)
            out.append(new_row)
        return out

    U_out = clean_half_int(U_out)
    V_out = clean_half_int(V_out)
    W_out = clean_half_int(W_out)

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
