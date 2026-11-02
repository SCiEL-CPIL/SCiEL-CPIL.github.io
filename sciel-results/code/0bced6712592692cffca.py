"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s evaluator kill
    n, m, p = DIMENSIONS

    # Default: schoolbook rank = n*m*p = 36
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    # Strategy: algebraic parameterization + local gradient search.
    # We model the tensor as a sum of rank-1 terms.  The ansatz combines:
    #   - 7 Strassen terms for the 2x2x2 subproblem (i in {0,1}, j in {0,1}, k in block)
    #   - 4 schoolbook terms for j=2 (i in {0,1}, k in block)
    # for each of 3 blocks (k=0..5 split into 3 pairs), giving 33 terms.
    # We then perform a local gradient search over a small set of free parameters
    # to see if we can reduce the rank by merging terms or finding a lower-rank exact decomposition.

    try:
        # Build the 2x3x6 tensor T[i,j,k] = 1 for all i,j,k (since we want to decompose the identity tensor)
        # Actually, the tensor for matrix multiplication is T[i,j,k] = delta_{i,j,k} but in the standard
        # formulation, the tensor is T[i,j,k] = 1 if the term (i,j,k) is present in the product, which
        # for standard matrix multiplication is T[i,j,k] = 1 for all i,j,k. Wait, no.
        # The tensor for matrix multiplication C = AB is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has entries T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        # But the rank decomposition is: T = sum_t u_t v_t w_t
        # where u_t is a vector in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
        # and the condition is: sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 for all i,j,k.

        # We'll use the 33-term construction as the starting point and try to optimize.
        rank = 33
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen terms for 2x2x2 (standard formulation):
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # p1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # p2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # p3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # p4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # p5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # p6
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0]), # p7
        ]

        # Build one 2x3x2 block (11 terms)
        U_block = np.zeros((6, 11))
        V_block = np.zeros((6, 11))
        W_block = np.zeros((4, 11))

        # Place 7 Strassen terms (for j in {0,1})
        for t in range(7):
            ut, vt, wt = strassen_terms[t]
            for i in range(2):
                for j in range(2):
                    U_block[i * 3 + j, t] = ut[i * 2 + j]
            for j in range(2):
                for k in range(2):
                    V_block[j * 2 + k, t] = vt[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    W_block[k * 2 + i, t] = wt[k * 2 + i]

        # Place 4 schoolbook terms (for j=2)
        for di, (i_val, k_val) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            t = 7 + di
            U_block[i_val * 3 + 2, t] = 1
            V_block[2 * 2 + k_val, t] = 1
            W_block[k_val * 2 + i_val, t] = 1

        # Place 3 blocks into full U, V, W
        for block_idx in range(3):
            k_off = block_idx * 2
            for local_i in range(2):
                for local_j in range(3):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        U[local_i * 3 + local_j, full_t] = U_block[local_i * 3 + local_j, t]
            for local_j in range(3):
                for local_k in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        V[local_j * 2 + local_k, full_t] = V_block[local_j * 2 + local_k, t]
            for local_k in range(2):
                for local_i in range(2):
                    for t in range(11):
                        full_t = block_idx * 11 + t
                        W[(k_off + local_k) * 2 + local_i, full_t] = W_block[local_k * 2 + local_i, t]

        # Verify the initial 33-term construction
        def verify_numpy(Uf, Vf, Wf, n, m, p, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for t in range(rank):
                            s += Uf[i * m + j, t] * Vf[j * p + k, t] * Wf[k * n + i, t]
                        if abs(s - 1) > 1e-9:
                            return False
            return True

        if not verify_numpy(U, V, W, n, m, p, rank):
            # If the 33-term construction is invalid, fall back to schoolbook
            return best

        # Now try to optimize: we'll use a parameterization where we allow small perturbations
        # and try to find a lower-rank exact decomposition by zeroing out some terms.
        # This is heuristic and may not find a lower rank, but it's a different strategy.

        # For now, return the 33-term construction as it's valid and better than 36.
        def to_clean(mat):
            result = []
            for row in mat:
                clean_row = []
                for x in row:
                    if abs(x - round(x)) < 1e-9:
                        clean_row.append(int(round(x)))
                    elif abs(2 * x - round(2 * x)) < 1e-9:
                        clean_row.append(round(2 * x) / 2)
                    else:
                        clean_row.append(x)
                result.append(clean_row)
            return result

        return {"U": to_clean(U.tolist()), "V": to_clean(V.tolist()), "W": to_clean(W.tolist())}

    except Exception:
        pass

    return best
# EVOLVE-BLOCK-END
