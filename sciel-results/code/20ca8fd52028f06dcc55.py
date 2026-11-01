"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen (rank 49) and attempts to reduce to rank 48 via linear solve.
    """
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def strassen_decomposition():
        num_blocks = p // 2
        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        rank = num_blocks * 7
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                a_c, b_c, c_c = strassen_terms[s]
                for a_idx, coeff in a_c.items():
                    U[(a_idx // m) * m + (a_idx % m), col] = coeff
                for b_idx, coeff in b_c.items():
                    b_j, b_col = b_idx // 2, b_idx % 2
                    V[b_j * p + 2 * t + b_col, col] = coeff
                for c_idx, coeff in c_c.items():
                    c_i, c_col = c_idx // 2, c_idx % 2
                    W[(2 * t + c_col) * n + c_i, col] = coeff
        return U, V, W

    def tensor_check(Uf, Vf, Wf, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    U49, V49, W49 = strassen_decomposition()
    best_U, best_V, best_W = U49.copy(), V49.copy(), W49.copy()
    best_rank = 49

    if not tensor_check(U49, V49, W49):
        rank = n * m * p
        U = np.zeros((nm, rank)); V = np.zeros((mp, rank)); W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0; V[j*p+k, col] = 1.0; W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 48: for each column to remove, solve linear system for W
    for c_remove in range(49):
        if time.time() > deadline - 10:
            break
        U48 = U49.copy(); V48 = V49.copy()
        U48[:, c_remove] = 0; V48[:, c_remove] = 0
        remaining = [r for r in range(49) if r != c_remove]
        
        # Build 56 x 1344 linear system: A * W_flat = b
        A = np.zeros((56, 1344))
        b = np.ones(56)
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ri, r in enumerate(remaining):
                        A[eq, (k*n+i)*48 + ri] = U49[i*m+j, r] * V49[j*p+k, r]
                    eq += 1
        
        try:
            W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W48 = W_flat.reshape(28, 48)
            W_full = np.zeros((28, 49))
            for ri, r in enumerate(remaining):
                W_full[:, r] = W48[:, ri]
            
            if tensor_check(U48, V48, W_full, tol=1e-6):
                # Check for integer/half-integer entries
                if all(abs(x - round(2*x))/2 < 1e-6 for row in W_full for x in row):
                    best_U, best_V, best_W = U48, V48, W_full
                    best_rank = 48
                    break
        except:
            pass

    U_list = best_U.tolist(); V_list = best_V.tolist(); W_list = best_W.tolist()
    U_list = [[int(round(x)) if abs(x-round(x))<1e-9 else x for x in row] for row in U_list]
    V_list = [[int(round(x)) if abs(x-round(x))<1e-9 else x for x in row] for row in V_list]
    W_list = [[int(round(x)) if abs(x-round(x))<1e-9 else x for x in row] for row in W_list]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
