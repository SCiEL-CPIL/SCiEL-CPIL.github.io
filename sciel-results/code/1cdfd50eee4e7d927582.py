"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 115.0

    # Build a 77-rank Strassen decomposition:
    # Part 1: 4x4x4 using two-level Strassen (7 * 7 = 49 terms)
    # Part 2: 4x4x2 using four 2x2x2 Strassen blocks (4 * 7 = 28 terms)
    # Total: 77 terms

    terms = []  # list of (a_dict, b_dict, c_dict)

    # Helper: add a 2x2x2 Strassen block at given offsets
    def add_strassen_2x2x2(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        # M1
        terms.append((
            {(i0, j0): 1, (i1, j1): 1},
            {(j0, k0): 1, (j1, k1): 1},
            {(i0, k0): 1, (i0, k1): 1, (i1, k0): 1, (i1, k1): 1}
        ))
        # M2
        terms.append((
            {(i0, j1): 1, (i1, j1): 1},
            {(j0, k0): 1},
            {(i0, k0): -1, (i1, k1): 1}
        ))
        # M3
        terms.append((
            {(i0, j0): 1},
            {(j1, k1): 1, (j0, k0): -1},
            {(i0, k0): 1, (i1, k0): -1}
        ))
        # M4
        terms.append((
            {(i1, j0): 1},
            {(j1, k0): 1, (j0, k0): -1},
            {(i1, k0): 1}
        ))
        # M5
        terms.append((
            {(i0, j0): 1, (i0, j1): 1},
            {(j1, k1): 1},
            {(i0, k1): -1, (i1, k1): 1}
        ))
        # M6
        terms.append((
            {(i1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0, k1): 1},
            {(i0, k0): 1}
        ))
        # M7
        terms.append((
            {(i0, j1): 1, (i1, j1): -1},
            {(j1, k0): 1, (j1, k1): 1},
            {(i0, k1): 1, (i1, k1): -1}
        ))

    # Block-level Strassen for 4x4x4
    block_strassen_4x4x4 = [
        # M1 = (A00+A11)*(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}),
        # M2 = (A10+A11)*B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): -1, (1,1): 1}),
        # M3 = A00*(B11-B00)
        ({(0,0): 1}, {(1,1): 1, (0,0): -1}, {(0,0): 1, (1,0): -1}),
        # M4 = A10*(B10-B00)
        ({(1,0): 1}, {(1,0): 1, (0,0): -1}, {(1,0): 1}),
        # M5 = (A00+A01)*B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): -1, (1,1): 1}),
        # M6 = (A10-A00)*(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 1}),
        # M7 = (A01-A11)*(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,1): 1, (1,1): -1}),
    ]

    # 2x2x2 Strassen coefficients
    strassen_2x2x2_coeffs = [
        # S1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}),
        # S2
        ({(0,1):1, (1,1):1}, {(0,0):1}, {(0,0):-1, (1,1):1}),
        # S3
        ({(0,0):1}, {(1,1):1, (0,0):-1}, {(0,0):1, (1,0):-1}),
        # S4
        ({(1,0):1}, {(1,0):1, (0,0):-1}, {(1,0):1}),
        # S5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):1}),
        # S6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1}),
        # S7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):-1}),
    ]

    # Build 4x4x4 part (49 terms)
    for ca, cb, cc in block_strassen_4x4x4:
        for alpha, beta, gamma in strassen_2x2x2_coeffs:
            a_coeffs = {}
            for (a, c), val_a in alpha.items():
                for (i_g, j_g), val_ca in ca.items():
                    i = 2 * i_g + a
                    j = 2 * j_g + c
                    a_coeffs[(i, j)] = a_coeffs.get((i, j), 0) + val_a * val_ca

            b_coeffs = {}
            for (c, d), val_b in beta.items():
                for (j_g, k_g), val_cb in cb.items():
                    j = 2 * j_g + c
                    k = 2 * k_g + d
                    b_coeffs[(j, k)] = b_coeffs.get((j, k), 0) + val_b * val_cb

            c_coeffs = {}
            for (a, d), val_g in gamma.items():
                for (i_g, k_g), val_cc in cc.items():
                    i = 2 * i_g + a
                    k = 2 * k_g + d
                    c_coeffs[(i, k)] = c_coeffs.get((i, k), 0) + val_g * val_cc

            terms.append((a_coeffs, b_coeffs, c_coeffs))

    # Build 4x4x2 part (28 terms)
    for i_g in range(2):
        for j_g in range(2):
            add_strassen_2x2x2(2 * i_g, 2 * j_g, 4)

    # Build matrices
    rank = len(terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for col, (a_coeffs, b_coeffs, c_coeffs) in enumerate(terms):
        for (i, j), coeff in a_coeffs.items():
            U[i * m + j, col] = coeff
        for (j, k), coeff in b_coeffs.items():
            V[j * p + k, col] = coeff
        for (i, k), coeff in c_coeffs.items():
            W[k * n + i, col] = coeff

    # Verify
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])

    if not np.allclose(T_reconstructed, 1.0, atol=1e-10):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Attempt rank reduction by finding linear dependencies
    # We can try to remove terms that are linear combinations of others
    # This is a complex optimization, so we'll try a simple greedy approach

    # Check for duplicate or proportional terms
    def try_reduce(U, V, W, max_time=10.0):
        """Try to reduce rank by finding linear dependencies."""
        start = time.time()
        current_rank = U.shape[1]
        
        # Try removing each term one at a time
        for col_to_remove in range(current_rank):
            if time.time() - start > max_time:
                break
            
            # Remove column col_to_remove
            mask = [i for i in range(current_rank) if i != col_to_remove]
            U_reduced = U[:, mask]
            V_reduced = V[:, mask]
            W_reduced = W[:, mask]
            
            # Verify
            T_test = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_test[i, j, k] = np.sum(U_reduced[i*m+j, :] * V_reduced[j*p+k, :] * W_reduced[k*n+i, :])
            
            if np.allclose(T_test, 1.0, atol=1e-10):
                return U_reduced, V_reduced, W_reduced, len(mask)
        
        return U, V, W, current_rank

    U_reduced, V_reduced, W_reduced, new_rank = try_reduce(U, V, W, max_time=10.0)
    
    if new_rank < rank:
        return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
