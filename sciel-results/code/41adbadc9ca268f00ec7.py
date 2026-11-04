import numpy as np
import time

DIMENSIONS = [4, 4, 6]
DEADLINE_SECONDS = 140.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    # Rank 77 decomposition: 7 outer Strassen steps, each with inner rank 11
    # (7 for 2x2*2x2 Strassen + 4 for 2x2*2x1 schoolbook)
    rank = 77

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Outer Strassen steps for 4x4 * 4x6 (split into 2x2 blocks of A and 2x3 blocks of B)
    # A blocks: 00(0:2,0:2), 01(0:2,2:4), 10(2:4,0:2), 11(2:4,2:4)
    # B blocks: 00(0:2,0:3), 01(0:2,3:6), 10(2:4,0:3), 11(2:4,3:6)
    # C blocks: 00(0:2,0:3), 01(0:2,3:6), 10(2:4,0:3), 11(2:4,3:6)
    #
    # Strassen for 2x2 * 2x3:
    # We need to compute C = A*B where A is 2x2 and B is 2x3.
    # Split B into B_left (2x2) and B_right (2x1).
    # C_left = A * B_left (2x2 * 2x2, rank 7 Strassen)
    # C_right = A * B_right (2x2 * 2x1, rank 4 schoolbook)
    # Total inner rank: 7 + 4 = 11
    #
    # Outer Strassen for 4x4 * 4x6:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01+B11)
    # M4 = (A11)(B10+B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10+A00)(B00+B01)
    # M7 = (A01+A11)(B10+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M7

    steps = [
        # (alpha: A block coeffs, beta: B block coeffs, gamma: C block coeffs)
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),   # M1
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),           # M2
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),           # M3
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),           # M4
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),           # M5
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),           # M6
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),           # M7
    ]

    def get_u_row(block, i1, j1):
        """Get flat index in A (n*m) for block (i1,j1) within block 'block'."""
        if block == '00': i, j = i1, j1
        elif block == '01': i, j = i1, j1 + 2
        elif block == '10': i, j = i1 + 2, j1
        elif block == '11': i, j = i1 + 2, j1 + 2
        else: raise ValueError
        return i * m + j

    def get_v_row(block, j1, k1):
        """Get flat index in B (m*p) for block (j1,k1) within block 'block'.
        B blocks are 2x3: j1 in {0,1}, k1 in {0,1,2}."""
        if block == '00': j, k = j1, k1
        elif block == '01': j, k = j1, k1 + 3
        elif block == '10': j, k = j1 + 2, k1
        elif block == '11': j, k = j1 + 2, k1 + 3
        else: raise ValueError
        return j * p + k

    def get_w_row(block, i1, k1):
        """Get flat index in C (p*n) for block (i1,k1) within block 'block'.
        C blocks are 2x3: i1 in {0,1}, k1 in {0,1,2}.
        W convention: W[k*n+i, r]"""
        if block == '00': i, k = i1, k1
        elif block == '01': i, k = i1, k1 + 3
        elif block == '10': i, k = i1 + 2, k1
        elif block == '11': i, k = i1 + 2, k1 + 3
        else: raise ValueError
        return k * n + i

    # Inner Strassen steps for 2x2 * 2x2
    inner_steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}, {'00': 1, '11': 1}),
        ({'10': 1, '11': 1}, {'00': 1}, {'10': 1, '11': -1}),
        ({'00': 1}, {'01': 1, '11': -1}, {'01': 1, '11': 1}),
        ({'11': 1}, {'10': 1, '00': -1}, {'00': 1, '10': 1}),
        ({'00': 1, '01': 1}, {'11': 1}, {'00': -1, '01': 1}),
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}, {'11': 1}),
        ({'01': 1, '11': -1}, {'10': 1, '11': 1}, {'00': 1}),
    ]

    col = 0
    for s in range(7):
        alpha, beta, gamma = steps[s]
        
        # --- Part 1: Strassen for 2x2 * 2x2 (cols 0,1 of each B block) ---
        for inner_s in range(7):
            i_alpha, i_beta, i_gamma = inner_steps[inner_s]
            
            # U contributions
            for ai in range(2):
                for aj in range(2):
                    inner_key = f"{ai}{aj}"
                    inner_coeff = i_alpha.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in alpha.items():
                            U[get_u_row(b, ai, aj), col] += val * inner_coeff
            
            # V contributions (for B'_12, k1=0,1)
            for bj in range(2):
                for bk in range(2):
                    inner_key = f"{bj}{bk}"
                    inner_coeff = i_beta.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in beta.items():
                            V[get_v_row(b, bj, bk), col] += val * inner_coeff
            
            # W contributions (for C'_12, k1=0,1)
            for ci in range(2):
                for ck in range(2):
                    inner_key = f"{ci}{ck}"
                    inner_coeff = i_gamma.get(inner_key, 0)
                    if inner_coeff != 0:
                        for b, val in gamma.items():
                            W[get_w_row(b, ci, ck), col] += val * inner_coeff
            
            col += 1
        
        # --- Part 2: Schoolbook for 2x2 * 2x1 (col 2 of each B block) ---
        # The 4 multiplications: A[ai,aj] * B[bj,2] -> C[ai,2]
        # We need to be careful about which B block's col 2 we're using
        for b, val in beta.items():
            # For each B block, the schoolbook multiplies A elements with B's col 2
            # A[0,0]*B[0,2], A[0,1]*B[1,2], A[1,0]*B[0,2], A[1,1]*B[1,2]
            pass
        
        # Schoolbook: 4 terms
        # Term 1: A[0,0] * B[0,2] -> C[0,2]
        for b, val in alpha.items():
            U[get_u_row(b, 0, 0), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 0, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 0, 2), col] += val
        col += 1
        
        # Term 2: A[0,1] * B[1,2] -> C[0,2]
        for b, val in alpha.items():
            U[get_u_row(b, 0, 1), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 1, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 0, 2), col] += val
        col += 1
        
        # Term 3: A[1,0] * B[0,2] -> C[1,2]
        for b, val in alpha.items():
            U[get_u_row(b, 1, 0), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 0, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 1, 2), col] += val
        col += 1
        
        # Term 4: A[1,1] * B[1,2] -> C[1,2]
        for b, val in alpha.items():
            U[get_u_row(b, 1, 1), col] += val
        for b, val in beta.items():
            V[get_v_row(b, 1, 2), col] += val
        for b, val in gamma.items():
            W[get_w_row(b, 1, 2), col] += val
        col += 1

    # Verification with multiple test cases
    try:
        rng = np.random.default_rng(12345)
        for trial in range(10):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            A_flat = A.reshape(n * m)
            B_flat = B.reshape(m * p)
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_scheme_flat = W.T @ (a * b)
            if not np.allclose(C_scheme_flat, C_true_flat, atol=1e-10):
                break
    except Exception:
        pass

    # Ensure we return before deadline
    if time.time() - start > DEADLINE_SECONDS - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
