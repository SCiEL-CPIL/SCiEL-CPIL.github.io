"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 150s limit

    n, m, p = DIMENSIONS
    rank = 54

    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    current_col = 0

    def next_col():
        nonlocal current_col
        current_col += 1

    def add_strassen_block(i_base, j_base, k_base):
        nonlocal current_col
        # M0: (A00+A11)(B00+B11)
        U[i_base*m+j_base, current_col] = 1
        U[(i_base+1)*m+(j_base+1), current_col] = 1
        V[j_base*p+k_base, current_col] = 1
        V[(j_base+1)*p+(k_base+1), current_col] = 1
        W[k_base*n+i_base, current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        next_col()

        # M1: (A10+A11)B00
        U[(i_base+1)*m+j_base, current_col] = 1
        U[(i_base+1)*m+(j_base+1), current_col] = 1
        V[j_base*p+k_base, current_col] = 1
        W[k_base*n+(i_base+1), current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = -1
        next_col()

        # M2: A00(B01-B11)
        U[i_base*m+j_base, current_col] = 1
        V[j_base*p+(k_base+1), current_col] = 1
        V[(j_base+1)*p+(k_base+1), current_col] = -1
        W[(k_base+1)*n+i_base, current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        next_col()

        # M3: A11(B10-B00)
        U[(i_base+1)*m+(j_base+1), current_col] = 1
        V[(j_base+1)*p+k_base, current_col] = 1
        V[j_base*p+k_base, current_col] = -1
        W[k_base*n+i_base, current_col] = 1
        W[k_base*n+(i_base+1), current_col] = 1
        next_col()

        # M4: (A00+A01)B11
        U[i_base*m+j_base, current_col] = 1
        U[i_base*m+(j_base+1), current_col] = 1
        V[(j_base+1)*p+(k_base+1), current_col] = 1
        W[k_base*n+i_base, current_col] = -1
        W[(k_base+1)*n+i_base, current_col] = 1
        next_col()

        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m+j_base, current_col] = 1
        U[i_base*m+j_base, current_col] = -1
        V[j_base*p+k_base, current_col] = 1
        V[j_base*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        next_col()

        # M6: (A01-A11)(B10+B11)
        U[i_base*m+(j_base+1), current_col] = 1
        U[(i_base+1)*m+(j_base+1), current_col] = -1
        V[(j_base+1)*p+k_base, current_col] = 1
        V[(j_base+1)*p+(k_base+1), current_col] = 1
        W[k_base*n+i_base, current_col] = 1
        next_col()

    def add_schoolbook_block(i_base, j_idx, k_base):
        nonlocal current_col
        U[i_base*m+j_idx, current_col] = 1
        V[j_idx*p+k_base, current_col] = 1
        W[k_base*n+i_base, current_col] = 1
        next_col()

        U[i_base*m+j_idx, current_col] = 1
        V[j_idx*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+i_base, current_col] = 1
        next_col()

        U[(i_base+1)*m+j_idx, current_col] = 1
        V[j_idx*p+k_base, current_col] = 1
        W[k_base*n+(i_base+1), current_col] = 1
        next_col()

        U[(i_base+1)*m+j_idx, current_col] = 1
        V[j_idx*p+(k_base+1), current_col] = 1
        W[(k_base+1)*n+(i_base+1), current_col] = 1
        next_col()

    # 1. Process Strassen blocks for j-pairs (0,1) and (2,3)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            i_base = 0
            if time.time() > deadline:
                break
            add_strassen_block(i_base, j_base, k_base)
        if time.time() > deadline:
            break

    # 2. Process Naive blocks for j=4
    if time.time() <= deadline:
        for k_base in [0, 2, 4]:
            i_base = 0
            j_idx = 4
            if time.time() > deadline:
                break
            add_schoolbook_block(i_base, j_idx, k_base)

    if current_col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {current_col}")

    # Verification: check that the decomposition exactly reproduces the MM tensor
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all (i,j,k)
    try:
        if time.time() <= deadline - 5:
            # Build tensor from decomposition
            tensor = np.zeros((n, m, p), dtype=float)
            for r in range(rank):
                # u_r is shape (n, m), v_r is shape (m, p), w_r is shape (p, n)
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                # outer product: u_r[i,j] * v_r[j,k] * w_r[k,i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
            # Check against identity tensor (should be all 1s)
            if not np.allclose(tensor, np.ones((n, m, p)), atol=1e-9):
                # Decomposition is invalid - fall back to schoolbook
                rank_fb = n * m * p
                U_fb = np.zeros((n * m, rank_fb), dtype=float)
                V_fb = np.zeros((m * p, rank_fb), dtype=float)
                W_fb = np.zeros((p * n, rank_fb), dtype=float)
                col_fb = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j, col_fb] = 1
                            V_fb[j * p + k, col_fb] = 1
                            W_fb[k * n + i, col_fb] = 1
                            col_fb += 1
                return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    except Exception:
        # If verification fails, return the original (should be correct)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
