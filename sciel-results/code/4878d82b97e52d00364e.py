import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 7

    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    # Try to find a better decomposition using rank-revealing approach
    # The current best is rank 50 from Strassen blocks.
    # Let's try to see if we can do better with a more clever approach.
    
    # For 2x4 * 4x7, we can think of this as computing 14 output elements.
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try a block approach where we use the fact that n=2 is small.
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # One approach: group the 4 middle dimensions into pairs.
    # For j in {0,1} and k in {0,1,2,3,4,5,6}:
    #   C[0,k] += (A[0,0]+A[0,1]) * (B[0,k]+B[1,k]) - A[0,1]*B[0,k] - A[0,0]*B[1,k]
    # This gives us Strassen-like savings.
    
    # Let's try a different block decomposition:
    # Split m=4 into two groups of 2: {0,1} and {2,3}
    # For each group, use Strassen on the 2x2 subproblem with p=7 columns
    
    # For group {0,1}:
    #   We compute contributions to C[0,0:7] and C[1,0:7] from A[0:2, 0:2] * B[0:2, 0:7]
    #   This is a 2x2 * 2x7 multiplication.
    #   
    # For a 2x2 * 2x7, we can use:
    #   M1 = (a+d)(e0+e1+...+e6) -- no, that's not right
    #   
    # Actually, for 2x2 * 2x7, we can do:
    #   C[0,k] = a*B[0,k] + c*B[1,k]  (for each k)
    #   C[1,k] = b*B[0,k] + d*B[1,k]  (for each k)
    #   
    # Using Strassen on the 2x2 part:
    #   M1 = (a+d)(B[0,k]+B[1,k]) for each k -- but this doesn't work directly
    #   
    # Better: for each k, we have a 2x2 * 2x1 problem (just one column of B)
    # But that's not efficient.
    
    # Let me think differently. The standard approach for 2x4 * 4x7:
    # Treat it as 2x(2+2) * (2+2)x7
    # = 2x2*2x7 + 2x2*2x7 (with appropriate row/column selections)
    
    # For 2x2 * 2x7:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #   
    # Using Strassen:
    #   M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k])
    #   M2 = (A[1,0]+A[1,1]) * B[0,k]
    #   M3 = A[0,0] * (B[1,k]-B[0,k])
    #   M4 = A[1,1] * (B[0,k]-B[1,k])
    #   M5 = (A[0,0]+A[0,1]) * B[1,k]
    #   M6 = (A[1,0]-A[0,0]) * (B[0,k]+B[1,k])
    #   M7 = (A[0,1]-A[1,1]) * (B[0,k]+B[1,k])
    #   
    # But these M's depend on k, so we can't share them across k values.
    # Each k gives a separate 2x2*2x1 problem, which has rank 4 (schoolbook) or 7 (Strassen, but that's worse).
    
    # So the current approach of using Strassen on 2x2x2 blocks (where we pair up k values too) is reasonable.
    
    # Let's try: can we do better than 50?
    # Lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Known: for 2x4*4x7, the best known is likely around 30-40.
    
    # Let's try a different strategy: use the fact that we have 2 rows.
    # Define for each k:
    #   x_k = B[0,k], y_k = B[1,k], z_k = B[2,k], w_k = B[3,k]
    #   C[0,k] = a*x_k + b*y_k + c*z_k + d*w_k  where a=A[0,0], b=A[0,1], c=A[0,2], d=A[0,3]
    #   C[1,k] = e*x_k + f*y_k + g*z_k + h*w_k  where e=A[1,0], f=A[1,1], g=A[1,2], h=A[1,3]
    
    # We can write this as:
    #   [C[0,k]] = [a b c d] * [x_k]
    #   [C[1,k]]   [e f g h]   [y_k]
    #                    [z_k]
    #                    [w_k]
    
    # This is a 2x4 matrix times a 4x1 vector, for each k.
    # The tensor is the product of the 2x4 matrix and 4x7 matrix.
    
    # One approach: use a low-rank factorization of the 4x7 matrix B.
    # But B is arbitrary, so we can't assume structure.
    
    # Let's just stick with the current rank-50 solution and verify it's correct.
    # The key question is whether the current implementation is correct.
    
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0

    def add_strassen_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return

        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        if time.time() > deadline:
            return

        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
        col += 1

        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
        col += 1

        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

    # Build the decomposition
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    if col == rank and verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    else:
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
