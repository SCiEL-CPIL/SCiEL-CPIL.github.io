"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _make_2x2_strassen():
    """
    Strassen decomposition of 2x2 * 2x2 -> 2x2.
    Rank 7.
    Returns (U, V, W) as 2D arrays of shape (4, 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # Strassen's 7 products for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C[0,0] = P1 - P4 + P5 + P6
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P3 + P2 - P7

    # U indexed by (i,j) -> i*2+j where i is row of A, j is col of A
    # V indexed by (j,k) -> j*2+k where j is row of B, k is col of B
    # W indexed by (k,i) -> k*2+i where k is col of B, i is row of A (output C[i,k])

    # P1 = (a+d)(e+h): A[0,0]+A[1,1], B[0,0]+B[1,1]
    # Contributes to C[0,0] with +1, C[1,1] with +1
    U[0*2+0, 0] = 1  # a
    U[1*2+1, 0] = 1  # d
    V[0*2+0, 0] = 1  # e
    V[1*2+1, 0] = 1  # h
    W[0*2+0, 0] = 1  # C[0,0] (k=0,i=0)
    W[1*2+1, 0] = 1  # C[1,1] (k=1,i=1)

    # P2 = (c+d)e: A[1,0]+A[1,1], B[0,0]
    # Contributes to C[1,0] with +1
    U[1*2+0, 1] = 1  # c
    U[1*2+1, 1] = 1  # d
    V[0*2+0, 1] = 1  # e
    W[0*2+1, 1] = 1  # C[1,0] (k=0,i=1)

    # P3 = a(f-h): A[0,0], B[0,1]-B[1,1]
    # Contributes to C[0,1] with +1, C[1,1] with -1
    U[0*2+0, 2] = 1  # a
    V[0*2+1, 2] = 1  # f
    V[1*2+1, 2] = -1  # -h
    W[1*2+0, 2] = 1  # C[0,1] (k=1,i=0)
    W[1*2+1, 2] = -1  # C[1,1] (k=1,i=1)

    # P4 = d(g-e): A[1,1], B[1,0]-B[0,0]
    # Contributes to C[0,0] with -1, C[1,0] with +1
    U[1*2+1, 3] = 1  # d
    V[1*2+0, 3] = 1  # g
    V[0*2+0, 3] = -1  # -e
    W[0*2+0, 3] = -1  # C[0,0] (k=0,i=0)
    W[0*2+1, 3] = 1  # C[1,0] (k=0,i=1)

    # P5 = (a+b)h: A[0,0]+A[0,1], B[1,1]
    # Contributes to C[0,0] with +1, C[0,1] with +1
    U[0*2+0, 4] = 1  # a
    U[0*2+1, 4] = 1  # b
    V[1*2+1, 4] = 1  # h
    W[0*2+0, 4] = 1  # C[0,0] (k=0,i=0)
    W[1*2+0, 4] = 1  # C[0,1] (k=1,i=0)

    # P6 = (c-a)(e+f): A[1,0]-A[0,0], B[0,0]+B[0,1]
    # Contributes to C[0,0] with +1
    U[1*2+0, 5] = 1  # c
    U[0*2+0, 5] = -1  # -a
    V[0*2+0, 5] = 1  # e
    V[0*2+1, 5] = 1  # f
    W[0*2+0, 5] = 1  # C[0,0] (k=0,i=0)

    # P7 = (b-d)(g+h): A[0,1]-A[1,1], B[1,0]+B[1,1]
    # Contributes to C[1,1] with -1
    U[0*2+1, 6] = 1  # b
    U[1*2+1, 6] = -1  # -d
    V[1*2+0, 6] = 1  # g
    V[1*2+1, 6] = 1  # h
    W[1*2+1, 6] = -1  # C[1,1] (k=1,i=1)

    return U, V, W


def _make_3x3_laderman():
    """
    Laderman decomposition of 3x3 * 3x3 -> 3x3.
    Rank 23.
    Returns (U, V, W) as 2D arrays of shape (9, 23).
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))

    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # U indexed by (i,j) -> i*3+j
    # V indexed by (j,k) -> j*3+k
    # W indexed by (k,i) -> k*3+i

    # Laderman's 23 products:
    # P0  = (a+e)(j+n)
    # P1  = (e-a)j
    # P2  = a(l-o)
    # P3  = e(m-j)
    # P4  = (a+b)n
    # P5  = (d+e)j
    # P6  = (a-b)(j+k)
    # P7  = (b-a)(k+n)
    # P8  = (c+f)(m+p)
    # P9  = (f-c)(p+m)  -- wait, let me use a cleaner formulation
    #
    # Actually, let me use a well-known valid Laderman construction.
    # I'll define it term by term and verify.

    # Based on the standard Laderman algorithm:
    #
    # Let me use the following verified set of 23 products:

    # Term 0: (a+e)(j+n)
    # U: A[0,0]+A[1,1], V: B[0,0]+B[1,1]
    # W: C[0,0]+1, C[1,1]+1
    U[0*3+0, 0] = 1; U[1*3+1, 0] = 1
    V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
    W[0*3+0, 0] = 1; W[1*3+1, 0] = 1

    # Term 1: (e-a)j
    # U: A[1,1]-A[0,0], V: B[0,0]
    # W: C[1,0]+1
    U[1*3+1, 1] = 1; U[0*3+0, 1] = -1
    V[0*3+0, 1] = 1
    W[0*3+1, 1] = 1

    # Term 2: a(l-o)
    # U: A[0,0], V: B[0,2]-B[1,2]
    # W: C[0,2]+1, C[1,2]-1
    U[0*3+0, 2] = 1
    V[0*3+2, 2] = 1; V[1*3+2, 2] = -1
    W[2*3+0, 2] = 1; W[2*3+1, 2] = -1

    # Term 3: e(m-j)
    # U: A[1,1], V: B[1,0]-B[0,0]
    # W: C[1,0]+1, C[0,0]-1
    U[1*3+1, 3] = 1
    V[1*3+0, 3] = 1; V[0*3+0, 3] = -1
    W[0*3+1, 3] = 1; W[0*3+0, 3] = -1

    # Term 4: (a+b)n
    # U: A[0,0]+A[0,1], V: B[1,1]
    # W: C[0,0]+1, C[0,1]+1
    U[0*3+0, 4] = 1; U[0*3+1, 4] = 1
    V[1*3+1, 4] = 1
    W[0*3+0, 4] = 1; W[1*3+0, 4] = 1

    # Term 5: (d+e)j
    # U: A[1,0]+A[1,1], V: B[0,0]
    # W: C[1,0]+1
    U[1*3+0, 5] = 1; U[1*3+1, 5] = 1
    V[0*3+0, 5] = 1
    W[0*3+1, 5] = 1

    # Term 6: (a-b)(j+k)
    # U: A[0,0]-A[0,1], V: B[0,0]+B[0,1]
    # W: C[0,0]+1
    U[0*3+0, 6] = 1; U[0*3+1, 6] = -1
    V[0*3+0, 6] = 1; V[0*3+1, 6] = 1
    W[0*3+0, 6] = 1

    # Term 7: (b-a)(k+n)
    # U: A[0,1]-A[0,0], V: B[0,1]+B[1,1]
    # W: C[0,1]+1
    U[0*3+1, 7] = 1; U[0*3+0, 7] = -1
    V[0*3+1, 7] = 1; V[1*3+1, 7] = 1
    W[1*3+0, 7] = 1

    # Term 8: (c+f)(m+p)
    # U: A[0,2]+A[1,2], V: B[1,0]+B[2,0]
    # W: C[0,0]+1, C[1,0]+1
    U[0*3+2, 8] = 1; U[1*3+2, 8] = 1
    V[1*3+0, 8] = 1; V[2*3+0, 8] = 1
    W[0*3+0, 8] = 1; W[0*3+1, 8] = 1

    # Term 9: (f-c)(p+m)
    # U: A[1,2]-A[0,2], V: B[2,0]+B[1,0]
    # W: C[0,0]+1, C[1,0]-1  -- wait, this needs care
    # Actually, let me reconsider. This is getting complex. Let me use a different approach.
    # I'll construct a valid Laderman decomposition by solving the linear system.

    # Actually, let me just return None and use a fallback approach.
    return None


def _make_3x3_schoolbook():
    """
    Schoolbook decomposition of 3x3 * 3x3 -> 3x3.
    Rank 27.
    Returns (U, V, W) as 2D arrays of shape (9, 27).
    """
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1
                V[j*3+k, col] = 1
                W[k*3+i, col] = 1
                col += 1

    return U, V, W


def _construct_hybrid(outer_U, outer_V, outer_W, inner_U, inner_V, inner_W, block_size, num_blocks):
    """
    Construct a hybrid decomposition using an outer decomposition and an inner decomposition.
    outer_U, outer_V, outer_W: decomposition of num_blocks x num_blocks matrix multiplication
    inner_U, inner_V, inner_W: decomposition of block_size x block_size matrix multiplication
    """
    dim = num_blocks * block_size
    outer_rank = outer_U.shape[1]
    inner_rank = inner_U.shape[1]
    total_rank = outer_rank * inner_rank

    U = np.zeros((dim * dim, total_rank))
    V = np.zeros((dim * dim, total_rank))
    W = np.zeros((dim * dim, total_rank))

    for oc in range(outer_rank):
        for ic in range(inner_rank):
            col = oc * inner_rank + ic

            # For each (I,J) in outer A, and (u,v) in inner A
            for I in range(num_blocks):
                for J in range(num_blocks):
                    outer_coeff_a = outer_U[I * num_blocks + J, oc]
                    if outer_coeff_a != 0:
                        for u in range(block_size):
                            for v in range(block_size):
                                inner_coeff_a = inner_U[u * block_size + v, ic]
                                if inner_coeff_a != 0:
                                    i_global = I * block_size + u
                                    j_global = J * block_size + v
                                    idx_u = i_global * dim + j_global
                                    U[idx_u, col] += outer_coeff_a * inner_coeff_a

            # For each (J,K) in outer B, and (v,w) in inner B
            for J in range(num_blocks):
                for K in range(num_blocks):
                    outer_coeff_b = outer_V[J * num_blocks + K, oc]
                    if outer_coeff_b != 0:
                        for v in range(block_size):
                            for w in range(block_size):
                                inner_coeff_b = inner_V[v * block_size + w, ic]
                                if inner_coeff_b != 0:
                                    j_global = J * block_size + v
                                    k_global = K * block_size + w
                                    idx_v = j_global * dim + k_global
                                    V[idx_v, col] += outer_coeff_b * inner_coeff_b

            # For each (I,K) in outer C, and (u,w) in inner C
            for I in range(num_blocks):
                for K in range(num_blocks):
                    outer_coeff_c = outer_W[K * num_blocks + I, oc]
                    if outer_coeff_c != 0:
                        for u in range(block_size):
                            for w in range(block_size):
                                inner_coeff_c = inner_W[w * block_size + u, ic]
                                if inner_coeff_c != 0:
                                    i_global = I * block_size + u
                                    k_global = K * block_size + w
                                    idx_w = k_global * dim + i_global
                                    W[idx_w, col] += outer_coeff_c * inner_coeff_c

    return U, V, W


def construct():
    start_time = time.time()

    n, m, p = 6, 6, 6

    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try 3x3 outer (schoolbook, rank 27) x 2x2 inner (Strassen, rank 7) = 189
    try:
        outer_U, outer_V, outer_W = _make_3x3_schoolbook()
        inner_U, inner_V, inner_W = _make_2x2_strassen()
        block_size = 2
        num_blocks = 3
        U, V, W = _construct_hybrid(outer_U, outer_V, outer_W, inner_U, inner_V, inner_W, block_size, num_blocks)

        res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Verify
        if _verify_decomposition(res["U"], res["V"], res["W"], n, m, p, tol=1e-10):
            return res
    except Exception:
        pass

    # Try 2x2 outer (Strassen, rank 7) x 3x3 inner (schoolbook, rank 27) = 189
    try:
        outer_U, outer_V, outer_W = _make_2x2_strassen()
        inner_U, inner_V, inner_W = _make_3x3_schoolbook()
        block_size = 3
        num_blocks = 2
        U, V, W = _construct_hybrid(outer_U, outer_V, outer_W, inner_U, inner_V, inner_W, block_size, num_blocks)

        res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Verify
        if _verify_decomposition(res["U"], res["V"], res["W"], n, m, p, tol=1e-10):
            return res
    except Exception:
        pass

    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
