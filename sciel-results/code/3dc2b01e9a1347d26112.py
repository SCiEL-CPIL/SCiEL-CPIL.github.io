"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication
    using Strassen's algorithm on 7 blocks of 2x2.
    
    Tensor convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We need T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,13}.
    
    Strategy: Split B (2x14) into 7 blocks of 2x2. For each block, use Strassen's
    7-term algorithm for 2x2 * 2x2. Total rank = 7 blocks * 7 terms = 49.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28
    num_blocks = p // 2  # 7
    rank = num_blocks * 7  # 49

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen's algorithm for 2x2 * 2x2:
    # A = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B = [[e, f], [g, h]] where e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # C = [[i, j], [k, l]] where i=C[0,0], j=C[0,1], k=C[1,0], l=C[1,1]
    #
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # i = M1 + M4 - M5 + M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M3 + M6

    # Data-driven Strassen terms:
    # Each term: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict from A index (0=a, 1=b, 2=c, 3=d) to coefficient
    # b_coeffs: dict from B index (0=e, 1=f, 2=g, 3=h) to coefficient
    # c_coeffs: dict from C index (0=i, 1=j, 2=k, 3=l) to coefficient
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # i += M1, l += M1
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),  # k += M2, l -= M2
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),  # j += M3, l += M3
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),  # i += M4, k += M4
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),  # i -= M5, j += M5
        # M6 = (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),  # l += M6
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),  # i += M7
    ]

    for t in range(num_blocks):
        base = t * 7
        for s in range(7):
            col = base + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
            
            # U: shape (n*m, rank) = (4, rank)
            # U[i*m + j, col] = coefficient of A[i,j] in term s
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m  # 0 or 1
                a_j = a_idx % m   # 0 or 1
                U[a_i * m + a_j, col] = coeff
            
            # V: shape (m*p, rank) = (28, rank)
            # V[j*p + k, col] = coefficient of B[j,k] in term s for block t
            # Block t covers columns 2*t and 2*t+1 of B
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2   # 0 or 1 (row of B block)
                b_col = b_idx % 2  # 0 or 1 (column within block)
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff
            
            # W: shape (p*n, rank) = (28, rank)
            # W[k*n + i, col] = coefficient of C[i,k] in term s for block t
            # Block t covers columns 2*t and 2*t+1 of C
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2   # 0 or 1 (row of C block)
                c_col = c_idx % 2  # 0 or 1 (column within block)
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Verify correctness
    def verify(Uf, Vf, Wf):
        for i in range(n):
            for j in range(m):
                u_row = Uf[i * m + j]
                for k in range(p):
                    v_row = Vf[j * p + k]
                    w_row = Wf[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook rank-56
        rank_sb = n * m * p  # 56
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Convert to lists with integer entries where possible
    def to_int_list(arr):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in arr.tolist()]

    U_list = to_int_list(U)
    V_list = to_int_list(V)
    W_list = to_int_list(W)

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
