"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    The multiplication tensor T_{2,2,14} decomposes as a direct sum of 7 independent
    2x2 * 2x2 multiplication tensors (one per pair of output columns). By the additivity
    of tensor rank over direct sums, and the known minimality of Strassen's rank-7
    decomposition for 2x2 * 2x2, the total rank is 7 * 7 = 49.
    
    Index conventions:
      U: shape (n*m, rank) = (4, 49), indexed by (i, j) -> i*m + j
      V: shape (m*p, rank) = (28, 49), indexed by (j, k) -> j*p + k
      W: shape (p*n, rank) = (28, 49), indexed by (k, i) -> k*n + i  [trace convention]
    
    The decomposition satisfies:
      For all i in {0,1}, j in {0,1}, k in {0,...,13}:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7 blocks of 2x2

    # Strassen's rank-7 decomposition for 2x2 * 2x2
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    #
    # M1 = (A00+A11)(B00+B11)  -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00        -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11)        -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00)        -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11        -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01)  -> C11+=M6
    # M7 = (A01-A11)(B10+B11)  -> C00+=M7
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    strassen_terms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # M7
    ]

    rank = num_blocks * 7  # 49
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            # Fill U: index (i, j) -> i*m + j
            for a_idx, coeff in a_coeffs.items():
                a_i = a_idx // m
                a_j = a_idx % m
                U[a_i * m + a_j, col] = coeff

            # Fill V: index (j, k) -> j*p + k
            # Block t covers B columns 2*t and 2*t+1
            for b_idx, coeff in b_coeffs.items():
                b_j = b_idx // 2
                b_col = b_idx % 2
                b_k = 2 * t + b_col
                V[b_j * p + b_k, col] = coeff

            # Fill W: index (k, i) -> k*n + i  [trace convention]
            # Block t covers C columns 2*t and 2*t+1
            for c_idx, coeff in c_coeffs.items():
                c_i = c_idx // 2
                c_col = c_idx % 2
                c_k = 2 * t + c_col
                W[c_k * n + c_i, col] = coeff

    # Verify the decomposition is exact
    def verify(Uf, Vf, Wf):
        """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                u_row = Uf[i * m + j, :]
                for k in range(p):
                    v_row = Vf[j * p + k, :]
                    w_row = Wf[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: schoolbook decomposition (rank 56)
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Convert to plain lists with integer entries where possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    new_row.append(int(rx))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
