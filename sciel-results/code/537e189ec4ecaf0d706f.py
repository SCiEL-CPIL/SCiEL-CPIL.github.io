"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    
    Strategy:
    The multiplication $C = AB$ where $A$ is $2 \times 2$ and $B$ is $2 \times 14$ can be viewed
    as 7 independent $2 \times 2$ matrix multiplications, one for each pair of columns in $B$ (and $C$).
    Specifically, let $B$ be split into blocks $B_0, \dots, B_6$ where $B_t$ is $2 \times 2$.
    Then $C_t = A B_t$.
    
    The rank of $2 \times 2 \times 2$ matrix multiplication is 7 (Strassen's algorithm).
    Since the blocks are independent (no cross-terms between different $t$ in the output $C$ relative to the input $A$ structure in a way that allows sharing rank-1 terms across blocks for the standard decomposition), the total rank is the sum of the ranks of the blocks.
    
    Total Rank = 7 blocks * 7 rank/block = 49.
    
    Index Conventions:
    U: shape (n*m, rank) = (4, 49). Index row = i*m + j.
    V: shape (m*p, rank) = (28, 49). Index row = j*p + k.
    W: shape (p*n, rank) = (28, 49). Index row = k*n + i. (Trace convention: W[k*n+i])
    
    The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i, j, k.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    nm = n * m  # 4
    mp = m * p  # 28
    pn = p * n  # 28

    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49

    # Initialize matrices as lists of lists of zeros
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Strassen's rank-7 decomposition for 2x2 * 2x2
    # Local indices for a block:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # We define the 7 terms (M1..M7) as tuples of (A_coeffs, B_coeffs, C_coeffs)
    # Keys are local indices 0..3 (0=00, 1=01, 2=10, 3=11)
    # Value is the coefficient (+1 or -1)
    
    strassen_terms = [
        # M1 = (A00 + A11)(B00 + B11)
        # C00 += M1, C11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        
        # M2 = (A10 + A11)B00
        # C10 += M2, C11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        
        # M3 = A00(B01 - B11)
        # C01 += M3, C11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        
        # M4 = A11(B10 - B00)
        # C00 += M4, C10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        
        # M5 = (A00 + A01)B11
        # C00 -= M5, C01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        
        # M6 = (A10 - A00)(B00 + B01)
        # C11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        
        # M7 = (A01 - A11)(B10 + B11)
        # C00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_blocks):
        # Block t corresponds to columns 2*t and 2*t+1 of B and C
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Global indices for B (V matrix)
        # V index: j*p + k
        # j=0: k0, k1 -> indices 0*p+k0, 0*p+k1
        # j=1: k0, k1 -> indices 1*p+k0, 1*p+k1
        v_idx = {
            0: 0 * p + k0,  # B00
            1: 0 * p + k1,  # B01
            2: 1 * p + k0,  # B10
            3: 1 * p + k1   # B11
        }
        
        # Global indices for C (W matrix)
        # W index: k*n + i
        # k=k0: i=0, i=1 -> indices k0*n+0, k0*n+1
        # k=k1: i=0, i=1 -> indices k1*n+0, k1*n+1
        w_idx = {
            0: k0 * n + 0,  # C00
            1: k1 * n + 0,  # C01
            2: k0 * n + 1,  # C10
            3: k1 * n + 1   # C11
        }
        
        # Global indices for A (U matrix) are constant across blocks
        # U index: i*m + j
        u_idx = {
            0: 0 * m + 0,  # A00
            1: 0 * m + 1,  # A01
            2: 1 * m + 0,  # A10
            3: 1 * m + 1   # A11
        }

        for s in range(7):
            col = t * 7 + s
            a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]

            # Fill U
            for a_local, coeff in a_coeffs.items():
                u_row = u_idx[a_local]
                U[u_row][col] += coeff

            # Fill V
            for b_local, coeff in b_coeffs.items():
                v_row = v_idx[b_local]
                V[v_row][col] += coeff

            # Fill W
            for c_local, coeff in c_coeffs.items():
                w_row = w_idx[c_local]
                W[w_row][col] += coeff

    # Verification
    # Check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # We use a simple loop. The matrices are small.
    is_valid = True
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                total = 0.0
                for r in range(rank):
                    total += u_row[r] * v_row[r] * w_row[r]
                if abs(total - 1.0) > 1e-9:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            break

    if not is_valid:
        # Fallback to Schoolbook (Rank 56) just in case, though it shouldn't happen
        rank_fb = n * m * p
        U = [[0] * rank_fb for _ in range(nm)]
        V = [[0] * rank_fb for _ in range(mp)]
        W = [[0] * rank_fb for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        # Return the fallback, but the primary path should be valid
        return {"U": U, "V": V, "W": W}

    # Ensure entries are integers where possible
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if isinstance(x, float):
                    rx = round(x)
                    if abs(x - rx) < 1e-9:
                        new_row.append(int(rx))
                    else:
                        new_row.append(x)
                else:
                    new_row.append(x)
            result.append(new_row)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
