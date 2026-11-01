import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen's algorithm on 7 blocks of 2x2.
    """
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    num_blocks = p // 2   # 7
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank  # 49

    # Initialize U, V, W as lists of lists
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Strassen terms for a single 2x2 block
    # Each term is a tuple: (u_coeffs, v_local_coeffs, w_local_coeffs)
    # u_coeffs: dict {u_index: coeff}
    # v_local_coeffs: dict {local_b_index: coeff}
    # w_local_coeffs: dict {local_c_index: coeff}
    
    # Local B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Local C indices: 0=C00, 1=C10, 2=C01, 3=C11
    # Note: The order in C for local indices needs to map to global W indices correctly.
    # Global W index for C[i][k] is k*n + i.
    # Local C00 is C[0][0] -> k=0, i=0 -> w_local 0
    # Local C10 is C[1][0] -> k=0, i=1 -> w_local 1
    # Local C01 is C[0][1] -> k=1, i=0 -> w_local 2
    # Local C11 is C[1][1] -> k=1, i=1 -> w_local 3
    
    # Strassen Formulas:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    terms = [
        # P1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # P3
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # P4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # P5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # P6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for t in range(num_blocks):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        for s in range(strassen_rank):
            col = t * strassen_rank + s
            u_coeffs, v_local, w_local = terms[s]
            
            # Fill U
            for u_idx, coeff in u_coeffs.items():
                U[u_idx][col] = coeff
                
            # Fill V
            # Local 0 -> (0, k0), 1 -> (0, k1), 2 -> (1, k0), 3 -> (1, k1)
            for v_loc, coeff in v_local.items():
                j = v_loc // 2
                k_col = k0 + (v_loc % 2)
                v_idx = j * p + k_col
                V[v_idx][col] = coeff
                
            # Fill W
            # Local 0 -> C(0, k0), 1 -> C(1, k0), 2 -> C(0, k1), 3 -> C(1, k1)
            # W index = k * n + i
            for w_loc, coeff in w_local.items():
                i = w_loc % 2
                k_col = k0 + (w_loc // 2)
                w_idx = k_col * n + i
                W[w_idx][col] = coeff

    # Verification (optional but good for safety)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-9:
        # Fallback to schoolbook rank-56
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj][col] = 1
                    V_fb[jj * p + kk][col] = 1
                    W_fb[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
