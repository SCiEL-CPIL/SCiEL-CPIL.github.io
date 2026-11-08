import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank expected: 66
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u: list of (idx, coeff) for A
    # v: list of (idx, coeff) for B
    # w: list of (idx, coeff) for C
    def add_term(u, v, w):
        nonlocal col
        for idx_u, coeff_u in u:
            U[idx_u][col] += coeff_u
        for idx_v, coeff_v in v:
            V[idx_v][col] += coeff_v
        for idx_w, coeff_w in w:
            W[idx_w][col] += coeff_w
        col += 1

    # 1. Split m=4 into m1=2, m2=2
    # Block 1: n=3, m=2, p=6 (j in 0,1)
    # Block 2: n=3, m=2, p=6 (j in 2,3)
    
    for block_m in range(2):
        # m_offset for j index
        j_off = block_m * 2
        
        # 2. Split p=6 into p1=2, p2=2, p3=2
        for block_p in range(3):
            # p_offset for k index
            k_off = block_p * 2
            
            # 3. Split n=3 into n1=1, n2=2
            # Part A: n=1, m=2, p=2 (i=0, j in j_off..j_off+1, k in k_off..k_off+1)
            # Part B: n=2, m=2, p=2 (i=1,2, j in j_off..j_off+1, k in k_off..k_off+1)
            
            # --- Part A: Naive 1x2x2 ---
            # A is 1x2: A[0, j0], A[0, j1]
            # B is 2x2: B[j0, k0], B[j0, k1], B[j1, k0], B[j1, k1]
            # C is 1x2: C[0, k0], C[0, k1]
            
            # Term 1: A[0, j0] * B[j0, k0] -> C[0, k0]
            add_term(
                [(0 * m + (j_off + 0), 1)],
                [((j_off + 0) * p + (k_off + 0), 1)],
                [((k_off + 0) * n + 0, 1)]
            )
            
            # Term 2: A[0, j1] * B[j1, k0] -> C[0, k0]
            add_term(
                [(0 * m + (j_off + 1), 1)],
                [((j_off + 1) * p + (k_off + 0), 1)],
                [((k_off + 0) * n + 0, 1)]
            )
            
            # Term 3: A[0, j0] * B[j0, k1] -> C[0, k1]
            add_term(
                [(0 * m + (j_off + 0), 1)],
                [((j_off + 0) * p + (k_off + 1), 1)],
                [((k_off + 1) * n + 0, 1)]
            )
            
            # Term 4: A[0, j1] * B[j1, k1] -> C[0, k1]
            add_term(
                [(0 * m + (j_off + 1), 1)],
                [((j_off + 1) * p + (k_off + 1), 1)],
                [((k_off + 1) * n + 0, 1)]
            )
            
            # --- Part B: Strassen 2x2x2 ---
            # A is 2x2: A[1, j0], A[1, j1], A[2, j0], A[2, j1]
            # B is 2x2: B[j0, k0], B[j0, k1], B[j1, k0], B[j1, k1]
            # C is 2x2: C[1, k0], C[1, k1], C[2, k0], C[2, k1]
            
            # Local indices for A:
            # a11 = A[1, j0], a12 = A[1, j1]
            # a21 = A[2, j0], a22 = A[2, j1]
            
            # Local indices for B:
            # b11 = B[j0, k0], b12 = B[j0, k1]
            # b21 = B[j1, k0], b22 = B[j1, k1]
            
            # Local indices for C:
            # c11 = C[1, k0], c12 = C[1, k1]
            # c21 = C[2, k0], c22 = C[2, k1]
            
            # Strassen Multiplications:
            
            # M1 = (a11 + a22) * (b11 + b22)
            add_term(
                [(1 * m + (j_off + 0), 1), (2 * m + (j_off + 1), 1)],
                [((j_off + 0) * p + (k_off + 0), 1), ((j_off + 1) * p + (k_off + 1), 1)],
                [((k_off + 0) * n + 1, 1), ((k_off + 1) * n + 2, 1)]
            )
            # Note: M1 contributes to c11 (+1), c12 (-1? No, see below), c21 (-1? No), c22 (+1)
            # Standard Strassen:
            # c11 = M1 + M4 - M5 + M7
            # c12 = M3 + M5
            # c21 = M2 + M4
            # c22 = M1 - M2 + M3 + M6
            
            # We need to add the contributions to W for each M.
            # Let's refine add_term to handle W coefficients properly.
            # The current add_term adds 1 to W.
            # We need to add specific coefficients to W for each M.
            
            # Let's restart the Strassen part with explicit W coefficients.
            
            # M1 = (a11+a22)(b11+b22)
            # W contributions: c11(+1), c22(+1)
            # We already added M1 above with W coeffs 1,1.
            # But we need to add -1 to c12? No, c12 doesn't get M1.
            # Wait, the W vector for a rank-1 term is fixed.
            # $T = \sum u_r \otimes v_r \otimes w_r$.
            # $C = \sum u_r (v_r \cdot B)$.
            # So $w_r$ is the output vector.
            # For M1, $u_1 = [1, 0, 1]^T$ (local), $v_1 = [1, 1]^T$ (local flattened? No, $v$ is $2 \times 2$ flattened).
            # $w_1$ should be such that $M_1 \cdot w_1$ gives the correct addition to $C$.
            # $M_1$ is a scalar.
            # $C_{11} += M_1$, $C_{22} += M_1$.
            # So $w_1$ has 1 at $c_{11}$ and 1 at $c_{22}$.
            # My previous add_term for M1 had W: `[((k_off + 0) * n + 1, 1), ((k_off + 1) * n + 2, 1)]`.
            # This is $c_{1, k0}$ and $c_{2, k1}$.
            # $c_{1, k0}$ is $C[1, k0]$. $c_{2, k1}$ is $C[2, k1]$.
            # This matches $c_{11}$ and $c_{22}$ in local coords.
            
            # M2 = (a21+a22) * b11
            # W: c21(+1)
            add_term(
                [(2 * m + (j_off + 0), 1), (2 * m + (j_off + 1), 1)],
                [((j_off + 0) * p + (k_off + 0), 1)],
                [((k_off + 0) * n + 2, 1)]
            )
            
            # M3 = a11 * (b21-b22)
            # W: c12(+1), c22(+1)
            add_term(
                [(1 * m + (j_off + 0), 1)],
                [((j_off + 1) * p + (k_off + 0), 1), ((j_off + 1) * p + (k_off + 1), -1)],
                [((k_off + 1) * n + 1, 1), ((k_off + 1) * n + 2, 1)]
            )
            
            # M4 = a22 * (b12-b11)
            # W: c11(+1), c21(+1)
            add_term(
                [(2 * m + (j_off + 1), 1)],
                [((j_off + 0) * p + (k_off + 1), 1), ((j_off + 0) * p + (k_off + 0), -1)],
                [((k_off + 0) * n + 1, 1), ((k_off + 0) * n + 2, 1)]
            )
            
            # M5 = (a11+a12) * b22
            # W: c11(-1), c12(+1)
            add_term(
                [(1 * m + (j_off + 0), 1), (1 * m + (j_off + 1), 1)],
                [((j_off + 1) * p + (k_off + 1), 1)],
                [((k_off + 0) * n + 1, -1), ((k_off + 1) * n + 1, 1)]
            )
            
            # M6 = (a21-a11) * (b11+b12)
            # W: c22(+1)
            add_term(
                [(2 * m + (j_off + 0), 1), (1 * m + (j_off + 0), -1)],
                [((j_off + 0) * p + (k_off + 0), 1), ((j_off + 0) * p + (k_off + 1), 1)],
                [((k_off + 1) * n + 2, 1)]
            )
            
            # M7 = (a12-a22) * (b21+b22)
            # W: c11(+1), c21(-1)
            add_term(
                [(1 * m + (j_off + 1), 1), (2 * m + (j_off + 1), -1)],
                [((j_off + 1) * p + (k_off + 0), 1), ((j_off + 1) * p + (k_off + 1), 1)],
                [((k_off + 0) * n + 1, 1), ((k_off + 0) * n + 2, -1)]
            )
            
    return {"U": U, "V": V, "W": W}
