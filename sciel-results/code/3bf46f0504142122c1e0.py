import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Build the matrix multiplication tensor T[a,b,c] = delta_{a,c} for a,c in [0,6), b in [0,6)
    # where a = row index, b = col-of-A / row-of-B, c = col index
    T = np.zeros((n, m, p), dtype=float)
    for a in range(n):
        for c in range(p):
            if a == c:
                for b in range(m):
                    T[a, b, c] = 1.0

    # Define Strassen decomposition at the 2x2 block level (block size = 3)
    bs = 3
    nblocks = n // bs  # 2

    # Strassen: 7 multiplications instead of 4 block multiplications
    # Each "multiplication" is a 3x3 x 3x3 schoolbook (27 rank-1 terms)
    # Total rank = 7 * 27 = 189

    # For each Strassen step s, define:
    #   A_s = sum_{(br,bc)} alpha_{s,br,bc} * A[br*bs:(br+1)*bs, bc*bs:(bc+1)*bs]
    #   B_s = sum_{(br,bc)} beta_{s,br,bc} * B[br*bs:(br+1)*bs, bc*bs:(bc+1)*bs]
    #   C += gamma_{s,cbr,cbc} * A_s @ B_s  (for each output block)

    # Strassen definitions: (list of (block_row, block_col, coefficient))
    strassen_A = [
        [(0,0,1.0), (1,1,1.0)],       # s=0: A00+A11
        [(1,0,1.0), (1,1,1.0)],       # s=1: A10+A11
        [(0,0,1.0)],                   # s=2: A00
        [(1,1,1.0)],                   # s=3: A11
        [(0,0,1.0), (0,1,1.0)],       # s=4: A00+A01
        [(1,0,1.0), (0,0,-1.0)],      # s=5: A10-A00
        [(0,1,1.0), (1,1,-1.0)],      # s=6: A01-A11
    ]
    strassen_B = [
        [(0,0,1.0), (1,1,1.0)],       # s=0: B00+B11
        [(0,0,1.0)],                   # s=1: B00
        [(0,1,1.0), (1,1,-1.0)],      # s=2: B01-B11
        [(1,0,1.0), (0,0,-1.0)],      # s=3: B10-B00
        [(1,1,1.0)],                   # s=4: B11
        [(0,0,1.0), (0,1,1.0)],       # s=5: B00+B01
        [(1,0,1.0), (1,1,1.0)],       # s=6: B10+B11
    ]
    # Reconstruction: C[cr,cc] += gamma * M_s
    recon_gamma = [
        # s=0: C00+=1, C11+=1
        {(0,0): 1.0, (1,1): 1.0},
        # s=1: C10+=1, C11+=-1
        {(1,0): 1.0, (1,1): -1.0},
        # s=2: C01+=1, C11+=1
        {(0,1): 1.0, (1,1): 1.0},
        # s=3: C00+=1, C10+=1
        {(0,0): 1.0, (1,0): 1.0},
        # s=4: C00+=-1, C01+=1
        {(0,0): -1.0, (0,1): 1.0},
        # s=5: C11+=1
        {(1,1): 1.0},
        # s=6: C00+=1
        {(0,0): 1.0},
    ]

    n_strassen = 7
    rank = n_strassen * bs * bs * bs  # 7 * 27 = 189

    # Build factor tensors as 3D arrays then flatten
    # U_tensor[a, b, r] : contribution of U
    # V_tensor[b, c, r] : contribution of V
    # W_tensor[c, a, r] : contribution of W (note: W index is [j*n+i, r] per convention)

    U_t = np.zeros((n, m, rank), dtype=float)
    V_t = np.zeros((m, p, rank), dtype=float)
    W_t = np.zeros((p, n, rank), dtype=float)

    r_base = 0
    for s in range(n_strassen):
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    r = r_base + s * (bs*bs*bs) + u*(bs*bs) + v*bs + j

                    # U: for each (a,b) in global, U[a,b,r] = sum over strassen_A entries
                    for (abr, abc, ca) in strassen_A[s]:
                        ga = abr * bs + u  # global row in A
                        gb = abc * bs + v  # global col in A
                        U_t[ga, gb, r] += ca

                    # V: for each (b,c) in global, V[b,c,r] = sum over strassen_B entries
                    for (bbr, bbc, cb) in strassen_B[s]:
                        gb = bbr * bs + v  # global row in B
                        gc = bbc * bs + j  # global col in B
                        V_t[gb, gc, r] += cb

                    # W: for each (c,a) in global, W[c,a,r] = sum over recon_gamma
                    for (cr, cc), g in recon_gamma[s].items():
                        gc = cc * bs + j  # global col
                        ga = cr * bs + u  # global row
                        W_t[gc, ga, r] += g

    # Verify: sum_r U[a,b,r] * V[b,c,r] * W[c,a,r] should equal T[a,b,c]
    # For efficiency, do a partial check
    if time.time() < deadline - 5:
        # Check all entries (n*m*p = 216 entries, rank=189)
        contraction = np.einsum('abr,bcr,car->abc', U_t, V_t, W_t)
        max_err = np.max(np.abs(contraction - T))
        if max_err > 1e-10:
            raise ValueError(f"Tensor decomposition verification failed: max_err={max_err}")

    # Convert to matrix form
    # U matrix: shape (n*m, rank), row index = a*m + b
    U_mat = np.zeros((n * m, rank), dtype=float)
    for a in range(n):
        for b in range(m):
            U_mat[a * m + b, :] = U_t[a, b, :]

    # V matrix: shape (m*p, rank), row index = b*p + c
    V_mat = np.zeros((m * p, rank), dtype=float)
    for b in range(m):
        for c in range(p):
            V_mat[b * p + c, :] = V_t[b, c, :]

    # W matrix: shape (p*n, rank), row index = c*n + a  (trace convention W[k*n+i, r])
    W_mat = np.zeros((p * n, rank), dtype=float)
    for c in range(p):
        for a in range(n):
            W_mat[c * n + a, :] = W_t[c, a, :]

    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
