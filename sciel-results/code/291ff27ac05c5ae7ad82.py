"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, num_trials=5, seed=12345):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for _ in range(num_trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        # Efficient verification using matrix multiplication of factor matrices
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can reshape and use einsum or simple loops for small sizes
        # For n=4, m=4, p=6, rank~77, this is fast enough
        for i in range(n):
            for k in range(p):
                s = 0.0
                # Extract vectors for this (i,k)
                # U row: i*m + j for j in 0..m-1 -> length m
                # V col: j*p + k for j in 0..m-1 -> length m
                # W row: k*n + i -> length 1 (scalar in W matrix row? No, W is (p*n, rank))
                # Wait, the tensor is sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
                # Let's just do the direct sum for clarity and safety
                w_vec = W_np[k * n + i, :]
                # Sum over j
                for j in range(m):
                    u_val = U_np[i * m + j, :]
                    v_val = V_np[j * p + k, :]
                    s += np.dot(np.dot(u_val, v_val), w_vec)
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _build_schoolbook(n, m, p):
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

def _strassen_2x2_factors():
    """
    Return the 7 Strassen factors for 2x2 * 2x2 matrix multiplication.
    Each factor is a dict with 'U', 'V', 'W' keys mapping (row, col) to value.
    """
    factors = []
    # M1 = (A11 + A22)(B11 + B22)
    factors.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): -1}
    })
    # M2 = (A21 + A22)B11
    factors.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): 1}
    })
    # M3 = A11(B21 - B11)
    factors.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4 = A22(B12 - B22)
    factors.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A11 + A12)B22
    factors.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): 1, (0,1): 1}
    })
    # M6 = (A21 - A11)(B11 + B12)
    factors.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7 = (A12 - A22)(B21 + B22)
    factors.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,1): 1}
    })
    return factors

def _build_strassen_4x4_4x4():
    """
    Build rank-49 Strassen decomposition for 4x4 * 4x4.
    Uses 2 levels of Strassen.
    """
    level1_factors = _strassen_2x2_factors()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    col = 0
    for l1 in range(7):
        for l2 in range(7):
            f1 = level1_factors[l1]
            f2 = level1_factors[l2]

            # U: A matrix
            for (bi, bj), val1 in f1['U'].items():
                for (si, sj), val2 in f2['U'].items():
                    ai = bi * 2 + si
                    aj = bj * 2 + sj
                    U[ai * 4 + aj][col] += val1 * val2

            # V: B matrix
            for (bi, bj), val1 in f1['V'].items():
                for (si, sj), val2 in f2['V'].items():
                    bri = bi * 2 + si
                    bcj = bj * 2 + sj
                    V[bri * 4 + bcj][col] += val1 * val2

            # W: C matrix
            for (bi, bj), val1 in f1['W'].items():
                for (si, sj), val2 in f2['W'].items():
                    cri = bi * 2 + si
                    ccj = bj * 2 + sj
                    # W index is k*n + i. Here ccj is k (col of C), cri is i (row of C)
                    W[ccj * 4 + cri][col] += val1 * val2

            col += 1

    return U, V, W

def _build_strassen_4x4_4x2():
    """
    Build rank-28 decomposition for 4x4 * 4x2.
    Splits into 2 (row blocks of A) * 1 (col block of B, which is 2 wide) * 7 (Strassen 2x2*2x2? No).
    Wait, 4x4 * 4x2.
    A: 2x2 grid of 2x2 blocks.
    B: 2x1 grid of 2x2 blocks.
    C: 2x1 grid of 2x2 blocks.
    
    For each A row block (2 options) and B col block (1 option, since B is 4x2, split into 2 rows x 2 cols? No, 4x2 is 2 row blocks of 2x2 and 1 col block of 2x2? No, 4x2 has 2 row blocks and 1 col block if we view it as 2x2 blocks? 
    Actually, 4x2 can be viewed as two 2x2 blocks stacked vertically.
    So B has 2 row blocks and 1 col block.
    A has 2 row blocks and 2 col blocks.
    C has 2 row blocks and 1 col block.
    
    The multiplication is C = A * B.
    C_{r, c} = sum_j A_{r, j} B_{j, c}.
    Here r is row block index (0,1), c is col block index (0).
    j is the intermediate index. For A, j is col block (0,1). For B, j is row block (0,1).
    So for each r in {0,1} and c in {0}:
      C_{r,c} = A_{r,0} B_{0,c} + A_{r,1} B_{1,c}
      
    This is a sum of two 2x2 * 2x2 multiplications.
    We can use Strassen for each 2x2 * 2x2 multiplication.
    Total rank = 2 (sum terms) * 7 (Strassen rank) * 2 (C row blocks) = 28.
    
    Let's implement this.
    """
    factors = _strassen_2x2_factors()
    rank = 28
    U = [[0] * rank for _ in range(16)]  # 4x4
    V = [[0] * rank for _ in range(8)]   # 4x2
    W = [[0] * rank for _ in range(8)]   # 4x2

    col = 0
    # r is C row block (0,1)
    # c is C col block (0)
    # j is intermediate (0,1)
    for r in range(2):
        for c in range(1):
            for j in range(2):
                for f in range(7):
                    factor = factors[f]
                    
                    # U: A block at (r, j)
                    # A is 4x4. Block (r, j) is rows 2r..2r+1, cols 2j..2j+1
                    for (ri, ci), val in factor['U'].items():
                        a_row = 2 * r + ri
                        a_col = 2 * j + ci
                        U[a_row * 4 + a_col][col] += val

                    # V: B block at (j, c)
                    # B is 4x2. Block (j, c) is rows 2j..2j+1, cols 2c..2c+1
                    # Note: B col index in the 4x2 matrix is 2c + ci. Since c=0, cols 0,1.
                    for (ri, ci), val in factor['V'].items():
                        b_row = 2 * j + ri
                        b_col = 2 * c + ci
                        V[b_row * 2 + b_col][col] += val

                    # W: C block at (r, c)
                    # C is 4x2. Block (r, c) is rows 2r..2r+1, cols 2c..2c+1
                    for (ri, ci), val in factor['W'].items():
                        c_row = 2 * r + ri
                        c_col = 2 * c + ci
                        # W index: k*n + i. Here k is col of C (c_col), i is row of C (c_row)
                        # n=4.
                        W[c_col * 4 + c_row][col] += val

                    col += 1

    return U, V, W

def _build_strassen_tiled_4x4_4x6():
    """
    Build a Strassen-tiled decomposition for 4x4 * 4x6.
    Split B into 4x4 (cols 0-3) and 4x2 (cols 4-5).
    Total rank: 49 + 28 = 77.
    """
    n, m, p = 4, 4, 6
    rank_1 = 49
    rank_2 = 28
    rank_total = rank_1 + rank_2

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    # Part 1: 4x4 * 4x4 (cols 0-3 of B)
    U1, V1, W1 = _build_strassen_4x4_4x4()
    for i in range(16):
        for c in range(rank_1):
            U[i][c] = U1[i][c]
            V[i][c] = V1[i][c]
            W[i][c] = W1[i][c]

    # Part 2: 4x4 * 4x2 (cols 4-5 of B)
    U2, V2, W2 = _build_strassen_4x4_4x2()
    offset = rank_1
    for i in range(16):
        for c in range(rank_2):
            U[i][c + offset] = U2[i][c]
    
    # V2 is 8x28 (4x2 matrix). We need to place it in V (24x77).
    # V2 row i corresponds to B row i (0..3).
    # V2 col j corresponds to B col j (0..1).
    # In the full B (4x6), these are cols 4 and 5.
    # So V2[i, c] goes to V[i * 6 + (4 + j), c + offset]?
    # Wait, V2 is built for a 4x2 matrix. Its V matrix is (m*p2) x rank = (4*2) x 28 = 8x28.
    # The rows of V2 are indexed by (row, col) of the 4x2 matrix.
    # Row index in V2: r2 * 2 + c2.
    # In full V, the row index is r_full * 6 + c_full.
    # r_full = r2. c_full = 4 + c2.
    for i in range(8):
        r2 = i // 2
        c2 = i % 2
        full_v_idx = r2 * 6 + (4 + c2)
        for c in range(rank_2):
            V[full_v_idx][c + offset] = V2[i][c]

    # W2 is 8x28 (4x2 matrix output). Its W matrix is (p2*n) x rank = (2*4) x 28 = 8x28.
    # Rows of W2 are indexed by (col, row) of the 4x2 output matrix.
    # Row index in W2: k2 * 4 + i2.
    # In full W, the row index is k_full * 4 + i_full.
    # k_full = 4 + k2. i_full = i2.
    for i in range(8):
        k2 = i // 4
        i2 = i % 4
        full_w_idx = (4 + k2) * 4 + i2
        for c in range(rank_2):
            W[full_w_idx][c + offset] = W2[i][c]

    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    best = baseline
    best_rank = n * m * p

    # Try Strassen-tiled approach (should give rank 77 if correct)
    try:
        strassen = _build_strassen_tiled_4x4_4x6()
        if strassen is not None:
            if _verify_decomposition(strassen["U"], strassen["V"], strassen["W"], n, m, p):
                rank = len(strassen["U"][0])
                if rank < best_rank:
                    best = strassen
                    best_rank = rank
                    print(f"Strassen-tiled verified: rank {rank}")
            else:
                print("Strassen-tiled FAILED verification")
    except Exception as e:
        print(f"Strassen-tiled exception: {e}")

    # If we still have time, we could try to optimize further, but 77 is a solid improvement over 96.
    # Given the 120s limit and the complexity of finding rank < 77 for this size,
    # we return the verified Strassen result.
    
    return best

# EVOLVE-BLOCK-END
