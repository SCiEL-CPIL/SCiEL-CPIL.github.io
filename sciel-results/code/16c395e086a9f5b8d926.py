import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 120.0  # wall-clock budget (s); evaluator kills at 150 s

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _check_deadline():
    if _time_left() < 5.0:
        raise TimeoutError("Deadline exceeded")

def construct():
    _check_deadline()
    # Dimensions: n=2 (rows of A), m=3 (inner), p=9 (cols of B)
    n, m, p = 2, 3, 9

    # Rank 50 explicit construction:
    #  4 Strassen blocks over B columns (0,1),(2,3),(4,5),(6,7):  4*7 = 28 terms
    #  1 naive 2x1x2x1 block   over B column 8:                   4      terms
    #  9 + 9 naive single-product terms for A_right * B_bottom:   18     terms
    # ---------------------------------------------------------------------------
    # Total rank = 50
    rank = 50

    # U has shape (n*m, rank) = (6, 50)
    # V has shape (m*p, rank) = (27, 50)
    # W has shape (p*n, rank) = (18, 50)
    # Indexing convention (matching the seed / task contract):
    #   A index  = i*m + j       (i in 0..n-1, j in 0..m-1)
    #   B index  = j*p + k       (j in 0..m-1, k in 0..p-1)
    #   C index  = k*n + i       (k in 0..p-1, i in 0..n-1)   ← trace convention

    U = [[0] * rank for _ in range(n * m)]   # 6 x 50
    V = [[0] * rank for _ in range(m * p)]   # 27 x 50
    W = [[0] * rank for _ in range(p * n)]   # 18 x 50

    # Verify construction correctness with a test case
    # Removed _verify_decomposition call as it was undefined.
    # We will perform a lightweight inline check after construction.

    col = 0

    # ------------------------------------------------------------------
    # Part 1: Strassen-style 2x2 blocks over A_top (2x2) and B_top (2x9)
    # ------------------------------------------------------------------
    # A_top indices:
    #   a = A_{0,0} -> U[0*3+0] = U[0]
    #   b = A_{0,1} -> U[0*3+1] = U[1]
    #   c = A_{1,0} -> U[1*3+0] = U[3]
    #   d = A_{1,1} -> U[1*3+1] = U[4]
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):  # 4 blocks over pairs of B columns
        k0 = 2 * b
        k1 = 2 * b + 1

        # B indices for these columns
        v_e = 0 * p + k0   # B_{0, k0}
        v_f = 0 * p + k1   # B_{0, k1}
        v_g = 1 * p + k0   # B_{1, k0}
        v_h = 1 * p + k1   # B_{1, k1}

        # W indices (trace convention C = k*n + i)
        w_00 = k0 * n + 0   # C_{0, k0}
        w_01 = k1 * n + 0   # C_{0, k1}
        w_10 = k0 * n + 1   # C_{1, k0}
        w_11 = k1 * n + 1   # C_{1, k1}

        # 7 Strassen terms --------------------------------------------------
        # M1 = (a + d)(e + h)
        U[u_a][col] = U[u_a][col] + 1
        U[u_d][col] = U[u_d][col] + 1
        V[v_e][col] = V[v_e][col] + 1
        V[v_h][col] = V[v_h][col] + 1

        # M2 = (c + d) * e
        U[u_c][col+1] = U[u_c][col+1] + 1
        U[u_d][col+1] = U[u_d][col+1] + 1
        V[v_e][col+1] = V[v_e][col+1] + 1

        # M3 = a * (f - h)
        U[u_a][col+2] = U[u_a][col+2] + 1
        V[v_f][col+2] = V[v_f][col+2] + 1
        V[v_h][col+2] = V[v_h][col+2] - 1

        # M4 = d * (g - e)
        U[u_d][col+3] = U[u_d][col+3] + 1
        V[v_g][col+3] = V[v_g][col+3] + 1
        V[v_e][col+3] = V[v_e][col+3] - 1

        # M5 = (a + b) * h
        U[u_a][col+4] = U[u_a][col+4] + 1
        U[u_b][col+4] = U[u_b][col+4] + 1
        V[v_h][col+4] = V[v_h][col+4] + 1

        # M6 = (c - a)(e + f)
        U[u_c][col+5] = U[u_c][col+5] + 1
        U[u_a][col+5] = U[u_a][col+5] - 1
        V[v_e][col+5] = V[v_e][col+5] + 1
        V[v_f][col+5] = V[v_f][col+5] + 1

        # M7 = (b - d)(g + h)
        U[u_b][col+6] = U[u_b][col+6] + 1
        U[u_d][col+6] = U[u_d][col+6] - 1
        V[v_g][col+6] = V[v_g][col+6] + 1
        V[v_h][col+6] = V[v_h][col+6] + 1

        # Reconstruct the 2x2 block output ----------------------------------
        # C_{0, k0} = M1 + M4 - M5 + M7
        W[w_00][col]   += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        # C_{0, k1} = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        # C_{1, k0} = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        # C_{1, k1} = M1 - M2 + M3 + M6
        W[w_11][col]   += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7  # 7 terms per block

    # ------------------------------------------------------------------
    # Part 2: Naive 2x1 block for last B column k = 8
    # A_top right column (A_{0,1}, A_{1,1}) times B column 8
    # Actually re-reading the seed: it uses A_top * B_col for k=8 with
    # the same four A entries (a,b,c,d) as above, and B row entries e=B_{0,8}, g=B_{1,8}
    # This is a 2x2 x 2x1 multiplication done naively: 4 scalar products.
    # ------------------------------------------------------------------
    k8 = 8
    v_e8 = 0 * p + k8   # B_{0, 8}
    v_g8 = 1 * p + k8   # B_{1, 8}

    w_08 = k8 * n + 0   # C_{0, 8}
    w_18 = k8 * n + 1   # C_{1, 8}

    # P1 = a * B_{0,8}
    U[u_a][col]   += 1
    V[v_e8][col]  += 1
    W[w_08][col]  += 1

    # P2 = b * B_{1,8}   (contributing to C_{0,8})
    # Wait – in a 2x2 * 2x1 multiply:
    #   C_{0,0} = a*e + b*g   (row 0 of A dot column of B)
    #   C_{1,0} = c*e + d*g   (row 1 of A dot column of B)
    # So:
    # P2 = b * B_{1,8} contributes to C_{0,8}
    U[u_b][col+1] += 1
    V[v_g8][col+1] += 1
    W[w_08][col+1] += 1

    # P3 = c * B_{0,8} contributes to C_{1,8}
    U[u_c][col+2] += 1
    V[v_e8][col+2] += 1
    W[w_18][col+2] += 1

    # P4 = d * B_{1,8} contributes to C_{1,8}
    U[u_d][col+3] += 1
    V[v_g8][col+3] += 1
    W[w_18][col+3] += 1

    col += 4

    # ------------------------------------------------------------------
    # Part 3: A_right (col 2) × B_bottom (row 2):  2 x 1 × 1 x 9  →  18 terms
    # ------------------------------------------------------------------
    u_ar0 = 0 * m + 2   # A_{0,2}
    u_ar1 = 1 * m + 2   # A_{1,2}

    for k in range(p):
        v_bk = 2 * p + k   # B_{2, k}
        w_0k = k * n + 0   # C_{0, k} from A_{0,2}
        w_1k = k * n + 1   # C_{1, k} from A_{1,2}

        # Q_k = A_{0,2} * B_{2,k}  → C_{0,k}
        U[u_ar0][col]   += 1
        V[v_bk][col]    += 1
        W[w_0k][col]    += 1

        # R_k = A_{1,2} * B_{2,k}  → C_{1,k}
        U[u_ar1][col+1] += 1
        V[v_bk][col+1]  += 1
        W[w_1k][col+1]  += 1

        col += 2

    # At this point col should equal rank (50). Assert for safety.
    # (kept as light check; no print on success path)
    assert col == rank, f"Rank mismatch: col={col}, rank={rank}"

    # Lightweight verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Generate a random test case
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = A_test @ B_test
    
    # Reconstruct C using the decomposition
    # C_{i,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # This is a tensor contraction.
    # Efficient way:
    # For each r, we have outer product U[:,r] (n*m) and V[:,r] (m*p)
    # This forms a (n*m) x (m*p) matrix.
    # We need to contract indices.
    # Let's just loop over r for simplicity in verification, as rank is small (50).
    C_recon = np.zeros((n, p))
    for r in range(rank):
        # U_col = U[:, r] shape (n*m,)
        # V_col = V[:, r] shape (m*p,)
        # We need to sum over j.
        # C[i, k] += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        # Reshape U_col to (n, m)
        U_col_reshaped = U_np[:, r].reshape(n, m)
        # Reshape V_col to (m, p)
        V_col_reshaped = V_np[:, r].reshape(m, p)
        # Product is (n, p)
        P_r = U_col_reshaped @ V_col_reshaped
        # Scale by W
        # W[:, r] is shape (p*n,). Reshape to (p, n) then transpose to (n, p)?
        # W index is k*n + i. So W[k*n+i, r].
        # We want C[i, k] += P_r[i, k] * W[k*n + i, r].
        # W_col = W[:, r]. Reshape to (p, n).
        W_col_reshaped = W_np[:, r].reshape(p, n)
        # Transpose to (n, p) to match P_r
        W_col_reshaped = W_col_reshaped.T
        C_recon += P_r * W_col_reshaped
        
    if not np.allclose(C_test, C_recon, atol=1e-10):
        raise ValueError("Verification failed")

    _check_deadline()
    return {"U": U, "V": V, "W": W}
