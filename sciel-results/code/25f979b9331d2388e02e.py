"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                e = abs(val - 1.0)
                if e > max_err:
                    max_err = e
    return max_err


def _strassen_222():
    """Rank-7 decomposition of 2x2x2 matrix multiplication (Strassen).
    U shape (2*2,7), V shape (2*2,7), W shape (2*2,7).
    T[i,j,k] = sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] == 1 for all i,j,k.
    """
    # Strassen 7 products for 2x2:
    # m1 = (a00) * (b01 - b11)
    # m2 = (a00 + a10) * b00
    # m3 = (a10 - a00) * (b11 + b01)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a11) * (b00 + b11)
    # m6 = (a10 + a11) * (b11 - b10)
    # m7 = (a00 + a01) * (b01 - b00)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # U rows: index i*2+j -> (0,0),(0,1),(1,0),(1,1)
    # V rows: index j*2+k -> (0,0),(0,1),(1,0),(1,1)
    # W rows: index k*2+i -> (0,0),(0,1),(1,0),(1,1)

    # m1: U[0,0]=1 ; V[0,1]=1,V[1,1]=-1 ; W[1,0]=1,W[0,0]=1
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = -1
    W[1, 0] = 1; W[0, 0] = 1
    # m2: U[0,1]=1,U[2,1]=1 ; V[0,1]=1 ; W[0,1]=1
    U[0, 1] = 1; U[2, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1
    # m3: U[2,2]=1,U[0,2]=-1 ; V[1,2]=1,V[0,2]=1 ; W[1,2]=1,W[0,2]=1
    U[2, 2] = 1; U[0, 2] = -1
    V[1, 2] = 1; V[0, 2] = 1
    W[1, 2] = 1; W[0, 2] = 1
    # m4: U[3,3]=1 ; V[1,3]=1,V[0,3]=-1 ; W[0,3]=1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    # m5: U[0,4]=1,U[3,4]=1 ; V[0,4]=1,V[1,4]=1 ; W[1,4]=1,W[0,4]=1
    U[0, 4] = 1; U[3, 4] = 1
    V[0, 4] = 1; V[1, 4] = 1
    W[1, 4] = 1; W[0, 4] = 1
    # m6: U[2,5]=1,U[3,5]=1 ; V[1,5]=1,V[0,5]=-1 ; W[1,5]=1
    U[2, 5] = 1; U[3, 5] = 1
    V[1, 5] = 1; V[0, 5] = -1
    W[1, 5] = 1
    # m7: U[0,6]=1,U[1,6]=1 ; V[0,6]=1,V[1,6]=-1 ; W[0,6]=1
    U[0, 6] = 1; U[1, 6] = 1
    V[0, 6] = 1; V[1, 6] = -1
    W[0, 6] = 1
    return U, V, W


def _embed_sub(n_sub, m_sub, p_sub, U_sub, V_sub, W_sub, i_off, j_off, k_off,
               n, m, p, rank_sub):
    """Place a sub-decomposition into full-size factor matrices (zero-padded)."""
    U = np.zeros((n * m, rank_sub))
    V = np.zeros((m * p, rank_sub))
    W = np.zeros((p * n, rank_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for c in range(rank_sub):
                U[(i + i_off) * m + (j + j_off), c] = U_sub[i * m_sub + j, c]
    for j in range(m_sub):
        for k in range(p_sub):
            for c in range(rank_sub):
                V[(j + j_off) * p + (k + k_off), c] = V_sub[j * p_sub + k, c]
    for k in range(p_sub):
        for i in range(n_sub):
            for c in range(rank_sub):
                W[(k + k_off) * n + (i + i_off), c] = W_sub[k * p_sub + i, c]
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin vs 120s

    # Schoolbook fallback (always valid, rank 75)
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)

    # ---- Block decomposition strategy ----
    # Split the 5x5 (m x p) into blocks along m and p:
    #   m = 2 + 3, p = 2 + 3
    # A is 3x5 -> split along m: A1 (3x2), A2 (3x3)
    # B is 5x5 -> 4 blocks: B11(2x2), B12(2x3), B21(3x2), B22(3x3)
    # C is 3x5 -> split along p: C1(3x2), C2(3x3)
    #
    # Sub-problems:
    #   S1: (3x2)*(2x2) -> 3x2   schoolbook rank 12
    #   S2: (3x2)*(2x3) -> 3x3   schoolbook rank 18
    #   S3: (3x3)*(3x2) -> 3x2   schoolbook rank 18
    #   S4: (3x3)*(3x3) -> 3x3   schoolbook rank 27
    # Total schoolbook: 75.
    #
    # Try to reduce S4 (3x3x3) using a rank-23 known decomposition? Not standard.
    # Instead, try to reduce S1 (3x2x2) and S2/S3 using Strassen-like tricks.
    #
    # A concrete improvement: S1 is (3x2)*(2x2)->3x2.
    # This is 3 independent 1x2 * 2x2 -> 1x2 multiplications stacked.
    # Each 1x2 * 2x2 -> 1x2 has rank 4 (schoolbook). 3 of them = 12.
    # Can we do better? 1x2*2x2: C_j = sum_i A_i B_ij. This is bilinear.
    # Actually each output C_j = A_0*B_0j + A_1*B_1j. That's 4 products (2x2).
    # For 3 rows: 12 products. Hard to reduce without structure.
    #
    # Better target: S4 (3x3x3). Known: 3x3x3 matrix mult has rank 23 (no, 
    # actually the rank of 3x3x3 is 23? No — rank of matrix mult 3x3x3 is 23?
    # The rank of the 3x3x3 multiplication tensor is 23? Let me recall: 
    # rank of 2x2x2 = 7, rank of 3x3x3 = 23 (Landsberg lower bound), 
    # and it is known that rank(3x3x3) <= 23 via various constructions.
    # Actually the best known upper bound for 3x3x3 is 23.
    #
    # Rather than risk an incorrect 3x3x3 rank-23, let's use a verified
    # Strassen-based block approach on a 4x4x4 sub-block embedded in 5x5.
    #
    # Alternative safe improvement: use Strassen on overlapping 2x2 blocks.
    # This is complex to verify. Given the risk, let's attempt a careful
    # rank reduction on S4 using a known 3x3x3 rank-23 construction IF we
    # can verify it numerically. Otherwise fall back to schoolbook.

    # Try: replace S4 (3x3x3, rank 27) with a candidate rank-23 decomposition.
    # We attempt to construct and verify it. If verification fails, keep 27.
    candidate_found = False
    U_total = None
    V_total = None
    W_total = None
    total_rank = 0

    # Build S1, S2, S3 with schoolbook (safe)
    s1_rank = 3 * 2 * 2   # 12
    s2_rank = 3 * 2 * 3   # 18
    s3_rank = 3 * 3 * 2   # 18
    s4_rank = 3 * 3 * 3   # 27 (will try to reduce)

    U1, V1, W1 = _schoolbook(3, 2, 2)
    U2, V2, W2 = _schoolbook(3, 2, 3)
    U3, V3, W3 = _schoolbook(3, 3, 2)
    U4, V4, W4 = _schoolbook(3, 3, 3)

    # Try to find a lower-rank 3x3x3 decomposition via a simple search.
    # We'll attempt a rank-23 candidate using a known pattern and verify.
    # Since constructing a correct rank-23 3x3x3 from scratch is hard,
    # we instead try a small local optimization to shave rank off S4.
    # Given time constraints, we do a quick random-restart least-squares
    # attempt for rank 26, 25, 24, 23 and verify.

    best_s4 = (27, U4, V4, W4)
    for target_r in [26, 25, 24, 23]:
        if time.time() > deadline:
            break
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            np.random.seed(1234 + target_r * 100 + attempt)
            U_c = np.random.randn(9, target_r) * 0.3
            V_c = np.random.randn(9, target_r) * 0.3
            W_c = np.random.randn(9, target_r) * 0.3
            # Alternating least squares for a few iterations
            for it in range(40):
                if time.time() > deadline:
                    break
                # Update U: for each (i,j), solve linear system
                for i in range(3):
                    for j in range(3):
                        # sum_k sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*3+i,c] = 1
                        # For fixed V,W, this is linear in U[i*3+j,:]:
                        # Let A_c = sum_k V[j*3+k,c]*W[k*3+i,c]
                        # Then sum_c U[i*3+j,c]*A_c = 1
                        A = np.zeros(target_r)
                        for k in range(3):
                            A += V[j * 3 + k] * W[k * 3 + i]
                        # Solve U[i*3+j,:] . A = 1 (least squares)
                        if np.dot(A, A) > 1e-12:
                            U_c[i * 3 + j] = A / np.dot(A, A)
                # Update V
                for j in range(3):
                    for k in range(3):
                        A = np.zeros(target_r)
                        for i in range(3):
                            A += U[i * 3 + j] * W[k * 3 + i]
                        if np.dot(A, A) > 1e-12:
                            V_c[j * 3 + k] = A / np.dot(A, A)
                # Update W
                for k in range(3):
                    for i in range(3):
                        A = np.zeros(target_r)
                        for j in range(3):
                            A += U[i * 3 + j] * V[j * 3 + k]
                        if np.dot(A, A) > 1e-12:
                            W_c[k * 3 + i] = A / np.dot(A, A)
                # Check error
                err = 0.0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            val = float(np.dot(U_c[i * 3 + j], V_c[j * 3 + k] * W_c[k * 3 + i]))
                            err += (val - 1.0) ** 2
                if err < 1e-10:
                    found = True
                    break
            if found:
                best_s4 = (target_r, U_c, V_c, W_c)
                candidate_found = True
                break
        if candidate_found:
            break

    s4_rank_final, U4_f, V4_f, W4_f = best_s4

    # Assemble full decomposition
    U1_e, V1_e, W1_e = _embed_sub(3, 2, 2, U1, V1, W1, 0, 0, 0, n, m, p, s1_rank)
    U2_e, V2_e, W2_e = _embed_sub(3, 2, 3, U2, V2, W2, 0, 0, 2, n, m, p, s2_rank)
    U3_e, V3_e, W3_e = _embed_sub(3, 3, 2, U3, V3, W3, 0, 2, 0, n, m, p, s3_rank)
    U4_e, V4_e, W4_e = _embed_sub(3, 3, 3, U4_f, V4_f, W4_f, 0, 2, 2, n, m, p, s4_rank_final)

    total_rank = s1_rank + s2_rank + s3_rank + s4_rank_final
    U_total = np.hstack([U1_e, U2_e, U3_e, U4_e])
    V_total = np.hstack([V1_e, V2_e, V3_e, V4_e])
    W_total = np.hstack([W1_e, W2_e, W3_e, W4_e])

    # Verify the full decomposition
    full_err = _verify(U_total, V_total, W_total, n, m, p, tol=1e-6)
    if full_err < 1e-6 and total_rank < n * m * p:
        U_out = U_total.tolist()
        V_out = V_total.tolist()
        W_out = W_total.tolist()
    else:
        # Fallback to schoolbook
        U_out = U_sb.tolist()
        V_out = V_sb.tolist()
        W_out = W_sb.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
