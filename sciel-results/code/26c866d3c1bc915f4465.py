"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

# Strassen 2x2x2 coefficients (verified correct):
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A01)B11
# M6 = (A10-A00)(B00+B01)
# M7 = (A01-A11)(B10+B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6
#
# Each M_r is a product of a linear form in A and a linear form in B.
# U entries: coefficient of A_{ij} in the A-linear form of M_r
# V entries: coefficient of B_{jk} in the B-linear form of M_r
# W entries: coefficient of C_{ik} in the output (each W_r is a single C entry with coeff 1)

STRASSEN_U = [
    [1, 0, 0, 0],   # M1: A00
    [0, 1, 0, 0],   # M1: A11
    [0, 0, 1, 0],   # M2: A10
    [0, 0, 0, 1],   # M2: A11
    [1, 0, 0, 0],   # M3: A00
    [0, 0, 0, 1],   # M4: A11
    [1, 0, 0, 0],   # M5: A00
    [0, 1, 0, 0],   # M5: A01
    [0, 0, 1, 0],   # M6: A10
    [1, 0, 0, 0],   # M6: A00 (negative, handled below)
    [0, 1, 0, 0],   # M7: A01
    [0, 0, 0, 1],   # M7: A11 (negative, handled below)
]

# Actually let me build this more carefully.
# 7 Strassen products, each M_r = (sum_j u_{r,j} A_j) * (sum_k v_{r,k} B_k)
# where A_j are the 4 entries A00,A01,A10,A11 (row-major)
# and B_k are the 4 entries B00,B01,B10,B11 (row-major)

# M1 = (A00 + A11)(B00 + B11)
# M2 = (A10 + A11) B00
# M3 = A00 (B01 - B11)
# M4 = A11 (B10 - B00)
# M5 = (A00 + A01) B11
# M6 = (A10 - A00)(B00 + B01)
# M7 = (A01 - A11)(B10 + B11)

# U: 4 rows (A00,A01,A10,A11) x 7 cols (M1..M7)
STRASSEN_U = [
    [1, 0, 1, 0, 1, -1, 0],   # A00
    [0, 0, 0, 0, 1,  0, 1],   # A01
    [0, 1, 0, 0, 0,  1, 0],   # A10
    [1, 1, 0, 1, 0,  0,-1],   # A11
]

# V: 4 rows (B00,B01,B10,B11) x 7 cols (M1..M7)
STRASSEN_V = [
    [1, 1, 0, -1, 0, 1, 0],   # B00
    [0, 0, 1,  0, 0, 1, 0],   # B01
    [0, 0, 0,  1, 0, 0, 1],   # B10
    [1, 0,-1,  0, 1, 0, 1],   # B11
]

# W: 4 rows (C00,C01,C10,C11) x 7 cols (M1..M7)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6
STRASSEN_W = [
    [1, 0, 0, 1, -1, 0, 1],   # C00
    [0, 0, 1, 0,  1, 0, 0],   # C01
    [0, 1, 0, 1,  0, 0, 0],   # C10
    [1,-1, 1, 0,  0, 1, 0],   # C11
]


def _verify_strassen(n_sub=2, m_sub=2, p_sub=2):
    """Verify Strassen decomposition for 2x2x2 multiplication.
    Check: delta(i,i')delta(j,j')delta(k,k') = sum_r U[i*m+j,r] V[j'*p+k',r] W[k'*n+i',r]
    for all i,j,k,i',j',k' in [0,2).
    """
    U = STRASSEN_U  # shape (2*2, 7)
    V = STRASSEN_V  # shape (2*2, 7)
    W = STRASSEN_W  # shape (2*2, 7)
    rank = 7
    for i in range(n_sub):
        for j in range(m_sub):
            for jp in range(m_sub):
                for k in range(p_sub):
                    for kp in range(p_sub):
                        for ip in range(n_sub):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i * m_sub + j][r] * V[jp * p_sub + kp][r] * W[kp * n_sub + ip][r]
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Structural approach: rank-14 decomposition using the fact that
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This gives 2*p = 14 terms, which is the known optimal rank for 2x2xp (p>=2).
    #
    # Term r corresponds to (i, j, k) where:
    #   i = r // p  (row of A, 0 or 1)
    #   j = (r // (p//2)) % 2 ... let's just use direct indexing
    #
    # For each (i, j, k) with i in {0,1}, j in {0,1}, k in {0..6}:
    #   U[i*m + j, r] = 1
    #   V[j*p + k, r] = 1
    #   W[k*n + i, r] = 1
    #
    # Total: 2*2*7 = 28... that's schoolbook. I need a better structure.
    #
    # Actually for 2x2xp, the known optimal is 2p+1 for small p? Let me just
    # try to find a rank-14 decomposition by exploiting linearity.
    #
    # Key: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #       C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Define for each k:
    #   M1_k = (A[0,0] + A[1,1]) * (B[0,k] + B[1,k])
    #   M2_k = (A[0,0] - A[1,1]) * (B[0,k] - B[1,k])
    #   Then A[0,0]*B[0,k] + A[1,1]*B[1,k] = (M1_k + M2_k)/2
    #   And A[1,1]*B[0,k] + A[0,0]*B[1,k] = (M1_k - M2_k)/2
    #
    # But we also have A[0,1]*B[1,k] and A[1,0]*B[0,k] which are "off-diagonal".
    #
    # Let me just use the direct 14-term construction:
    # For each k in 0..6, and for each of the 2 "rows" of the 2x1 multiplication:
    #   Row 0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]  -> 2 terms
    #   Row 1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]  -> 2 terms
    # Total: 4 * 7 = 28. Still schoolbook.
    #
    # I think 14 is not achievable with simple structure. Let me just return
    # the best I can do: the Strassen-block approach giving rank 25, or try
    # a randomized search for a lower rank.
    #
    # Let me try numpy-based search: build the tensor as a matrix and find its rank.

    import numpy as np

    # Build the multiplication tensor as a 3D array T[i*m+j, jp*p+k, kp*n+ip]
    # T[a, b, c] = 1 if a corresponds to (i,j), b to (j',k'), c to (k'',i'')
    # and i==i', j==j', k==k''
    # More precisely: T[i*m+j, jp*p+kp, kp*n+ip] = 1 iff i==ip and j==jp and k==kp... 
    # Wait, the tensor is T[i,j,k; i',j',k'] = delta(i,i')delta(j,j')delta(k,k')
    # Flattened: T[(i*m+j), (j'*p+k'), (k'*n+i')] where the three indices are
    # A-index, B-index, C-index.
    # T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j',k'), C_idx=(k'',i'')
    # and i==i' and j==j' and k==k''... but k is in A? No.
    #
    # A is n x m: A[i,j], index i*m+j
    # B is m x p: B[j,k], index j*p+k  
    # C is n x p: C[i,k], index k*n+i (per trace convention W[k*n+i,r])
    #
    # T[i*m+j, j'*p+k', k'*n+i'] = 1 iff i==i' and j==j' and k==k'
    # But k is the column index of B and C. So for T to be nonzero, we need
    # the k from B's index to equal the k from C's index, and j from A to equal
    # j' from B, and i from A to equal i' from C.
    #
    # Let me build it:
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Reshape to a matrix: (n*m) x (m*p * p*n)
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # Find the rank via SVD
    try:
        sv = np.linalg.svd(T_mat, full_matrices=False)
        tol = 1e-9
        rank_numeric = int(np.sum(sv > tol))
    except Exception:
        rank_numeric = n * m * p

    # The numerical rank tells us the minimum possible rank.
    # For 2x2x7, this should be 14 (if that's the true rank).
    #
    # Now we need to find an explicit decomposition. Let's use the SVD to get
    # a rank-r decomposition, but we need integer/half-integer entries.
    #
    # Alternative: just use the known structure. Let me try rank 14 with a
    # clever construction.
    #
    # Actually, let me just return the Strassen-block result (rank 25) if it
    # verifies, or fall back to schoolbook (rank 28). The SVD rank tells us
    # we might be able to do better, but finding explicit low-rank factors
    # with rational entries is hard.
    #
    # Let me try a different structural approach: use the fact that the tensor
    # has a "bilinear" structure and find a decomposition via the SVD, then
    # round to nearest integers/half-integers.

    if rank_numeric < n * m * p and time.time() < deadline - 30:
        # Try to extract a rank-r decomposition from SVD
        U_svd, S, Vt = np.linalg.svd(T_mat, full_matrices=False)
        r = rank_numeric
        # T_mat ≈ U_svd[:, :r] * diag(S[:r]) * Vt[:r, :]
        # This gives U_svd[:, :r] * S[:r] as left factors and Vt[:r, :] as right
        # But we need a 3-way decomposition, not 2-way.
        #
        # This approach won't directly give us the 3-factor decomposition.
        # Let me just use the Strassen-block approach.
        pass

    # Use the Strassen-block decomposition (rank 25)
    strassen_valid = _verify_strassen()
    
    if not strassen_valid:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    num_strassen = p // 2
    remaining = p % 2
    rank = num_strassen * 7 + remaining * n * m

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block in range(num_strassen):
        k_base = block * 2
        for r in range(7):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = STRASSEN_U[i * m + j][r]
            for j in range(m):
                for kk in range(2):
                    V[j * p + (k_base + kk)][col] = STRASSEN_V[j * 2 + kk][r]
            for i in range(n):
                for kk in range(2):
                    W[(k_base + kk) * n + i][col] = STRASSEN_W[i * 2 + kk][r]
            col += 1

    for k in range(num_strassen * 2, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    if time.time() < deadline:
        valid = True
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                rhs = 0.0
                                for r in range(rank):
                                    rhs += U[i * m + j][r] * V[jp * p + kp][r] * W[kp * n + ip][r]
                                if abs(lhs - rhs) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            return {"U": U, "V": V, "W": W}

    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
