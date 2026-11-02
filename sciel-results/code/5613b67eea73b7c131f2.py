"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Fallback: Schoolbook
    schoolbook_rank = n * m * p
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Try Hybrid 4+2 partition (Rank 172)
    hybrid_rank = 172
    if hybrid_rank < best_rank:
        success = _build_hybrid_42(n, m, p, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = hybrid_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found

    # Try Block-Strassen (Rank 189)
    strassen_rank = 189
    if strassen_rank < best_rank:
        success = _build_block_strassen(n, m, p, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = strassen_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _build_hybrid_42(n, m, p, deadline):
    nm, mp, pn = n * m, m * p, p * n
    total_rank = 172
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    I4 = [0, 1, 2, 3]
    I2 = [4, 5]
    
    col = 0
    
    # Helper to add rectangular terms: A (nA x k) * B (k x pB) -> C (nA x pB)
    def add_rect(nA, k, pB, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # Iterate over middle dimension k
        for t in range(k):
            # Iterate over output rows and cols
            for r in range(nA):
                for c in range(pB):
                    # U index: a_rows[r] * m + a_cols[t]
                    # V index: b_rows[t] * p + b_cols[c]
                    # W index: c_cols[c] * n + c_rows[r]
                    ua = a_rows[r] * m + a_cols[t]
                    vb = b_rows[t] * p + b_cols[c]
                    wc = c_cols[c] * n + c_rows[r]
                    U[ua, col] = 1
                    V[vb, col] = 1
                    W[wc, col] = 1
                    col += 1

    # Helper to add Strassen 2x2
    def add_strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        terms = [
            ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
            ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
            ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
            ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),
            ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
            ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
            ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
        ]
        for a_fac, b_fac, c_fac in terms:
            for r in range(2):
                for c in range(2):
                    idx = r * 2 + c
                    if a_fac[idx] != 0:
                        U[a_rows[r] * m + a_cols[c], col] = a_fac[idx]
                    if b_fac[idx] != 0:
                        V[b_rows[r] * p + b_cols[c], col] = b_fac[idx]
                    if c_fac[idx] != 0:
                        W[c_cols[c] * n + c_rows[r], col] = c_fac[idx]
            col += 1

    # Helper to add Laderman 4x4
    def add_laderman_4x4(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # Laderman 21 terms
        # Indices 0..3
        terms = [
            # (A_idx, B_idx, C_idx, coeff)
            ((0,0), (0,0), (0,0), 1),
            ((0,0), (1,1), (0,1), 1),
            ((0,1), (0,0), (1,0), 1),
            ((0,1), (1,1), (1,1), 1),
            ((1,0), (0,0), (0,0), 1),
            ((1,0), (1,1), (0,1), 1),
            ((1,1), (0,0), (1,0), 1),
            ((1,1), (1,1), (1,1), 1),
            ((0,0), (0,1), (0,0), 1),
            ((0,0), (1,0), (0,1), 1),
            ((0,1), (0,1), (1,0), 1),
            ((0,1), (1,0), (1,1), 1),
            ((1,0), (0,1), (0,0), 1),
            ((1,0), (1,0), (0,1), 1),
            ((1,1), (0,1), (1,0), 1),
            ((1,1), (1,0), (1,1), 1),
            ((0,0), (0,0), (0,0), 1), # Placeholder, need correct Laderman
            # Correct Laderman is complex. Using Strassen 4x4 (Rank 49) instead if Laderman fails?
            # But 49 > 21.
            # Let's use Strassen 4x4 for now to ensure correctness, rank 49.
            # Total rank: 49 + 32 + 32 + 16 + 32 + 16 + 16 + 7 = 200.
            # This is worse than 189.
            # So we MUST use Laderman.
            # I will use a known 21-term decomposition.
        ]
        # Since I don't have the exact 21 terms hardcoded, I will use a 4x4 decomposition
        # that is known to be rank 21? 
        # Actually, I can use 2 Strassen 2x2 (14) + 7 cross terms?
        # Let's stick to Block Strassen (189) if Laderman is not easily available.
        # But 172 is the target.
        # I will implement Laderman terms from a reliable source if possible.
        # For now, I will return None to fall back to Strassen 189.
        return False

    # C44: A44 B44 (21) + A42 B24 (32)
    if not add_laderman_4x4(I4, I4, I4, I4, I4, I4):
        return None
    add_rect(4, 2, 4, I4, I2, I2, I4, I4, I4)
    
    # C42: A44 B42 (32) + A42 B22 (16)
    add_rect(4, 4, 2, I4, I4, I4, I2, I4, I2)
    add_rect(4, 2, 2, I4, I2, I2, I2, I4, I2)
    
    # C24: A24 B44 (32) + A22 B24 (16)
    add_rect(2, 4, 4, I2, I4, I4, I4, I2, I4)
    add_rect(2, 2, 4, I2, I2, I2, I4, I2, I4)
    
    # C22: A24 B42 (16) + A22 B22 (7)
    add_rect(2, 4, 2, I2, I4, I4, I2, I2, I2)
    add_strassen_2x2(I2, I2, I2, I2, I2, I2)
    
    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    return U, V, W

def _build_block_strassen(n, m, p, deadline):
    # ... (same as before) ...
    nm, mp, pn = n * m, m * p, p * n
    num_blocks = 3
    total_rank = 189
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    strassen_terms = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (0, 1, -1, 0), (1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]
    
    col = 0
    for bi in range(num_blocks):
        for bk in range(num_blocks):
            for bj in range(num_blocks):
                for s in range(7):
                    a_fac, b_fac, c_fac = strassen_terms[s]
                    for r in range(2):
                        for s_idx in range(2):
                            idx = r * 2 + s_idx
                            if a_fac[idx] != 0:
                                U[(bi * 2 + r) * m + (bk * 2 + s_idx), col] = a_fac[idx]
                            if b_fac[idx] != 0:
                                V[(bk * 2 + r) * p + (bj * 2 + s_idx), col] = b_fac[idx]
                            if c_fac[idx] != 0:
                                W[(bj * 2 + s_idx) * n + (bi * 2 + r), col] = c_fac[idx]
                    col += 1
    
    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    np.random.seed(123)
    for _ in range(2000):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True
# EVOLVE-BLOCK-END
