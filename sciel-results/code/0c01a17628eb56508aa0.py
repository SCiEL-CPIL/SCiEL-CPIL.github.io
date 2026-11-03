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

    # Schoolbook fallback
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

    # Try Block-Strassen: rank 189
    strassen_rank = 189
    if strassen_rank < best_rank:
        success = _build_block_strassen(n, m, p, deadline)
        if success is not None:
            best_rank = strassen_rank
            best_U, best_V, best_W = success

    # Try alternative: 2x2 blocks of 3x3 with Strassen on the 2x2 super-matrix
    # This gives the same rank 189 but different structure; skip if not better.

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    return {"U": U_out, "V": V_out, "W": W_out}


def _build_block_strassen(n, m, p, deadline):
    """Corrected Block-Strassen: 3x3 blocks of 2x2, rank 189."""
    nm, mp, pn = n * m, m * p, p * n
    num_blocks = 3
    total_rank = num_blocks ** 3 * 7  # 189

    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))

    # Corrected Strassen terms: (A_factor, B_factor, C_factor)
    # c00 = M1 - M4 + M5 - M7
    # c01 = M3 - M5 + M7
    # c10 = M2 - M4 + M6
    # c11 = M1 - M2 + M3 + M6
    strassen_terms = [
        # M1: (a00+a11)(b00+b11)
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2: (a10+a11)b00
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        # M3: a00(b10-b11)
        ((1, 0, 0, 0), (0, 0, 1, -1), (0, 1, 0, 1)),
        # M4: a11(b01-b00)  -> c00 -= M4, c10 += M4
        ((0, 0, 0, 1), (0, 1, -1, 0), (-1, 0, 1, 0)),
        # M5: (a00+a01)b11  -> c00 += M5, c01 -= M5
        ((1, 1, 0, 0), (0, 0, 0, 1), (1, -1, 0, 0)),
        # M6: (a10-a00)(b00+b01) -> c11 += M6
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
        # M7: (a01-a11)(b10+b11) -> c00 -= M7, c01 += M7
        ((0, 1, 0, -1), (0, 0, 1, 1), (-1, 1, 0, 0)),
    ]

    col = 0
    for bi in range(num_blocks):
        for bk in range(num_blocks):
            for bj in range(num_blocks):
                for s in range(7):
                    a_fac, b_fac, c_fac = strassen_terms[s]

                    for r in range(2):
                        for sc in range(2):
                            idx = r * 2 + sc
                            if a_fac[idx] != 0:
                                U[(bi * 2 + r) * m + (bk * 2 + sc), col] = a_fac[idx]

                    for r in range(2):
                        for sc in range(2):
                            idx = r * 2 + sc
                            if b_fac[idx] != 0:
                                V[(bk * 2 + r) * p + (bj * 2 + sc), col] = b_fac[idx]

                    for t in range(2):
                        for u in range(2):
                            idx = t * 2 + u
                            if c_fac[idx] != 0:
                                W[(bj * 2 + u) * n + (bi * 2 + t), col] = c_fac[idx]

                    col += 1

    if not _verify_decomposition(n, m, p, U, V, W):
        return None
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Fast verification using vectorized numpy."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False

    # Check valid triples: U[a,:] * V[b,:] * W[c,:] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(np.dot(U[a], V[b] * W[c]) - 1.0) > tol:
                    return False

    # Check invalid triples with vectorized approach
    # For each (a,b), the valid c must satisfy: c = (b%p)*n + (a//m)
    # and a%m == b//p. If not, all c should give 0.
    for a in range(nm):
        for b in range(mp):
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k1 = b % p
            if j1 != j2:
                # All c are invalid
                vals = np.einsum('i,i,i->', U[a], V[b], W)
                # Actually need per-c check
                prod = U[a] * V[b]  # shape (rank,)
                for c in range(pn):
                    if abs(prod[c if False else 0] * 0) > tol:  # placeholder
                        pass
                # Better: check sum over all c at once isn't enough; check each
                # Use: for all c, U[a,:]*V[b,:]*W[c,:] should be 0
                # Vectorized: (U[a]*V[b]) @ W.T gives all c values
                cvals = (U[a] * V[b]) @ W.T  # shape (pn,)
                if np.max(np.abs(cvals)) > tol:
                    return False
            else:
                # Only c = k1*n + i1 is valid
                valid_c = k1 * n + i1
                prod = U[a] * V[b]
                cvals = prod @ W.T
                # valid_c should be 1, others 0
                if abs(cvals[valid_c] - 1.0) > tol:
                    return False
                mask = np.ones(pn, dtype=bool)
                mask[valid_c] = False
                if np.max(np.abs(cvals[mask])) > tol:
                    return False

    return True
# EVOLVE-BLOCK-END
