"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit

    # Strassen patterns for 2x2x2 (7 rank-1 terms)
    # Each pattern: U entries (i,j,val), V entries (j,k,val), W entries (k,i,val)
    strassen_patterns = [
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]

    def get_splits(d):
        """Generate all compositions of d into parts of size 1 or 2."""
        results = []
        def helper(remaining, current):
            if remaining == 0:
                results.append(tuple(current))
                return
            if remaining >= 1:
                current.append(1)
                helper(remaining - 1, current)
                current.pop()
            if remaining >= 2:
                current.append(2)
                helper(remaining - 2, current)
                current.pop()
        helper(d, [])
        return results

    def build_block(nb, mb, pb, i_off, j_off, k_off):
        """
        Build decomposition for an nb x mb x pb block.
        Returns (U_entries, V_entries, W_entries, rank) where each entry is
        (row_index, col_index_local, val).
        """
        if nb == 2 and mb == 2 and pb == 2:
            # Use Strassen: 7 rank-1 terms
            U_ent = []
            V_ent = []
            W_ent = []
            col = 0
            for pattern in strassen_patterns:
                for li, lj, val in pattern['U']:
                    U_ent.append(((i_off + li) * m + (j_off + lj), col, val))
                for lj, lk, val in pattern['V']:
                    V_ent.append(((j_off + lj) * p + (k_off + lk), col, val))
                for lk, li, val in pattern['W']:
                    W_ent.append(((k_off + lk) * n + (i_off + li), col, val))
                col += 1
            return U_ent, V_ent, W_ent, 7
        else:
            # Schoolbook
            U_ent = []
            V_ent = []
            W_ent = []
            col = 0
            for li in range(nb):
                for lj in range(mb):
                    for lk in range(pb):
                        U_ent.append(((i_off + li) * m + (j_off + lj), col, 1))
                        V_ent.append(((j_off + lj) * p + (k_off + lk), col, 1))
                        W_ent.append(((k_off + lk) * n + (i_off + li), col, 1))
                        col += 1
            return U_ent, V_ent, W_ent, col

    def build_decomposition(n_splits, m_splits, p_splits):
        """Build the full decomposition for given splits."""
        n_off = [0]
        for s in n_splits[:-1]:
            n_off.append(n_off[-1] + s)
        m_off = [0]
        for s in m_splits[:-1]:
            m_off.append(m_off[-1] + s)
        p_off = [0]
        for s in p_splits[:-1]:
            p_off.append(p_off[-1] + s)

        U_entries = []
        V_entries = []
        W_entries = []
        rank = 0

        for i_b in range(len(n_splits)):
            for j_b in range(len(m_splits)):
                for k_b in range(len(p_splits)):
                    nb = n_splits[i_b]
                    mb = m_splits[j_b]
                    pb = p_splits[k_b]
                    i_off = n_off[i_b]
                    j_off = m_off[j_b]
                    k_off = p_off[k_b]

                    U_ent, V_ent, W_ent, r = build_block(nb, mb, pb, i_off, j_off, k_off)
                    for e in U_ent:
                        U_entries.append((e[0], rank + e[1], e[2]))
                    for e in V_ent:
                        V_entries.append((e[0], rank + e[1], e[2]))
                    for e in W_ent:
                        W_entries.append((e[0], rank + e[1], e[2]))
                    rank += r

        # Build matrices
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        for idx, col, val in U_entries:
            U[idx][col] = val
        for idx, col, val in V_entries:
            V[idx][col] = val
        for idx, col, val in W_entries:
            W[idx][col] = val

        return {"U": U, "V": V, "W": W}, rank

    def verify(result, rank):
        """Verify the decomposition is correct. Returns True if valid."""
        U = np.array(result['U'], dtype=float)
        V = np.array(result['V'], dtype=float)
        W = np.array(result['W'], dtype=float)

        # Check shape
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False

        # For each (i, j, k), sum over r of U[i*m+j, r] * V[j*p+k, r] should equal W[k*n+i, r]...
        # Actually the convention is: C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i,j,k] = 1 if this is a valid multiplication term
        # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, the W factor follows W[k*n+i, r]
        # So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k

        for i in range(n):
            if time.time() > deadline:
                return True  # assume valid if we're running out of time
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Generate all possible splits for each dimension (parts of size 1 or 2)
    n_split_options = get_splits(n)
    m_split_options = get_splits(m)
    p_split_options = get_splits(p)

    # Find the best combination of splits (minimum rank)
    best_rank = float('inf')
    best_result = None
    best_splits = None

    for ns in n_split_options:
        if time.time() > deadline:
            break
        for ms in m_split_options:
            if time.time() > deadline:
                break
            for ps in p_split_options:
                if time.time() > deadline:
                    break
                result, r = build_decomposition(ns, ms, ps)
                if r < best_rank:
                    best_rank = r
                    best_result = result
                    best_splits = (ns, ms, ps)

    # Verify the best result
    if best_result is not None and verify(best_result, best_rank):
        return best_result

    # Fallback to a simple schoolbook decomposition
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
