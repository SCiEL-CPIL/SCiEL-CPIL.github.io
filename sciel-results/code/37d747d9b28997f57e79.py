"""Tensor decomposition for 2x3x9 matrix multiplication using Strassen blocks."""
DIMENSIONS = [2, 3, 9]

import numpy as np
import time
from fractions import Fraction


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 150s kill, leave 10s margin

    # Build target tensor for verification
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def _build_strassen_222(offset_i, offset_j, offset_k):
        """Build rank-7 Strassen decomposition for a 2x2x2 block.
        offset_i: starting row index in A (and C)
        offset_j: starting col index in A (and row in B)
        offset_k: starting col index in B (and C)
        Returns list of (U_vec, V_vec, W_vec) where each is a dict mapping
        (local_index) -> value, to be placed into global matrices.
        """
        # Strassen for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6

        # For each Strassen term, we need U (on A indices), V (on B indices), W (on C indices)
        # U acts on (i,j) in {0,1}x{0,1}, V on (j,k) in {0,1}x{0,1}, W on (k,i) in {0,1}x{0,1}

        terms = []

        # M1 = (A11+A22)(B11+B22)
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11=1, C22=1 (since C11 += M1, C22 += M1)
        terms.append((
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ))

        # M2 = (A21+A22)B11
        # U: A21=1, A22=1
        # V: B11=1
        # W: C21=1, C22=-1
        terms.append((
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1}
        ))

        # M3 = A11(B21-B22)
        # U: A11=1
        # V: B21=1, B22=-1
        # W: C12=1, C22=1
        terms.append((
            {(0,0): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ))

        # M4 = A22(B12-B11)
        # U: A22=1
        # V: B12=1, B11=-1
        # W: C11=1, C21=1
        terms.append((
            {(1,1): 1},
            {(0,1): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ))

        # M5 = (A11+A12)B22
        # U: A11=1, A12=1
        # V: B22=1
        # W: C11=-1, C12=1
        terms.append((
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ))

        # M6 = (A21-A11)(B11+B12)
        # U: A21=1, A11=-1
        # V: B11=1, B12=1
        # W: C22=1
        terms.append((
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ))

        # M7 = (A12-A22)(B21+B22)
        # U: A12=1, A22=-1
        # V: B21=1, B22=1
        # W: C11=1
        terms.append((
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        ))

        # Now convert to global indices
        global_terms = []
        for u_dict, v_dict, w_dict in terms:
            U_g = {}
            V_g = {}
            W_g = {}
            for (li, lj), val in u_dict.items():
                gi = offset_i + li
                gj = offset_j + lj
                U_g[(gi, gj)] = U_g.get((gi, gj), 0) + val
            for (lj, lk), val in v_dict.items():
                gj = offset_j + lj
                gk = offset_k + lk
                V_g[(gj, gk)] = V_g.get((gj, gk), 0) + val
            for (lk, li), val in w_dict.items():
                gk = offset_k + lk
                gi = offset_i + li
                W_g[(gk, gi)] = W_g.get((gk, gi), 0) + val
            global_terms.append((U_g, V_g, W_g))

        return global_terms

    def _build_schoolbook_block(offset_i, i_size, offset_j, j_size, offset_k, k_size):
        """Build schoolbook decomposition for a block of size i_size x j_size x k_size.
        Returns list of (U_dict, V_dict, W_dict) with global indices.
        """
        terms = []
        for li in range(i_size):
            for lj in range(j_size):
                for lk in range(k_size):
                    gi = offset_i + li
                    gj = offset_j + lj
                    gk = offset_k + lk
                    terms.append((
                        {(gi, gj): 1},
                        {(gj, gk): 1},
                        {(gk, gi): 1}
                    ))
        return terms

    # Partition: m=3 into [2,1], p=9 into [2,2,2,2,1]
    m_parts = [2, 1]
    p_parts = [2, 2, 2, 2, 1]

    all_terms = []

    # Build blocks
    j_offset = 0
    for j_size in m_parts:
        k_offset = 0
        for k_size in p_parts:
            if j_size == 2 and k_size == 2:
                # Use Strassen
                all_terms.extend(_build_strassen_222(0, j_offset, k_offset))
            else:
                # Use schoolbook
                all_terms.extend(_build_schoolbook_block(0, n, j_offset, j_size, k_offset, k_size))
            k_offset += k_size
        j_offset += j_size

    rank = len(all_terms)
    print(f"Constructed rank {rank} decomposition")

    # Build U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for c, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (gi, gj), val in u_dict.items():
            U[gi * m + gj, c] = val
        for (gj, gk), val in v_dict.items():
            V[gj * p + gk, c] = val
        for (gk, gi), val in w_dict.items():
            W[gk * n + gi, c] = val

    # Verify
    D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    max_err = np.max(np.abs(D - T))
    print(f"Max error: {max_err}")

    if max_err > 1e-10:
        # Fallback to schoolbook
        print("Strassen construction failed, falling back to schoolbook")
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        max_err = np.max(np.abs(D - T))
        print(f"Schoolbook max error: {max_err}")

    # Try to find even lower rank with optimization if time permits
    if max_err < 1e-10 and time.time() < deadline - 20:
        # Try rank-1 optimization from the current solution
        # This is unlikely to find integer/half-integer solutions, but try
        pass  # Skip for now; the Strassen result is already an improvement

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
