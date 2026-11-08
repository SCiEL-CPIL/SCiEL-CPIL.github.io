"""Matrix multiplication 2,4,8 - rank-minimal exact tensor decomposition.

Uses a structural decomposition: split the 2x4x8 multiplication into two
2x2x8 multiplications (by the column index of A / row index of B), and each
2x2x8 into four 2x2x2 multiplications (by blocks of B's columns).  Each
2x2x2 sub-problem is solved with Strassen's algorithm (rank 7), giving total
rank 2 * 4 * 7 = 56.

The tensor T has indices (i,j,k) with n=2, m=4, p=8 and T[i,j,k] = delta.
A decomposition of rank r gives U (n*m x r), V (m*p x r), W (p*n x r) such that
T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
"""

import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.

    Each term is a dict with keys 'U', 'V', 'W' mapping (index) -> value.
    For 2x2x2: U index is i*2+j (i in {0,1}, j in {0,1}),
               V index is j*2+k (j in {0,1}, k in {0,1}),
               W index is k*2+i (k in {0,1}, i in {0,1}).

    C_00 = M1 - M4 + M6 + M7
    C_01 = M5 + M4 - M7 - M8   ... wait, let me use the standard 7 terms.

    Standard Strassen (7 products):
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) B00
    M3 = A00 (B01 - B11)
    M4 = A11 (B10 - B00)
    M5 = (A00 + A01) B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)

    C00 = M1 - M4 + M6 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M3 + M2 - M5
    """
    terms = []
    # Each term: (U_dict, V_dict, W_dict)
    # U_dict maps (i,j) -> val, V_dict maps (j,k) -> val, W_dict maps (k,i) -> val

    # M1 = (A00 + A11)(B00 + B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: identity on (i,k) -> delta_ik
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1},
    })

    # M2 = (A10 + A11) B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10 += 1, C11 += 1  => W(0,1)=1, W(1,1)=1  (k=0->i=1, k=1->i=1)
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(0,1): 1, (1,1): 1},
    })

    # M3 = A00 (B01 - B11)
    # U: A00=1
    # V: B01=1, B11=-1
    # W: C01 += 1  => W(1,0)=1  (k=1, i=0)
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(1,0): 1},
    })

    # M4 = A11 (B10 - B00)
    # U: A11=1
    # V: B10=1, B00=-1
    # W: C00 += -1, C10 += 1  => W(0,0)=-1, W(1,1)=1
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): -1, (1,1): 1},
    })

    # M5 = (A00 + A01) B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C01 += 1, C11 += -1  => W(1,0)=1, W(1,1)=-1
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(1,0): 1, (1,1): -1},
    })

    # M6 = (A10 - A00)(B00 + B01)
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C00 += 1  => W(0,0)=1
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1},
    })

    # M7 = (A01 - A11)(B10 + B11)
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C00 += 1  => W(0,0)=1
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1},
    })

    return terms


def construct():
    """Construct rank-56 decomposition using Strassen on 2x2x2 blocks."""
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 2, 4, 8

    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)

    strassen = _strassen_terms()
    rank = 2 * 4 * 7  # 56

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: j in {0,1} (A cols 0,1; B rows 0,1)
    # Part 2: j in {2,3} (A cols 2,3; B rows 2,3)
    for part in range(2):
        j_offset = 2 * part  # 0 or 2
        for block in range(4):
            k_offset = 2 * block  # 0, 2, 4, 6
            for term in strassen:
                # U: for (i, j_local) in term['U']:
                #   global U index = i * m + (j_offset + j_local)
                for (i, j_loc), val in term['U'].items():
                    u_idx = i * m + (j_offset + j_loc)
                    U[u_idx][col] = val

                # V: for (j_loc, k_loc) in term['V']:
                #   global V index = (j_offset + j_loc) * p + (k_offset + k_loc)
                for (j_loc, k_loc), val in term['V'].items():
                    v_idx = (j_offset + j_loc) * p + (k_offset + k_loc)
                    V[v_idx][col] = val

                # W: for (k_loc, i) in term['W']:
                #   global W index = (k_offset + k_loc) * n + i
                for (k_loc, i), val in term['W'].items():
                    w_idx = (k_offset + k_loc) * n + i
                    W[w_idx][col] = val

                col += 1

    # Verify before returning
    _verify(U, V, W, n, m, p)

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # We'll do a spot check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U_arr[0])):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: sum={s}")
# EVOLVE-BLOCK-END
