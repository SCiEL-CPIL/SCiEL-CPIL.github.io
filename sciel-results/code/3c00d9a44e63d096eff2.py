"""Matrix multiplication 2x4 * 4x5 tensor decomposition - rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # leave 10s margin

    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is valid using einsum."""
        r = U.shape[1]
        U_idx = U.reshape(n, m, r)
        V_idx = V.reshape(m, p, r)
        W_idx = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_idx, V_idx, W_idx)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    # Build constructive block decomposition
    # Split j into two groups of 2: {0,1} and {2,3}
    # For each group, split k into {0,1}, {2,3}, {4}
    # Each 2x2x2 uses Strassen (rank 7), 2x2x1 uses schoolbook (rank 4)
    # Total rank = 2 * (7 + 7 + 4) = 36

    # Strassen 2x2x2 local templates (7 terms)
    # Each term: (U_local[2x2], V_local[2x2], W_local[2x2])
    strassen_terms = [
        # P1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        {'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}},
        # P2 = (A10+A11)B00 -> C10+=1
        {'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1}},
        # P3 = A00(B10-B11) -> C01+=1, C11+=1
        {'U': {(0,0):1}, 'V': {(1,0):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}},
        # P4 = A11(B01-B00) -> C00+=1, C10+=1
        {'U': {(1,1):1}, 'V': {(0,1):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}},
        # P5 = (A00+A01)B11 -> C01+=1, C00-=1, C11-=1
        {'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(1,0):1, (0,0):-1, (1,1):-1}},
        # P6 = (A10-A00)(B00+B01) -> C11+=1
        {'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}},
        # P7 = (A01-A11)(B10+B11) -> C00+=1
        {'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}},
    ]

    # Schoolbook 2x2x1 local templates (4 terms)
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    schoolbook_221_terms = [
        {'U': {(0,0):1}, 'V': {(0,0):1}, 'W': {(0,0):1}},
        {'U': {(0,1):1}, 'V': {(1,0):1}, 'W': {(0,0):1}},
        {'U': {(1,0):1}, 'V': {(0,0):1}, 'W': {(0,1):1}},
        {'U': {(1,1):1}, 'V': {(1,0):1}, 'W': {(0,1):1}},
    ]

    # Count total rank
    total_rank = 2 * (7 + 7 + 4)  # = 36
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for j_offset in [0, 2]:
        # j' = 0,1 maps to j = j_offset, j_offset+1
        # For k in {0,1}: Strassen 2x2x2
        for k_offset in [0, 2]:
            # k' = 0,1 maps to k = k_offset, k_offset+1
            for term in strassen_terms:
                for (li, lj), val in term['U'].items():
                    U[li * m + (j_offset + lj), col] = val
                for (lj, lk), val in term['V'].items():
                    V[(j_offset + lj) * p + (k_offset + lk), col] = val
                for (lk, li), val in term['W'].items():
                    W[(k_offset + lk) * n + li, col] = val
                col += 1
        # For k = 4: schoolbook 2x2x1
        for term in schoolbook_221_terms:
            for (li, lj), val in term['U'].items():
                U[li * m + (j_offset + lj), col] = val
            for (lj, lk), val in term['V'].items():
                V[(j_offset + lj) * p + 4, col] = val
            for (lk, li), val in term['W'].items():
                W[4 * n + li, col] = val
            col += 1

    # Verify
    assert col == total_rank, f"Column count {col} != total_rank {total_rank}"
    if not verify(U, V, W, tol=1e-9):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
