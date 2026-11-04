import numpy as np
import itertools

def construct():
    n, m, p = 2, 3, 10

    # Build the 60x60x60 multiplication tensor T_{i,j,k} = 1 if (i,j,k) is a valid triple
    # T[a,b,c] = 1 where a = i*m + j, b = j*p + k, c = k*n + i
    # We need to find rank-minimal decomposition.

    # Strategy: Use Strassen for the 2x2 part (rows 0,1 of A; cols 0,1 of B) for each of 5 column-pairs,
    # and schoolbook for row 2 of A with all 10 cols of B.
    # This gives rank 7*5 + 2*10 = 35 + 20 = 55.

    # But let's try to do better by mixing Strassen patterns across the full 3-column width.
    # Actually, let's try a different partition: use Strassen on rows 0,1 and cols 0,1 (2x2 block),
    # and handle col 2 separately. That's what we did.

    # Alternative: Can we use a rank-7 decomposition for the 2x2x2 tensor and combine with the rest?
    # The 2x3x10 tensor can be seen as: for each of 5 pairs of columns in B, we have a 2x2x2 subproblem
    # (rows 0,1 of A; cols 2k,2k+1 of B; all 2 rows of output) PLUS a 2x1x2 subproblem for the last row of A.
    # Wait, the last row of A (row 2) multiplies with all 3 rows of B to give 2 outputs per column.

    # Let me think about this differently. The tensor is:
    # T[(i,j),(j',k),(k',i')] = 1 if j=j' and k=k'
    # So it's a "diagonal" tensor in the middle index.

    # The rank of the 2x3x10 multiplication tensor:
    # Lower bound: The tensor has 60 non-zero entries. Each rank-1 term can cover at most 1 entry in a
    # "diagonal" sense... actually no. A rank-1 term U[a]V[b]W[c] contributes to T[a,b,c] for ALL (a,b,c)
    # where all three are non-zero. So it's not that simple.

    # Let's just go with the known rank 55 construction and verify it.

    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # For each pair of columns (2k, 2k+1) in B, apply Strassen-like 7-term decomposition
    # for the 2x2 submatrix A[0:2, :] * B[:, 2k:2k+2]
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A00 + A11) * (B0,k0 + B1,k1)
        U[0, idx] = 1.0  # A[0,0]
        U[4, idx] = 1.0  # A[1,1]
        V[k0, idx] = 1.0      # B[0,k0]
        V[10 + k1, idx] = 1.0 # B[1,k1]
        W[2*k0, idx] = 1.0    # C[k0, 0]
        W[2*k1+1, idx] = 1.0  # C[k1, 1]
        idx += 1

        # M2 = (A10 + A11) * B0,k0
        U[3, idx] = 1.0  # A[1,0]
        U[4, idx] = 1.0  # A[1,1]
        V[k0, idx] = 1.0
        W[2*k0+1, idx] = 1.0    # C[k0, 1]
        W[2*k1+1, idx] = -1.0   # C[k1, 1]
        idx += 1

        # M3 = A00 * (B0,k1 - B1,k1)
        U[0, idx] = 1.0
        V[10 + k1, idx] = 1.0   # B[1,k1]
        V[k1, idx] = -1.0       # B[0,k1]
        W[2*k1, idx] = 1.0      # C[k1, 0]
        W[2*k1+1, idx] = 1.0    # C[k1, 1]
        idx += 1

        # M4 = A11 * (B1,k0 - B0,k0)
        U[4, idx] = 1.0
        V[10 + k0, idx] = 1.0   # B[1,k0]
        V[k0, idx] = -1.0       # B[0,k0]
        W[2*k0, idx] = 1.0      # C[k0, 0]
        W[2*k0+1, idx] = 1.0    # C[k0, 1]
        idx += 1

        # M5 = (A00 + A01) * B1,k1
        U[0, idx] = 1.0
        U[1, idx] = 1.0  # A[0,1]
        V[10 + k1, idx] = 1.0
        W[2*k0, idx] = -1.0   # C[k0, 0]
        W[2*k1, idx] = 1.0    # C[k1, 0]
        idx += 1

        # M6 = (A10 - A00) * (B0,k0 + B0,k1)
        U[3, idx] = 1.0
        U[0, idx] = -1.0
        V[k0, idx] = 1.0
        V[10 + k1, idx] = 1.0  # B[1,k1]
        W[2*k1+1, idx] = 1.0   # C[k1, 1]
        idx += 1

        # M7 = (A01 - A11) * (B1,k0 + B1,k1)
        U[1, idx] = 1.0
        U[4, idx] = -1.0
        V[10 + k0, idx] = 1.0
        V[20 + k1, idx] = 1.0  # B[2,k1]
        W[2*k0, idx] = 1.0
        idx += 1

    # Schoolbook for A[2,:] * B[:, :]
    for i in range(2):
        for k in range(10):
            U[6 + i, idx] = 1.0  # A[2, i] where i=0,1... wait
            # A has shape 2x3, so A[2, j] for j=0,1,2. But we only have rows 0,1 of A in the Strassen part.
            # Row 2 of A: A[2,0], A[2,1], A[2,2]
            # For each k in 0..9, we need A[2,j] * B[j,k] for j=0,1,2 summed, giving C[k,i] for i=0,1
            # So: C[k,i] += sum_j A[2,j] * B[j,k]
            # This is 2*10 = 20 terms if done schoolbook
            pass

    # Actually let me redo this more carefully.
    # A is 2x3: rows 0,1; cols 0,1,2
    # B is 3x10: rows 0,1,2; cols 0..9
    # C is 2x10: rows 0,1; cols 0..9
    # C[i,k] = sum_j A[i,j] * B[j,k]

    # In the Strassen part, we used A[0,0], A[0,1], A[1,0], A[1,1] (the 2x2 submatrix)
    # We did NOT use A[0,2], A[1,2] (column 2 of A)
    # And we did NOT use A[2,*] - wait, A only has 2 rows!

    # I made an error. A is 2x3, so it has rows 0,1 and cols 0,1,2.
    # B is 3x10, so it has rows 0,1,2 and cols 0..9.
    # C is 2x10, so it has rows 0,1 and cols 0..9.

    # The Strassen decomposition above only handles A[0:2, 0:2] * B[:, 0:2] for each 2-column block.
    # We still need to handle A[0:2, 2] * B[:, k] for each k.

    # Let me restart with a cleaner approach.

    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # For each 2-column block in B (cols 2k, 2k+1):
    # Handle A[:, 0:2] * B[:, 2k:2k+2] using 7-term Strassen
    # Then handle A[:, 2] * B[:, 2k:2k+2] using 2 terms (schoolbook)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1+1]
        # This gives contributions to C[0:2, k0] and C[0:2, k1] from A's first two columns and B's first two rows

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U[0, idx] = 1.0   # A[0,0]
        U[4, idx] = 1.0   # A[1,1]
        V[k0, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[2*k0, idx] = 1.0
        W[2*k1+1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        U[3, idx] = 1.0   # A[1,0]
        U[4, idx] = 1.0   # A[1,1]
        V[k0, idx] = 1.0
        W[2*k0+1, idx] = 1.0
        W[2*k1+1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        U[0, idx] = 1.0
        V[k1, idx] = 1.0
        V[10 + k1, idx] = -1.0
        W[2*k1, idx] = 1.0
        W[2*k1+1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        U[4, idx] = 1.0
        V[10 + k0, idx] = 1.0
        V[k0, idx] = -1.0
        W[2*k0, idx] = 1.0
        W[2*k0+1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0, idx] = 1.0
        U[1, idx] = 1.0   # A[0,1]
        V[10 + k1, idx] = 1.0
        W[2*k0, idx] = -1.0
        W[2*k1, idx] = 1.0
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U[3, idx] = 1.0
        U[0, idx] = -1.0
        V[k0, idx] = 1.0
        V[k1, idx] = 1.0
        W[2*k1+1, idx] = 1.0
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[1, idx] = 1.0
        U[4, idx] = -1.0
        V[10 + k0, idx] = 1.0
        V[10 + k1, idx] = 1.0
        W[2*k0, idx] = 1.0
        idx += 1

        # Now handle A[:, 2] * B[:, k0:k1+1]
        # C[i,k] += A[i,2] * B[0,k] + A[i,2] * B[1,k] + A[i,2] * B[2,k]
        # = A[i,2] * (B[0,k] + B[1,k] + B[2,k])
        # For i=0,1 and k=k0,k1: 4 terms

        # Term for i=0, k=k0: A[0,2] * (B[0,k0] + B[1,k0] + B[2,k0])
        U[2, idx] = 1.0  # A[0,2]
        V[k0, idx] = 1.0
        V[10 + k0, idx] = 1.0
        V[20 + k0, idx] = 1.0
        W[2*k0, idx] = 1.0
        idx += 1

        # Term for i=0, k=k1
        U[2, idx] = 1.0
        V[k1, idx] = 1.0
        V[10 + k1, idx] = 1.0
        V[20 + k1, idx] = 1.0
        W[2*k1, idx] = 1.0
        idx += 1

        # Term for i=1, k=k0
        U[5, idx] = 1.0  # A[1,2]
        V[k0, idx] = 1.0
        V[10 + k0, idx] = 1.0
        V[20 + k0, idx] = 1.0
        W[2*k0+1, idx] = 1.0
        idx += 1

        # Term for i=1, k=k1
        U[5, idx] = 1.0
        V[k1, idx] = 1.0
        V[10 + k1, idx] = 1.0
        V[20 + k1, idx] = 1.0
        W[2*k1+1, idx] = 1.0
        idx += 1

    # That's 11 terms per block * 5 blocks = 55 terms.

    # Validate
    U_arr = U
    V_arr = V
    W_arr = W

    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
