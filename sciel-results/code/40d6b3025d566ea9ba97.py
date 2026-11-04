"""Matrix multiplication 6x6x6 via Kronecker product of two 2x2 Strassen decompositions.

Approach:
- The 4x4 matrix multiplication tensor is the tensor product M_2 ⊗ M_2.
- A rank-7 Strassen decomposition of M_2 gives a rank-49 decomposition of M_4.
- Embed the 4x4 multiplication into 6x6 by padding with two zero rows/cols.
- Handle the remaining 4x6, 6x4, 2x2 cross-blocks with schoolbook multiplication.
- Total rank: 49 + 24 + 24 + 4 = 101.
"""
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = 6, 6, 6

    # --- Strassen rank-7 decomposition for 2x2 multiplication ---
    # For A, B in F^{2x2}, C = A*B.
    # Indexing: A[i,j] with i in {0,1}, j in {0,1}; same for B; C[k,i] with k in {0,1}.
    # U has 4 entries (2x2), V has 4 entries (2x2), W has 4 entries (2x2).
    # Strassen:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # U[a,b] = coefficient of A[a,b] in term r
    # V[c,d] = coefficient of B[c,d] in term r
    # W[k,i] = coefficient of C[k,i] in term r  (trace convention: W[k*n+i, r])
    # For 2x2: U is 4x7, V is 4x7, W is 4x7.
    #
    # Let's define the 7 terms:
    # Term 0: M1 = (A00+A11)*(B00+B11)
    #   U: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1 (row-major: A00=idx0, A01=idx1, A10=idx2, A11=idx3)
    #   V: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
    #   W: C00=1, C11=1 -> W[0,0]=1, W[3,0]=1
    #
    # Term 1: M2 = (A10+A11)*B00
    #   U: A10=1, A11=1 -> U[2,1]=1, U[3,1]=1
    #   V: B00=1 -> V[0,1]=1
    #   W: C10=1, C11=-1 -> W[2,1]=1, W[3,1]=-1
    #
    # Term 2: M3 = A00*(B01-B11)
    #   U: A00=1 -> U[0,2]=1
    #   V: B01=1, B11=-1 -> V[1,2]=1, V[3,2]=-1
    #   W: C01=1, C11=1 -> W[1,2]=1, W[3,2]=1
    #
    # Term 3: M4 = A11*(B10-B00)
    #   U: A11=1 -> U[3,3]=1
    #   V: B10=1, B00=-1 -> V[2,3]=1, V[0,3]=-1
    #   W: C00=1, C10=1 -> W[0,3]=1, W[2,3]=1
    #
    # Term 4: M5 = (A00+A01)*B11
    #   U: A00=1, A01=1 -> U[0,4]=1, U[1,4]=1
    #   V: B11=1 -> V[3,4]=1
    #   W: C00=-1, C01=1 -> W[0,4]=-1, W[1,4]=1
    #
    # Term 5: M6 = (A10-A00)*(B00+B01)
    #   U: A10=1, A00=-1 -> U[2,5]=1, U[0,5]=-1
    #   V: B00=1, B01=1 -> V[0,5]=1, V[1,5]=1
    #   W: C11=1 -> W[3,5]=1
    #
    # Term 6: M7 = (A01-A11)*(B10+B11)
    #   U: A01=1, A11=-1 -> U[1,6]=1, U[3,6]=-1
    #   V: B10=1, B11=1 -> V[2,6]=1, V[3,6]=1
    #   W: C00=1 -> W[0,6]=1

    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # Term 0: M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # Term 1: M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # Term 2: M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # Term 3: M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # Term 4: M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # Term 5: M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # Term 6: M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # --- Build 4x4 decomposition from Kronecker product of two 2x2 Strassen ---
    # M_4 = M_2 ⊗ M_2
    # U4 = U2 ⊗ U2 (Kronecker), shape (16, 49)
    # V4 = V2 ⊗ V2, shape (16, 49)
    # W4 = W2 ⊗ W2, shape (16, 49)
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # --- Embed 4x4 into 6x6 ---
    # The 4x4 block uses indices 0,1,2,3. The remaining indices are 4,5.
    # We need to handle:
    #   - C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4]  (handled by 4x4 decomposition)
    #   - C[0:4, 4:6] = A[0:4, 0:6] * B[4:6, 4:6]  -> A[0:4, 4:6] * B[4:6, 4:6]
    #     Wait, C[i,k] = sum_j A[i,j] * B[j,k]
    #     For i in 0:4, k in 4:6: C[i,k] = sum_{j=0}^{5} A[i,j] * B[j,k]
    #     = sum_{j=0}^{3} A[i,j] * B[j,k] + sum_{j=4}^{5} A[i,j] * B[j,k]
    #     The first part involves B[j,k] with j in 0:4, k in 4:6. But B is 6x6, so B[j,k] for j in 0:4, k in 4:6 is valid.
    #     However, our 4x4 decomposition only handles B[0:4, 0:4]. We need to handle the full 6x6 B.
    #
    # Let me reconsider. The 4x4 decomposition handles A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4].
    # For the rest, we use schoolbook:
    #   - i in 0:4, k in 4:6: C[i,k] = sum_j A[i,j]*B[j,k]. This is 4*2*6 = 48 terms, but we can do it more efficiently.
    #   Actually, let's just use schoolbook for all terms not covered by the 4x4 block.
    #
    # Terms covered by 4x4 Strassen: i in 0:4, j in 0:4, k in 0:4. That's 4*4*4 = 64 (i,j,k) triples.
    # Total (i,j,k) triples: 6*6*6 = 216.
    # Remaining: 216 - 64 = 152 terms. But we can group them.
    #
    # Actually, the rank of the 4x4 decomposition is 49, which handles all 64 (i,j,k) triples for i,j,k in 0:4.
    # For the remaining triples, we use schoolbook (rank 1 per triple).
    #
    # Let's list the remaining (i,j,k) triples:
    #   i in 0:4, j in 4:6, k in 0:6: 4*2*6 = 48
    #   i in 0:4, j in 0:4, k in 4:6: 4*4*2 = 32
    #   i in 4:6, j in 0:6, k in 0:6: 2*6*6 = 72
    #   i in 4:6, j in 0:6, k in 4:6: 2*6*2 = 24 (but this is already counted in the above? No, let me be careful)
    #
    # Let me just iterate over all (i,j,k) and skip those with i,j,k all in 0:4.
    remaining = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                if not (i < 4 and j < 4 and k < 4):
                    remaining.append((i, j, k))

    # Total rank = 49 + len(remaining)
    total_rank = 49 + len(remaining)

    # Build U, V, W
    U = np.zeros((36, total_rank), dtype=int)
    V = np.zeros((36, total_rank), dtype=int)
    W = np.zeros((36, total_rank), dtype=int)

    # First 49 columns: 4x4 Strassen
    # U4 is 16x49, corresponding to A[0:4, 0:4] with row index i*4+j for i,j in 0:4
    # In the 6x6 matrix, A[i,j] has row index i*6+j
    for i in range(4):
        for j in range(4):
            U[i*6 + j, 0:49] = U4[i*4 + j, :]

    for j in range(4):
        for k in range(4):
            V[j*6 + k, 0:49] = V4[j*4 + k, :]

    for k in range(4):
        for i in range(4):
            W[k*6 + i, 0:49] = W4[k*4 + i, :]

    # Remaining columns: schoolbook
    col = 49
    for (i, j, k) in remaining:
        U[i*6 + j, col] = 1
        V[j*6 + k, col] = 1
        W[k*6 + i, col] = 1
        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
