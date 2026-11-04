"""Matrix multiplication 4x4x4 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 4
    deadline = time.time() + 110  # safety margin before 120s limit

    # We'll build a rank-48 decomposition using block structure.
    # The 4x4x4 tensor is divided into 8x8x8 = 512 entries.
    # Block structure: 2x2x2 blocks of size 2x2x2.
    # 4 diagonal blocks (where block indices are equal) use rank-6 degenerate Strassen.
    # 4 off-diagonal blocks where i-block != k-block but j-block varies use rank-4.
    # Actually: blocks (bi, bj, bk) where bi=bj=bk are diagonal (4 of them: (0,0,0),(0,0,1),(0,1,0),(0,1,1) etc.)
    # Wait: 2x2x2 block grid gives 8 blocks. Diagonal: (0,0,0),(1,1,1) -> 2 blocks.
    # Off-diagonal: 6 blocks.
    # Total: 2*6 + 6*4 = 12 + 24 = 36? That seems too low.
    # 
    # Let me reconsider. The 4x4x4 tensor T[i,j,k] = delta_{i,k} (since C[i,k] = sum_j A[i,j]B[j,k]).
    # Actually T[i,j,k] = 1 if i==k else 0... no.
    # T_{ijk} = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # The tensor is T[i,j,k] = delta_{i,k}? No.
    # T[i,j,k] = 1 for all i,j,k such that... 
    # Actually the matrix multiplication tensor has T[i,j,k] = 1 if i==k? No.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) — it's the full tensor with T[i,j,k]=1 when... 
    # No! T[i,j,k] = delta_{i,k} is wrong.
    # The tensor is T[i,j,k] = 1 for ALL i,j,k? No.
    # T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] appears in C[i,k].
    # It appears for every (i,j,k) triple. So T[i,j,k] = 1 for all 64 entries? 
    # No! The tensor T is defined by T[i,j,k] = 1 if i==k... 
    # 
    # Let me be precise. The bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k].
    # The associated tensor T has components T[i,j,k] such that C[i,k] = sum_{j} A[i,j] B[j,k].
    # In tensor notation: C = T(A,B) where T[i,j,k] = delta_{i,k}? No.
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) of C.
    # This is true for ALL i,j,k. So T[i,j,k] = 1 for all 4*4*4=64 entries.
    # Wait, that's the schoolbook: rank = 64.
    # 
    # Hmm, but that can't be right because Strassen reduces 2x2x2 from 8 to 7.
    # For 2x2x2: C[i,k] = sum_j A[i,j]B[j,k], i,k in {0,1}, j in {0,1}.
    # T[i,j,k] = 1 for all 8 combinations. Rank of this tensor is 7 (Strassen).
    # So yes, T[i,j,k] = 1 for all entries, and the rank is the tensor rank.
    #
    # OK so for 4x4x4, T[i,j,k] = 1 for all 64 entries, and we want minimum rank.
    #
    # Block decomposition: partition indices into blocks of size 2.
    # Block (bi,bj,bk) contains T entries with i in {2bi, 2bi+1}, j in {2bj, 2bj+1}, k in {2bk, 2bk+1}.
    # There are 8 blocks: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1).
    #
    # For block (bi,bj,bk): the sub-tensor is 2x2x2 with all entries = 1.
    # If bi == bk (diagonal in block sense), this is a "diagonal" block.
    # If bi != bk, this is an "off-diagonal" block.
    #
    # Diagonal blocks: (0,0,0), (0,0,1) -> wait, bi==bk means bi=bk.
    # (0,0,0): bi=0,bk=0 -> diagonal
    # (0,0,1): bi=0,bk=1 -> off-diagonal
    # (0,1,0): bi=0,bk=0 -> diagonal
    # (0,1,1): bi=0,bk=1 -> off-diagonal
    # (1,0,0): bi=1,bk=0 -> off-diagonal
    # (1,0,1): bi=1,bk=1 -> diagonal
    # (1,1,0): bi=1,bk=0 -> off-diagonal
    # (1,1,1): bi=1,bk=1 -> diagonal
    #
    # So 4 diagonal blocks and 4 off-diagonal blocks.
    # Each is a full 2x2x2 tensor (all 8 entries = 1).
    # Rank of full 2x2x2 tensor = 7 (Strassen).
    # Total: 8 * 7 = 56. That's worse than 49.
    #
    # The key insight: off-diagonal blocks (bi != bk) have a special structure.
    # For block (bi,bj,bk) with bi != bk, the sub-tensor maps:
    # A[2bi:2bi+2, 2bj:2bj+2] * B[2bj:2bj+2, 2bk:2bk+2] -> C[2bi:2bi+2, 2bk:2bk+2]
    # This is a 2x2 matrix product, which has rank 7 (Strassen) or can be done in rank 4 
    # if we use the fact that... no, 2x2 matrix multiplication has rank 7.
    #
    # Hmm, I'm confusing myself. Let me just use the known rank-49 construction or 
    # implement a clean Strassen-based approach.
    #
    # Actually, the simplest correct approach: use Kronecker product of Strassen.
    # 4x4 = 2x2 ⊗ 2x2, so rank(4x4x4) <= rank(2x2x2)^2 = 7*7 = 49.
    # This gives rank 49.
    #
    # For rank 48, we need a better construction. Let me try the block approach
    # where diagonal 2x2x2 blocks use a rank-6 "degenerate" decomposition.
    #
    # For a diagonal block (bi,bj,bk) with bi=bj=bk (say bi=0):
    # The sub-tensor is the full 2x2x2 tensor. Its rank is 7.
    # But if we can exploit that this block is "diagonal" in the larger structure,
    # maybe we can share terms with other blocks.
    #
    # This is getting complex. Let me just implement rank 49 via Kronecker product
    # of Strassen, which is known to be correct.

    # Strassen's 7-term decomposition for 2x2x2:
    # M1 = A00 * B00
    # M2 = (A00+A01) * B10
    # M3 = A10 * (B01-B11)
    # M4 = A11 * B11
    # M5 = (A00+A11) * (B00+B11)
    # M6 = (A01-A11) * (B10+B11)
    # M7 = (A00+A10) * (B01-B11)
    #
    # C00 = M1 + M5 - M4 - M6
    # C01 = M3 + M4
    # C10 = M1 + M2
    # C11 = M5 - M1 + M3 + M7
    #
    # For 4x4x4 via Kronecker: rank = 7*7 = 49.
    # Each factor has 49 columns.
    # U has 16 rows, V has 16 rows, W has 16 rows.

    # Build Strassen factors for 2x2x2
    # U_2[i*2+j, c], V_2[j*2+k, c], W_2[k*2+i, c]
    # Note: W uses trace convention W[k*n+i, r] for 2x2: W[k*2+i, r]

    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)

    # Column 0: M1 = A00 * B00
    U2[0, 0] = 1  # A00
    V2[0, 0] = 1  # B00
    W2[0, 0] = 1  # C00 (k=0, i=0)

    # Column 1: M2 = (A00+A01) * B10
    U2[0, 1] = 1  # A00
    U2[1, 1] = 1  # A01
    V2[2, 1] = 1  # B10 (j=1, k=0)
    W2[0, 1] = 1  # C00 (k=0, i=0)

    # Column 2: M3 = A10 * (B01-B11)
    U2[2, 2] = 1  # A10 (i=1, j=0)
    V2[1, 2] = 1  # B01 (j=0, k=1)
    V2[3, 2] = -1 # B11 (j=1, k=1)
    W2[1, 2] = 1  # C01 (k=1, i=0)
    W2[3, 2] = 1  # C11 (k=1, i=1)

    # Column 3: M4 = A11 * B11
    U2[3, 3] = 1  # A11 (i=1, j=1)
    V2[3, 3] = 1  # B11 (j=1, k=1)
    W2[3, 3] = 1  # C11 (k=1, i=1)

    # Column 4: M5 = (A00+A11) * (B00+B11)
    U2[0, 4] = 1  # A00
    U2[3, 4] = 1  # A11
    V2[0, 4] = 1  # B00
    V2[3, 4] = 1  # B11
    W2[0, 4] = 1  # C00
    W2[3, 4] = 1  # C11

    # Column 5: M6 = (A01-A11) * (B10+B11)
    U2[1, 5] = 1  # A01
    U2[3, 5] = -1 # A11
    V2[2, 5] = 1  # B10
    V2[3, 5] = 1  # B11
    W2[0, 5] = -1 # C00
    W2[3, 5] = -1 # C11

    # Column 6: M7 = (A00+A10) * (B01-B11)
    U2[0, 6] = 1  # A00
    U2[2, 6] = 1  # A10
    V2[1, 6] = 1  # B01
    V2[3, 6] = -1 # B11
    W2[1, 6] = 1  # C01
    W2[3, 6] = 1  # C11

    # Now build 4x4x4 via Kronecker product
    # Index mapping: for 4x4, i = i1*2 + i0, j = j1*2 + j0, k = k1*2 + k0
    # U4[i*4+j, c] where c = c1*7 + c0
    # U4[i1*4 + i0*2 + j1*2 + j0, c1*7+c0] = U2[i1*2+j1, c1] * U2[i0*2+j0, c0]
    # Wait, need to be careful with index ordering.
    # 
    # For the Kronecker product of tensor decompositions:
    # U4[i,j, c1,c0] = U2[i1,j1, c1] * U2[i0,j0, c0]
    # where i = i1*2+i0, j = j1*2+j0
    # So U4[i1*2+i0, j1*2+j0, c1*7+c0] = U2[i1,j1,c1] * U2[i0,j0,c0]
    # In flat index: U4[(i1*2+i0)*4 + (j1*2+j0), c1*7+c0]
    # = U2[i1*2+j1, c1] * U2[i0*2+j0, c0]

    rank = 49
    U4 = np.zeros((16, rank), dtype=float)
    V4 = np.zeros((16, rank), dtype=float)
    W4 = np.zeros((16, rank), dtype=float)

    for c1 in range(7):
        for c0 in range(7):
            c = c1 * 7 + c0
            for i1 in range(2):
                for j1 in range(2):
                    for i0 in range(2):
                        for j0 in range(2):
                            i = i1 * 2 + i0
                            j = j1 * 2 + j0
                            u_val = U2[i1 * 2 + j1, c1] * U2[i0 * 2 + j0, c0]
                            if u_val != 0:
                                U4[i * 4 + j, c] = u_val
                            v_val = V2[j1 * 2 + (k1 := 0) * 0, c1]  # placeholder, fix below
            # Redo V and W properly
            for j1 in range(2):
                for k1 in range(2):
                    for j0 in range(2):
                        for k0 in range(2):
                            j = j1 * 2 + j0
                            k = k1 * 2 + k0
                            v_val = V2[j1 * 2 + k1, c1] * V2[j0 * 2 + k0, c0]
                            if v_val != 0:
                                V4[j * 4 + k, c] = v_val
            for k1 in range(2):
                for i1 in range(2):
                    for k0 in range(2):
                        for i0 in range(2):
                            k = k1 * 2 + k0
                            i = i1 * 2 + i0
                            w_val = W2[k1 * 2 + i1, c1] * W2[k0 * 2 + i0, c0]
                            if w_val != 0:
                                W4[k * 4 + i, c] = w_val

    # Verify correctness
    # C = sum_r U[:,r] * V[:,r] * W[:,r] should give the matmul tensor
    # Check: for random A, B, C[i,k] = sum_j A[i,j]B[j,k]
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test

    C_computed = np.zeros((4, 4))
    for r in range(rank):
        A_factor = U4[:, r].reshape(4, 4)
        B_factor = V4[:, r].reshape(4, 4)
        C_factor = W4[:, r].reshape(4, 4)
        C_computed += np.einsum('ij,jk,kl->il', A_factor, B_factor, C_factor)

    # Actually, the decomposition is: C[i,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] summed over j
    # Let me verify properly
    C_check = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            for j in range(4):
                C_check[i, k] += sum(U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] for r in range(rank))

    if not np.allclose(C_check, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Convert to lists
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
