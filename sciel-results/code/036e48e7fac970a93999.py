"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    
    Strategy: Use a block-partitioned approach with Strassen's algorithm for 2x2 blocks
    and schoolbook for the remaining 1x1 blocks. This gives rank 117 (vs 125 schoolbook).
    
    Partition 5 = 2 + 2 + 1:
    - 4 blocks of size 2x2x2: use Strassen (rank 7 each) = 28
    - 8 blocks of size 2x1x1 or 1x2x1: schoolbook (rank 2 each) = 16
    - 1 block of size 1x1x1: schoolbook (rank 1) = 1
    Total: 28 + 16 + 1 = 45... wait, let me recalculate.
    
    Actually, for matrix multiplication C = A * B where A is 5x5 and B is 5x5:
    Partition rows of A and columns of B into blocks.
    
    Let's think of it as: A is partitioned into row-blocks, B into column-blocks.
    A = [A11 A12 A13] where A11 is 2x2, A12 is 2x2, A13 is 2x1
        [A21 A22 A23] where A21 is 2x2, A22 is 2x2, A23 is 2x1
        [A31 A32 A33] where A31 is 1x2, A32 is 1x2, A33 is 1x1
    
    B = [B11 B12 B13] where B11 is 2x2, B12 is 2x2, B13 is 2x1
        [B21 B22 B23] where B21 is 2x2, B22 is 2x2, B23 is 2x1
        [B31 B32 B33] where B31 is 1x2, B32 is 1x2, B33 is 1x1
    
    C = A * B, so C[i,j] = sum_k A[i,k] * B[k,j]
    
    C is partitioned into blocks:
    C = [C11 C12 C13]
        [C21 C22 C23]
        [C31 C32 C33]
    
    where C11 = A11*B11 + A12*B21 + A13*B31 (2x2)
          C12 = A11*B12 + A12*B22 + A13*B32 (2x2)
          C13 = A11*B13 + A12*B23 + A13*B33 (2x1)
          C21 = A21*B11 + A22*B21 + A23*B31 (2x2)
          C22 = A21*B12 + A22*B22 + A23*B32 (2x2)
          C23 = A21*B13 + A22*B23 + A23*B33 (2x1)
          C31 = A31*B11 + A32*B21 + A33*B31 (1x2)
          C32 = A31*B12 + A32*B22 + A33*B32 (1x2)
          C33 = A31*B13 + A32*B23 + A33*B33 (1x1)
    
    For each 2x2 block multiplication, we can use Strassen's algorithm with rank 7.
    For 2x1 or 1x2 block multiplications, we need rank 2 (schoolbook).
    For 1x1, rank 1.
    
    Let me count the total rank:
    - 4 full 2x2 block products (C11, C12, C21, C22): each is sum of 3 terms, each term is 2x2*2x2
      But wait, each C block is a sum of 3 products. Each product like A11*B11 is a 2x2*2x2 multiplication.
    
    Actually, I need to think about this differently. The tensor for 5x5 matrix mult has rank at most 125 (schoolbook).
    
    Let me use a different approach: directly construct the decomposition using known algorithms.
    
    For the 2+2+1 partition, the total rank is:
    - For each of the 9 output blocks (3x3 grid), we compute a sum of 3 matrix products.
    - The 4 corner/edge 2x2 output blocks each involve 3 products of 2x2 matrices.
    - But we can't just add up ranks because the same input elements are shared.
    
    Let me just implement the known result: rank 117 for 5x5 using Strassen on 2x2 subblocks.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Use block structure: partition into 2+2+1
    # We'll build the U, V, W matrices explicitly.
    
    # The matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,j] = sum_k A[i,k]*B[k,j]
    # In the standard convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # where the tensor is T[a, b, c] with a = i*m+j (row i, col j of A), b = j*p+k (row j, col k of B), c = k*n+i (row k, col i of C)
    # Wait, the task says W[k*n+i, r] so c = k*n + i where k is the column of C (which is the row of B) and i is the row of C (which is the row of A).
    
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So for C[i,j] = sum_k A[i,k] * B[k,j]:
    # U index: i*m + k (row i, col k of A) -- wait, that's n*m entries
    # V index: k*p + j (row k, col j of B) -- that's m*p entries  
    # W index: j*n + i (row j, col i of C) -- that's p*n entries
    
    # So T[i*m+k, k*p+j, j*n+i] = 1 for all i,j,k in range(5)
    
    # For the schoolbook, rank = 125, with one term for each (i,j,k).
    
    # For Strassen-based reduction, I'll use the following approach:
    # Partition indices into groups: {0,1}, {2,3}, {4}
    # For each combination of groups (I, J, K) where I is row group of A, K is col group of A / row group of B, J is col group of B / col group of C:
    #   If |I|=2, |K|=2, |J|=2: use Strassen (7 terms instead of 8)
    #   Otherwise: use schoolbook
    
    # Let me enumerate all (i, k, j) triples and group them:
    # Groups: G0 = {0,1}, G1 = {2,3}, G2 = {4}
    
    groups = [[0, 1], [2, 3], [4]]
    
    # Build the decomposition
    U_entries = []  # list of (index, value) pairs
    V_entries = []
    W_entries = []
    
    col = 0
    
    for gi, I in enumerate(groups):
        for gk, K in enumerate(groups):
            for gj, J in enumerate(groups):
                # For each i in I, k in K, j in J, we have a term
                if len(I) == 2 and len(K) == 2 and len(J) == 2:
                    # Use Strassen for this 2x2x2 block
                    # Strassen's algorithm for 2x2 matrix multiplication:
                    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
                    # C = [[p, q], [r, s]]
                    # m1 = (a+d)*(e+h)
                    # m2 = (c+d)*e
                    # m3 = a*(f-h)
                    # m4 = d*(g-e)
                    # m5 = (a+b)*h
                    # m6 = (c-a)*(e+f)
                    # m7 = (b-d)*(g+h)
                    # p = m1 + m4 - m5 + m7
                    # q = m3 + m5
                    # r = m2 + m4
                    # s = m1 - m2 + m3 + m6
                    
                    i0, i1 = I[0], I[1]
                    k0, k1 = K[0], K[1]
                    j0, j1 = J[0], J[1]
                    
                    # A elements: A[i0,k0]=a, A[i0,k1]=b, A[i1,k0]=c, A[i1,k1]=d
                    # B elements: B[k0,j0]=e, B[k0,j1]=f, B[k1,j0]=g, B[k1,j1]=h
                    # C elements: C[i0,j0]=p, C[i0,j1]=q, C[i1,j0]=r, C[i1,j1]=s
                    
                    a_idx = i0 * m + k0
                    b_idx = i0 * m + k1
                    c_idx = i1 * m + k0
                    d_idx = i1 * m + k1
                    
                    e_idx = k0 * p + j0
                    f_idx = k0 * p + j1
                    g_idx = k1 * p + j0
                    h_idx = k1 * p + j1
                    
                    p_idx = j0 * n + i0
                    q_idx = j1 * n + i0
                    r_idx = j0 * n + i1
                    s_idx = j1 * n + i1
                    
                    # m1 = (a+d)*(e+h)
                    U_entries.append((a_idx, 1)); V_entries.append((e_idx, 1)); W_entries.append((p_idx, 1))
                    U_entries.append((d_idx, 1)); V_entries.append((h_idx, 1)); W_entries.append((p_idx, 1))
                    U_entries.append((a_idx, 1)); V_entries.append((h_idx, 1)); W_entries.append((q_idx, 1))
                    U_entries.append((d_idx, 1)); V_entries.append((e_idx, 1)); W_entries.append((s_idx, 1))
                    col += 1
                    
                    # m2 = (c+d)*e
                    U_entries.append((c_idx, 1)); V_entries.append((e_idx, 1)); W_entries.append((r_idx, 1))
                    U_entries.append((d_idx, 1)); V_entries.append((e_idx, 1)); W_entries.append((r_idx, 1))
                    U_entries.append((c_idx, 1)); V_entries.append((e_idx, 1)); W_entries.append((s_idx, -1))
                    U_entries.append((d_idx, 1)); V_entries.append((e_idx, 1)); W_entries.append((s_idx, -1))
                    col += 1
                    
                    # m3 = a*(f-h)
                    U_entries.append((a_idx, 1)); V_entries.append((f_idx, 1)); W_entries.append((q_idx, 1))
                    U_entries.append((a_idx, 1)); V_entries.append((h_idx, -1)); W_entries.append((q_idx, 1))
                    U_entries.append((a_idx, 1)); V_entries.append((f_idx, 1)); W_entries.append((s_idx, 1))
                    U_entries.append((a_idx, 1)); V_entries.append((h_idx, -1)); W_entries.append((s_idx, 1))
                    col += 1
                    
                    # m4 = d*(g-e)
                    U_entries.append((d_idx, 1)); V_entries.append((g_idx, 1)); W_entries.append((p_idx, 1))
                    U_entries.append((d_idx, 1)); V_entries.append((e_idx, -1)); W_entries.append((p_idx, 1))
                    U_entries.append((d_idx, 1)); V_entries.append((g_idx, 1)); W_entries.append((r_idx, 1))
                    U_entries.append((d_idx, 1)); V_entries.append((e_idx, -1)); W_entries.append((r_idx, 1))
                    col += 1
                    
                    # m5 = (a+b)*h
                    U_entries.append((a_idx, 1)); V_entries.append((h_idx, 1)); W_entries.append((p_idx, -1))
                    U_entries.append((b_idx, 1)); V_entries.append((h_idx, 1)); W_entries.append((p_idx, -1))
                    U_entries.append((a_idx, 1)); V_entries.append((h_idx, 1)); W_entries.append((q_idx, 1))
                    U_entries.append((b_idx, 1)); V_entries.append((h_idx, 1)); W_entries.append((q_idx, 1))
                    col += 1
                    
                    # m6 = (c-a)*(e+f)
                    U_entries.append((c_idx, 1)); V_entries.append((e_idx, 1)); W_entries.append((s_idx, 1))
                    U_entries.append((a_idx, -1)); V_entries.append((e_idx, 1)); W_entries.append((s_idx, 1))
                    U_entries.append((c_idx, 1)); V_entries.append((f_idx, 1)); W_entries.append((s_idx, 1))
                    U_entries.append((a_idx, -1)); V_entries.append((f_idx, 1)); W_entries.append((s_idx, 1))
                    col += 1
                    
                    # m7 = (b-d)*(g+h)
                    U_entries.append((b_idx, 1)); V_entries.append((g_idx, 1)); W_entries.append((p_idx, 1))
                    U_entries.append((d_idx, -1)); V_entries.append((g_idx, 1)); W_entries.append((p_idx, 1))
                    U_entries.append((b_idx, 1)); V_entries.append((h_idx, 1)); W_entries.append((p_idx, 1))
                    U_entries.append((d_idx, -1)); V_entries.append((h_idx, 1)); W_entries.append((p_idx, 1))
                    col += 1
                else:
                    # Schoolbook for this block
                    for i in I:
                        for k in K:
                            for j in J:
                                U_entries.append((i * m + k, 1))
                                V_entries.append((k * p + j, 1))
                                W_entries.append((j * n + i, 1))
                                col += 1
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_idx, u_val = U_entries[c]
        v_idx, v_val = V_entries[c]
        w_idx, w_val = W_entries[c]
        U[u_idx][c] = u_val
        V[v_idx][c] = v_val
        W[w_idx][c] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
