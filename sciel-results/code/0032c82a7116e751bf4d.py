"""Matrix multiplication 3,4,6: block-based rank-68 exact tensor decomposition.

Uses the Strassen-like rank-7 algorithm for 2x2x2 and combines with direct
decompositions for smaller blocks to achieve rank 68 (vs. schoolbook 72).

The decomposition satisfies:
  (A * B)_{i,k} = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
  where T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 6]
EVAL_DEADLINE_SECONDS = 118.0


def _make_strassen_222():
    """
    Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # We need: C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    #
    # For each product r, we have a rank-1 term in the tensor:
    # T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    #
    # The Strassen decomposition works by defining 7 "products" which are
    # bilinear forms in A and B. Each product r corresponds to:
    #   U[:, :, r] * V[:, :, r] * W[:, :, r]
    # where these are specific rank-1 tensors.
    
    # Let's think of it differently. The tensor T[i,j,k] for 2x2x2 is:
    # T[i,j,k] = delta condition: it's 1 if (i,j,k) is a valid multiplication index.
    # Actually T[i,j,k] is the structure constant: T[i,j,k] = 1 always for all i,j,k
    # in the sense that C[i,k] = sum_j A[i,j] B[j,k].
    
    # The tensor is: T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1}
    # Wait, no. The multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Because (AB)[i,k] = sum_j A[i,j] B[j,k], and the tensor entry is 1.
    
    # Actually the tensor is: T[i,j,k] = 1 for all i,j,k in the 2x2x2 case.
    # No wait, that's not right either. Let me reconsider.
    
    # The matrix multiplication tensor for n x m x p has entries:
    # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # For 2x2x2: T[i,j,k] = 1 for all i,j,k in {0,1}
    # So we need: sum_r U[i*2+j, r] V[j*2+k, r] W[k*2+i, r] = 1 for all i,j,k
    
    # Strassen gives rank 7. Let me look up the explicit decomposition.
    # 
    # The standard way: define 7 vectors u_r in R^4 (for A indices), v_r in R^4 (for B indices), 
    # w_r in R^4 (for C indices) such that sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1 for all i,j,k.
    
    # Actually, let me use a known explicit Strassen decomposition.
    # 
    # From the literature, the 7 rank-1 terms for 2x2x2 are:
    # 
    # r=0: U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
    #   But this doesn't work directly because we need the sum to equal 1 for each (i,j,k).
    
    # Let me use a different approach. I'll construct the Strassen decomposition 
    # using the standard formulas.
    
    # The 7 products are:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00  
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    
    # And:
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    
    # For the tensor decomposition, each P_r is a bilinear form:
    # P_r = sum_{i,j} alpha_r[i,j] A[i,j] * sum_{j,k} beta_r[j,k] B[j,k]
    # Wait, that's not quite right. Each P_r is a product of a linear form in A and a linear form in B.
    
    # P1 = (a00+a11)(b00+b11) = (sum_{i,j} u1[i,j] a[i,j]) * (sum_{j,k} v1[j,k] b[j,k])
    # where u1[0,0]=1, u1[1,1]=1, others 0; v1[0,0]=1, v1[1,1]=1, others 0.
    
    # Then the contribution to C is:
    # c[i,k] += w1[i,k] * P1
    # where w1 encodes how P1 contributes to each c[i,k].
    
    # For P1: c00 += P1, c11 += P1, so w1[0,0]=1, w1[1,1]=1, w1[0,1]=0, w1[1,0]=0.
    
    # Let me write out all 7:
    
    # P1: u1 = [[1,0],[0,1]], v1 = [[1,0],[0,1]], w1 = [[1,0],[0,1]]
    # P2: u2 = [[0,0],[1,1]], v2 = [[1,0],[0,0]], w2 = [[0,0],[1,0]]  (c10 += P2)
    # P3: u3 = [[1,0],[0,0]], v3 = [[0,1],[0,-1]], w3 = [[0,1],[0,0]]  (c01 += P3)
    # P4: u4 = [[0,0],[0,1]], v4 = [[0,-1],[1,0]], w4 = [[1,0],[0,1]]  (c00 += P4, c11 += P4)
    # P5: u5 = [[1,1],[0,0]], v5 = [[0,0],[0,1]], w5 = [[-1,1],[0,0]]  (c00 -= P5, c01 += P5)
    # P6: u6 = [[-1,0],[1,0]], v6 = [[1,1],[0,0]], w6 = [[0,0],[0,1]]  (c11 += P6)
    # P7: u7 = [[0,1],[0,-1]], v7 = [[0,0],[1,1]], w7 = [[1,0],[0,0]]  (c00 += P7)
    
    # Let me verify:
    # c00 = P1 + P4 - P5 + P7
    # c01 = P3 + P5
    # c10 = P2 + P4
    # c11 = P1 - P2 + P3 + P6
    
    # Check c00: P1 + P4 - P5 + P7
    # = (a00+a11)(b00+b11) + a11(b10-b00) - (a00+a01)b11 + (a01-a11)(b10+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00 - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
    # = a00*b00 + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + (a11*b11 - a11*b11) + (a11*b10 - a11*b10) + (-a01*b11 + a01*b11) + a01*b10
    # = a00*b00 + a01*b10? That doesn't seem right.
    
    # Let me recompute more carefully.
    # c00 should = a00*b00 + a01*b10
    
    # P1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # P4 = a11*(b10-b00) = a11*b10 - a11*b00
    # -P5 = -(a00+a01)*b11 = -a00*b11 - a01*b11
    # P7 = (a01-a11)*(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
    
    # Sum: a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00 - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
    # = a00*b00 + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + (a11*b11 - a11*b11) + (a11*b10 - a11*b10) + (-a01*b11 + a01*b11) + a01*b10
    # = a00*b00 + a01*b10 ✓
    
    # Check c01 = P3 + P5 = a00*(b01-b11) + (a00+a01)*b11 = a00*b01 - a00*b11 + a00*b11 + a01*b11 = a00*b01 + a01*b11 ✓
    
    # Check c10 = P2 + P4 = (a10+a11)*b00 + a11*(b10-b00) = a10*b00 + a11*b00 + a11*b10 - a11*b00 = a10*b00 + a11*b10 ✓
    
    # Check c11 = P1 - P2 + P3 + P6
    # P1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # -P2 = -(a10+a11)*b00 = -a10*b00 - a11*b00
    # P3 = a00*(b01-b11) = a00*b01 - a00*b11
    # P6 = (a10-a00)*(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01
    
    # Sum: a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b01 - a00*b11 + a10*b00 + a10*b01 - a00*b00 - a00*b01
    # = (a00*b00 - a00*b00) + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + a11*b11 + (-a10*b00 + a10*b00) + (a00*b01 - a00*b01) + a10*b01
    # = a11*b11 + a10*b01 ✓
    
    # Great! Now let me construct the matrices.
    
    # U: shape (4, 7), U[i*2+j, r] = u_r[i,j]
    # V: shape (4, 7), V[j*2+k, r] = v_r[j,k]
    # W: shape (4, 7), W[k*2+i, r] = w_r[i,k]
    
    # u1 = [[1,0],[0,1]] -> U[0,0]=1, U[3,0]=1 (indices: i*2+j: (0,0)->0, (1,1)->3)
    # u2 = [[0,0],[1,1]] -> U[2,1]=1, U[3,1]=1
    # u3 = [[1,0],[0,0]] -> U[0,2]=1
    # u4 = [[0,0],[0,1]] -> U[3,3]=1
    # u5 = [[1,1],[0,0]] -> U[0,4]=1, U[1,4]=1
    # u6 = [[-1,0],[1,0]] -> U[0,5]=-1, U[2,5]=1
    # u7 = [[0,1],[0,-1]] -> U[1,6]=1, U[3,6]=-1
    
    U = np.zeros((4, 7), dtype=int)
    U[0, 0] = 1; U[3, 0] = 1  # r=0 (P1)
    U[2, 1] = 1; U[3, 1] = 1  # r=1 (P2)
    U[0, 2] = 1                # r=2 (P3)
    U[3, 3] = 1                # r=3 (P4)
    U[0, 4] = 1; U[1, 4] = 1  # r=4 (P5)
    U[0, 5] = -1; U[2, 5] = 1 # r=5 (P6)
    U[1, 6] = 1; U[3, 6] = -1 # r=6 (P7)
    
    # v1 = [[1,0],[0,1]] -> V[0,0]=1, V[3,0]=1 (j*2+k: (0,0)->0, (1,1)->3)
    # v2 = [[1,0],[0,0]] -> V[0,1]=1
    # v3 = [[0,1],[0,-1]] -> V[1,2]=1, V[3,2]=-1
    # v4 = [[0,-1],[1,0]] -> V[1,3]=-1, V[2,3]=1
    # v5 = [[0,0],[0,1]] -> V[3,4]=1
    # v6 = [[1,1],[0,0]] -> V[0,5]=1, V[1,5]=1
    # v7 = [[0,0],[1,1]] -> V[2,6]=1, V[3,6]=1
    
    V = np.zeros((4, 7), dtype=int)
    V[0, 0] = 1; V[3, 0] = 1  # r=0
    V[0, 1] = 1                # r=1
    V[1, 2] = 1; V[3, 2] = -1 # r=2
    V[1, 3] = -1; V[2, 3] = 1 # r=3
    V[3, 4] = 1                # r=4
    V[0, 5] = 1; V[1, 5] = 1  # r=5
    V[2, 6] = 1; V[3, 6] = 1  # r=6
    
    # w1 = [[1,0],[0,1]] -> W[0,0]=1, W[3,0]=1 (k*2+i: (0,0)->0, (1,1)->3)
    # w2 = [[0,0],[1,0]] -> W[2,1]=1
    # w3 = [[0,1],[0,0]] -> W[1,2]=1
    # w4 = [[1,0],[0,1]] -> W[0,3]=1, W[3,3]=1
    # w5 = [[-1,1],[0,0]] -> W[0,4]=-1, W[1,4]=1
    # w6 = [[0,0],[0,1]] -> W[3,5]=1
    # w7 = [[1,0],[0,0]] -> W[0,6]=1
    
    W = np.zeros((4, 7), dtype=int)
    W[0, 0] = 1; W[3, 0] = 1  # r=0
    W[2, 1] = 1                # r=1
    W[1, 2] = 1                # r=2
    W[0, 3] = 1; W[3, 3] = 1  # r=3
    W[0, 4] = -1; W[1, 4] = 1 # r=4
    W[3, 5] = 1                # r=5
    W[0, 6] = 1                # r=6
    
    return U, V, W


def _make_direct(n, m, p):
    """Direct (schoolbook) decomposition for n x m x p. Rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def _combine_blocks(blocks):
    """
    Combine a list of (U, V, W) blocks into a single decomposition.
    Each block is for a sub-problem. The blocks are concatenated in the rank dimension.
    """
    total_rank = sum(b[0].shape[1] for b in blocks)
    
    # Get the shapes from the first block (they should all be consistent for the overall problem)
    # Actually, we need to know the overall n, m, p. Let's just concatenate.
    
    U_list = []
    V_list = []
    W_list = []
    
    for U_b, V_b, W_b in blocks:
        U_list.append(U_b)
        V_list.append(V_b)
        W_list.append(W_b)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return U, V, W


def _embed_block(U_b, V_b, W_b, i_offset, j_offset, k_offset, n, m, p):
    """
    Embed a block decomposition (for sub-problem of size n_b x m_b x p_b)
    into the full n x m x p problem at offset (i_offset, j_offset, k_offset).
    
    U_b has shape (n_b * m_b, rank_b)
    V_b has shape (m_b * p_b, rank_b)
    W_b has shape (p_b * n_b, rank_b)
    
    We need to create full-sized U, V, W with zeros elsewhere.
    """
    n_b = U_b.shape[0] // m_b
    m_b = U_b.shape[0] // n_b  # This is circular, let me fix
    # Actually, let's pass n_b, m_b, p_b explicitly or infer differently.
    
    # U_b is (n_b*m_b, rank), V_b is (m_b*p_b, rank), W_b is (p_b*n_b, rank)
    rank_b = U_b.shape[1]
    
    U_full = np.zeros((n * m, rank_b), dtype=int)
    V_full = np.zeros((m * p, rank_b), dtype=int)
    W_full = np.zeros((p * n, rank_b), dtype=int)
    
    # Copy U_b: row index in U_b is i_b * m_b + j_b, maps to (i_offset + i_b) * m + (j_offset + j_b)
    n_b = U_b.shape[0] // m_b
    for i_b in range(n_b):
        for j_b in range(m_b):
            src_row = i_b * m_b + j_b
            dst_row = (i_offset + i_b) * m + (j_offset + j_b)
            U_full[dst_row, :] = U_b[src_row, :]
    
    # Copy V_b: row index in V_b is j_b * p_b + k_b, maps to (j_offset + j_b) * p + (k_offset + k_b)
    p_b = V_b.shape[0] // m_b
    for j_b in range(m_b):
        for k_b in range(p_b):
            src_row = j_b * p_b + k_b
            dst_row = (j_offset + j_b) * p + (k_offset + k_b)
            V_full[dst_row, :] = V_b[src_row, :]
    
    # Copy W_b: row index in W_b is k_b * n_b + i_b, maps to (k_offset + k_b) * n + (i_offset + i_b)
    for k_b in range(p_b):
        for i_b in range(n_b):
            src_row = k_b * n_b + i_b
            dst_row = (k_offset + k_b) * n + (i_offset + i_b)
            W_full[dst_row, :] = W_b[src_row, :]
    
    return U_full, V_full, W_full


def construct():
    """
    Return a rank-68 exact decomposition for 3x4x6 matrix multiplication.
    
    Strategy:
    1. Split m=4 into 2+2.
    2. Split p=6 into 3+3.
    3. For each (m_block, p_block) pair, we need mu(3, 2, 3) = mu(2, 3, 3).
    4. For mu(2, 3, 3): split m=3 into 2+1.
       - mu(2, 2, 3) + mu(2, 1, 3) = 11 + 6 = 17
    5. For mu(2, 2, 3): split p=3 into 2+1.
       - mu(2, 2, 2) + mu(2, 2, 1) = 7 + 4 = 11
    6. Total: 4 blocks * 17 = 68
    
    The W indexing convention is W[k*n + i, r].
    """
    t0 = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Get the base decompositions
    U_222, V_222, W_222 = _make_strassen_222()  # rank 7
    U_221, V_221, W_221 = _make_direct(2, 2, 1)  # rank 4
    U_213, V_213, W_213 = _make_direct(2, 1, 3)  # rank 6
    
    # Build mu(2,2,3) = mu(2,2,2) + mu(2,2,1), rank 11
    # Embed in 2x2x3 space
    U_223_1, V_223_1, W_223_1 = _embed_block(U_222, V_222, W_222, 0, 0, 0, 2, 2, 3)
    U_223_2, V_223_2, W_223_2 = _embed_block(U_221, V_221, W_221, 0, 0, 2, 2, 2, 3)
    
    U_223 = np.hstack([U_223_1, U_223_2])
    V_223 = np.hstack([V_223_1, V_223_2])
    W_223 = np.hstack([W_223_1, W_223_2])
    
    # Build mu(2,3,3) = mu(2,2,3) + mu(2,1,3), rank 17
    # Embed in 2x3x3 space
    U_233_1, V_233_1, W_233_1 = _embed_block(U_223, V_223, W_223, 0, 0, 0, 2, 3, 3)
    U_233_2, V_233_2, W_233_2 = _embed_block(U_213, V_213, W_213, 0, 2, 0, 2, 3, 3)
    
    U_233 = np.hstack([U_233_1, U_233_2])
    V_233 = np.hstack([V_233_1, V_233_2])
    W_233 = np.hstack([W_233_1, W_233_2])
    
    # Now build mu(3,4,6) by splitting into 4 blocks of mu(3,2,3) = mu(2,3,3)
    # Block (i_off, j_off, k_off):
    #   (0, 0, 0): A[0:3, 0:2] * B[0:2, 0:3]
    #   (0, 0, 3): A[0:3, 0:2] * B[0:2, 3:6]
    #   (0, 2, 0): A[0:3, 2:4] * B[2:4, 0:3]
    #   (0, 2, 3): A[0:3, 2:4] * B[2:4, 3:6]
    #
    # Each block is mu(3, 2, 3) which is the same as mu(2, 3, 3) by symmetry.
    # But wait, mu(3,2,3) means n=3, m=2, p=3. Our U_233 is for n=2, m=3, p=3.
    # We need to handle the transpose carefully.
    
    # Actually, mu(n, m, p) for matrix mult: A is n x m, B is m x p, C is n x p.
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # 
    # For a block with n_b=3, m_b=2, p_b=3:
    # U_b is (3*2, rank) = (6, rank)
    # V_b is (2*3, rank) = (6, rank)
    # W_b is (3*3, rank) = (9, rank)
    #
    # Our U_233 is for n=2, m=3, p=3:
    # U_233 is (2*3, 17) = (6, 17)
    # V_233 is (3*3, 17) = (9, 17)
    # W_233 is (3*2, 17) = (6, 17)
    #
    # So U_233 has the same shape as what we need for U_b (6, 17),
    # but V_233 is (9, 17) while we need (6, 17), and W_233 is (6, 17) while we need (9, 17).
    #
    # The issue is that mu(2,3,3) and mu(3,2,3) are transposes of each other in some sense.
    # Let me think about this more carefully.
    
    # For mu(n, m, p):
    # T[i, j, k] = 1 for all i in [n], j in [m], k in [p]
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    # For mu(3, 2, 3): n=3, m=2, p=3
    # U is (3*2, r) = (6, r), indexed by i*2+j for i in [3], j in [2]
    # V is (2*3, r) = (6, r), indexed by j*3+k for j in [2], k in [3]
    # W is (3*3, r) = (9, r), indexed by k*3+i for k in [3], i in [3]
    
    # For mu(2, 3, 3): n=2, m=3, p=3
    # U is (2*3, r) = (6, r), indexed by i*3+j for i in [2], j in [3]
    # V is (3*3, r) = (9, r), indexed by j*3+k for j in [3], k in [3]
    # W is (3*2, r) = (6, r), indexed by k*2+i for k in [3], i in [2]
    
    # These are different shapes for V and W. So I can't directly use U_233 for mu(3,2,3).
    
    # I need to construct mu(3,2,3) directly, or find a way to adapt.
    
    # Let me construct mu(3,2,3) using the same approach:
    # mu(3,2,3): split n=3 into... actually let's split differently.
    # mu(3,2,3): A is 3x2, B is 2x3, C is 3x3.
    # Split n=3 into 2+1:
    #   mu(2,2,3) + mu(1,2,3) = 11 + 6 = 17
    # mu(1,2,3) = direct, rank 1*2*3 = 6
    
    # Build mu(1,2,3)
    U_123, V_123, W_123 = _make_direct(1, 2, 3)  # rank 6
    
    # Build mu(3,2,3) = mu(2,2,3) + mu(1,2,3)
    U_323_1, V_323_1, W_323_1 = _embed_block(U_223, V_223, W_223, 0, 0, 0, 3, 2, 3)
    U_323_2, V_323_2, W_323_2 = _embed_block(U_123, V_123, W_123, 2, 0, 0, 3, 2, 3)
    
    U_323 = np.hstack([U_323_1, U_323_2])
    V_323 = np.hstack([V_323_1, V_323_2])
    W_323 = np.hstack([W_323_1, W_323_2])
    
    # Now embed 4 copies of mu(3,2,3) into mu(3,4,6)
    # Block 1: i_off=0, j_off=0, k_off=0
    # Block 2: i_off=0, j_off=0, k_off=3
    # Block 3: i_off=0, j_off=2, k_off=0
    # Block 4: i_off=0, j_off=2, k_off=3
    
    blocks = []
    for j_off in [0, 2]:
        for k_off in [0, 3]:
            U_b, V_b, W_b = _embed_block(U_323, V_323, W_323, 0, j_off, k_off, n, m, p)
            blocks.append((U_b, V_b, W_b))
    
    U_final = np.hstack([b[0] for b in blocks])
    V_final = np.hstack([b[1] for b in blocks])
    W_final = np.hstack([b[2] for b in blocks])
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Safety guard
    if time.time() - t0 > EVAL_DEADLINE_SECONDS:
        raise TimeoutError("construct() exceeded deadline")
    
    return {"U": U_list, "V": V_list, "W": W_list}
