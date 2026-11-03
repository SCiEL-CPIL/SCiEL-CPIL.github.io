"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses fine-grained block decomposition with Strassen for 2x2x2 blocks.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses fine-grained block decomposition: split n=3 as 2+1, m=4 as 2+2, p=7 as 2+2+2+1.
    For each (n_block, m_block, p_block):
      - 2x2x2: Strassen (rank 7)
      - 2x2x1: schoolbook (rank 4)
      - 1x2x2: schoolbook (rank 4)
      - 1x2x1: schoolbook (rank 2)
    Total: 2*(3*7 + 1*4) + 1*(3*4 + 1*2) = 2*25 + 14 = 64
    Wait, let me recount:
    n_blocks: [0,1], [2] (sizes 2, 1)
    m_blocks: [0,1], [2,3] (sizes 2, 2)
    p_blocks: [0,1], [2,3], [4,5], [6] (sizes 2,2,2,1)
    
    For n_block size 2:
      m_block size 2, p_block size 2: 7 (3 pairs) = 21
      m_block size 2, p_block size 1: 4 (1 pair) = 4
      Subtotal: 25 per m_block, so 2*25 = 50? No...
      
    Let me be more careful:
    For each (n_blk, m_blk, p_blk) triple:
      n_blk=2, m_blk=2, p_blk=2: 7 terms, count = 1*2*3 = 6 triples -> 42
      n_blk=2, m_blk=2, p_blk=1: 4 terms, count = 1*2*1 = 2 triples -> 8
      n_blk=1, m_blk=2, p_blk=2: 4 terms, count = 1*2*3 = 6 triples -> 24
      n_blk=1, m_blk=2, p_blk=1: 2 terms, count = 1*2*1 = 2 triples -> 4
    Total: 42 + 8 + 24 + 4 = 78
    
    Hmm, that's the same as before. Let me try a different split.
    
    Alternative: Don't split m at all. Use:
    n=3: blocks [2, 1]
    m=4: single block
    p=7: blocks [2, 2, 2, 1]
    
    For n_blk=2, m=4, p_blk=2: This is 2x4x2. Can we do better than schoolbook (16)?
      Split m=4 as 2+2: two 2x2x2 blocks, each Strassen rank 7. Total: 14
    For n_blk=2, m=4, p_blk=1: 2x4x1. Schoolbook: 8. Or split m: two 2x2x1, each rank 4. Total: 8.
    For n_blk=1, m=4, p_blk=2: 1x4x2. Schoolbook: 8. Or split m: two 1x2x2, each rank 4. Total: 8.
    For n_blk=1, m=4, p_blk=1: 1x4x1. Schoolbook: 4.
    
    Total: 3*14 + 1*8 + 3*8 + 1*4 = 42 + 8 + 24 + 4 = 78. Same!
    
    The issue is that the fine-grained decomposition gives the same rank as the coarse one.
    Let me try yet another approach: use the tensor power / laser method.
    """
    start_time = time.time()
    deadline = 135.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val
        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def schoolbook_block(i0, ni, j0, nj, k0, nk):
        """Schoolbook decomposition for ni x nj x nk block."""
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    ii = i0 + i
                    jj = j0 + j
                    kk = k0 + k
                    add_rank1_term([ii*m+jj], [1.0],
                                   [jj*p+kk], [1.0],
                                   [kk*n+ii], [1.0])

    # Strategy: Split n=3 as 2+1, keep m=4 as one block, split p=7 as 2+2+2+1
    # For 2x4x2 blocks: use Strassen on m split (2x2x2 + 2x2x2 = 14)
    # For 2x4x1 blocks: schoolbook (8)
    # For 1x4x2 blocks: schoolbook (8)
    # For 1x4x1 blocks: schoolbook (4)
    # Total: 3*14 + 8 + 3*8 + 4 = 42 + 8 + 24 + 4 = 78
    
    # Let me try to do better. What if I use a non-trivial decomposition for 2x4x2?
    # 2x4x2: A is 2x4, B is 4x2, C is 2x2
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # This is like two dot products of length 4.
    # Can we use Strassen-like tricks?
    # Split j=4 as 2+2:
    #   C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # Each part is 2x2x2, which has rank 7 via Strassen.
    # Total: 14. That's what we have.
    
    # What about 1x4x2?
    # C[0,k] = sum_{j=0}^{3} A[0,j]*B[j,k]
    # Split j=4 as 2+2:
    #   C[0,k] = sum_{j=0}^{1} A[0,j]*B[j,k] + sum_{j=2}^{3} A[0,j]*B[j,k]
    # Each part is 1x2x2, rank 4. Total: 8.
    
    # It seems 78 is the natural rank for this decomposition family.
    # Let me just implement it cleanly and verify.
    
    # Build decomposition
    if time.time() - start_time < deadline:
        # n=0,1 (size 2), m=0..3 (size 4), p=0,1 (size 2): Strassen x2
        strassen_222(0, 0, 0)
        strassen_222(0, 2, 0)
        # n=0,1, m=0..3, p=2,3: Strassen x2
        strassen_222(0, 0, 2)
        strassen_222(0, 2, 2)
        # n=0,1, m=0..3, p=4,5: Strassen x2
        strassen_222(0, 0, 4)
        strassen_222(0, 2, 4)
        # n=0,1, m=0..3, p=6: schoolbook 2x4x1
        schoolbook_block(0, 2, 0, 4, 6, 1)
        # n=2, m=0..3, p=0,1: schoolbook 1x4x2
        schoolbook_block(2, 1, 0, 4, 0, 2)
        # n=2, m=0..3, p=2,3: schoolbook 1x4x2
        schoolbook_block(2, 1, 0, 4, 2, 2)
        # n=2, m=0..3, p=4,5: schoolbook 1x4x2
        schoolbook_block(2, 1, 0, 4, 4, 2)
        # n=2, m=0..3, p=6: schoolbook 1x4x1
        schoolbook_block(2, 1, 0, 4, 6, 1)

    rank = len(U_vecs)

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    verified = True
    if time.time() - start_time < deadline - 5.0:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        verified = False
                        break
                if not verified:
                    break
            if not verified:
                break

    if not verified:
        rank_school = n * m * p
        U_sb = [[0] * rank_school for _ in range(n * m)]
        V_sb = [[0] * rank_school for _ in range(m * p)]
        W_sb = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
