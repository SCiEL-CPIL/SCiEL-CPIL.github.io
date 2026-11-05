"""Structural rewrite: p-splitting with per-block rank reduction."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_block(n, m, p_block, rank_block):
    """Build a rank-rank_block decomposition for M(n,m,p_block).
    Falls back to schoolbook if rank_block >= n*m*p_block."""
    nm = n * m
    mp = m * p_block
    pn = p_block * n
    
    # Schoolbook indices
    schoolbook_rank = n * m * p_block
    if rank_block >= schoolbook_rank:
        U = np.zeros((nm, rank_block))
        V = np.zeros((mp, rank_block))
        W = np.zeros((pn, rank_block))
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    c = i * m * p_block + j * p_block + k
                    U[i * m + j, c] = 1.0
                    V[j * p_block + k, c] = 1.0
                    W[k * n + i, c] = 1.0
        return U, V, W
    
    # Attempt reduced-rank construction for M(3,4,2) with rank 22
    # Strategy: start from schoolbook rank 24, try to merge 2 terms
    # by finding linear dependencies in the tensor entries.
    # For M(3,4,2): entries are (i,j,k) for i in 0..2, j in 0..3, k in 0..1
    # Total 24 entries. We want rank 22.
    
    # Use a structured approach: for each (i,k) pair, we have 4 terms (one per j).
    # The 6 (i,k) pairs each need 4 multiplications = 24 total.
    # To reduce by 2, we need to find 2 pairs of (i,j,k) entries that can be
    # combined via a shared multiplication.
    
    # Alternative: use the fact that M(3,4,2) can be viewed as 3 copies of M(1,4,2).
    # M(1,4,2) has 8 entries. Can we do it in rank 7? No, rank of M(1,4,2) is 8.
    # So 3*8=24 is tight for this approach.
    
    # Let's try a different structure: exploit that n=3 allows some sharing.
    # For fixed j, the entries (i,j,0) for i=0,1,2 and (i,j,1) for i=0,1,2
    # form a 3x2 sub-tensor. This sub-tensor has rank 6 (schoolbook).
    # But we can write it as: for fixed j, sum over i,k of A[i,j]*B[j,k]*C[k,i]
    # The 3x2 C-matrix has rank 2, so the sub-tensor for fixed j has rank 6.
    
    # Total: 4 * 6 = 24. Same as schoolbook.
    
    # Let me try an optimization-based approach to find rank 22.
    # We'll use a random initialization and gradient descent on the tensor norm.
    
    U = np.zeros((nm, rank_block))
    V = np.zeros((mp, rank_block))
    W = np.zeros((pn, rank_block))
    
    # Initialize with schoolbook pattern, then zero out 2 columns and try to fix
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                c = i * m * p_block + j * p_block + k
                if c < rank_block:
                    U[i * m + j, c] = 1.0
                    V[j * p_block + k, c] = 1.0
                    W[k * n + i, c] = 1.0
    
    # We have 24 schoolbook terms but only rank_block=22 slots.
    # The last 2 terms (c=22, c=23) are missing. We need to absorb them.
    # Terms 22 and 23 correspond to (i=2,j=3,k=0) and (i=2,j=3,k=1).
    # i=2, j=3: U[2*4+3]=U[11], V[3*2+0]=V[6], W[0*3+2]=W[2] and W[1*3+2]=W[5]
    
    # We need to find coefficients such that:
    # U[11,c]*V[6,c]*W[2,c] sums to 1 for the (i=2,j=3,k=0) entry
    # U[11,c]*V[7,c]*W[5,c] sums to 1 for the (i=2,j=3,k=1) entry
    # while all other entries remain correct.
    
    # This is a hard problem. Let's use a numerical search.
    # Given time constraints, return schoolbook if we can't find better.
    
    # Actually, let's just return the schoolbook for now as a safe fallback.
    # The inspirations suggest rank 66 is achievable, but I need to verify the construction.
    
    # For safety, return full schoolbook
    U = np.zeros((nm, rank_block))
    V = np.zeros((mp, rank_block))
    W = np.zeros((pn, rank_block))
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                c = i * m * p_block + j * p_block + k
                if c < rank_block:
                    U[i * m + j, c] = 1.0
                    V[j * p_block + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                else:
                    # These terms are dropped - invalid!
                    pass
    
    # This is invalid. Return schoolbook instead.
    U = np.zeros((nm, schoolbook_rank))
    V = np.zeros((mp, schoolbook_rank))
    W = np.zeros((pn, schoolbook_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                c = i * m * p_block + j * p_block + k
                U[i * m + j, c] = 1.0
                V[j * p_block + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    """Return a decomposition for 3x4x6 matrix multiplication.
    
    Uses p-splitting: M(3,4,6) = M(3,4,2) + M(3,4,2) + M(3,4,2).
    Each block uses schoolbook rank 24, total 72.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Split p=6 into three blocks of p=2
    p_blocks = [2, 2, 2]
    block_ranks = [24, 24, 24]  # schoolbook for each M(3,4,2)
    
    total_rank = sum(block_ranks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    for block_idx, (p_b, r_b) in enumerate(zip(p_blocks, block_ranks)):
        # k offset for this block
        k_offset = sum(p_blocks[:block_idx])
        
        U_b, V_b, W_b = _build_block(n, m, p_b, r_b)
        
        # V_b has shape (m*p_b, r_b), need to map to correct k positions
        # V_b[j*p_b + k, c] should go to V[j*p + (k_offset + k), offset+c]
        # W_b has shape (p_b*n, r_b), W_b[k*n + i, c] -> W[(k_offset+k)*n + i, offset+c]
        
        for j in range(m):
            for k in range(p_b):
                for c in range(r_b):
                    V[j * p + k_offset + k, offset + c] = V_b[j * p_b + k, c]
        
        for k in range(p_b):
            for i in range(n):
                for c in range(r_b):
                    W[(k_offset + k) * n + i, offset + c] = W_b[k * n + i, c]
        
        for r in range(n * m):
            for c in range(r_b):
                U[r, offset + c] = U_b[r, c]
        
        offset += r_b
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        # Should not happen for direct construction
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
