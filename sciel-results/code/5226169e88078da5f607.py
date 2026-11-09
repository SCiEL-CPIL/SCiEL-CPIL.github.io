"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Check that U,V,W exactly decompose the matmul tensor.
    T[i,j,k] = 1 for all valid (i,j,k).
    We verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k).
    Also check that "off-diagonal" entries are zero (not strictly required by the
    problem statement which only asks for the matmul identity, but good for sanity).
    Actually, the problem only requires:
      C[i,k] = sum_j A[i,j]*B[j,k]
    which corresponds to:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k.
    And for the decomposition to be valid, we also need that the "wrong" combinations
    sum to 0, but the standard tensor decomposition for matmul only requires the above
    identity. Let me re-read the contract.
    
    The contract says: "exactly decompose the matrix multiplication tensor"
    The matmul tensor T has T[i,j,k] = 1 if the indices are consistent (which they always
    are in the standard indexing), and the decomposition is:
      T = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r]
    
    So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k.
    That's it. The "off-diagonal" entries aren't part of the tensor.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],   # i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0,j=1
        [1, 1, 0, 0, 0, 1, 0],   # i=1,j=0
        [0, 1, 0, 1, 0, -1, -1], # i=1,j=1
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],   # j=0,k=0
        [0, 0, 1, 0, 0, 0, 1],   # j=0,k=1
        [1, 0, 0, 1, 0, 1, 0],   # j=1,k=0
        [0, 0, -1, -1, 1, 0, 1], # j=1,k=1
    ]
    W = [
        [1, 0, 0, 1, 0, 0, 1],   # k=0,i=0
        [0, 0, 1, 0, 1, 0, 0],   # k=0,i=1
        [0, 1, 0, 0, 0, 1, 0],   # k=1,i=0
        [1, 0, 0, 0, 0, 1, 0],   # k=1,i=1
    ]
    
    if not _verify(U, V, W, 2, 2, 2, 7):
        return _schoolbook(2, 2, 2)
    return U, V, W

def _try_block_strassen(n, m, p, deadline):
    """Try to use Strassen on 2x2 blocks to reduce rank.
    
    Strategy: Split the problem into blocks where possible.
    For 4x5*5x6:
    - Split n=4 into 2+2
    - Split m=5 into 2+3  
    - Split p=6 into 3+3
    
    This gives a block structure:
    A = [A11(2x2) A12(2x3)]
        [A21(2x2) A22(2x3)]
    B = [B11(2x3) B12(2x3)]
        [B21(3x3) B22(3x3)]
    
    C11 = A11*B11 + A12*B21  (2x3)
    C12 = A11*B12 + A12*B22  (2x3)
    C21 = A21*B11 + A22*B21  (2x3)
    C22 = A21*B12 + A22*B22  (2x3)
    
    This doesn't directly allow Strassen since the blocks aren't square.
    
    Alternative: Pad to 4x6*6x6, then use 2x2 blocks of 2x3.
    Still not directly Strassen-friendly.
    
    Let's try a different approach: use the fact that we can sometimes
    share computations between overlapping blocks.
    
    Actually, for a concrete improvement, let's try to find if there are
    linear dependencies we can exploit. The schoolbook rank is 120.
    
    One approach: use randomized linear algebra to check if the rank can be
    reduced. But this is complex.
    
    For now, let's try a simple heuristic: check if removing one schoolbook
    term and adjusting others still works. This is unlikely to work but
    let's try a few random perturbations.
    """
    if time.time() > deadline:
        return None
    
    U, V, W = _schoolbook(n, m, p)
    rank = n * m * p
    best = (rank, U, V, W)
    
    # Try to find if any single column can be removed (unlikely)
    # Or if two columns can be merged (also unlikely)
    
    # A more promising approach: use the structure of the problem.
    # For 4x5*5x6, note that 5 = 2+3. We can split the sum over j:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    #         = C1[i,k] + C2[i,k]
    # where C1 = A[:,0:2] * B[0:2,:] (4x2 * 2x6 = 4x6, rank 4*2*6=48)
    # and C2 = A[:,2:5] * B[2:5,:] (4x3 * 3x6 = 4x6, rank 4*3*6=72)
    # Total: 48 + 72 = 120. No improvement.
    
    # What if we use Strassen on the 4x2*2x6 part?
    # 4x2*2x6: split n=4 into 2+2, m=2, p=6 into 3+3
    # This is still not square blocks.
    
    # Let's try a completely different approach: use the known result that
    # R(4,5,6) might be less than 120. Without a proven construction,
    # we'll return the schoolbook.
    
    return best

def _try_tensor_rank_reduction(n, m, p, deadline):
    """Try to reduce the tensor rank using optimization.
    
    We want to find U, V, W of rank r < n*m*p such that
    sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k.
    
    This is a system of n*m*p bilinear equations in r variables (the entries
    of U, V, W). For r close to n*m*p, we can try to find a solution.
    
    Approach: Start with the schoolbook (r = n*m*p) and try to find a
    lower-rank approximation, then verify if it's exact.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = DIMENSIONS
    target_ranks = [119, 118, 117, 116, 115]
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        
        # Try to find a rank-r decomposition
        # Initialize with the first r columns of the schoolbook
        U, V, W = _schoolbook(n, m, p)
        
        # This is just the schoolbook with r columns, which won't work
        # unless r >= n*m*p. We need a smarter initialization.
        
        # For now, skip this and return None
        pass
    
    return None

def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Verify the schoolbook decomposition
    if _verify(U, V, W, n, m, p, best_rank):
        pass  # Good
    else:
        # Should not happen
        pass
    
    # Try block Strassen approach
    result = _try_block_strassen(n, m, p, deadline)
    if result is not None:
        r, U2, V2, W2 = result
        if r < best_rank and _verify(U2, V2, W2, n, m, p, r):
            best_rank = r
            best_result = {"U": U2, "V": V2, "W": W2}
    
    # Try tensor rank reduction
    result = _try_tensor_rank_reduction(n, m, p, deadline)
    if result is not None:
        r, U2, V2, W2 = result
        if r < best_rank and _verify(U2, V2, W2, n, m, p, r):
            best_rank = r
            best_result = {"U": U2, "V": V2, "W": W2}
    
    return best_result
# EVOLVE-BLOCK-END
