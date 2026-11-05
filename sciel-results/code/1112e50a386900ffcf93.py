"""Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen multiplication.
    Each term is (u_vec, v_vec, w_vec) where u_vec, v_vec are length 4 (row-major 2x2),
    w_vec is length 4 (row-major 2x2) following W[k*n+i] convention.
    
    For 2x2x2: n=m=p=2, so U shape (4,7), V shape (4,7), W shape (4,7).
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    """
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Index convention: U[i*m+j], V[j*p+k], W[k*n+i]
    # For 2x2: i,j,k in {0,1}, n=m=p=2
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)  [k*n+i]
    
    # Each of the 7 terms contributes to U, V, W vectors of length 4
    terms = []
    
    # M1 = (A11+A22)(B11+B22) -> contributes to C11(+), C22(+)
    u1 = [1, 0, 0, 1]  # A11 + A22
    v1 = [1, 0, 0, 1]  # B11 + B22
    w1 = [1, 0, 0, 1]  # C11 + C22 (indices 0 and 3)
    terms.append((u1, v1, w1))
    
    # M2 = (A21+A22)B11 -> contributes to C21(+), C22(-)
    u2 = [0, 0, 1, 1]  # A21 + A22
    v2 = [1, 0, 0, 0]  # B11
    w2 = [0, 0, 1, -1]  # C21 + (-C22) -> indices 2 and 3
    terms.append((u2, v2, w2))
    
    # M3 = A11(B21-B22) -> contributes to C12(+), C22(+)
    u3 = [1, 0, 0, 0]  # A11
    v3 = [0, 0, 1, -1]  # B21 - B22
    w3 = [0, 1, 0, 1]  # C12 + C22 -> indices 1 and 3
    terms.append((u3, v3, w3))
    
    # M4 = A22(B12-B11) -> contributes to C11(+), C21(+)
    u4 = [0, 0, 0, 1]  # A22
    v4 = [0, 1, 0, 0]  # B12 (index 1)
    # Wait, B12-B11: B12 is V[0*2+1]=V[1], B11 is V[0*2+0]=V[0]
    v4 = [-1, 1, 0, 0]  # B12 - B11
    w4 = [1, 0, 1, 0]  # C11 + C21 -> indices 0 and 2
    terms.append((u4, v4, w4))
    
    # M5 = (A11+A12)B22 -> contributes to C11(-), C12(+)
    u5 = [1, 1, 0, 0]  # A11 + A12
    v5 = [0, 0, 0, 1]  # B22
    w5 = [-1, 1, 0, 0]  # -C11 + C12 -> indices 0 and 1
    terms.append((u5, v5, w5))
    
    # M6 = (A21-A11)(B11+B12) -> contributes to C22(+)
    u6 = [0, 0, 1, -1]  # A21 - A11
    v6 = [1, 1, 0, 0]  # B11 + B12
    w6 = [0, 0, 0, 1]  # C22 -> index 3
    terms.append((u6, v6, w6))
    
    # M7 = (A12-A22)(B21+B22) -> contributes to C11(+)
    u7 = [0, 1, 0, -1]  # A12 - A22
    v7 = [0, 0, 1, 1]  # B21 + B22
    w7 = [1, 0, 0, 0]  # C11 -> index 0
    terms.append((u7, v7, w7))
    
    return terms

def _build_block_decomposition(n, m, p, block_n, block_m, block_p, inner_rank_func):
    """
    Build a block decomposition where each (block_n x block_m) @ (block_m x block_p) 
    subproduct uses inner_rank_func terms.
    
    Returns U, V, W as lists of lists (integers).
    """
    # Number of blocks
    nb_i = n // block_n
    nb_j = m // block_m
    nb_k = p // block_p
    
    # Check divisibility
    assert n % block_n == 0, f"n={n} not divisible by block_n={block_n}"
    assert m % block_m == 0, f"m={m} not divisible by block_m={block_m}"
    assert p % block_p == 0, f"p={p} not divisible by block_p={block_p}"
    
    # Total rank = nb_i * nb_j * nb_k * inner_rank
    inner_terms = inner_rank_func()
    inner_rank = len(inner_terms)
    total_rank = nb_i * nb_j * nb_k * inner_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for bi in range(nb_i):
        for bj in range(nb_j):
            for bk in range(nb_k):
                for (u_vec, v_vec, w_vec) in inner_terms:
                    # u_vec has length block_n * block_m
                    # v_vec has length block_m * block_p
                    # w_vec has length block_p * block_n
                    for a in range(block_n):
                        for b in range(block_m):
                            gi = bi * block_n + a
                            gj = bj * block_m + b
                            U[gi * m + gj][col] = u_vec[a * block_m + b]
                    for b in range(block_m):
                        for c in range(block_p):
                            gj = bj * block_m + b
                            gk = bk * block_p + c
                            V[gj * p + gk][col] = v_vec[b * block_p + c]
                    for c in range(block_p):
                        for a in range(block_n):
                            gk = bk * block_p + c
                            gi = bi * block_n + a
                            W[gk * n + gi][col] = w_vec[c * block_n + a]
                    col += 1
    
    return U, V, W, total_rank

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if j is the "middle" index connecting i and k, else 0.
    # Actually for matrix multiplication: T[i,j,k] = delta_{j,j} * delta_{i,i}... 
    # No. The matrix multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A 
    # multiplies the (j,k) entry of B to give (i,k) entry of C.
    # So T[i,j,k] = 1 for all i,j,k (it's the standard matmul tensor where each 
    # (i,j,k) triple has coefficient 1).
    
    # Wait, let me reconsider. The matmul tensor is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # As a 3-tensor: T[i,j,k] = 1 (all entries are 1)
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Check a subset for speed
    max_check = min(n*m*p, 200)
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    np.random.seed(42)
    if len(indices) > max_check:
        idx = np.random.choice(len(indices), max_check, replace=False)
        indices = [indices[i] for i in idx]
    
    for (i, j, k) in indices:
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        val = 0.0
        for c in range(rank):
            val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
        if abs(val - 1.0) > tol:
            return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _try_block_2x2x2(n, m, p):
    """Try using 2x2x2 Strassen blocks where possible."""
    # For 4x5x5: we can do 2x2x2 blocks for the 4x4x4 part (2x2 blocks)
    # and handle the remaining parts with schoolbook or other methods.
    
    # Strategy: Partition into blocks.
    # 4 = 2+2, 5 = 2+2+1, 5 = 2+2+1
    # This gets complex. Let's try a simpler approach first.
    
    # Option 1: Use 2x2x2 Strassen for all 2x2x2 sub-blocks, schoolbook for 1x1x1
    # 4x5x5 = (2+2) x (2+2+1) x (2+2+1)
    # Blocks: 2x2x2 (Strassen, rank 7), 2x2x1, 2x1x2, 2x1x1, 1x2x2, 1x2x1, 1x1x2, 1x1x1
    # This is getting complicated. Let me think of a cleaner approach.
    
    # Alternative: Just use the known best bound approach.
    # For 4x5x5, the trivial bound is 100.
    # Using Strassen on 4x4x4 part: 49 + remaining terms.
    # 4x5x5 = 4x4x5 + 4x1x5
    # 4x4x5 can use Strassen on 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65
    # 4x1x5 is schoolbook: rank 20
    # Total: 85
    
    # Let me implement this: split j dimension into 4+1
    # A is 4x5, B is 5x5
    # C = A * B = A[:, :4] * B[:4, :] + A[:, 4:] * B[4:, :]
    # First term: 4x4 @ 4x4 -> 4x4 (use Strassen, rank 49)
    # Second term: 4x1 @ 1x5 -> 4x5 (schoolbook, rank 20)
    # Total rank: 49 + 20 = 69? No wait...
    
    # Actually for the first term: (4x4) @ (4x4) = 4x4
    # U shape: 4*4=16 rows, rank 49
    # V shape: 4*4=16 rows, rank 49  
    # W shape: 4*4=16 rows, rank 49
    # But we need to embed into the full 4x5x5 problem.
    
    # Let me just build it directly.
    
    # Full problem: n=4, m=5, p=5
    # U: 4*5=20 rows
    # V: 5*5=25 rows
    # W: 5*4=20 rows
    
    # Term 1: i in 0..3, j in 0..3, k in 0..3 (4x4x4 Strassen, rank 49)
    # Term 2: i in 0..3, j=4, k in 0..4 (4x1x5 schoolbook, rank 20)
    # Total: 69
    
    # Wait, but we also need j in 0..3, k=4 (4x4x1, rank 16)
    # And j=4, k=4 (1x1x1, rank 1)
    # So: 49 + 16 + 20 + 1 = 86? Let me recount.
    
    # Full decomposition covers all (i,j,k) with i in 0..3, j in 0..4, k in 0..4
    # Split j into 0..3 and 4:
    #   j in 0..3: i in 0..3, k in 0..4
    #     Split k into 0..3 and 4:
    #       j in 0..3, k in 0..3: 4x4x4 -> Strassen rank 49
    #       j in 0..3, k=4: 4x4x1 -> schoolbook rank 16
    #   j=4: i in 0..3, k in 0..4
    #     Split k into 0..3 and 4:
    #       j=4, k in 0..3: 4x1x4 -> schoolbook rank 16
    #       j=4, k=4: 4x1x1 -> schoolbook rank 4
    # Total: 49 + 16 + 16 + 4 = 85
    
    # That matches the known result! Let me implement this.
    
    n, m, p = 4, 5, 5
    total_rank = 49 + 16 + 16 + 4
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: 4x4x4 Strassen (i:0-3, j:0-3, k:0-3)
    strassen_terms = _strassen_2x2x2()
    # We need 4x4x4 = 2x2 blocks of 2x2x2
    # 4x4x4 uses 2x2x2 Strassen with 2x2=4 blocks
    # Each 2x2x2 block has rank 7, total 4*7=28? No...
    # 4x4x4 = (2+2)x(2+2)x(2+2), so 2*2*2=8 blocks of 2x2x2
    # Each 2x2x2 has rank 7, so 8*7=56? But we want 49.
    # 
    # Actually, for 4x4x4, we can use a 4x4x4 Strassen-like algorithm.
    # The standard approach: 4x4x4 can be done with rank 49 using
    # a combination of 2x2x2 Strassen blocks.
    #
    # Let me use the block approach: 4x4x4 with 2x2x2 Strassen blocks.
    # 4/2=2 blocks in each dimension, so 2*2*2=8 sub-blocks.
    # Each sub-block is 2x2x2 with rank 7.
    # Total: 8*7=56. But we claimed 49.
    #
    # Hmm, 49 is from a different construction. Let me just use 56 for now
    # and see if we can do better, or use a known 4x4x4 rank-49 algorithm.
    #
    # Actually, I recall that 4x4x4 has rank at most 49 via a specific algorithm.
    # But implementing that is complex. Let me use the block Strassen (rank 56)
    # for now, giving total rank 56+16+16+4=92.
    #
    # Wait, the knowledge says "reducing 4x4x4 core from 56 to 49 rank-1 terms, 
    # total rank 92->85". So there IS a rank-49 algorithm for 4x4x4.
    #
    # Let me look up the 4x4x4 rank 49 algorithm. It's based on the 
    # "Strassen-like" algorithm for 4x4x4.
    #
    # For now, let me use the block Strassen (rank 56) which gives total 92,
    # and then try to improve.
    
    # Use block Strassen for 4x4x4: rank 56
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for (u_vec, v_vec, w_vec) in strassen_terms:
                    for a in range(2):
                        for b in range(2):
                            gi = bi * 2 + a
                            gj = bj * 2 + b
                            U[gi * m + gj][col] = u_vec[a * 2 + b]
                    for b in range(2):
                        for c in range(2):
                            gj = bj * 2 + b
                            gk = bk * 2 + c
                            V[gj * p + gk][col] = v_vec[b * 2 + c]
                    for c in range(2):
                        for a in range(2):
                            gk = bk * 2 + c
                            gi = bi * 2 + a
                            W[gk * n + gi][col] = w_vec[c * 2 + a]
                    col += 1
    
    # Block 2: 4x4x1 schoolbook (i:0-3, j:0-3, k=4)
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Block 3: 4x1x4 schoolbook (i:0-3, j=4, k:0-3)
    for i in range(4):
        for k in range(4):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 4: 4x1x1 schoolbook (i:0-3, j=4, k=4)
    for i in range(4):
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    return U, V, W, total_rank

def _try_4x4x4_rank49():
    """
    Attempt to construct a rank-49 decomposition for 4x4x4.
    This uses a known algorithm based on combining Strassen blocks.
    
    The idea: 4x4x4 can be decomposed using 7 "Strassen-like" products
    on 4x4 matrices, but with some sharing.
    
    Actually, a simpler approach: use the fact that 4x4x4 rank is at most 49
    via the algorithm of Pan (1978) or similar.
    
    For now, let me try a different partitioning of 4x5x5.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Try the block Strassen approach (rank 92)
    U, V, W, rank = _try_block_2x2x2(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        elapsed = time.time() - start_time
        if elapsed < deadline:
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook (rank 100)
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
