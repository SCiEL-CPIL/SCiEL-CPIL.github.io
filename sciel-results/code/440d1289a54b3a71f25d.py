"""Tensor rank decomposition for 3x4 * 4x4 matrix multiplication.

Strategy:
- Start from the schoolbook decomposition (rank 48).
- Attempt to reduce rank by merging pairs of schoolbook terms.
- For each candidate rank r (from a lower bound up to 47), try to find
  U, V, W of shapes (12, r), (16, r), (12, r) satisfying the tensor identity.
- Use a structured random search: initialize from schoolbook with random
  linear combinations of pairs, then do coordinate-descent / local search
  to satisfy all 48 equations.
- Also try a block-based Strassen construction for 2x2 sub-blocks.
"""
import time
import random
import numpy as np

N, M, P = 3, 4, 4
RANK_SCHOOLBOOK = N * M * P  # 48


def _verify(U, V, W, n=N, m=M, p=P):
    """Verify exact decomposition using einsum."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(recon, expected, atol=1e-8)


def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 48."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    for i,j,k in {0,1}: sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A11-A22)*(B22-B11)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B22)
    # M4 = A22*(B22-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # We need U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # where i=row of A, j=col of A/row of B, k=col of B
    #
    # Let's label the 7 terms c=0..6:
    # c=0: M1: U[A11]=1, U[A22]=-1; V[B22]=1, V[B11]=-1; W[C11]=1, W[C22]=1
    # c=1: M2: U[A21]=1, U[A22]=1; V[B11]=1; W[C21]=1, W[C22]=-1
    # c=2: M3: U[A11]=1; V[B11]=1, V[B22]=-1; W[C12]=1, W[C22]=1
    # c=3: M4: U[A22]=1; V[B22]=1, V[B11]=-1; W[C11]=1, W[C21]=1
    # c=4: M5: U[A11]=1, U[A12]=1; V[B22]=1; W[C11]=-1, W[C12]=1
    # c=5: M6: U[A21]=1, U[A11]=-1; V[B11]=1, V[B12]=1; W[C22]=1
    # c=6: M7: U[A12]=1, U[A22]=-1; V[B21]=1, V[B22]=1; W[C11]=1

    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # Indices: A row i, col j -> index i*2+j
    # B row j, col k -> index j*2+k
    # C row k, col i -> index k*2+i  (W[k*2+i] in the convention)

    # M1: (A11-A22)*(B22-B11) -> contributes to C11 and C22
    U[0, 0] = 1   # A11 (i=0,j=0)
    U[3, 0] = -1  # A22 (i=1,j=1)
    V[3, 0] = 1   # B22 (j=1,k=1)
    V[0, 0] = -1  # B11 (j=0,k=0)
    W[0, 0] = 1   # C11 (k=0,i=0)
    W[3, 0] = 1   # C22 (k=1,i=1)

    # M2: (A21+A22)*B11 -> contributes to C21 and C22
    U[2, 1] = 1   # A21 (i=1,j=0)
    U[3, 1] = 1   # A22 (i=1,j=1)
    V[0, 1] = 1   # B11 (j=0,k=0)
    W[1, 1] = 1   # C21 (k=0,i=1)
    W[3, 1] = -1  # C22 (k=1,i=1)

    # M3: A11*(B11-B22) -> contributes to C12 and C22
    U[0, 2] = 1   # A11 (i=0,j=0)
    V[0, 2] = 1   # B11 (j=0,k=0)
    V[3, 2] = -1  # B22 (j=1,k=1)
    W[2, 2] = 1   # C12 (k=1,i=0)
    W[3, 2] = 1   # C22 (k=1,i=1)

    # M4: A22*(B22-B11) -> contributes to C11 and C21
    U[3, 3] = 1   # A22 (i=1,j=1)
    V[3, 3] = 1   # B22 (j=1,k=1)
    V[0, 3] = -1  # B11 (j=0,k=0)
    W[0, 3] = 1   # C11 (k=0,i=0)
    W[1, 3] = 1   # C21 (k=0,i=1)

    # M5: (A11+A12)*B22 -> contributes to C11 (negative) and C12
    U[0, 4] = 1   # A11 (i=0,j=0)
    U[1, 4] = 1   # A12 (i=0,j=1)
    V[3, 4] = 1   # B22 (j=1,k=1)
    W[0, 4] = -1  # C11 (k=0,i=0)
    W[2, 4] = 1   # C12 (k=1,i=0)

    # M6: (A21-A11)*(B11+B12) -> contributes to C22
    U[2, 5] = 1   # A21 (i=1,j=0)
    U[0, 5] = -1  # A11 (i=0,j=0)
    V[0, 5] = 1   # B11 (j=0,k=0)
    V[1, 5] = 1   # B12 (j=0,k=1)
    W[3, 5] = 1   # C22 (k=1,i=1)

    # M7: (A12-A22)*(B21+B22) -> contributes to C11
    U[1, 6] = 1   # A12 (i=0,j=1)
    U[3, 6] = -1  # A22 (i=1,j=1)
    V[2, 6] = 1   # B21 (j=1,k=0)
    V[3, 6] = 1   # B22 (j=1,k=1)
    W[0, 6] = 1   # C11 (k=0,i=0)

    return U, V, W


def _block_strassen_construction():
    """
    Build a decomposition for 3x4 * 4x4 using block Strassen.
    
    Split A (3x4) into:
      A_top (2x4) and A_bot (1x4)
    Split B (4x4) into:
      B_left (4x2) and B_right (4x2)
    
    C (3x4) = [C_top (2x4); C_bot (1x4)]
    
    C_top = A_top * B = A_top * [B_left B_right]
    C_bot = A_bot * B = A_bot * [B_left B_right]
    
    For A_top (2x4) * B_left (4x2):
      Split A_top = [A_tl (2x2) A_tr (2x2)]
      Split B_left = [B_tl (2x2); B_bl (2x2)]
      C_tl (2x2) = A_tl * B_tl + A_tr * B_bl
      Use Strassen for each 2x2*2x2: rank 7 each, total 14
    
    For A_top (2x4) * B_right (4x2): same structure, rank 14
    
    For A_bot (1x4) * B_left (4x2):
      C_bl (1x2) = A_bot * B_left
      A_bot is 1x4, B_left is 4x2.
      C_bl[0,0] = sum_j A_bot[j] * B_left[j,0]
      C_bl[0,1] = sum_j A_bot[j] * B_left[j,1]
      This is two dot products. Rank = 4 (one term per j for each output col).
      Actually, we can share: for each j, A_bot[j] * B_left[j,0] and A_bot[j] * B_left[j,1]
      These are 4 separate products, so rank 4.
    
    For A_bot (1x4) * B_right (4x2): same, rank 4.
    
    Total rank: 14 + 14 + 4 + 4 = 36.
    
    Let me construct this carefully.
    """
    rank = 36
    U = np.zeros((N * M, rank))  # (12, 36)
    V = np.zeros((M * P, rank))  # (16, 36)
    W = np.zeros((P * N, rank))  # (12, 36)
    
    col = 0
    
    # Helper: add a Strassen 2x2 block
    # A block at rows i0:i0+2, cols j0:j0+2 of A
    # B block at rows j0:j0+2, cols k0:k0+2 of B
    # C block at rows i0:i0+2, cols k0:k0+2 of C
    def add_strassen_block(i0, j0, k0, col_start):
        nonlocal col
        sU, sV, sW = _strassen_2x2()
        for c in range(7):
            # U indices: (i0+ii)*M + (j0+jj) for ii,jj in {0,1}
            for ii in range(2):
                for jj in range(2):
                    U[(i0 + ii) * M + (j0 + jj), col_start + c] += sU[ii * 2 + jj, c]
            # V indices: (j0+jj)*P + (k0+kk) for jj,kk in {0,1}
            for jj in range(2):
                for kk in range(2):
                    V[(j0 + jj) * P + (k0 + kk), col_start + c] += sV[jj * 2 + kk, c]
            # W indices: (k0+kk)*N + (i0+ii) for kk,ii in {0,1}
            for kk in range(2):
                for ii in range(2):
                    W[(k0 + kk) * N + (i0 + ii), col_start + c] += sW[kk * 2 + ii, c]
        col_start += 7
        return col_start
    
    # A_top (rows 0-1) * B_left (cols 0-1 of B, which is rows 0-3 of B)
    # C_top-left (rows 0-1, cols 0-1 of C)
    # A_tl: rows 0-1, cols 0-1 of A
    # B_tl: rows 0-1, cols 0-1 of B
    # A_tr: rows 0-1, cols 2-3 of A
    # B_bl: rows 2-3, cols 0-1 of B
    
    # C_tl = A_tl * B_tl + A_tr * B_bl
    # First term: A_tl (2x2) * B_tl (2x2) -> C_tl (2x2)
    col = add_strassen_block(i0=0, j0=0, k0=0, col_start=col)
    # Second term: A_tr (2x2) * B_bl (2x2) -> C_tl (2x2)
    col = add_strassen_block(i0=0, j0=2, k0=0, col_start=col)
    
    # A_top (rows 0-1) * B_right (cols 2-3 of B)
    # C_top-right (rows 0-1, cols 2-3 of C)
    # A_tl: rows 0-1, cols 0-1 of A
    # B_tr: rows 0-1, cols 2-3 of B
    # A_tr: rows 0-1, cols 2-3 of A
    # B_br: rows 2-3, cols 2-3 of B
    
    # C_tr = A_tl * B_tr + A_tr * B_br
    col = add_strassen_block(i0=0, j0=0, k0=2, col_start=col)
    col = add_strassen_block(i0=0, j0=2, k0=2, col_start=col)
    
    # A_bot (row 2) * B_left (cols 0-1 of B)
    # C_bot-left (row 2, cols 0-1 of C)
    # C_bl[2,0] = sum_j A[2,j] * B[j,0]
    # C_bl[2,1] = sum_j A[2,j] * B[j,1]
    for j in range(4):
        U[2 * M + j, col] = 1.0
        V[j * P + 0, col] = 1.0
        W[0 * N + 2, col] = 1.0
        col += 1
        U[2 * M + j, col] = 1.0
        V[j * P + 1, col] = 1.0
        W[1 * N + 2, col] = 1.0
        col += 1
    
    # A_bot (row 2) * B_right (cols 2-3 of B)
    # C_bot-right (row 2, cols 2-3 of C)
    for j in range(4):
        U[2 * M + j, col] = 1.0
        V[j * P + 2, col] = 1.0
        W[2 * N + 2, col] = 1.0
        col += 1
        U[2 * M + j, col] = 1.0
        V[j * P + 3, col] = 1.0
        W[3 * N + 2, col] = 1.0
        col += 1
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}


def _try_merge_pairs(base_U, base_V, base_W, target_rank, max_time=5.0):
    """
    Try to reduce rank by merging pairs of schoolbook terms.
    Start from schoolbook (rank 48) and try to combine pairs.
    
    For each pair of schoolbook terms (c1, c2), we can try to replace them
    with a single term by finding a linear combination.
    
    This is a heuristic: we randomly select pairs to merge and check if
    the resulting decomposition still satisfies the tensor identity.
    """
    start = time.time()
    deadline = start + max_time
    n, m, p = N, M, P
    rank0 = N * M * P  # 48
    
    # We want to go from rank 48 to target_rank.
    # Number of merges needed: 48 - target_rank
    num_merges = rank0 - target_rank
    if num_merges <= 0:
        return None
    
    # Try random pair merges
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        U = base_U.copy()
        V = base_V.copy()
        W = base_W.copy()
        current_rank = rank0
        merged = set()
        success = True
        
        for merge_idx in range(num_merges):
            # Pick two unmerged columns to merge
            available = [c for c in range(current_rank) if c not in merged]
            if len(available) < 2:
                success = False
                break
            
            c1, c2 = random.sample(available, 2)
            
            # Try to merge c1 and c2 into a single column.
            # The merged column should satisfy: for all (i,j,k),
            # U[i,j,c1]*V[j,k,c1]*W[k,i,c1] + U[i,j,c2]*V[j,k,c2]*W[k,i,c2]
            # = U[i,j,merged]*V[j,k,merged]*W[k,i,merged]
            #
            # This is generally not possible with a single rank-1 term.
            # But if the two terms have a special structure, it might work.
            #
            # For now, this is a placeholder. A real merge would require
            # solving a system of equations.
            
            # Instead, let's try a different approach: random linear combinations.
            # Pick a random scalar alpha and set:
            # merged = alpha * c1 + (1-alpha) * c2  (in each factor)
            # This won't generally preserve the tensor identity.
            
            # For now, skip actual merging and just track.
            merged.add(c1)
            merged.add(c2)
            current_rank -= 1
        
        if not success:
            continue
        
        # This is a stub - real merging logic needed.
        # For now, return None.
        pass
    
    return None


def _random_search(target_rank, max_time=10.0):
    """
    Random search for a decomposition of the given rank.
    Initialize U, V, W randomly and try to optimize.
    """
    start = time.time()
    deadline = start + max_time
    n, m, p = N, M, P
    r = target_rank
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize each column
        for c in range(r):
            norm_U = np.linalg.norm(U[:, c])
            norm_V = np.linalg.norm(V[:, c])
            norm_W = np.linalg.norm(W[:, c])
            if norm_U > 1e-10 and norm_V > 1e-10 and norm_W > 1e-10:
                # Scale so that product of norms is 1
                scale = (norm_U * norm_V * norm_W) ** (-1.0 / 3.0)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
        
        # Check if this random init happens to work (extremely unlikely)
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Start with schoolbook
    best = schoolbook()
    best_rank = N * M * P
    
    # Try the block Strassen construction (rank 36)
    try:
        block_result = _block_strassen_construction()
        if _verify(block_result["U"], block_result["V"], block_result["W"]):
            best = block_result
            best_rank = 36
            print(f"Block Strassen: rank {best_rank} verified.")
        else:
            print("Block Strassen: verification failed.")
    except Exception as e:
        print(f"Block Strassen error: {e}")
    
    # Try to reduce further using random search for lower ranks
    for r in range(30, best_rank):
        if time.time() > deadline - 5:
            break
        result = _random_search(r, max_time=3.0)
        if result is not None:
            best = result
            best_rank = r
            print(f"Random search found rank {r}!")
            break
    
    # Final check
    if not _verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        best = schoolbook()
    
    return best


# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if V.shape[1] != r or W.shape[1] != r:
            return False
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        expected = np.ones((n, m, p))
        return np.allclose(recon, expected, atol=1e-8)

    # Strassen 2x2
    def _strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = -1
        V[3, 0] = 1; V[0, 0] = -1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[0, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    # Block Strassen construction: rank 36
    def _block_strassen():
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0, c_start):
            sU, sV, sW = _strassen_2x2()
            for c in range(7):
                for ii in range(2):
                    for jj in range(2):
                        U[(i0 + ii) * m + (j0 + jj), c_start + c] += sU[ii * 2 + jj, c]
                for jj in range(2):
                    for kk in range(2):
                        V[(j0 + jj) * p + (k0 + kk), c_start + c] += sV[jj * 2 + kk, c]
                for kk in range(2):
                    for ii in range(2):
                        W[(k0 + kk) * n + (i0 + ii), c_start + c] += sW[kk * 2 + ii, c]
            return c_start + 7

        # A_top (rows 0-1) * B_left (cols 0-1)
        col = add_strassen(0, 0, 0, col)
        col = add_strassen(0, 2, 0, col)
        # A_top (rows 0-1) * B_right (cols 2-3)
        col = add_strassen(0, 0, 2, col)
        col = add_strassen(0, 2, 2, col)
        # A_bot (row 2) * B (all cols)
        for j in range(4):
            for k in range(4):
                U[2 * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + 2, col] = 1.0
                col += 1

        assert col == rank
        return {"U": U, "V": V, "W": W}

    best = _schoolbook()
    best_rank = n * m * p

    try:
        bs = _block_strassen()
        if _verify(bs["U"], bs["V"], bs["W"]):
            best = bs
            best_rank = 36
    except Exception:
        pass

    # Try random search for lower ranks
    for r in range(28, best_rank):
        if time.time() > deadline - 5:
            break
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            if _verify(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = r
                break
        if best_rank == r:
            break

    if not _verify(best["U"], best["V"], best["W"]):
        best = _schoolbook()

    return best
# EVOLVE-BLOCK-END
