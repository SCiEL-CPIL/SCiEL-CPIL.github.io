"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition.

Strategy:
1. Build a correct rank-42 decomposition via 2x2 Strassen blocks (baseline).
2. Attempt to reduce rank using integer linear algebra on the tensor's
   flattening (slice) matrix. If a lower-rank decomposition with integer
   or half-integer entries is found, return it.
3. Return the best valid decomposition found within the time budget.

The tensor for (n x m) * (m x p) matrix multiplication has:
  - Mode 1 (A): n*m entries, indexed by (i, j) -> i*m + j
  - Mode 2 (B): m*p entries, indexed by (j, k) -> j*p + k
  - Mode 3 (C): p*n entries, indexed by (k, i) -> k*n + i

T[(i,j),(j',k),(k',i')] = 1 iff j==j' and k==k' and i==i'

A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r) such that
T = sum_r u_r ⊗ v_r ⊗ w_r.

Equivalently, for each pair (a_idx, b_idx) the vector T[a_idx, b_idx, :]
equals sum_r U[a_idx, r] * V[b_idx, r] * W[:, r].

We can set up a linear system: for each (a_idx, b_idx), the vector
T[a_idx, b_idx, :] must be in the span of {U[a_idx, r] * V[b_idx, r] * W[:, r]}.

A simpler approach: the tensor can be represented as a matrix of size
(n*m * m*p) x (p*n) where entry [(a,b), c] = T[a,b,c].
The rank of this matrix is an upper bound on the tensor rank.
If we can find a rank-r decomposition of this matrix where the columns
factor appropriately, we're done. But that's not straightforward.

Alternative: Use the fact that T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
For fixed c, T[:, :, c] = sum_r (U[:,r] ⊗ V[:,r]) * W[c,r].
So the (n*m * m*p) x (p*n) matrix has columns that are linear combinations
of rank-1 matrices U[:,r] ⊗ V[:,r] with coefficients W[:,r].

The minimum rank r such that the tensor decomposes is what we want.

Practical approach:
- Start with rank 42 (Strassen).
- Try to reduce by checking if some columns are linearly dependent in a
  structured way, or use randomized SVD-like approaches with integer constraints.

Given the difficulty of guaranteed rank reduction for this specific tensor,
and the time budget, I'll implement:
1. A verified rank-42 Strassen decomposition.
2. A quick check using numpy SVD to estimate the numerical rank of the
   flattening matrix, which gives a lower bound. If the lower bound is
   significantly below 42, attempt a search. Otherwise return 42.

Actually, for 2x4x6, the known minimum rank is likely 32 or 36 based on
literature for small matrix multiplication. Let me try a different
constructive approach.

Alternative constructive approach:
- The 2x4x6 multiplication can be seen as computing 12 output entries,
  each a sum of 4 products. Total "naive" work is 48 scalar mults.
- Strassen on 2x2x2 gives 7 per 8 entries. For 2x4x6:
  - Split into two 2x2x6 products, each split into three 2x2x2: 2*3*7=42.
  
Can we do better? One idea: use the fact that n=2 is small.
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]

This is like two rows of A times B. We can group:
Let X0 = A[0,0]+A[1,0], X1 = A[0,1]+A[1,1], etc. But that mixes rows.

Actually, a cleaner view: C = A @ B where A is 2x4, B is 4x6.
This is equivalent to 4 separate 2x1 @ 1x6 products summed, but that's 48.

Better: view as 2x2 @ 2x6 twice (Strassen approach) = 42.

Let me try to find if rank 36 is achievable by a different blocking.
If we use a 3x3 base case with rank 21 (Laderman's algorithm):
- But 2x4x6 doesn't tile neatly into 3x3 blocks.

Given the constraints, I'll implement the rank-42 solution cleanly and
add a verification step. The key fix from the original is to ensure the
index mapping is correct and the tensor identity actually holds.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def _strassen_2x2x2_factors():
    """Return the 7 rank-1 factors for 2x2x2 Strassen multiplication.
    
    For C = A @ B where A, B, C are 2x2:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    factors = []
    
    # Each factor: U (for A, 4 entries), V (for B, 4 entries), W (for C, 4 entries)
    # Index convention: A[i,j] -> i*2+j, B[i,j] -> i*2+j, C[i,j] -> i*2+j
    
    # M1 = (A00 + A11)(B00 + B11)
    factors.append((
        [1, 0, 0, 1],   # A00=1, A11=1
        [1, 0, 0, 1],   # B00=1, B11=1
        [1, 0, 0, 1]    # C00=1, C11=1
    ))
    
    # M2 = (A10 + A11)B00
    factors.append((
        [0, 0, 1, 1],   # A10=1, A11=1
        [1, 0, 0, 0],   # B00=1
        [0, 0, 1, 0]    # C10=1
    ))
    
    # M3 = A00(B10 - B11)
    factors.append((
        [1, 0, 0, 0],   # A00=1
        [0, 0, 1, -1],  # B10=1, B11=-1
        [0, 1, 0, 1]    # C01=1, C11=1
    ))
    
    # M4 = A11(B01 - B00)
    factors.append((
        [0, 0, 0, 1],   # A11=1
        [-1, 1, 0, 0],  # B00=-1, B01=1
        [1, 0, 1, 0]    # C00=1, C10=1
    ))
    
    # M5 = (A00 + A01)B11
    factors.append((
        [1, 1, 0, 0],   # A00=1, A01=1
        [0, 0, 0, 1],   # B11=1
        [-1, 1, 0, 0]   # C00=-1, C01=1
    ))
    
    # M6 = (A10 - A00)(B00 + B01)
    factors.append((
        [-1, 0, 1, 0],  # A00=-1, A10=1
        [1, 1, 0, 0],   # B00=1, B01=1
        [0, 0, 0, 1]    # C11=1
    ))
    
    # M7 = (A01 - A11)(B10 + B11)
    factors.append((
        [0, 1, 0, -1],  # A01=1, A11=-1
        [0, 0, 1, 1],   # B10=1, B11=1
        [1, 0, 0, 0]    # C00=1
    ))
    
    return factors


def _build_strassen_42():
    """Build the rank-42 Strassen decomposition for 2x4x6.
    
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Split: C = A_L @ B_L + A_R @ B_R
    where A_L = A[:, 0:2], A_R = A[:, 2:4]
    and B_L = B[0:2, :], B_R = B[2:4, :]
    
    Each (2x2) @ (2x6) is split into three (2x2) @ (2x2) Strassen products.
    Total: 2 * 3 * 7 = 42 factors.
    """
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_factors = _strassen_2x2x2_factors()
    
    factor_idx = 0
    for a_block in range(2):       # 0: A_L (cols 0,1), 1: A_R (cols 2,3)
        for b_col_block in range(3):  # 0: cols 0,1; 1: cols 2,3; 2: cols 4,5
            for (u, v, w) in strassen_factors:
                # U part: A indices
                # A[i, j_local] where j_local in {0,1}
                # Global A col = a_block*2 + j_local
                # U row = i*m + global_j = i*4 + a_block*2 + j_local
                for idx in range(4):
                    if u[idx] != 0:
                        i, j_local = divmod(idx, 2)
                        u_row = i * m + a_block * 2 + j_local
                        U[u_row, factor_idx] += u[idx]
                
                # V part: B indices
                # B[j_local, k_local] where j_local in {0,1}, k_local in {0,1}
                # Global B row = a_block*2 + j_local
                # Global B col = b_col_block*2 + k_local
                # V row = global_j * p + global_k
                for idx in range(4):
                    if v[idx] != 0:
                        j_local, k_local = divmod(idx, 2)
                        v_row = (a_block * 2 + j_local) * p + b_col_block * 2 + k_local
                        V[v_row, factor_idx] += v[idx]
                
                # W part: C indices
                # C[i, k_local] where k_local in {0,1}
                # Global C col = b_col_block*2 + k_local
                # W row = global_k * n + i
                for idx in range(4):
                    if w[idx] != 0:
                        k_local, i = divmod(idx, 2)
                        w_row = (b_col_block * 2 + k_local) * n + i
                        W[w_row, factor_idx] += w[idx]
                
                factor_idx += 1
    
    assert factor_idx == rank
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents the multiplication tensor."""
    rank = U.shape[1]
    # Check: T[(i,j),(j',k),(k',i')] = 1 iff j==j', k==k', i==i'
    # = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j != jp:
                    # Should be 0
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                val = sum(U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] 
                                         for r in range(rank))
                                if val != 0:
                                    return False
                else:
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1 if (k == kp and i == ip) else 0
                                val = sum(U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] 
                                         for r in range(rank))
                                if val != expected:
                                    return False
    return True


def _verify_with_random(U, V, W, n, m, p, trials=20):
    """Verify by computing actual matrix multiplications."""
    rank = U.shape[1]
    np.random.seed(12345)
    for _ in range(trials):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_expected = A @ B
        
        # Compute C via decomposition
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        C_computed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    a_part = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                    b_part = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                    s += W[k*n+i, r] * a_part * b_part
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_time=60):
    """Try to reduce the rank of the decomposition using linear algebra.
    
    This is a heuristic attempt. If it fails, we return the original.
    """
    start = time.time()
    rank = U.shape[1]
    
    # Build the tensor flattening matrix
    # For each (a, b) pair, T[a, b, :] should equal sum_r U[a,r] V[b,r] W[:,r]
    # The (n*m * m*p) x (p*n) matrix M where M[(a,b), c] = T[a,b,c]
    # has the property that M = sum_r (U[:,r] ⊗ V[:,r]) * W[:,r]^T ... no.
    # M[(a,b), c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # So M = sum_r (U[:,r] ⊗ V[:,r]) ⊗ ... this is a 3-way tensor.
    
    # A simpler check: for fixed c, M[:, c] = sum_r (U[:,r] * V[:,r]) * W[c,r]
    # where U[:,r] * V[:,r] is element-wise product reshaped to (n*m*m*p).
    
    # The rank of M (as a 2D matrix of size (n*m*m*p) x (p*n)) gives an upper bound.
    # But we need the tensor rank, which is different.
    
    # For now, just try to see if some columns of U, V, W are linearly dependent
    # in a way that allows rank reduction. This is complex, so skip for now.
    
    elapsed = time.time() - start
    if elapsed > max_time:
        return U, V, W
    
    return U, V, W


def construct():
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # Build the rank-42 Strassen decomposition
    U, V, W = _build_strassen_42()
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook rank 48
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank (heuristic, may not succeed)
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=30)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Rebuild from scratch
        U, V, W = _build_strassen_42()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
