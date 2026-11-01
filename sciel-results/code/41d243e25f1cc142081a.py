import time
import numpy as np

DIMENSIONS = [2, 2, 11]


def _verify_numpy(U, V, W, n, m, p, trials=20, seed=42):
    """Verify the bilinear identity using numpy on random rational inputs.
    
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Identity: for all A (n x m), B (m x p):
        C[i,k] = sum_j A[i,j]*B[j,k]
        must equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    
    More precisely, the tensor T[i,j,k] = delta (the multiplication tensor)
    must equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    """
    rng = np.random.RandomState(seed)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the target tensor T of shape (n, m, p) where T[i,j,k] = 1 if it's
    # a valid multiplication term... actually the multiplication tensor is:
    # M[i, j, k] = 1 (for all i in [0,n), j in [0,m), k in [0,p))
    # The decomposition requires: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape U, V, W into 3D tensors
    # U_flat[i*m+j, r] -> U3[i, j, r]
    U3 = U.reshape(n, m, rank)
    # V_flat[j*p+k, r] -> V3[j, k, r]
    V3 = V.reshape(m, p, rank)
    # W_flat[k*n+i, r] -> W3[k, i, r]
    W3 = W.reshape(p, n, rank)
    
    # The reconstructed tensor at (i,j,k) is sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # We need this to equal 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    
    # Compute efficiently: for each r, the contribution is outer product structure
    # Reconstructed[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Build full tensor
    rec = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        # U3[:, :, r] has shape (n, m)
        # V3[:, :, r] has shape (m, p)  
        # W3[:, :, r] has shape (p, n)
        # Contribution: for each (i,j,k): U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This is a 3-way product. We can compute as:
        # For fixed r: term[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # = (U3[:,:,r] * W3.T[:,:,r]) * V3[:,:,r]  ... not quite
        # Let's just do it directly for small sizes
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rec[i, j, k] += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    
    # Target: all ones
    target = np.ones((n, m, p), dtype=float)
    
    if not np.allclose(rec, target, atol=1e-9):
        return False
    
    # Additional spot-check with actual matrix multiplication
    for _ in range(trials):
        A = rng.randint(-3, 4, size=(n, m)).astype(float)
        B = rng.randint(-3, 4, size=(m, p)).astype(float)
        C_true = A @ B
        # Reconstruct: C_rec[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        C_rec = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
                C_rec[i, k] = s
        if not np.allclose(C_rec, C_true, atol=1e-9):
            return False
    return True


def _build_strassen_39(n, m, p):
    """Build rank-39 decomposition using Strassen on 2x2 blocks.
    
    The 2x2 matrix A has 4 entries. The 2x11 matrix B has 2 rows of 11 entries.
    We can view B as two 2x2 block-columns plus a 2x7 remainder... 
    Actually, let's think more carefully.
    
    A is 2x2: A = [[a00, a01], [a10, a11]]
    B is 2x11: B = [[b00,...,b0,10], [b10,...,b1,10]]
    C = A @ B is 2x11: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Schoolbook: 2*2*11 = 44 multiplications.
    
    Strassen idea: group the 11 columns into blocks. Since we have 2 rows in A,
    we can use Strassen's 7 products for 2x2 * 2x2 blocks.
    
    Split B's 11 columns into: one 2x2 block + one 2x7 block? That's awkward.
    Better: think of it as 2x2 * (2x11). The 2x2 * 2x2 part uses Strassen (7 mults).
    The 2x2 * 2x7 part uses 2*2*7 = 28 mults schoolbook.
    Total: 7 + 28 = 35? Wait, but the 2x7 part isn't square.
    
    Actually, for 2x2 * 2xq, the rank is known to be at most 2q - 1 for q >= 2
    (using a variant of Strassen). For q=11, that's 21.
    For the full 2x2 * 2x11, since A is 2x2 and B is 2x11:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Let me use a different approach. The tensor for 2x2 * 2x11 multiplication:
    T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,10}
    
    We can decompose this as:
    For each k, the 2x2 sub-tensor T[:,:,k] is the identity-like tensor for 2x2*2x1,
    which has rank 4 (schoolbook) or... actually 2x2 * 2x1 is just:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    This is 4 bilinear terms per k, so 44 total.
    
    To reduce rank, we need to share products across k values.
    
    Let me try a direct construction. Define rank-r factors:
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    For the "Strassen-like" approach on the 2x2 A matrix:
    Define 7 Strassen products:
    M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k])  -- but B has two different k's...
    
    This is getting complex. Let me just use the known result:
    R(2,2,11) <= 21 via the formula R(n,m,p) <= (n*m) * ceil(p/2) * 2 - ... 
    
    Actually, the simplest known bound: R(2,2,p) <= 2p+1 for p odd? 
    Or R(2,2,p) <= 4p/3 * something...
    
    Let me just implement a concrete rank-39 construction and verify it.
    
    Alternative: use the fact that for 2x2 * 2x11, we can write:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    Introduce 7 Strassen products for each pair of consecutive k's (k, k+1):
    For k = 0,2,4,6,8 (5 pairs) + k=10 (1 single):
    Each pair (k, k+1): treat B[:,k] and B[:,k+1] as a 2x2 matrix.
    Then A (2x2) * B_pair (2x2) uses Strassen: 7 products.
    5 pairs * 7 = 35 products.
    1 remaining column: 2x2 * 2x1 uses 4 products.
    Total: 35 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]  # 22 x 39
    
    col = 0
    # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    
    for (k1, k2) in pairs:
        # Strassen for A(2x2) * B_pair(2x2) where B_pair[j, t] = B[j, k1+t]
        # Strassen products:
        # M1 = (A00+A11) * (B00+B11)  where B00=B[0,k1], B11=B[1,k2]
        # M2 = (A10+A11) * B00
        # M3 = A00 * (B01+B11)  where B01=B[0,k2], B11=B[1,k2]
        # M4 = A11 * (B10-B00)  where B10=B[1,k1], B00=B[0,k1]
        # M5 = (A00+A10) * B10
        # M6 = (A11-A10) * (B01+B10)  where B01=B[0,k2], B10=B[1,k1]
        # M7 = (A01-A11) * (B11+B01)  where B11=B[1,k2], B01=B[0,k2]
        
        # C[0,k1] = M1 - M4 + M6 - M7  (standard Strassen reconstruction)
        # C[0,k2] = M3 + M7
        # C[1,k1] = M2 + M4
        # C[1,k2] = M1 - M3 + M5 + M7
        
        # Wait, standard Strassen for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Using 0-indexed: A[i,j] with i,j in {0,1}
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # M3 = A[0,0] * (B[1,0]-B[0,0])
        # M4 = A[1,1] * (B[0,1]-B[1,1])
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        # C[0,0] = M1+M4-M5+M7
        # C[0,1] = M3+M5
        # C[1,0] = M2+M4
        # C[1,1] = M1-M2+M3+M6
        
        # In our notation, B_pair[0,0]=B[0,k1], B_pair[0,1]=B[0,k2],
        # B_pair[1,0]=B[1,k1], B_pair[1,1]=B[1,k2]
        
        # M1
        U[0][col] = 1.0; U[3][col] = 1.0  # A[0,0]+A[1,1]
        V[0*p+k1][col] = 1.0; V[1*p+k2][col] = 1.0  # B[0,k1]+B[1,k2]
        W[k1*n+0][col] = 1.0; W[k2*n+1][col] = 1.0  # contributes to C[0,k1] and C[1,k2]
        # Actually, W[k*n+i] means the (k,i) entry of C.
        # M1 contributes +1 to C[0,k1] and -1 to C[0,k2]... no.
        # C[0,k1] = M1+M4-M5+M7 => W[k1*n+0] += 1
        # C[1,k2] = M1-M2+M3+M6 => W[k2*n+1] += 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k1]
        U[2][col] = 1.0; U[3][col] = 1.0  # A[1,0]+A[1,1]
        V[0*p+k1][col] = 1.0  # B[0,k1]
        W[k1*n+1][col] = 1.0  # C[1,k1] += M2
        col += 1
        
        # M3 = A[0,0] * (B[1,k1]-B[0,k1])
        U[0][col] = 1.0  # A[0,0]
        V[1*p+k1][col] = 1.0; V[0*p+k1][col] = -1.0  # B[1,k1]-B[0,k1]
        W[k2*n+0][col] = 1.0  # C[0,k2] += M3
        col += 1
        
        # M4 = A[1,1] * (B[0,k2]-B[1,k2])
        U[3][col] = 1.0  # A[1,1]
        V[0*p+k2][col] = 1.0; V[1*p+k2][col] = -1.0  # B[0,k2]-B[1,k2]
        W[k1*n+0][col] = 1.0; W[k2*n+1][col] = 1.0  # C[0,k1] += M4, C[1,k2] += M4
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k2]
        U[0][col] = 1.0; U[1][col] = 1.0  # A[0,0]+A[0,1]
        V[1*p+k2][col] = 1.0  # B[1,k2]
        W[k1*n+0][col] = -1.0; W[k2*n+0][col] = 1.0  # C[0,k1] -= M5, C[0,k2] += M5
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k1]+B[0,k2])
        U[2][col] = 1.0; U[0][col] = -1.0  # A[1,0]-A[0,0]
        V[0*p+k1][col] = 1.0; V[0*p+k2][col] = 1.0  # B[0,k1]+B[0,k2]
        W[k2*n+1][col] = 1.0  # C[1,k2] += M6
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k1]+B[1,k2])
        U[1][col] = 1.0; U[3][col] = -1.0  # A[0,1]-A[1,1]
        V[1*p+k1][col] = 1.0; V[1*p+k2][col] = 1.0  # B[1,k1]+B[1,k2]
        W[k1*n+0][col] = 1.0; W[k2*n+1][col] = 1.0  # C[0,k1] += M7, C[1,k2] += M7
        col += 1
    
    # Remaining column k=10: schoolbook 2x2 * 2x1
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    
    # Term 1: A[0,0]*B[0,10]
    U[0][col] = 1.0
    V[0*p+10][col] = 1.0
    W[10*n+0][col] = 1.0
    col += 1
    
    # Term 2: A[0,1]*B[1,10]
    U[1][col] = 1.0
    V[1*p+10][col] = 1.0
    W[10*n+0][col] = 1.0
    col += 1
    
    # Term 3: A[1,0]*B[0,10]
    U[2][col] = 1.0
    V[0*p+10][col] = 1.0
    W[10*n+1][col] = 1.0
    col += 1
    
    # Term 4: A[1,1]*B[1,10]
    U[3][col] = 1.0
    V[1*p+10][col] = 1.0
    W[10*n+1][col] = 1.0
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}


def _try_rank_reduction(result, max_rank_target=21, deadline=None):
    """Attempt to reduce the rank by finding linear dependencies.
    
    If two rank-1 terms are linearly dependent, we can merge them.
    More generally, if the columns of U, V, W have a linear relation,
    we can reduce the rank.
    """
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    n, m, p = 2, 2, 11
    current_rank = U.shape[1]
    
    if deadline is None:
        deadline = time.time() + 30.0
    
    # Try to find a subset of columns that can be removed while preserving
    # the tensor identity. This is hard in general, but for small cases
    # we can try greedy removal.
    
    # For each column r, check if removing it still gives a valid decomposition
    # (it won't, since each column contributes uniquely). But we can check if
    # two columns can be merged into one.
    
    # Actually, a better approach: check if the current decomposition has
    # any zero columns or duplicate columns that can be removed.
    
    # Check for zero columns
    valid_cols = []
    for r in range(current_rank):
        if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
            valid_cols.append(r)
    
    if len(valid_cols) < current_rank:
        U = U[:, valid_cols]
        V = V[:, valid_cols]
        W = W[:, valid_cols]
        current_rank = len(valid_cols)
    
    # Check for duplicate columns (same U, V, W pattern up to scaling)
    # This is complex; skip for now.
    
    # Try to verify the reduced decomposition
    if _verify_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p, trials=5, seed=999):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return result


def _schoolbook(n, m, p):
    """Guaranteed-valid rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s limit
    
    # Step 1: Build the structured rank-39 Strassen-based decomposition
    result = _build_strassen_39(n, m, p)
    
    # Verify
    if not _verify_numpy(result['U'], result['V'], result['W'], n, m, p, trials=10, seed=42):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    # Step 2: Try rank reduction
    result = _try_rank_reduction(result, max_rank_target=21, deadline=deadline)
    
    # Final verification
    if not _verify_numpy(result['U'], result['V'], result['W'], n, m, p, trials=10, seed=123):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
    
    return result


if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    print(f"U shape: ({len(res['U'])}, {rank})")
    print(f"V shape: ({len(res['V'])}, {rank})")
    print(f"W shape: ({len(res['W'])}, {rank})")
