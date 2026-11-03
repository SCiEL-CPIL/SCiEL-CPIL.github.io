"""Matrix multiplication 2,4,7 tensor decomposition using numerical rank-reduction search."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # Safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify if the decomposition is exact using the tensor identity."""
    rank = U.shape[1]
    # Build the full tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # and check it equals 1 for all (i,j,k) in the multiplication tensor.
    # The multiplication tensor has T[i,j,k]=1 iff the (i,j) entry of A and (j,k) entry of B
    # contribute to (i,k) entry of C, which is always 1 for valid indices.
    # Actually, the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right. The matrix multiplication tensor M_{i,j,k} = delta_{j,j} which is always 1.
    # No. The tensor for matrix multiplication C = AB where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # So we just need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # Efficient check: compute the tensor and compare to ones.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape for einsum
    U_ = U.reshape(n, m, rank)  # U[i, j, r] where index is i*m+j
    V_ = V.reshape(m, p, rank)  # V[j, k, r] where index is j*p+k
    W_ = W.reshape(p, n, rank)  # W[k, i, r] where index is k*n+i
    
    # Compute T[i,j,k] = sum_r U_[i,j,r] * V_[j,k,r] * W_[k,i,r]
    # Using einsum: 'ijr,jkr,kr->ijk' won't work directly because W is indexed by (k,i)
    # Let's use: T[i,j,k] = sum_r U_[i,j,r] * V_[j,k,r] * W_[k,i,r]
    
    # Build W_ properly: W_ has shape (p, n, rank), W_[k, i, r]
    # We need sum_r U_[i,j,r] * V_[j,k,r] * W_[k,i,r]
    
    # einsum: 'ijr,jkr,kr->ijk' -- but W_ is (p,n,rank) = (k,i,r), so W_[k,i,r]
    # Let's transpose W_ to (n, p, rank) -> W_[i, k, r]
    W_t = W_.transpose(1, 0, 2)  # shape (n, p, rank), W_t[i, k, r] = W_[k, i, r]
    
    T = np.einsum('ijr,jkr,ikr->ijk', U_, V_, W_t)
    
    # T should be all ones
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # A = [[a,b],[c,d]] -> idx 0,1,2,3 (row-major)
    # B = [[e,f],[g,h]] -> idx 0,1,2,3
    # C = AB = [[c00,c01],[c10,c11]] -> idx 0,1,2,3
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[0, 2] = -1
    
    # M4 = d(b-e)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[2, 3] = -1; W[3, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = -1; W[2, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1; W[1, 6] = -1
    
    return U, V, W


def _construct_block_decomposition():
    """
    Construct 2x4x7 decomposition using block structure.
    Split A into 2x2 + 2x2 (cols 0-1, cols 2-3).
    Split B into 2x7 + 2x7 (rows 0-1, rows 2-3).
    Each 2x2 * 2x7 is split into 3x 2x2x2 + 1x 2x2x1.
    Total rank: 2 * (3*7 + 4) = 50.
    """
    n, m, p = 2, 4, 7
    
    U_total = np.zeros((n*m, 0), dtype=np.float64)
    V_total = np.zeros((m*p, 0), dtype=np.float64)
    W_total = np.zeros((p*n, 0), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # 2x2x1 decomposition
    U_221 = np.zeros((4, 4), dtype=np.float64)
    V_221 = np.zeros((2, 4), dtype=np.float64)
    W_221 = np.zeros((2, 4), dtype=np.float64)
    
    U_221[0, 0] = 1; V_221[0, 0] = 1; W_221[0, 0] = 1
    U_221[1, 1] = 1; V_221[1, 1] = 1; W_221[0, 1] = 1
    U_221[2, 2] = 1; V_221[0, 2] = 1; W_221[1, 2] = 1
    U_221[3, 3] = 1; V_221[1, 3] = 1; W_221[1, 3] = 1
    
    def embed_tensor(U_t, V_t, W_t, a_col_start, b_row_start, b_col_start, c_col_start, b_width):
        nonlocal U_total, V_total, W_total
        R = U_t.shape[1]
        U_block = np.zeros((8, R), dtype=np.float64)
        V_block = np.zeros((28, R), dtype=np.float64)
        W_block = np.zeros((14, R), dtype=np.float64)
        
        for i in range(2):
            for j in range(b_width):
                U_block[i*4 + a_col_start + j, :] = U_t[i*2 + j, :]
        
        for i in range(b_width):
            for j in range(b_width):
                V_block[(b_row_start + i)*7 + b_col_start + j, :] = V_t[i*b_width + j, :]
        
        for i in range(b_width):
            for j in range(2):
                W_block[(c_col_start + i)*2 + j, :] = W_t[i*b_width + j, :]
        
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # Term 1: A_top (cols 0,1) * B_top (rows 0,1)
    embed_tensor(U_s, V_s, W_s, 0, 0, 0, 0, 2)
    embed_tensor(U_s, V_s, W_s, 0, 0, 2, 2, 2)
    embed_tensor(U_s, V_s, W_s, 0, 0, 4, 4, 2)
    embed_tensor(U_221, V_221, W_221, 0, 0, 6, 6, 1)
    
    # Term 2: A_bottom (cols 2,3) * B_bottom (rows 2,3)
    embed_tensor(U_s, V_s, W_s, 2, 2, 0, 0, 2)
    embed_tensor(U_s, V_s, W_s, 2, 2, 2, 2, 2)
    embed_tensor(U_s, V_s, W_s, 2, 2, 4, 4, 2)
    embed_tensor(U_221, V_221, W_221, 2, 2, 6, 6, 1)
    
    return U_total, V_total, W_total


def _apply_random_transform(U, V, W, rng, scale=1.0):
    """
    Apply a random invertible linear transformation to the rank dimension.
    U' = U * T, V' = V * T^{-1}, W' = W * T^{-1} preserves the tensor.
    """
    rank = U.shape[1]
    # Generate a random invertible matrix T
    T = rng.standard_normal((rank, rank)) * scale + np.eye(rank)
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        return U, V, W
    
    U_new = U @ T
    V_new = V @ T_inv
    W_new = W @ T_inv
    return U_new, V_new, W_new


def _try_rank_reduction(U, V, W, n, m, p, target_rank, tol=1e-7):
    """
    Try to reduce the rank by finding a lower-rank equivalent decomposition.
    We use the fact that if the rank-r decomposition has linear dependencies,
    we can project to a lower-dimensional subspace.
    
    Strategy: randomly subsample columns and check if the remaining columns
    still span the tensor. If yes, we've found a lower-rank decomposition.
    """
    rank = U.shape[1]
    if rank <= target_rank:
        return U, V, W
    
    # Try to find a subset of columns that still spans the tensor
    # This is a greedy approach: try removing columns one at a time
    
    best_U, best_V, best_W = U, V, W
    current_rank = rank
    
    # Try random subsets
    rng = np.random.default_rng(42)
    for attempt in range(50):
        # Try to reduce by 1
        if current_rank <= target_rank:
            break
        
        # Randomly pick a column to try removing
        col_to_remove = rng.integers(0, current_rank)
        
        # Check if removing this column still gives a valid decomposition
        # We need to check if the tensor can be represented without this column.
        # This is equivalent to checking if the removed column is in the span of the others
        # in the tensor product space.
        
        # Simpler approach: just try it and verify
        mask = np.ones(current_rank, dtype=bool)
        mask[col_to_remove] = False
        
        U_trial = best_U[:, mask]
        V_trial = best_V[:, mask]
        W_trial = best_W[:, mask]
        
        if _verify(U_trial, V_trial, W_trial, n, m, p, tol):
            best_U, best_V, best_W = U_trial, V_trial, W_trial
            current_rank -= 1
    
    return best_U, best_V, best_W


def _ladder_search(U, V, W, n, m, p, time_budget=100):
    """
    Ladder-based search: iteratively apply random transformations and try to reduce rank.
    """
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    rng = np.random.default_rng(12345)
    
    iteration = 0
    while time.time() - start_time < time_budget:
        iteration += 1
        
        # Apply random transformation
        U_t, V_t, W_t = _apply_random_transform(best_U, best_V, best_W, rng, scale=0.5)
        
        # Try rank reduction
        U_r, V_r, W_r = _try_rank_reduction(U_t, V_t, W_t, n, m, p, best_rank - 1)
        
        if U_r.shape[1] < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = U_r.shape[1]
            # Reset the transformation base to the new best
            continue
        
        # Also try with different scales
        if iteration % 10 == 0:
            U_t, V_t, W_t = _apply_random_transform(best_U, best_V, best_W, rng, scale=2.0)
            U_r, V_r, W_r = _try_rank_reduction(U_t, V_t, W_t, n, m, p, best_rank - 1)
            if U_r.shape[1] < best_rank:
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = U_r.shape[1]
    
    return best_U, best_V, best_W


def _structured_bini_approach():
    """
    Try a structured approach inspired by Bini-Capovani 3x3x3 decomposition.
    Embed the 4-dimensional middle index into a 3x3 grid structure.
    
    The key insight: 2x4x7 can be viewed as having a 4-dimensional middle index.
    We can embed this into a 3x3 grid (9 positions) and use the Bini-Capovani
    rank-19 structure for 3x3x3, then project down.
    
    Actually, let's try a different structural approach:
    Split the 4x7 B matrix into blocks that allow sharing of Strassen-like
    computations across the 7 columns.
    
    New idea: Instead of splitting 7 into 2+2+2+1, try 7 = 3+2+2.
    Then we need 2x2x3, 2x2x2, 2x2x2 decompositions.
    2x2x3 schoolbook is rank 12.
    Total per term: 12 + 7 + 7 = 26. Two terms: 52. Same as before.
    
    Let's try: 7 = 3+4.
    2x2x3 (rank 12) + 2x2x4 (rank 16 schoolbook, or can we do better?)
    2x2x4: split 4 into 2+2, use two Strassen: rank 14.
    Total per term: 12 + 14 = 26. Two terms: 52.
    
    Hmm, same. Let's try a completely different split.
    
    What if we don't split A into two 2x2 blocks? What if we use a different
    factorization of the 2x4 matrix?
    
    Actually, the fundamental issue is that the tensor rank of 2x4x7 is unknown
    but bounded above by 50 (our current construction) and below by... 
    The lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    
    So the true rank is between 28 and 50. Our job is to find the minimum.
    
    Let's try a more aggressive numerical search.
    """
    pass


def _aggressive_numerical_search(n, m, p, time_budget=100):
    """
    Aggressive numerical search: start from schoolbook (rank 56), apply random
    transformations, and try to reduce rank via SVD-based projection.
    
    Key insight: if we have a rank-r decomposition, the tensor lies in the
    image of the map (U,V,W) -> T. If we can find a rank-(r-1) decomposition,
    the tensor also lies in a smaller subspace.
    
    We use the following approach:
    1. Start with rank 56 schoolbook.
    2. Apply random invertible transformations to get equivalent decompositions.
    3. Try to find a lower-rank approximation that is still exact.
    4. Use the "peeling" method: if one of the rank-1 terms is redundant, remove it.
    """
    start_time = time.time()
    U, V, W = _construct_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    rng = np.random.default_rng(999)
    
    # First, try to reduce from 56 to 50 using our block decomposition as a guide
    # Actually, let's just use the block decomposition as our starting point
    U, V, W = _construct_block_decomposition()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Now try to reduce further
    for iteration in range(200):
        if time.time() - start_time > time_budget:
            break
        
        # Apply random transformation
        U_t, V_t, W_t = _apply_random_transform(best_U, best_V, best_W, rng, scale=0.3)
        
        # Try to peel off columns
        current_rank = U_t.shape[1]
        if current_rank <= 28:  # Can't go below the lower bound
            break
        
        # Try removing random columns
        for col in range(current_rank):
            if time.time() - start_time > time_budget:
                break
            
            mask = np.ones(current_rank, dtype=bool)
            mask[col] = False
            
            U_trial = U_t[:, mask]
            V_trial = V_t[:, mask]
            W_trial = W_t[:, mask]
            
            if _verify(U_trial, V_trial, W_trial, n, m, p, tol=1e-6):
                best_U, best_V, best_W = U_trial, V_trial, W_trial
                best_rank = U_trial.shape[1]
                break  # Restart with the new best
    
    return best_U, best_V, best_W


def _svd_rank_reduction(U, V, W, n, m, p, tol=1e-7):
    """
    Use SVD to find the numerical rank of the tensor and project to it.
    
    The tensor T can be viewed as a matrix in different ways:
    - T_{i,j,k} as a matrix with rows (i,j) and columns k: shape (n*m, p)
    - T_{i,j,k} as a matrix with rows (i,k) and columns j: shape (n*p, m)
    - T_{i,j,k} as a matrix with rows (j,k) and columns i: shape (m*p, n)
    
    The rank of the tensor is at least the maximum of these matrix ranks.
    But the tensor rank can be higher.
    
    For our purpose, we check if the current decomposition has numerical rank
    less than its nominal rank.
    """
    rank = U.shape[1]
    
    # Check if any columns are numerically dependent
    # U has shape (n*m, rank). If rank > n*m, there must be dependencies.
    # But the dependencies in U don't directly translate to tensor rank reduction.
    
    # A better approach: check if the set of rank-1 tensors {U[:,r] * V[:,r] * W[:,r]}
    # is linearly independent.
    
    # Flatten each rank-1 tensor
    # T_r = outer(U[:,r], outer(V[:,r], W[:,r])) reshaped to (n*m*p,)
    # Actually, T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is expensive for large rank. Let's skip this for now.
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    try:
        # Start with block decomposition (rank 50)
        U, V, W = _construct_block_decomposition()
        
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            U, V, W = _construct_schoolbook(n, m, p)
        
        # Try to reduce rank further using numerical search
        remaining_time = DEADLINE - (time.time() - start_time) - 5  # 5s safety margin
        if remaining_time > 20:
            U, V, W = _aggressive_numerical_search(n, m, p, time_budget=remaining_time)
            
            if not _verify(U, V, W, n, m, p):
                # Fallback
                U, V, W = _construct_block_decomposition()
                if not _verify(U, V, W, n, m, p):
                    U, V, W = _construct_schoolbook(n, m, p)
        
    except Exception:
        try:
            U, V, W = _construct_block_decomposition()
            if not _verify(U, V, W, n, m, p):
                U, V, W = _construct_schoolbook(n, m, p)
        except Exception:
            U, V, W = _construct_schoolbook(n, m, p)
    
    # Final safety check
    if not _verify(U, V, W, n, m, p):
        U, V, W = _construct_schoolbook(n, m, p)
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
