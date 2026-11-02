"""Matrix multiplication 4,4,5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 120.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _build_tensor(n, m, p):
    """Build the 3D tensor T[i,j,k] = 1 if output (i,k) uses input (i,j)*(j,k).
    Convention: W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]
    T[i*m+j, j*p+k, k*n+i] = 1
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-6):
    """Check if U,V,W exactly decompose T (within tolerance for float)."""
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    rank = U.shape[1]
    # Efficient: for each r, outer product U[:,r] * V[:,r] * W[:,r]
    # Reshape: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # Use einsum
    reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(reconstructed, T, atol=tol, rtol=0)

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2.0) / 2.0

def _als_decompose(T, rank, max_iters=200, lr=0.1, tol=1e-8):
    """Alternating least squares to find rank-r decomposition of tensor T.
    T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    """
    n1, n2, n3 = T.shape
    # Initialize with random
    np.random.seed(42)
    U = np.random.randn(n1, rank) * 0.1
    V = np.random.randn(n2, rank) * 0.1
    W = np.random.randn(n3, rank) * 0.1
    
    prev_err = np.inf
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        # Update U: for each i, U[i,:] solves T[i,:,:] = sum_r U[i,r] * outer(V[:,r], W[:,r])
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For fixed i: T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])
        # So we need: for each r, matrix M_r = outer(V[:,r], W[:,r])
        # T[i,:,:] = sum_r U[i,r] * M_r
        # Flatten: T[i].flatten() = U[i,:] @ M_flat where M_flat is n2*n3 x rank
        M_flat = np.zeros((n2*n3, rank))
        for r in range(rank):
            M_flat[:, r] = (V[:, r] * W[:, r]).flatten()
        
        # Solve least squares for U
        T_flat = T.reshape(n1, n2*n3)
        U, _, _, _ = np.linalg.lstsq(M_flat, T_flat.T, rcond=None)
        U = U.T  # back to (n1, rank)
        
        # Update V
        M_flat = np.zeros((n1*n3, rank))
        for r in range(rank):
            M_flat[:, r] = (U[:, r] * W[:, r]).flatten()
        T_flat = T.transpose(1, 0, 2).reshape(n2, n1*n3)
        V, _, _, _ = np.linalg.lstsq(M_flat, T_flat.T, rcond=None)
        V = V.T
        
        # Update W
        M_flat = np.zeros((n1*n2, rank))
        for r in range(rank):
            M_flat[:, r] = (U[:, r] * V[:, r]).flatten()
        T_flat = T.transpose(2, 0, 1).reshape(n3, n1*n2)
        W, _, _, _ = np.linalg.lstsq(M_flat, T_flat.T, rcond=None)
        W = W.T
        
        # Compute error
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(reconstructed - T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, prev_err

def _strassen_decompose(n, m, p):
    """Build Strassen-style decomposition. For 4x4*4x5, use block Strassen.
    Split into 2x2 blocks where possible.
    4x4 * 4x5: split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2 (with padding for the 5)
    Actually 4x5 doesn't split evenly. Let's use a different approach.
    
    For 4x4 * 4x5:
    We can write this as computing C = A*B where A is 4x4, B is 4x5.
    
    Let's use the standard approach: treat as 2x2 block matrix multiplication
    A = [[A11, A12], [A21, A22]] each 2x2
    B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x2, B21 is 2x2, B22 is 2x2
    Wait, B is 4x5, so blocks would be 2x2 and 2x3, etc.
    
    Let me just use a direct approach: split into 2x2 blocks.
    A (4x4) -> 4 blocks of 2x2
    B (4x5) -> split columns: first 4 cols as 2x2 blocks, last col separate
    Actually this gets complicated. Let me use a simpler reduction.
    
    Alternative: Use the fact that rank(4x4*4x4) <= 64 (Strassen on 2x2 blocks of 2x2).
    For 4x4*4x5, we can pad B to 4x8 (two copies or zero pad) but that changes rank.
    
    Let me just try: decompose 4x4*4x4 with Strassen (rank 64), then extend to 4x5.
    For the extra column of B, we need additional terms.
    
    Actually, let me think differently. The schoolbook rank is 80.
    Strassen for 2x2 gives rank 7 for 2x2*2x2.
    For 4x4*4x4: split into 2x2 blocks, use Strassen on each pair: rank = 7*4 = 28? No.
    
    4x4*4x4: 4 block rows x 4 block cols of 2x2.
    C[i,j] = sum_k A[i,k]*B[k,j]
    With 2x2 blocks: C_block[i,j] = sum_k A_block[i,k]*B_block[k,j]
    This is a 4x4 block matrix mult where each entry is 2x2.
    Using Strassen on the 4x4 block structure: rank = 4 * 7 = 28? 
    No, Strassen applies to 2x2. For 4x4, we'd need to recursively apply.
    
    Let me just use a simpler baseline and focus on the search.
    """
    n, m, p = 4, 4, 5
    # Schoolbook as safe baseline
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, rank

def _try_rank(T, rank, max_restarts=5):
    """Try to find exact rank-r decomposition via ALS + rounding."""
    n1, n2, n3 = T.shape
    best_result = None
    best_err = np.inf
    
    for restart in range(max_restarts):
        if _time_left() < 10.0:
            break
        np.random.seed(42 + restart * 1000)
        U, V, W, err = _als_decompose(T, rank, max_iters=300)
        
        if err < 1e-4:
            # Try rounding to half-integers
            U_r = _round_to_half(U)
            V_r = _round_to_half(V)
            W_r = _round_to_half(W)
            
            if _verify(U_r, V_r, W_r, T, tol=1e-6):
                return U_r, V_r, W_r, rank, True
            
            # Also try rounding to integers
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if _verify(U_i, V_i, W_i, T, tol=1e-6):
                return U_i, V_i, W_i, rank, True
            
            # Try various rounding offsets
            for offset in [0, 0.5]:
                U_o = _round_to_half(U + offset)
                V_o = _round_to_half(V + offset)
                W_o = _round_to_half(W + offset)
                if _verify(U_o, V_o, W_o, T, tol=1e-6):
                    return U_o, V_o, W_o, rank, True
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
    
    return best_result[0], best_result[1], best_result[2], rank, False

def _construct_from_factors(U, V, W):
    """Convert numpy arrays to list-of-lists format."""
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _strassen_decompose(n, m, p)
    best_result = _construct_from_factors(U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Known bounds: rank(4x4*4x5) is between some values
    # Try ranks from 50 down to 80 (below schoolbook)
    for rank in range(50, rank_sb + 1):
        if _time_left() < 20.0:
            break
        
        U_t, V_t, W_t, r, success = _try_rank(T, rank, max_restarts=3)
        
        if success and r < best_rank:
            best_rank = r
            best_result = _construct_from_factors(U_t, V_t, W_t)
            break  # Found valid lower rank, stop
    
    # If we have time, also try some specific known constructions
    if _time_left() > 30.0:
        # Try rank 64 (Strassen-like)
        for rank in [64, 60, 56]:
            if _time_left() < 15.0:
                break
            if rank < best_rank:
                U_t, V_t, W_t, r, success = _try_rank(T, rank, max_restarts=10)
                if success and r < best_rank:
                    best_rank = r
                    best_result = _construct_from_factors(U_t, V_t, W_t)
    
    return best_result
# EVOLVE-BLOCK-END
