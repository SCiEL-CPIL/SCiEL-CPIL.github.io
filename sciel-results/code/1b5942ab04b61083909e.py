import numpy as np
import time

DIMENSIONS = [3, 4, 5]
TIME_LIMIT = 140  # safety margin from 150s

def _verify(U, V, W):
    """Check if U,V,W exactly decompose the 3x4x5 matmul tensor."""
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # T = einsum('ur,vr,wr->uvw', U, V, W)
    T = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Expected: T[i*m+l, l*p+k, k*n+i] = 1, else 0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                if abs(T[u_idx, v_idx, w_idx] - 1.0) > 1e-9:
                    return False
    
    # Check that T is zero elsewhere
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_idx = i * m + l
                v_idx = l * p + k
                w_idx = k * n + i
                mask[u_idx, v_idx, w_idx] = True
    
    invalid = ~mask
    if np.any(np.abs(T[invalid]) > 1e-9):
        return False
    
    return True

def _seed_solution():
    """Return the seed 56-rank solution as U,V,W numpy arrays."""
    n, m, p = 3, 4, 5
    U_parts = []
    V_parts = []
    W_parts = []
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_blk[i, r]
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_blk[i, r]
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_blk[i, r]
    
    add_block([0,1,4,5], [0,1,5,6], [0,3,1,4], *strassen_2x2x2())
    add_block([0,1,4,5], [2,3,7,8], [6,9,7,10], *strassen_2x2x2())
    add_block([0,1,4,5], [4,9], [12,13], *schoolbook_2x2x1())
    add_block([2,3,6,7], [10,11,15,16], [0,3,1,4], *strassen_2x2x2())
    add_block([2,3,6,7], [12,13,17,18], [6,9,7,10], *strassen_2x2x2())
    add_block([2,3,6,7], [14,19], [12,13], *schoolbook_2x2x1())
    add_block([8,9], [0,1,5,6], [2,5], *schoolbook_1x2x2())
    add_block([8,9], [2,3,7,8], [8,11], *schoolbook_1x2x2())
    add_block([8,9], [4,9], [14], *schoolbook_1x2x1())
    add_block([10,11], [10,11,15,16], [2,5], *schoolbook_1x2x2())
    add_block([10,11], [12,13,17,18], [8,11], *schoolbook_1x2x2())
    add_block([10,11], [14,19], [14], *schoolbook_1x2x1())
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def _reduce_rank_by_linear_dependence(U, V, W, tol=1e-10):
    """
    Reduce rank by identifying linear dependencies among the rank-1 components.
    """
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W

    # Flatten each rank-1 component z_r = u_r \otimes v_r \otimes w_r
    # z_r is a vector of length n*m*p*n = 3600
    # We form a matrix Z of shape (3600, rank)
    Z = np.zeros((n * m * p * n, rank))
    
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # Outer product
        # u shape (12,), v shape (20,), w shape (15,)
        # Reshape to (12, 1, 1), (1, 20, 1), (1, 1, 15)
        tensor = np.outer(np.outer(u, v), w)
        Z[:, r] = tensor.flatten()

    # Perform QR with column pivoting to find rank
    # scipy.linalg.qr with pivoting is not always available or easy to use for rank detection directly in all versions
    # We can use SVD to determine rank more robustly
    U_svd, s, Vt = np.linalg.svd(Z, full_matrices=False)
    
    # Determine numerical rank
    # s is sorted in descending order
    rank_num = np.sum(s > tol)
    
    if rank_num >= rank:
        return U, V, W

    # We have a lower rank.
    # We need to express the original sum T = sum z_r as a sum of rank_num terms.
    # The columns of Z span a subspace of dimension rank_num.
    # Let P be the indices of the linearly independent columns.
    # We can find P by looking at the pivots of QR or by selecting columns of Z that are "large" in the SVD basis?
    # Actually, SVD doesn't directly give a subset of columns.
    # QR with pivoting is better for this.
    
    # Let's use a simple greedy approach or QR pivoting.
    # Since rank is small (56), we can use a simple algorithm to find a basis.
    
    # Use QR with column pivoting from scipy if available, else manual.
    # Let's try to use numpy's lstsq to find coefficients for a subset.
    
    # A simpler approach:
    # If the rank is R', we can select any R' linearly independent columns from Z.
    # Let's find them.
    independent_cols = []
    # We can use the fact that the null space of Z is non-trivial.
    # Or, we can just iterate and check if a column is in the span of previous ones.
    
    # Better: Use the pivots from QR.
    # scipy.linalg.qr(Z, mode='r', pivoting=True) returns R and pivots.
    # Let's use that.
    try:
        from scipy.linalg import qr
        R, pivots = qr(Z, mode='r', pivoting=True)
        # The rank is the number of diagonal elements of R that are > tol
        diag = np.abs(np.diag(R))
        rank_qr = np.sum(diag > tol)
        
        if rank_qr >= rank:
             return U, V, W
             
        # The first rank_qr columns in the permuted order (pivots) form a basis.
        # Let P be the indices of the pivot columns that are part of the basis.
        # The pivots array indicates the permutation of columns.
        # The first rank_qr elements of pivots are the indices of the linearly independent columns.
        P = pivots[:rank_qr]
        
        # Now, express the other columns as linear combinations of the columns in P.
        # Let Z_P be Z[:, P] (shape 3600 x R')
        # Let Z_N be Z[:, N] where N are the other columns.
        # We want Z_N = Z_P * C
        # C = pinv(Z_P) * Z_N
        
        Z_P = Z[:, P]
        # Solve for C using least squares (or exact if dependent)
        # C has shape (R', rank - R')
        
        N = [i for i in range(rank) if i not in P]
        if len(N) == 0:
             return U, V, W
             
        Z_N = Z[:, N]
        
        # Compute C
        # Z_P C = Z_N
        # C = (Z_P^T Z_P)^{-1} Z_P^T Z_N
        # Use lstsq for stability
        C, residuals, rank_c, s_c = np.linalg.lstsq(Z_P, Z_N, rcond=None)
        
        # Now, T = sum_{r in P} z_r + sum_{j in N} z_j
        # z_j = sum_{i in P} C_{i, j_idx} z_i
        # T = sum_{i in P} z_i + sum_{j in N} sum_{k in P} C_{k, j_idx} z_k
        # T = sum_{k in P} (1 + sum_{j in N} C_{k, j_idx}) z_k
        
        # Let alpha_k be the coefficient for z_k.
        # alpha_k = 1 + sum_{j in N} C_{k, j_idx}
        
        # We can form a new decomposition with only the columns in P.
        # The new U, V, W will have columns corresponding to P, scaled by alpha_k.
        
        new_U = []
        new_V = []
        new_W = []
        
        # Map original indices in P to new indices
        # P is an array of original indices.
        
        # Calculate alpha for each k in P
        # C is (R', len(N))
        # We need sum over j in N for each k.
        # Sum_C = C.sum(axis=1)
        
        Sum_C = C.sum(axis=1)
        alphas = 1.0 + Sum_C
        
        # Construct new matrices
        for idx, orig_idx in enumerate(P):
            alpha = alphas[idx]
            # Multiply the column by alpha. We can distribute this among U, V, W.
            # Simplest: multiply U column by alpha.
            u_col = U[:, orig_idx] * alpha
            v_col = V[:, orig_idx]
            w_col = W[:, orig_idx]
            
            new_U.append(u_col)
            new_V.append(v_col)
            new_W.append(w_col)
            
        new_U = np.array(new_U).T
        new_V = np.array(new_V).T
        new_W = np.array(new_W).T
        
        # Verify the new solution
        if _verify(new_U, new_V, new_W):
            return new_U, new_V, new_W
        else:
            # If verification fails, it might be due to numerical issues or incorrect logic.
            # Return original
            return U, V, W

    except ImportError:
        # Fallback if scipy is not available (though it should be)
        pass
    except Exception:
        return U, V, W

def _try_remove_columns(U, V, W, max_tries=200):
    """Try to remove columns by checking linear dependence."""
    n, m, p = 3, 4, 5
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try removing each column
    for r in range(rank):
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        if _verify(U_test, V_test, W_test):
            return U_test, V_test, W_test
    return None

def _mix_and_reduce(U, V, W, n_trials=50):
    """Apply random invertible transformations and try to reduce rank."""
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    for t in range(n_trials):
        # Generate random invertible matrix M
        # Use sparse perturbation to keep numbers small
        M = np.eye(best_rank)
        # Add random sparse entries
        for _ in range(best_rank // 2):
            i, j = np.random.randint(0, best_rank, 2)
            if i != j:
                M[i, j] = np.random.choice([-1, 1])
        
        # Check if M is invertible
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
            
        # Transform
        U_new = best_U @ M
        V_new = best_V @ M_inv
        W_new = best_W
        
        # Try to remove columns
        result = _try_remove_columns(U_new, V_new, W_new)
        if result is not None:
            U_new, V_new, W_new = result
            if U_new.shape[1] < best_rank:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = best_U.shape[1]
                
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start from seed
    U, V, W = _seed_solution()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]
    
    # Verify seed
    if not _verify(best[0], best[1], best[2]):
        raise ValueError("Seed solution failed verification")
    
    # Apply linear dependence reduction first
    U_red, V_red, W_red = _reduce_rank_by_linear_dependence(best[0], best[1], best[2])
    if U_red.shape[1] < best_rank:
        best = (U_red, V_red, W_red)
        best_rank = best[0].shape[1]
        
    # Iteratively try to reduce rank using mix-and-reduce
    for iteration in range(100):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
        
        U_c, V_c, W_c = best[0].copy(), best[1].copy(), best[2].copy()
        
        # Apply mix and reduce
        U_new, V_new, W_new = _mix_and_reduce(U_c, V_c, W_c, n_trials=20)
        
        if U_new.shape[1] < best_rank:
            best = (U_new, V_new, W_new)
            best_rank = best[0].shape[1]
            
        # Also try linear dependence reduction on the new result
        U_red2, V_red2, W_red2 = _reduce_rank_by_linear_dependence(best[0], best[1], best[2])
        if U_red2.shape[1] < best_rank:
            best = (U_red2, V_red2, W_red2)
            best_rank = best[0].shape[1]

        # If we hit a very low rank, we can stop or continue
        if best_rank <= 50:
            break

    # Final verification
    if not _verify(best[0], best[1], best[2]):
        # Fallback to seed if something went wrong
        best = (U.copy(), V.copy(), W.copy())

    U_out, V_out, W_out = best
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
