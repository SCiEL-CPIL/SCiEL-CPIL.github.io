"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for 2x4x7 matrix multiplication
    using local search on the tensor factors.
    
    Strategy:
    1. Initialize with a random hypercube partition (Strassen-style).
    2. Iteratively optimize U, V, W using least squares to minimize reconstruction error.
    3. Snap entries to the nearest half-integer.
    4. Prune zero columns to minimize rank.
    5. Return the best valid decomposition found within the time limit.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline to be safe against 150s limit
    
    # The matrix multiplication tensor T has indices (i, j, k) for A[i,j], B[j,k], C[i,k].
    # However, the interface defines:
    # U: (n*m, rank) where row index is i*m + j
    # V: (m*p, rank) where row index is j*p + k
    # W: (p*n, rank) where row index is k*n + i
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,j,k}?
    # No, the tensor for matrix multiplication is T_{i,j,k} = A_{i,j} B_{j,k} C_{i,k} (in dual space).
    # Actually, the standard decomposition is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This corresponds to a tensor T with indices (i, j, k) such that the "multiplication" is exact.
    # The evaluator likely checks:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 if (i,j,k) is a valid triple?
    # Wait, the task says "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor T has non-zero entries at (i, j, k) corresponding to the term A_{i,j}B_{j,k} contributing to C_{i,k}.
    # In the standard basis, T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} is part of the multiplication, else 0.
    # But which triples? All (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # So T is a 3-tensor of size n x m x p.
    # The decomposition is T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    # Note the index mapping in W: k*n + i.
    
    # Let's construct the target tensor T as a flat vector or 3D array.
    # T[i, j, k] = 1 for all valid i, j, k.
    # Wait, is T_{i,j,k} always 1?
    # The multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T represents the bilinear form.
    # Yes, for every (i, j, k), the term A_{i,j}B_{j,k} exists.
    # So T[i, j, k] = 1 for all i, j, k.
    
    # Let's verify the index conventions.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Product: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r.
    # This must equal T[i, j, k].
    # For standard matrix multiplication, T[i, j, k] = 1.
    
    # Target tensor T
    T = np.ones((n, m, p))
    
    # We will use a local search.
    # Initial rank: Start with a reasonable upper bound.
    # Schoolbook rank is n*m*p = 56.
    # Strassen-like might be lower.
    # Let's try to find a rank R such that we can decompose T.
    # We'll start with R = 56 and try to reduce it, or start lower and increase if it fails?
    # Better: Start with R = 56, optimize, prune. If error > 0, increase R?
    # Or start with a random R and optimize.
    # Given the small size, we can afford to try several R values.
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from a lower bound up to schoolbook.
    # Lower bound for 2x4x7?
    # 2x2x2 is 7.
    # 2x4x7 is likely > 2*4*7 / 2 = 28?
    # Let's try a range.
    
    for R in range(20, n * m * p + 1):
        if time.time() > deadline:
            break
            
        # Multiple restarts for each R
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random values
            # Using random initialization in [0, 1] or {-1, 1}
            # Hypercube partitioning suggests using +/- 1
            U = np.random.choice([-1, 1], size=(n * m, R))
            V = np.random.choice([-1, 1], size=(m * p, R))
            W = np.random.choice([-1, 1], size=(p * n, R))
            
            # Local search iterations
            # We alternate minimizing U, V, W.
            # Minimize || T - U V^T W (in some sense) ||
            # The term for (i, j, k) is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # This is a tensor product.
            # We can fix V and W, and solve for U.
            # For a fixed r, U[:, r] is a vector.
            # The contribution of U[:, r] to the tensor is U[:, r] outer V[:, r] outer W[:, r].
            # This is linear in U[:, r].
            # So we can solve for each column of U independently?
            # No, the sum is over r.
            # T = sum_r U[:, r] outer V[:, r] outer W[:, r]
            # If we fix V and W, we want to find U such that T - sum_{r != r0} ... = U[:, r0] outer V[:, r0] outer W[:, r0].
            # This is a linear system for U[:, r0].
            # Let's flatten the tensor.
            # T_flat of size (n*m*p).
            # The term for column r is a vector v_r of size (n*m*p).
            # v_r[i*m+j, j*p+k, k*n+i] ... wait, the indices are coupled.
            # Let's construct the matrix A_r such that U[:, r] = A_r^{-1} b_r?
            
            # Let's define the linear operator.
            # For a fixed r, let u = U[:, r], v = V[:, r], w = W[:, r].
            # The contribution to T[i, j, k] is u[i*m+j] * v[j*p+k] * w[k*n+i].
            # We want to approximate T.
            # If we fix all columns except r, let T_res = T - sum_{r' != r} u_{r'} outer v_{r'} outer w_{r'}.
            # We want to find u_r such that u_r outer v_r outer w_r approximates T_res.
            # This is a rank-1 tensor approximation problem.
            # The optimal u_r is given by:
            # u_r = (T_res * v_r * w_r) / (||v_r||^2 ||w_r||^2) ?
            # Let's check.
            # Minimize || T_res - u outer v outer w ||_F^2
            # Derivative wrt u: -2 (T_res * v * w) + 2 u (||v||^2 ||w||^2) = 0
            # So u = (T_res * v * w) / (||v||^2 ||w||^2)
            # Where * denotes element-wise multiplication with broadcasting.
            # T_res has shape (n, m, p).
            # v has shape (m*p). We need to reshape v to (m, p) to multiply with T_res.
            # w has shape (p*n). We need to reshape w to (p, n) to multiply with T_res.
            # Note the index mapping:
            # v index j*p+k -> v_reshaped[j, k]
            # w index k*n+i -> w_reshaped[k, i]
            
            # Let's implement one step of the local search.
            # We will iterate R times, updating each column.
            
            U_cur = U.copy()
            V_cur = V.copy()
            W_cur = W.copy()
            
            # Precompute reshaped V and W for faster access
            # V_cur shape (m*p, R) -> V_cur_reshaped (m, p, R)
            # W_cur shape (p*n, R) -> W_cur_reshaped (p, n, R)
            
            # We will do a few sweeps.
            for sweep in range(10):
                if time.time() > deadline:
                    break
                # Update U
                for r in range(R):
                    # Calculate residual T_res
                    # T_res = T - sum_{r' != r} U_cur[:, r'] outer V_cur[:, r'] outer W_cur[:, r']
                    # This is expensive if done naively.
                    # We can maintain a running sum?
                    # Or just compute the full tensor sum each time?
                    # R is small (<= 56). n*m*p = 56.
                    # 56 * 56 * 56 operations per update? 56*56 = 3136. 3136*56 = 175k.
                    # 10 sweeps * 56 cols * 175k = 98M ops.
                    # Python loops are slow. Numpy is fast.
                    
                    # Let's use Numpy to compute the sum for all r at once?
                    # No, we update one r at a time.
                    
                    # Efficient way:
                    # Compute full tensor S = sum_{r'} U[:, r'] outer V[:, r'] outer W[:, r']
                    # S[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                    # This is a tensor contraction.
                    # We can do this with einsum.
                    # U_flat: (n*m, R)
                    # V_flat: (m*p, R)
                    # W_flat: (p*n, R)
                    
                    # Reshape U to (n, m, R)
                    # Reshape V to (m, p, R)
                    # Reshape W to (p, n, R)
                    
                    # S[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
                    
                    # Let's compute S using einsum.
                    # U_reshaped: (n, m, R)
                    # V_reshaped: (m, p, R)
                    # W_reshaped: (p, n, R)
                    
                    # term = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                    
                    # If we fix r, we want to remove the r-th term from S.
                    # S = S_r + S_other
                    # S_r[i, j, k] = U[i, j, r] V[j, k, r] W[k, i, r]
                    
                    # So T_res = T - S_other = T - (S - S_r) = T - S + S_r
                    # We want to find u_new such that u_new outer v_r outer w_r approximates T_res.
                    # u_new = (T_res * v_r * w_r) / (||v_r||^2 ||w_r||^2)
                    
                    # Let's compute S once per sweep?
                    # If we update U[:, r], S changes.
                    # So we must recompute S or update it incrementally.
                    
                    # Incremental update:
                    # S_old is known.
                    # S_new = S_old - S_r_old + S_r_new
                    # We need S_old to compute T_res.
                    
                    # Let's maintain S.
                    # Initialize S.
                    if sweep == 0 and r == 0:
                        # Compute initial S
                        U_r = U_cur.reshape(n, m, R)
                        V_r = V_cur.reshape(m, p, R)
                        W_r = W_cur.reshape(p, n, R)
                        S = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    else:
                        # S is already computed from previous step?
                        # No, S was updated in the previous r iteration.
                        pass
                    
                    # Compute S_r for current r
                    u_r = U_cur[:, r].reshape(n, m)
                    v_r = V_cur[:, r].reshape(m, p)
                    w_r = W_cur[:, r].reshape(p, n)
                    S_r = u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :] * w_r[:, np.newaxis, :]
                    # Wait, indices:
                    # u_r[i, j]
                    # v_r[j, k]
                    # w_r[k, i]
                    # S_r[i, j, k] = u_r[i, j] * v_r[j, k] * w_r[k, i]
                    # w_r is (p, n). w_r[k, i] means row k, col i.
                    # So S_r[i, j, k] = u_r[i, j] * v_r[j, k] * w_r.T[i, k]? No.
                    # w_r[k, i].
                    # Let's check einsum:
                    # 'ij,jk,ki->ijk'
                    # u: ij, v: jk, w: ki.
                    # Yes.
                    
                    # T_res = T - (S - S_r)
                    T_res = T - S + S_r
                    
                    # Compute u_new
                    # u_new[i, j] = sum_k T_res[i, j, k] * v_r[j, k] * w_r[k, i]
                    # This is a contraction.
                    # u_new = np.einsum('ijk,jk,ki->ij', T_res, v_r, w_r)
                    # Denominator = ||v_r||^2 * ||w_r||^2
                    norm_v = np.sum(v_r**2)
                    norm_w = np.sum(w_r**2)
                    denom = norm_v * norm_w
                    
                    if denom > 1e-9:
                        u_new = u_new / denom
                    else:
                        # If v or w is zero, we can't solve. Keep old u? Or set to 0?
                        # If v or w is zero, the term contributes nothing.
                        # We can set u_new to 0.
                        u_new = np.zeros((n, m))
                    
                    # Update S
                    S = S - S_r + u_new[:, :, np.newaxis] * v_r[np.newaxis, :, :] * w_r[:, np.newaxis, :]
                    
                    # Update U_cur
                    U_cur[:, r] = u_new.flatten()
                
                # Update V
                for r in range(R):
                    u_r = U_cur[:, r].reshape(n, m)
                    v_r = V_cur[:, r].reshape(m, p)
                    w_r = W_cur[:, r].reshape(p, n)
                    S_r = u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :] * w_r[:, np.newaxis, :]
                    
                    T_res = T - S + S_r
                    
                    # v_new[j, k] = sum_i T_res[i, j, k] * u_r[i, j] * w_r[k, i]
                    v_new = np.einsum('ijk,ij,ki->jk', T_res, u_r, w_r)
                    
                    norm_u = np.sum(u_r**2)
                    norm_w = np.sum(w_r**2)
                    denom = norm_u * norm_w
                    
                    if denom > 1e-9:
                        v_new = v_new / denom
                    else:
                        v_new = np.zeros((m, p))
                    
                    S = S - S_r + u_r[:, :, np.newaxis] * v_new[np.newaxis, :, :] * w_r[:, np.newaxis, :]
                    V_cur[:, r] = v_new.flatten()
                
                # Update W
                for r in range(R):
                    u_r = U_cur[:, r].reshape(n, m)
                    v_r = V_cur[:, r].reshape(m, p)
                    w_r = W_cur[:, r].reshape(p, n)
                    S_r = u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :] * w_r[:, np.newaxis, :]
                    
                    T_res = T - S + S_r
                    
                    # w_new[k, i] = sum_j T_res[i, j, k] * u_r[i, j] * v_r[j, k]
                    w_new = np.einsum('ijk,ij,jk->ki', T_res, u_r, v_r)
                    
                    norm_u = np.sum(u_r**2)
                    norm_v = np.sum(v_r**2)
                    denom = norm_u * norm_v
                    
                    if denom > 1e-9:
                        w_new = w_new / denom
                    else:
                        w_new = np.zeros((p, n))
                    
                    S = S - S_r + u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :] * w_new[:, np.newaxis, :]
                    W_cur[:, r] = w_new.flatten()
            
            # After local search, snap to half-integers
            U_snap = np.round(U_cur * 2) / 2
            V_snap = np.round(V_cur * 2) / 2
            W_snap = np.round(W_cur * 2) / 2
            
            # Check error
            U_r = U_snap.reshape(n, m, R)
            V_r = V_snap.reshape(m, p, R)
            W_r = W_snap.reshape(p, n, R)
            S_final = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            error = np.linalg.norm(T - S_final)
            
            if error < 1e-6:
                # Valid decomposition
                # Prune zero columns
                # A column r is zero if U[:, r] is 0 OR V[:, r] is 0 OR W[:, r] is 0?
                # No, the product must be zero.
                # If any of U[:, r], V[:, r], W[:, r] is the zero vector, the term is zero.
                # But if they are non-zero, the term might still be zero?
                # Unlikely with random init.
                # Let's just check if the column contributes to the sum.
                # Simpler: check if U[:, r] is all 0s, or V[:, r] is all 0s, or W[:, r] is all 0s.
                # If so, remove it.
                
                mask = np.ones(R, dtype=bool)
                for r in range(R):
                    if np.all(U_snap[:, r] == 0) or np.all(V_snap[:, r] == 0) or np.all(W_snap[:, r] == 0):
                        mask[r] = False
                
                U_pruned = U_snap[:, mask]
                V_pruned = V_snap[:, mask]
                W_pruned = W_snap[:, mask]
                
                rank = U_pruned.shape[1]
                
                if rank < best_rank:
                    best_rank = rank
                    best_result = {
                        "U": U_pruned.tolist(),
                        "V": V_pruned.tolist(),
                        "W": W_pruned.tolist()
                    }
                    
                    # If we reach a low rank, we might stop early?
                    # But we want minimal rank.
                    # Let's continue to see if we can get lower.
                    
            # End restart loop
        # End R loop
        
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
