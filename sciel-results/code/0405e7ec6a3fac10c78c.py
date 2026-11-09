"""Direct approach for 3x3 by 3x6 matrix multiplication tensor decomposition.

Strategy:
  1. Build the schoolbook (rank-54) decomposition as a guaranteed fallback.
  2. Attempt to find a lower-rank exact decomposition via structured search:
     a. Use the Kronecker/block structure: 3x3*3x6 = 3 blocks of 3x3*3x2.
     b. For each 3x3*3x2 block, the minimum rank is known to be 8 (from 3x3*3x3=23,
        but 3x3*3x2 is smaller). Actually, let's think more carefully.
     
     The tensor for 3x3 * 3x6 multiplication:
       C[i,k] = sum_j A[i,j] * B[j,k]
     where i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4,5}.
     
     Total terms: 3*3*6 = 54 rank-1 tensors in schoolbook.
     
     Key insight: We can decompose B into 6 columns, each of length 3.
     For each output column k, C[:,k] = A * B[:,k] is a 3x3 * 3x1 product.
     The rank of 3x3 * 3x1 matrix multiplication is 3 (each output row is a dot product).
     So 6 columns * rank 3 = 18? No, that's not right either.
     
     Actually, for 3x3 * 3x1: C[i] = sum_j A[i,j] * B[j,0]. 
     This is 3 separate dot products of length 3. The tensor T[(i,j),(j,k),(k,i)] 
     for k=0 only. T[(i,j),(j,0),(0,i)] = 1. This is rank 3 (one per j, with i as output).
     
     Wait, more carefully: T[a,b,c] where a=(i,j), b=(j',k'), c=(k'',i').
     For 3x3*3x1: b has only j' (no k' dimension, or k'=0), c has only i' (k''=0).
     T[(i,j),(j',0),(0,i')] = 1 iff j=j' and i=i'.
     So T = sum_j e_j ⊗ e_j ⊗ (sum_i e_i ⊗ e_i) ... no.
     T = sum_{i,j} e_{(i,j)} ⊗ e_{(j,0)} ⊗ e_{(0,i)}
     This is a sum of 9 rank-1 tensors, but they might not be independent.
     
     Actually, for the full 3x3*3x6 problem, let me just try a smarter numerical approach.
     
     New strategy: Use the structure that C = A*B where B is 3x6.
     We can write B = [B1 | B2] where B1, B2 are 3x3.
     Then C = A*B1 + A*B2.
     If we have a rank-r decomposition for 3x3*3x3, then A*B1 uses rank r and A*B2 uses rank r.
     Total rank 2r. For r=23 (Laderman), that's 46.
     
     But we can do better by noting that the two products share the same A.
     Specifically, if we use the same "intermediate" computations where possible.
     
     Alternative: Use the fact that 3x3*3x6 can be viewed as computing 3 independent
     1x3 * 3x6 products (one per row of A). Each 1x3*3x6 product:
       C[k] = sum_j a_j * b_jk
     This is 6 dot products of length 3. The tensor for this:
       T[j, (j',k), k] = 1 iff j=j'.
     This tensor has rank 3 (sum_j e_j ⊗ e_j ⊗ ones_6).
     
     So for 3 rows: total rank = 3 * 3 = 9? But that would mean we can compute
     A*B with rank 9, which seems too good. Let me check.
     
     For row i of A: C[i,k] = sum_j A[i,j] * B[j,k].
     The tensor for this row: T[(i,j), (j',k), (k,i)] = 1 iff j=j'.
     For fixed i, this is sum_j e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}.
     
     Wait, the output index c = k*n + i = 3k + i. So for fixed i, the output indices
     are {i, i+3, i+6} (k=0,1,2). No wait, k goes from 0 to 5, so c = 3k+i.
     For i=0: c in {0,3,6,9,12,15}. For i=1: c in {1,4,7,10,13,16}. For i=2: c in {2,5,8,11,14,17}.
     
     So for fixed i, the output vector has 6 components. The tensor for row i:
     T[(i,j), (j',k), (3k+i)] = 1 iff j=j'.
     = sum_j e_{(i,j)} ⊗ (sum_k e_{(j,k)} ⊗ e_{(3k+i)})
     = sum_j e_{(i,j)} ⊗ v_{j,i}
     where v_{j,i} = sum_k e_{(j,k)} ⊗ e_{(3k+i)}.
     
     This is rank 3 for each i (one term per j). Total rank = 3*3 = 9.
     
     But wait, this would mean the entire 3x3*3x6 multiplication has rank 9, which
     contradicts the lower bound of max(9,18,18) = 18. Let me re-examine.
     
     The issue is that v_{j,i} for different i might not be independent. Let's check
     if the 9 vectors {e_{(i,j)} ⊗ v_{j,i}} for i,j in {0,1,2} are linearly independent.
     
     Actually, I think the error is that the output index c = k*n + i means that
     different (i,k) pairs give different c values, so the outputs are all distinct.
     The tensor is:
     T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(3k+i) for some i,j,k.
     
     This is a sum of 3*3*6 = 54 rank-1 tensors: sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(3k+i)}.
     
     The question is whether these 54 tensors are linearly independent. If they are,
     rank = 54. If not, we can find a lower-rank representation.
     
     Given the structure, I suspect the rank might be lower. Let me try a numerical
     approach with better initialization and more aggressive search.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
TIME_LIMIT = 145  # safety margin before 150s kill


def _build_schoolbook(n=3, m=3, p=6):
    """Return (U, V, W) for the schoolbook decomposition, rank = n*m*p."""
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


def _tensor_from_factors(U, V, W):
    """Reconstruct the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    a_dim = U.shape[0]
    b_dim = V.shape[0]
    c_dim = W.shape[0]
    r = U.shape[1]
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.float64)
    for r_idx in range(r):
        T += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
    return T


def _target_tensor(n=3, m=3, p=6):
    """Build the target multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    T = np.zeros((a_dim, b_dim, c_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, target, tol=1e-10):
    """Check that the factorisation reproduces the target tensor exactly."""
    T = _tensor_from_factors(U, V, W)
    return np.allclose(T, target, atol=tol, rtol=tol)


def _try_structured_decomposition(n=3, m=3, p=6):
    """
    Try to construct a lower-rank decomposition using the block structure.
    
    For 3x3 * 3x6, view B as [B1 | B2] where B1, B2 are 3x3.
    C = A*B1 + A*B2.
    
    Use the Laderman decomposition for 3x3*3x3 (rank 23) for each block.
    This gives rank 46. We can potentially share some terms if the
    intermediate values overlap.
    
    Actually, a simpler approach: since the two products share A, we can
    compute A*B1 and A*B2 separately, each with rank 23, total rank 46.
    
    But let's try to do better. The key observation is that for each row i of A,
    C[i,:] = A[i,:] * B is a 1x3 * 3x6 product. This has rank 3 (as argued above).
    So total rank = 3 * 3 = 9? But this seems too optimistic.
    
    Let me just try a numerical search with better methods.
    """
    return None


def _als_step(U, V, W, target):
    """Perform one ALS step: solve for U, V, W in sequence."""
    a_dim = U.shape[0]
    b_dim = V.shape[0]
    c_dim = W.shape[0]
    r = U.shape[1]
    
    # Solve for U: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed V,W: sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
    # Build matrix B of shape (b_dim*c_dim, r): B[(b,c), r] = V[b,r]*W[c,r]
    B = np.zeros((b_dim * c_dim, r))
    for b_idx in range(b_dim):
        for c_idx in range(c_dim):
            B[b_idx * c_dim + c_idx, :] = V[b_idx, :] * W[c_idx, :]
    
    # T_flat of shape (a_dim, b_dim*c_dim)
    T_flat = target.reshape(a_dim, b_dim * c_dim)
    
    # Solve: T_flat[a,:] = U[a,:] @ B
    # So: U[a,:] = T_flat[a,:] @ B^+ (pseudo-inverse)
    try:
        U_new = T_flat @ np.linalg.pinv(B)
    except:
        U_new = U
    
    # Solve for V: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed U,W: sum_r V[b,r] * (U[a,r]*W[c,r]) = T[a,b,c]
    # Build matrix C of shape (a_dim*c_dim, r): C[(a,c), r] = U[a,r]*W[c,r]
    C = np.zeros((a_dim * c_dim, r))
    for a_idx in range(a_dim):
        for c_idx in range(c_dim):
            C[a_idx * c_dim + c_idx, :] = U_new[a_idx, :] * W[c_idx, :]
    
    # T_flat_V of shape (b_dim, a_dim*c_dim)
    T_flat_V = np.zeros((b_dim, a_dim * c_dim))
    for b_idx in range(b_dim):
        T_flat_V[b_idx, :] = target[:, b_idx, :].reshape(-1)
    
    try:
        V_new = T_flat_V @ np.linalg.pinv(C)
    except:
        V_new = V
    
    # Solve for W: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed U,V: sum_r W[c,r] * (U[a,r]*V[b,r]) = T[a,b,c]
    # Build matrix D of shape (a_dim*b_dim, r): D[(a,b), r] = U[a,r]*V[b,r]
    D = np.zeros((a_dim * b_dim, r))
    for a_idx in range(a_dim):
        for b_idx in range(b_dim):
            D[a_idx * b_dim + b_idx, :] = U_new[a_idx, :] * V_new[b_idx, :]
    
    # T_flat_W of shape (c_dim, a_dim*b_dim)
    T_flat_W = np.zeros((c_dim, a_dim * b_dim))
    for c_idx in range(c_dim):
        T_flat_W[c_idx, :] = target[:, :, c_idx].reshape(-1)
    
    try:
        W_new = T_flat_W @ np.linalg.pinv(D)
    except:
        W_new = W
    
    return U_new, V_new, W_new


def _snap_to_exact(U, V, W, target, tol=1e-8):
    """Try to snap values to integers or half-integers if close."""
    U_snap = U.copy()
    V_snap = V.copy()
    W_snap = W.copy()
    
    # Try snapping to nearest half-integer
    U_snap = np.round(U_snap * 2) / 2
    V_snap = np.round(V_snap * 2) / 2
    W_snap = np.round(W_snap * 2) / 2
    
    if _verify(U_snap, V_snap, W_snap, target, tol=tol):
        return U_snap, V_snap, W_snap
    
    # Try snapping to nearest integer
    U_snap = np.round(U_snap)
    V_snap = np.round(V_snap)
    W_snap = np.round(W_snap)
    
    if _verify(U_snap, V_snap, W_snap, target, tol=tol):
        return U_snap, V_snap, W_snap
    
    # Try mixed: some half-integers, some integers
    # This is complex, so just return the snapped version if it's close
    T_check = _tensor_from_factors(U_snap, V_snap, W_snap)
    if np.allclose(T_check, target, atol=tol, rtol=tol):
        return U_snap, V_snap, W_snap
    
    return U, V, W


def _numerical_search(n=3, m=3, p=6, time_limit=140):
    """
    Try to find a lower-rank decomposition using ALS optimization.
    """
    start_time = time.time()
    target = _target_tensor(n, m, p)
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 54
    
    # Schoolbook baseline
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    if _verify(U_fb, V_fb, W_fb, target):
        best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()
        best_rank = n * m * p
    
    # Try ranks from 40 down to 18
    for r in range(40, 17, -1):
        if time.time() - start_time > time_limit:
            break
        if r >= best_rank:
            continue
        
        success = False
        # Multiple initializations
        n_inits = 5
        for init in range(n_inits):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize
            if init == 0:
                # Structured initialization: use schoolbook-like pattern
                U_init = np.zeros((n * m, r))
                V_init = np.zeros((m * p, r))
                W_init = np.zeros((p * n, r))
                # For each (i,j,k), assign a column
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < r:
                                U_init[i * m + j, col] = 1.0
                                V_init[j * p + k, col] = 1.0
                                W_init[k * n + i, col] = 1.0
                            col += 1
            elif init == 1:
                # Perturbed schoolbook
                U_init, V_init, W_init = _build_schoolbook(n, m, p)
                U_init = U_init[:, :r] + 0.1 * np.random.randn(n * m, r)
                V_init = V_init[:, :r] + 0.1 * np.random.randn(m * p, r)
                W_init = W_init[:, :r] + 0.1 * np.random.randn(p * n, r)
            else:
                # Random initialization with varying scales
                scale = 0.5 + 0.5 * init / n_inits
                U_init = scale * np.random.randn(n * m, r)
                V_init = scale * np.random.randn(m * p, r)
                W_init = scale * np.random.randn(p * n, r)
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            # ALS iterations
            max_als_iters = 100
            prev_loss = np.inf
            for it in range(max_als_iters):
                if time.time() - start_time > time_limit:
                    break
                
                U_cur, V_cur, W_cur = _als_step(U_cur, V_cur, W_cur, target)
                
                # Check convergence
                T_cur = _tensor_from_factors(U_cur, V_cur, W_cur)
                loss = np.sum((T_cur - target) ** 2)
                
                if loss < 1e-12:
                    break
                
                # Check for stagnation
                if abs(prev_loss - loss) < 1e-15:
                    break
                prev_loss = loss
            
            T_final = _tensor_from_factors(U_cur, V_cur, W_cur)
            final_loss = np.sum((T_final - target) ** 2)
            
            if final_loss < 1e-9 and _verify(U_cur, V_cur, W_cur, target, tol=1e-9):
                # Final polish: do more ALS iterations with tighter tolerance
                for polish_it in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    U_cur, V_cur, W_cur = _als_step(U_cur, V_cur, W_cur, target)
                    T_check = _tensor_from_factors(U_cur, V_cur, W_cur)
                    if np.sum((T_check - target) ** 2) < 1e-14:
                        break
                
                # Try to snap to exact values
                U_snap, V_snap, W_snap = _snap_to_exact(U_cur, V_cur, W_cur, target)
                best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                best_rank = r
                success = True
                break
        
        if success:
            break
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build fallback
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    
    # Try numerical search for lower rank
    try:
        U_opt, V_opt, W_opt, opt_rank = _numerical_search(n, m, p, time_limit=TIME_LIMIT - 5)
        if U_opt is not None and opt_rank < n * m * p:
            return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    except Exception:
        pass
    
    # Return schoolbook fallback
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


# EVOLVE-BLOCK-START
# (construct is already defined above; this block is kept for interface compatibility)
# EVOLVE-BLOCK-END
