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
     
     Wait, the output index c = k*n + i means that
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


def _numerical_search(n=3, m=3, p=6, time_limit=140):
    """
    Try to find a lower-rank decomposition using numerical optimisation with
    multiple initializations and adaptive learning rates.
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
        # Multiple random initializations
        for init in range(3):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random values
            scale = 0.5
            U_init = scale * np.random.randn(n * m, r)
            V_init = scale * np.random.randn(m * p, r)
            W_init = scale * np.random.randn(p * n, r)
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            lr = 0.05
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                T_cur = _tensor_from_factors(U_cur, V_cur, W_cur)
                loss = np.sum((T_cur - target) ** 2)
                
                if loss < 1e-12:
                    break
                
                diff = T_cur - target
                gU = np.einsum('abc,br,cr->ar', diff, V_cur, W_cur)
                gV = np.einsum('abc,ar,cr->br', diff, U_cur, W_cur)
                gW = np.einsum('abc,ar,br->cr', diff, U_cur, V_cur)
                
                U_cur -= lr * gU
                V_cur -= lr * gV
                W_cur -= lr * gW
                
                if it % 25 == 0 and it > 0:
                    new_loss = np.sum((_tensor_from_factors(U_cur, V_cur, W_cur) - target) ** 2)
                    if new_loss > loss * 1.05:
                        lr *= 0.5
                    elif new_loss < loss * 0.95:
                        lr = min(lr * 1.2, 0.5)
            
            T_final = _tensor_from_factors(U_cur, V_cur, W_cur)
            final_loss = np.sum((T_final - target) ** 2)
            
            if final_loss < 1e-9 and _verify(U_cur, V_cur, W_cur, target, tol=1e-9):
                best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
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
