"""Matrix multiplication 2x2x10: rank-minimal exact tensor decomposition.

Strategy: Use the Kronecker product structure of the 2x2x10 multiplication tensor
as 2x2x2 ⊗ (1x1x5 identity). Apply Laderman's rank-7 decomposition to the 2x2x2
component, yielding a rank-35 baseline. Then attempt to reduce rank by finding
linear combinations of the 35 terms that can be merged, using structured
alternating least squares with multiple restarts and a high-precision polish pass.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 150.0
_SAFETY_MARGIN = 15.0


def _build_laderman_baseline():
    """Build the rank-35 Laderman-based decomposition.
    
    The 2x2x10 multiplication can be viewed as 5 independent 2x2x2 multiplications
    (since the 2x10 matrix has 5 disjoint 2-column blocks, each multiplying with
    the 2x2 matrix to give a 2x2 output block).
    
    Actually, let me reconsider. C[i,k] = sum_j A[i,j] * B[j,k].
    For fixed i, C[i,k] depends on all B[j,k] for j=0,1 and all A[i,j].
    The tensor T[a_idx, b_idx, c_idx] = 1 where:
      a_idx = i*m + j = 2*i + j (for A[i,j])
      b_idx = j*p + k = 10*j + k (for B[j,k])
      c_idx = k*n + i = 2*k + i (for C[i,k])
    Non-zero when: T[2*i+j, 10*j+k, 2*k+i] = 1 for i in {0,1}, j in {0,1}, k in {0,...,9}
    
    That's 40 non-zero entries.
    
    The structure: for each k, we have a 2x2x2 sub-tensor (over i, j, and the k-index).
    But k appears in both B and C indices, so it's not simply 5 independent 2x2x2 blocks.
    
    Let me think again. For fixed k:
      T[2*i+j, 10*j+k, 2*k+i] = 1
    The B index is 10*j+k, which for fixed k varies over j=0,1 giving 10*0+k=k and 10*1+k=10+k.
    The C index is 2*k+i, which for fixed k varies over i=0,1 giving 2*k and 2*k+1.
    
    So for each k, we have a 2x2x2 tensor over (i, j) for A, (j) for B (but B has two entries per k),
    and (i) for C. This is exactly a 2x2x2 matrix multiplication tensor!
    
    Specifically, for each k:
      A_sub = A (2x2), B_sub = column k of B (2x1), C_sub = column k of C (2x1)
      C_sub[i] = sum_j A[i,j] * B_sub[j]
    
    So we have 10 independent 2x2x1 multiplications, not 5 blocks of 2x2x2.
    
    Wait, but the tensor structure T[a,b,c] = 1 when a=2i+j, b=10j+k, c=2k+i.
    For each (i,j,k), there's one non-zero entry. That's 40 entries.
    
    The 2x2x2 matrix multiplication tensor has 8 non-zero entries (for a 2x2 * 2x2 = 2x2).
    Here we have 40 = 5 * 8, suggesting 5 copies of the 2x2x2 tensor.
    
    Let me verify: if we group by k mod 2:
    - For k even (k=0,2,4,6,8): c = 2k+i, so c is even (2k) or odd (2k+1)
    - For k odd (k=1,3,5,7,9): c = 2k+i, so c is even (2k) or odd (2k+1)
    
    Actually, let me just build it directly and use Laderman on each 2x2x2 block.
    
    The key insight: T can be written as sum over k of T_k, where T_k is the tensor
    for the k-th column of the multiplication. Each T_k has 4 non-zero entries:
      T_k[2*i+j, 10*j+k, 2*k+i] = 1 for i,j in {0,1}
    
    This is a 2x2x2 tensor (over i, j, and the "output index" which is i for fixed k).
    It's exactly the 2x2x1 matrix-vector multiplication tensor, which has rank 4 (schoolbook)
    or can be done with rank 3? No, 2x2x1 matrix-vector mult has rank 4.
    
    Actually, the 2x2x2 matrix multiplication tensor (2x2 * 2x2 = 2x2) has rank 7 (Laderman).
    The 2x2x1 matrix-vector multiplication (2x2 * 2x1 = 2x1) has rank 4.
    
    So 10 * 4 = 40 for schoolbook, and we want to do better.
    
    Let me just build the rank-35 baseline using Strassen on 2x2x2 blocks properly.
    """
    n, m, p = 2, 2, 10
    # We'll use the Strassen decomposition applied to 2x2x2 sub-problems
    # The 2x10 matrix can be split into 5 blocks of 2x2
    # Each 2x2 * 2x2 = 2x2 multiplication uses rank 7 (Strassen)
    # Total rank = 5 * 7 = 35
    
    num_blocks = p // 2  # 5
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen's 7 products for 2x2 * 2x2 = 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For block b (columns 2b, 2b+1 of B, and 2b, 2b+1 of C):
    # A is the same 2x2 matrix
    # B_block = B[:, 2b:2b+2]
    # C_block = C[:, 2b:2b+2]
    
    strassen_terms = [
        # (U coefficients for A, V coefficients for B_block, W coefficients for C_block)
        # M1 = (a+d)(e+h): U has a=1,d=1; V has e=1,h=1; W contributes to C11,C22
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (c+d)e: U has c=1,d=1; V has e=1; W contributes to C21,C22
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = a(f-h): U has a=1; V has f=1,h=-1; W contributes to C12,C22
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = d(g-e): U has d=1; V has g=1,e=-1; W contributes to C11,C21
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (a+b)h: U has a=1,b=1; V has h=1; W contributes to C11,C12
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
        # M6 = (c-a)(e+f): U has c=1,a=-1; V has e=1,f=1; W contributes to C22
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (b-d)(g+h): U has b=1,d=-1; V has g=1,h=1; W contributes to C11
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_terms[t]
            for u_row, u_coeff in u_side:
                U[u_row][col] = u_coeff
            for v_koff, v_coeff in v_side:
                # v_koff: 0=e(B[0,2b]), 1=f(B[0,2b+1]), 2=g(B[1,2b]), 3=h(B[1,2b+1])
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k][col] = v_coeff
            for w_koff, w_coeff in w_side:
                # w_koff: 0=C[0,2b], 1=C[0,2b+1], 2=C[1,2b], 3=C[1,2b+1]
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i][col] = w_coeff
    
    return U, V, W, rank


def _verify_decomposition(U, V, W, rank, n, m, p, num_tests=5):
    """Verify the decomposition by testing with random matrices."""
    rng = np.random.default_rng(12345)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(num_tests):
        A_test = rng.integers(-5, 6, size=(n, m)).astype(float)
        B_test = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        C_computed = np.zeros(p * n)
        for r in range(rank):
            uA = np.dot(U_arr[:, r], A_flat)
            vB = np.dot(V_arr[:, r], B_flat)
            C_computed += uA * vB * W_arr[:, r]
        
        # Reorder C_expected to match W's convention: W[k*n+i, r]
        C_expected_reorder = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_expected_reorder[k * n + i] = C_expected[i, k]
        
        if not np.allclose(C_computed, C_expected_reorder, atol=1e-9):
            return False
    
    return True


def _build_tensor(n, m, p):
    """Build the multiplication tensor T[a_idx, b_idx, c_idx]."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T


def _tensor_error(T, U, V, W):
    """Compute the relative error of the tensor reconstruction."""
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    err = np.linalg.norm(T_recon - T)
    norm_T = np.linalg.norm(T)
    if norm_T < 1e-15:
        return err
    return err / norm_T


def _als_step(T, U, V, W, target_rank, n, m, p):
    """One step of alternating least squares."""
    # Update U
    C_mat = np.einsum('jr,kr->rjk', V, W)  # (r, m*p, p*n)
    C_flat = C_mat.reshape(target_rank, -1)  # (r, m*p*p*n)
    T_vec = T.reshape(n * m, -1)
    try:
        U_new = T_vec @ C_flat.T @ np.linalg.pinv(C_flat @ C_flat.T)
        U = U_new
    except:
        pass
    
    # Update V
    C_mat = np.einsum('ir,kr->rik', U, W)  # (r, n*m, p*n)
    C_flat = C_mat.reshape(target_rank, -1)
    T_trans = T.transpose(1, 0, 2).reshape(m * p, -1)
    try:
        V_new = T_trans @ C_flat.T @ np.linalg.pinv(C_flat @ C_flat.T)
        V = V_new.reshape(m * p, target_rank)
    except:
        pass
    
    # Update W
    C_mat = np.einsum('ir,jr->rij', U, V)  # (r, n*m, m*p)
    C_flat = C_mat.reshape(target_rank, -1)
    T_trans = T.transpose(2, 0, 1).reshape(p * n, -1)
    try:
        W_new = T_trans @ C_flat.T @ np.linalg.pinv(C_flat @ C_flat.T)
        W = W_new.reshape(p * n, target_rank)
    except:
        pass
    
    return U, V, W


def _polish(T, U, V, W, target_rank, n, m, p, max_iter=200, tol=1e-14):
    """High-precision polish pass using more iterations and tighter tolerance."""
    for _ in range(max_iter):
        U, V, W = _als_step(T, U, V, W, target_rank, n, m, p)
        err = _tensor_error(T, U, V, W)
        if err < tol:
            break
    return U, V, W, err


def _try_rank_reduction(T, U_init, V_init, W_init, target_rank, n, m, p, 
                        start_time, deadline, max_restarts=3):
    """Try to find a rank-target_rank decomposition starting from initial factors."""
    for restart in range(max_restarts):
        if time.time() - start_time > deadline:
            break
        
        if restart == 0:
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
        else:
            # Perturb the initial solution
            rng = np.random.default_rng(100 + restart * 7)
            U_curr = U_init + rng.normal(scale=0.1, size=U_init.shape)
            V_curr = V_init + rng.normal(scale=0.1, size=V_init.shape)
            W_curr = W_init + rng.normal(scale=0.1, size=W_init.shape)
        
        prev_err = _tensor_error(T, U_curr, V_curr, W_curr)
        
        for iteration in range(150):
            if time.time() - start_time > deadline:
                break
            
            U_curr, V_curr, W_curr = _als_step(T, U_curr, V_curr, W_curr, target_rank, n, m, p)
            err = _tensor_error(T, U_curr, V_curr, W_curr)
            
            if err < 1e-12:
                # Polish for high precision
                U_curr, V_curr, W_curr, final_err = _polish(
                    T, U_curr, V_curr, W_curr, target_rank, n, m, p,
                    max_iter=100, tol=1e-15
                )
                if final_err < 1e-10:
                    return U_curr, V_curr, W_curr, final_err
                break
            
            if abs(prev_err - err) < 1e-15 and iteration > 30:
                # Stagnant
                break
            
            prev_err = err
        
        # Check if we got close enough
        err = _tensor_error(T, U_curr, V_curr, W_curr)
        if err < 1e-8:
            U_curr, V_curr, W_curr, final_err = _polish(
                T, U_curr, V_curr, W_curr, target_rank, n, m, p,
                max_iter=100, tol=1e-15
            )
            if final_err < 1e-10:
                return U_curr, V_curr, W_curr, final_err
    
    return None, None, None, None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY_MARGIN
    
    # Start with the safe rank-35 baseline
    U_base, V_base, W_base, rank_base = _build_laderman_baseline()
    
    # Verify baseline
    if not _verify_decomposition(U_base, V_base, W_base, rank_base, n, m, p):
        # Should not happen, but fall back to schoolbook if needed
        U = [[0] * (n * m * p) for _ in range(n * m)]
        V = [[0] * (n * m * p) for _ in range(m * p)]
        W = [[0] * (n * m * p) for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W, best_rank = U_base, V_base, W_base, rank_base
    
    # Try to find lower-rank decompositions
    T = _build_tensor(n, m, p)
    
    # Convert baseline to numpy arrays for use as initialization
    U_base_arr = np.array(U_base, dtype=float)
    V_base_arr = np.array(V_base, dtype=float)
    W_base_arr = np.array(W_base, dtype=float)
    
    for target_rank in [34, 33, 32, 31, 30]:
        if time.time() - start_time > deadline:
            break
        
        # Try to reduce from the rank-35 baseline by zeroing out one term
        # and re-optimizing
        found = False
        
        for term_to_remove in range(rank_base):
            if time.time() - start_time > deadline:
                break
            if found:
                break
            
            # Initialize by removing one term from the baseline
            U_init = U_base_arr[:, term_to_remove].copy()
            V_init = V_base_arr[:, term_to_remove].copy()
            W_init = W_base_arr[:, term_to_remove].copy()
            
            # Create initial factors of size target_rank by removing one column
            U_init_full = np.delete(U_base_arr, term_to_remove, axis=1)
            V_init_full = np.delete(V_base_arr, term_to_remove, axis=1)
            W_init_full = np.delete(W_base_arr, term_to_remove, axis=1)
            
            # But we need rank target_rank, so we can't just delete one column
            # Instead, try to find a rank-target_rank decomposition
            # by starting from the baseline and projecting
            
            # Approach: start with random initialization biased toward the baseline
            rng = np.random.default_rng(42 + target_rank + term_to_remove)
            
            # Scale the baseline down slightly and add noise
            scale = (target_rank / rank_base) ** 0.5
            U_curr = U_base_arr * scale + rng.normal(scale=0.05, size=(n * m, target_rank))
            V_curr = V_base_arr * scale + rng.normal(scale=0.05, size=(m * p, target_rank))
            W_curr = W_base_arr * scale + rng.normal(scale=0.05, size=(p * n, target_rank))
            
            # Truncate to target_rank columns
            U_curr = U_curr[:, :target_rank]
            V_curr = V_curr[:, :target_rank]
            W_curr = W_curr[:, :target_rank]
            
            prev_err = _tensor_error(T, U_curr, V_curr, W_curr)
            
            for iteration in range(100):
                if time.time() - start_time > deadline:
                    break
                
                U_curr, V_curr, W_curr = _als_step(T, U_curr, V_curr, W_curr, target_rank, n, m, p)
                err = _tensor_error(T, U_curr, V_curr, W_curr)
                
                if err < 1e-12:
                    # Polish
                    U_curr, V_curr, W_curr, final_err = _polish(
                        T, U_curr, V_curr, W_curr, target_rank, n, m, p,
                        max_iter=100, tol=1e-15
                    )
                    if final_err < 1e-10:
                        U_list = U_curr.tolist()
                        V_list = V_curr.tolist()
                        W_list = W_curr.tolist()
                        if _verify_decomposition(U_list, V_list, W_list, target_rank, n, m, p):
                            best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                            found = True
                            break
                    break
                
                if abs(prev_err - err) < 1e-15 and iteration > 20:
                    break
                
                prev_err = err
        
        if found:
            break
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, best_rank, n, m, p):
        # Fall back to baseline
        best_U, best_V, best_W, best_rank = U_base, V_base, W_base, rank_base
    
    return {"U": best_U, "V": best_V, "W": best_W}
