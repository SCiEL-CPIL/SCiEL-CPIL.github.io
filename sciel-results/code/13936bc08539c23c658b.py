import time
import random
import numpy as np

DIMENSIONS = [2, 2, 10]

def _build_strassen_35():
    """Return the known rank-35 Strassen decomposition."""
    n, m, p = 2, 2, 10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U[idx][col] = a_vec[idx]
            V[0 * p + 2 * k][col] = b_vec[0]
            V[0 * p + 2 * k + 1][col] = b_vec[1]
            V[1 * p + 2 * k][col] = b_vec[2]
            V[1 * p + 2 * k + 1][col] = b_vec[3]
            W[2 * k * n + 0][col] = c_vec[0]
            W[2 * k * n + 1][col] = c_vec[1]
            W[(2 * k + 1) * n + 0][col] = c_vec[2]
            W[(2 * k + 1) * n + 1][col] = c_vec[3]
            col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n=2, m=2, p=10, tol=1e-6):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    # Check: sum_k U[i,k] * V[j,k] * W[l,k] should equal delta[i,j,l]
    # where i indexes A (n*m), j indexes B (m*p), l indexes C (p*n)
    # C[i, l] = sum_j A[i,j] * B[j, l]
    # In tensor form: T[i,j,l] = 1 if the (i,j) element of A times (j,l) element of B contributes to (i,l) of C
    # Actually T[i,j,l] = delta_{i, i'} * delta_{j, j'} where C[i', l] gets A[i', j'] * B[j', l]
    # So T[i,j,l] = 1 if j is the inner index connecting A[i,j] and B[j,l], which is always true
    # T[i,j,l] = 1 for all valid i,j,l where the multiplication is defined
    # Wait, let me think more carefully.
    
    # A is n x m = 2 x 2, B is m x p = 2 x 10, C is n x p = 2 x 10
    # C[i, l] = sum_{j=0}^{m-1} A[i, j] * B[j, l]
    
    # Tensor T has indices (i, j, l) where:
    # i in [0, n*m) = [0, 4) -- but wait, the tensor for matrix multiplication has 3 indices
    # Actually for matrix multiplication, the tensor is T[i, j, l] = delta_{i, i'} * delta_{j, j'}
    # where the sum is over the middle index.
    
    # Let me re-read the problem. U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # n=2, m=2, p=10
    # U: 4 x rank
    # V: 20 x rank  
    # W: 20 x rank
    
    # The tensor T has 3 indices: (a, b, c) where:
    # a in [0, n*m) = [0, 4): a = i*m + j where i is row of A, j is col of A (= row of B)
    # b in [0, m*p) = [0, 20): b = j*p + l where j is row of B, l is col of B (= col of C)
    # c in [0, p*n) = [0, 20): c = l*n + i where l is col of C, i is row of C
    
    # T[a, b, c] = 1 if a, b, c are consistent, i.e., the middle index matches
    # a = i*m + j, b = j'*p + l, c = l'*n + i'
    # T = 1 if j == j' and l == l' and i == i'
    
    # So T[a, b, c] = 1 iff:
    # (a % m) == (b // p) and (b % p) == (c // n) and (a // m) == (c % n)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # T_recon[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
    # This is a 3-tensor contraction
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For efficiency, compute for each k: outer product of U[:,k], V[:,k], W[:,k]
    # But that's rank * nm * mp * pn which could be large
    # rank=35, nm=4, mp=20, pn=20 -> 35*4*20*20 = 56000, fine
    
    T_recon = np.zeros((nm, mp, pn))
    for k in range(rank):
        uk = U_arr[:, k]
        vk = V_arr[:, k]
        wk = W_arr[:, k]
        T_recon += np.einsum('a,b,c->abc', uk, vk, wk)
    
    # Build the target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a = i * m + j
                b = j * p + l
                c = l * n + i
                T_target[a, b, c] = 1.0
    
    diff = np.abs(T_recon - T_target)
    return np.max(diff) < tol

def _try_rank_reduction(U, V, W, target_rank, n=2, m=2, p=10, max_iters=50, tol=1e-6):
    """Try to reduce rank by finding linear dependencies among columns."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    cur_rank = U_arr.shape[1]
    if cur_rank <= target_rank:
        return U, V, W
    
    # Try to find columns that can be eliminated
    # Strategy: randomly select subsets of columns to try to eliminate
    # by expressing one column as a linear combination of others
    
    best_U, best_V, best_W = U, V, W
    best_r = cur_rank
    
    for attempt in range(max_iters):
        U_try = U_arr.copy()
        V_try = V_arr.copy()
        W_try = W_arr.copy()
        
        r = cur_rank
        # Try to eliminate (cur_rank - target_rank) columns
        to_eliminate = cur_rank - target_rank
        
        success = True
        for elim in range(to_eliminate):
            if r <= target_rank:
                break
            # Pick a random column to try to eliminate
            col_to_elim = random.randint(0, r - 1)
            
            # Try to express column col_to_elim as linear combo of others
            # Using least squares
            if r == 1:
                success = False
                break
            
            # Build the system: U_try[:, col_to_elim] = U_try[:, other_cols] @ x
            mask = [i for i in range(r) if i != col_to_elim]
            A_mat = U_try[:, mask]  # (nm, r-1)
            b_vec = U_try[:, col_to_elim]  # (nm,)
            
            # Least squares
            x, residuals, rank_lsq, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Check if the fit is good
            fit_err = np.linalg.norm(A_mat @ x - b_vec)
            if fit_err > 1e-4:
                continue
            
            # Verify with V and W
            V_fit_err = np.linalg.norm(V_try[:, mask] @ x - V_try[:, col_to_elim])
            W_fit_err = np.linalg.norm(W_try[:, mask] @ x - W_try[:, col_to_elim])
            
            if V_fit_err > 1e-4 or W_fit_err > 1e-4:
                continue
            
            # Eliminate this column
            new_U = np.delete(U_try, col_to_elim, axis=1)
            new_V = np.delete(V_try, col_to_elim, axis=1)
            new_W = np.delete(W_try, col_to_elim, axis=1)
            U_try = new_U
            V_try = new_V
            W_try = new_W
            r -= 1
        
        if r < best_r:
            # Verify the decomposition
            if _verify_decomposition(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p, tol):
                best_U = U_try.tolist()
                best_V = V_try.tolist()
                best_W = W_try.tolist()
                best_r = r
                if best_r <= target_rank:
                    break
    
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin from 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-35 solution
    result = _build_strassen_35()
    best_rank = 35
    
    # Try to reduce rank
    for target_rank in [34, 33, 32]:
        if time.time() > deadline - 10:
            break
        
        U, V, W = result["U"], result["V"], result["W"]
        max_iters = 100 if target_rank == 34 else 50
        
        new_U, new_V, new_W = _try_rank_reduction(U, V, W, target_rank, n, m, p, max_iters)
        
        if len(new_U[0]) < best_rank:
            best_rank = len(new_U[0])
            result = {"U": new_U, "V": new_V, "W": new_W}
    
    return result
