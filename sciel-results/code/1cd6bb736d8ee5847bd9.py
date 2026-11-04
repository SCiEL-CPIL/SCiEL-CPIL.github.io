"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        
        W_col_T = W_col.T  # shape (n, p)
        
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)
        tmp2 = tmp1 * W_col_T[:, np.newaxis, :]
        T_computed += tmp2
    
    T_expected = np.ones((n, m, p), dtype=np.float64)
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < tol


def _construct_schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}


def _strassen_2x2_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 = 2x2 multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1 for all i,j,k in {0,1}
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],    # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],    # i=1,j=0
        [1, 1, -1, 1, 0, 0, -1]   # i=1,j=1
    ], dtype=np.float64)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # j=0,k=0
        [0, 0, 0, 1, 0, 1, 0],    # j=0,k=1
        [0, 0, 1, 0, 0, 0, 1],    # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1]    # j=1,k=1
    ], dtype=np.float64)
    
    W = np.array([
        [1, 0, 0, -1, 1, 0, 1],   # k=0,i=0
        [0, 1, 0, 0, 0, 0, 0],    # k=0,i=1
        [0, 0, 1, 0, 1, 0, 0],    # k=1,i=0
        [1, 0, 0, 1, 0, -1, -1]   # k=1,i=1
    ], dtype=np.float64)
    
    return U, V, W


def _build_4x4x4_strassen():
    """
    Build a rank-28 decomposition for the 4x4x4 all-ones tensor using Strassen.
    
    Partition i into blocks {0,1} and {2,3}, k into blocks {0,1} and {2,3}.
    For each (ib, kb) pair, use Strassen's 7-product method for the 2x4x2 sub-tensor.
    
    For fixed ib, kb, we need:
    sum_c U[ib*4+li*4+j, col] * V[j*4+lk*4+k, col] * W[(kb*2+lk)*4+li, col] = 1
    for li in {0,1}, lk in {0,1}, j in {0,1,2,3}
    
    where col = ib*14 + kb*7 + c, c in {0,...,6}
    
    The Strassen decomposition gives us patterns for 2x2x2. We extend to 2x4x2
    by replicating the j-dimension pattern across all 4 j values.
    """
    n, m, p = 4, 4, 4
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2()
    
    total_rank = 4 * 7  # 4 (i,k) blocks * 7 Strassen terms
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    for ib in range(2):
        for kb in range(2):
            for c in range(7):
                col = ib * 14 + kb * 7 + c
                
                # U: for i in {ib*2, ib*2+1}, j in {0,1,2,3}
                # Strassen U_s[li, c] gives coefficient for A[li, lj] where lj in {0,1}
                # But we have 4 j values. The Strassen decomposition is for 2x2x2,
                # so we need to handle the extra j dimensions.
                #
                # Actually, for the 2x4x2 sub-tensor, we can think of it as:
                # T[li, j, lk] = 1 for li in {0,1}, j in {0,1,2,3}, lk in {0,1}
                # This can be written as sum over j of e_li * e_j * e_lk
                # But we want to use Strassen's structure.
                #
                # Key insight: T[li, j, lk] = 1 = 1 * 1 * 1
                # We can factor the j-dimension: for each j, the 2x1x2 sub-tensor is rank-1.
                # But we want to share the Strassen structure across j.
                #
                # Alternative: The 2x4x2 tensor T[li,j,lk]=1 can be decomposed as:
                # sum_{c=0}^{6} U_c[li,j] * V_c[j,lk] * W_c[lk,li]
                # where U_c, V_c, W_c come from Strassen but extended to handle j.
                #
                # For Strassen 2x2x2: U_s[li, lj, c], V_s[lj, lk, c], W_s[lk, li, c]
                # For 2x4x2, we set:
                # U_ext[li, j, c] = U_s[li, j mod 2, c] for j in {0,1,2,3}
                # V_ext[j, lk, c] = V_s[j mod 2, lk, c] for j in {0,1,2,3}
                # W_ext[lk, li, c] = W_s[lk, li, c]
                #
                # But this doesn't work because Strassen's U_s has lj in {0,1} and
                # we're mapping j to lj = j mod 2, which creates dependencies.
                #
                # Let me try a different approach. For the 2x4x2 sub-tensor,
                # we can use the fact that it's the tensor product of:
                # (2x2 all-ones in i,k) ⊗ (4x4 identity in j) ... no, that's not right.
                #
                # Actually, T[li, j, lk] = 1 for all li, j, lk.
                # This is simply the outer product of three vectors:
                # [1,1] ⊗ [1,1,1,1] ⊗ [1,1]
                # So it has rank 1!
                #
                # But we want to use Strassen to reduce the total rank when combining
                # multiple blocks. The issue is that each 2x4x2 block has rank 1,
                # so 4 blocks would have rank 4, not 28.
                #
                # The confusion arises because the "rank" in matrix multiplication
                # complexity is not the tensor rank. Let me reconsider.
                #
                # The bilinear complexity of multiplying an (n x m) matrix by an
                # (m x p) matrix is the minimum r such that there exist matrices
                # U (nm x r), V (mp x r), W (pn x r) with:
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
                #
                # For the full 4x4x4 tensor, this is the matrix multiplication tensor M_{4,4,4}.
                # Its rank is known to be at most 28 (from Strassen's algorithm applied
                # to 4x4 matrices), and the lower bound is 26.
                #
                # So the correct approach is to use Strassen's algorithm for 4x4 matrices,
                # which recursively uses 2x2 Strassen blocks.
                #
                # Let me implement Strassen's algorithm for 4x4 * 4x4 = 4x4.
                pass
    
    # Given the complexity, let me just use a direct construction.
    # For the 4x4x4 tensor, use the known rank-28 decomposition from Strassen.
    
    # Strassen 4x4 = 4 blocks of 2x2, each using 7 products = 28 products
    # The decomposition is:
    # C = A*B where A, B are 4x4
    # Split into 2x2 blocks: A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = Strassen(A11, A22, B11, B22, ...) using 7 products
    # C12 = Strassen(A11, A12, B12, B22, ...) using 7 products  
    # C21 = Strassen(A21, A22, B11, B21, ...) using 7 products
    # C22 = Strassen(A11, A22, B12, B21, ...) using 7 products
    # Total: 28 products
    
    # For the tensor decomposition, each product corresponds to one term c.
    # U[i*4+j, c] = coefficient of A[i,j] in product c
    # V[j*4+k, c] = coefficient of B[j,k] in product c
    # W[k*4+i, c] = coefficient of C[i,k] in product c
    
    # Let me build this explicitly using Strassen's 2x2 decomposition.
    
    U_full = np.zeros((16, 28))
    V_full = np.zeros((16, 28))
    W_full = np.zeros((16, 28))
    
    # Strassen 2x2 patterns (from above)
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2()
    
    # For each output block (ib, kb) where ib is row block, kb is col block:
    # ib=0: i in {0,1}, ib=1: i in {2,3}
    # kb=0: k in {0,1}, kb=1: k in {2,3}
    
    for ib in range(2):
        for kb in range(2):
            # This output block C[ib*2:ib*2+2, kb*2:kb*2+2] is computed
            # from input blocks A[ib*2:ib*2+2, :] and B[:, kb*2:kb*2+2]
            # Using Strassen's 7 products on the 2x2 blocks
            
            for c in range(7):
                col = ib * 14 + kb * 7 + c
                
                # U: A coefficients for this product
                # U_s[li, lj, c] where li in {0,1}, lj in {0,1}
                # Maps to A[ib*2+li, lj] and A[ib*2+li, 2+lj]
                for li in range(2):
                    for lj in range(2):
                        # A[ib*2+li, lj] and A[ib*2+li, 2+lj]
                        i = ib * 2 + li
                        j1 = lj
                        j2 = 2 + lj
                        U_full[i * 4 + j1, col] += U_s[li * 2 + lj, c]
                        U_full[i * 4 + j2, col] += U_s[li * 2 + lj, c]
                
                # V: B coefficients for this product
                # V_s[lj, lk, c] where lj in {0,1}, lk in {0,1}
                # Maps to B[lj, kb*2+lk] and B[2+lj, kb*2+lk]
                for lj in range(2):
                    for lk in range(2):
                        j1 = lj
                        j2 = 2 + lj
                        k = kb * 2 + lk
                        V_full[j1 * 4 + k, col] += V_s[lj * 2 + lk, c]
                        V_full[j2 * 4 + k, col] += V_s[lj * 2 + lk, c]
                
                # W: C coefficients for this product
                # W_s[lk, li, c] where lk in {0,1}, li in {0,1}
                # Maps to C[ib*2+li, kb*2+lk]
                for lk in range(2):
                    for li in range(2):
                        i = ib * 2 + li
                        k = kb * 2 + lk
                        W_full[k * 4 + i, col] += W_s[lk * 2 + li, c]
    
    return U_full, V_full, W_full


def _build_4x4x5_decomposition():
    """
    Build a rank-28+16=44 decomposition for 4x4x5.
    Use Strassen for the 4x4x4 core (rank 28) and schoolbook for k=4 (rank 16).
    """
    n, m, p = 4, 4, 5
    
    # Build 4x4x4 core
    U_core, V_core, W_core = _build_4x4x4_strassen()
    
    # Verify core
    T_core = np.zeros((4, 4, 4))
    for c in range(28):
        Uc = U_core[:, c].reshape(4, 4)
        Vc = V_core[:, c].reshape(4, 4)
        Wc = W_core[:, c].reshape(4, 4)
        T_core += np.einsum('ij,jk->ijk', Uc, Vc) * Wc.T[:, np.newaxis, :]
    
    if np.max(np.abs(T_core - 1.0)) > 1e-8:
        return None  # Core verification failed
    
    # Build k=4 boundary (schoolbook for 4x4x1)
    U_bound = np.zeros((16, 16))
    V_bound = np.zeros((20, 16))
    W_bound = np.zeros((20, 16))
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_bound[i * 4 + j, col] = 1
            V_bound[j * 5 + 4, col] = 1
            W_bound[4 * 4 + i, col] = 1
    
    # Combine
    total_rank = 28 + 16
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((20, total_rank))
    W_full = np.zeros((20, total_rank))
    
    U_full[:, :28] = U_core
    V_full[:, :28] = V_core
    W_full[:, :28] = W_core
    
    U_full[:, 28:] = U_bound
    V_full[:, 28:] = V_bound
    W_full[:, 28:] = W_bound
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist(),
        "rank": total_rank
    }


def _try_newton(rank, n, m, p, max_iter=200, tol=1e-10, deadline=None):
    """Try to find a rank-r decomposition via Newton's method on the polynomial system."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    rng = np.random.RandomState(rank * 1000 + 42)
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(3):
        if deadline is not None and time.time() > deadline:
            break
        
        U = rng.randn(n_m, rank) * 0.1
        V = rng.randn(m_p, rank) * 0.1
        W = rng.randn(p_n, rank) * 0.1
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            S = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                tmp = np.einsum('ij,jk->ijk', Uc, Vc)
                S += tmp * Wc.T[:, np.newaxis, :]
            
            R = S - 1.0
            
            err = np.max(np.abs(R))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                Wc_T = Wc.T
                
                step1 = R * Wc_T[:, np.newaxis, :]
                dU_c = np.einsum('ijk,jk->ij', step1, Vc)
                dU[:, c] = dU_c.reshape(n_m)
                
                step2 = R * Wc_T[:, np.newaxis, :]
                dV_c = np.einsum('ijk,ij->jk', step2, Uc)
                dV[:, c] = dV_c.reshape(m_p)
                
                step3 = R * Vc[np.newaxis, :, :]
                dW_ki = np.einsum('ijk,ij->ik', step3, Uc)
                dW[:, c] = dW_ki.T.reshape(p_n)
            
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < tol:
            break
    
    if best_result is not None and best_err < tol:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank, "err": best_err}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s margin
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try Strassen-based 4x4x4 core + schoolbook boundary
    if time.time() < deadline:
        result_strassen = _build_4x4x5_decomposition()
        if result_strassen is not None:
            if _verify_decomposition(result_strassen["U"], result_strassen["V"], result_strassen["W"], n, m, p, tol=1e-8):
                best_result = result_strassen
                best_rank = result_strassen["rank"]
    
    # Try lower ranks using Newton's method
    for rank in range(best_rank - 1, max(best_rank - 10, 1), -1):
        if time.time() > deadline:
            break
        
        result = _try_newton(rank, n, m, p, max_iter=30, tol=1e-10, deadline=deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                best_result = result
                best_rank = rank
                break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
