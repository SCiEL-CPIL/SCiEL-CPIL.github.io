"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 115  # Safety margin before 120s hard limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Build the Strassen-based rank-54 solution
    # This partitions the m=5 dimension into pairs (0,1), (2,3) and single (4)
    # For each j-pair and k-pair, we use Strassen (7 mults)
    # For j=4, we use schoolbook (4 mults per k-pair)
    
    rank_54 = 54
    U = np.zeros((n * m, rank_54), dtype=int)
    V = np.zeros((m * p, rank_54), dtype=int)
    W = np.zeros((p * n, rank_54), dtype=int)
    
    col = 0
    
    def add_strassen(i_base, j_base, k_base):
        """Add a 2x2x2 Strassen block at the given base indices."""
        nonlocal col
        
        # M0: (A00+A11)(B00+B11)
        U[i_base*m + j_base, col] += 1
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[j_base*p + k_base, col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[k_base*n + i_base, col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1)*m + j_base, col] += 1
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[j_base*p + k_base, col] += 1
        W[k_base*n + (i_base+1), col] += 1
        W[(k_base+1)*n + (i_base+1), col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base*m + j_base, col] += 1
        V[j_base*p + (k_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] -= 1
        W[(k_base+1)*n + i_base, col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1)*m + (j_base+1), col] += 1
        V[(j_base+1)*p + k_base, col] += 1
        V[j_base*p + k_base, col] -= 1
        W[k_base*n + i_base, col] += 1
        W[k_base*n + (i_base+1), col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base*m + j_base, col] += 1
        U[i_base*m + (j_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[k_base*n + i_base, col] -= 1
        W[(k_base+1)*n + i_base, col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m + j_base, col] += 1
        U[i_base*m + j_base, col] -= 1
        V[j_base*p + k_base, col] += 1
        V[j_base*p + (k_base+1), col] += 1
        W[(k_base+1)*n + (i_base+1), col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base*m + (j_base+1), col] += 1
        U[(i_base+1)*m + (j_base+1), col] -= 1
        V[(j_base+1)*p + k_base, col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[k_base*n + i_base, col] += 1
        col += 1
    
    def add_naive_2x1x2(i_base, j_idx, k_base):
        """Add a 2x1x2 schoolbook block at the given base indices."""
        nonlocal col
        for di in range(2):
            for dk in range(2):
                U[(i_base+di)*m + j_idx, col] += 1
                V[j_idx*p + (k_base+dk), col] += 1
                W[(k_base+dk)*n + (i_base+di), col] += 1
                col += 1
    
    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    # k-pairs are (0,1), (2,3), (4,5)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen(0, j_base, k_base)
    
    # Apply naive blocks for j=4
    for k_base in [0, 2, 4]:
        add_naive_2x1x2(0, 4, k_base)
    
    def verify(U_mat, V_mat, W_mat):
        """Verify the decomposition is correct."""
        r = U_mat.shape[1]
        U_np = np.array(U_mat, dtype=float)
        V_np = np.array(V_mat, dtype=float)
        W_np = np.array(W_mat, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_reshaped = U_np.reshape(n, m, r)
        V_reshaped = V_np.reshape(m, p, r)
        W_reshaped = W_np.reshape(p, n, r)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        C_computed = np.einsum('ijk,ij,jk->ik', T, A_test, B_test)
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # First verify the base decomposition
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school), dtype=int)
        V_school = np.zeros((m * p, rank_school), dtype=int)
        W_school = np.zeros((p * n, rank_school), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_school[i*m+j, idx] = 1
                    V_school[j*p+k, idx] = 1
                    W_school[k*n+i, idx] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Now try to reduce rank by finding linearly dependent columns
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build matrix M where M[(i,j,k), r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    T_size = n * m * p  # 60
    M = np.zeros((T_size, col))
    for r in range(col):
        U_col = U_arr[:, r]
        V_col = V_arr[:, r]
        W_col = W_arr[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    M[idx, r] = U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
    
    # Check rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    best_keep = list(range(col))
    
    if rank_M < col:
        # Greedy removal with multiple passes and random restarts
        keep = list(range(col))
        changed = True
        while changed and _time_remaining() > 30:
            changed = False
            # Try removing columns in a fixed order
            for idx in list(keep):
                if _time_remaining() < 30:
                    break
                test_keep = [k for k in keep if k != idx]
                if len(test_keep) == 0:
                    break
                M_test = M[:, test_keep]
                if np.linalg.matrix_rank(M_test, tol=1e-10) == rank_M:
                    keep = test_keep
                    changed = True
                    if len(keep) < len(best_keep):
                        best_keep = keep.copy()
                    break
        
        # Try random order removals for a few iterations
        np.random.seed(123)
        for restart in range(5):
            if _time_remaining() < 30:
                break
            keep = list(range(col))
            np.random.shuffle(keep)
            changed = True
            while changed and _time_remaining() > 30:
                changed = False
                for idx in list(keep):
                    if _time_remaining() < 30:
                        break
                    test_keep = [k for k in keep if k != idx]
                    if len(test_keep) == 0:
                        break
                    M_test = M[:, test_keep]
                    if np.linalg.matrix_rank(M_test, tol=1e-10) == rank_M:
                        keep = test_keep
                        changed = True
                        if len(keep) < len(best_keep):
                            best_keep = keep.copy()
                        break
    
    if len(best_keep) < col:
        new_rank = len(best_keep)
        U_new = U_arr[:, best_keep]
        V_new = V_arr[:, best_keep]
        W_new = W_arr[:, best_keep]
        
        # Verify the reduced decomposition with multiple seeds
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            U_reshaped = U_new.reshape(n, m, new_rank)
            V_reshaped = V_new.reshape(m, p, new_rank)
            W_reshaped = W_new.reshape(p, n, new_rank)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            C_computed = np.einsum('ijk,ij,jk->ik', T, A_test, B_test)
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                break
        else:
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Return the original 54-rank decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
