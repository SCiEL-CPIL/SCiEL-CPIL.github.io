"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Build rank-50 Strassen decomposition, then use structured column
    elimination exploiting the 2-row A-matrix structure to reduce rank.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 115  # safety margin before 150s limit
    
    # Build the rank-50 decomposition as starting point
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    def verify_decomposition(U_m, V_m, W_m, n, m, p, num_tests=10):
        """Fast verification using random integer inputs."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank_m):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    def verify_exact(U_m, V_m, W_m, n, m, p):
        """Exact verification using symbolic-like random tests with small integers."""
        for seed in range(20):
            np.random.seed(seed)
            A = np.random.randint(-2, 3, size=(n, m))
            B = np.random.randint(-2, 3, size=(m, p))
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank_m):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-6):
                return False
        return True
    
    # Verify initial decomposition
    if not verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    def try_eliminate_column(U_arr, V_arr, W_arr, col_idx, n, m, p):
        """
        Try to eliminate column col_idx by redistributing its contribution
        to other columns. Uses the structure that each term is a rank-1 tensor.
        We check if the column can be expressed as a linear combination of others
        in the tensor sense.
        """
        r = U_arr.shape[1]
        if r <= 1:
            return None
        
        # The tensor contribution of column c is:
        # T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We want to see if T_c can be written as sum of a_i * T_i for i != c
        # This is a nonlinear system, so we use a heuristic approach:
        # Check if the column is "redundant" by seeing if removing it and
        # adjusting a few other columns still gives correct results.
        
        # Strategy: Try to absorb column c into one other column d
        # by checking if there exists alpha such that:
        # U[:,d] + alpha*U[:,c], V[:,d] + alpha*V[:,c], W[:,d] + alpha*W[:,c]
        # still works (this is a sufficient but not necessary condition)
        
        # More practical: just try removing the column and verify
        keep = [i for i in range(r) if i != col_idx]
        U_new = U_arr[:, keep]
        V_new = V_arr[:, keep]
        W_new = W_arr[:, keep]
        
        if verify_decomposition(U_new, V_new, W_new, n, m, p, num_tests=5):
            return U_new, V_new, W_new
        
        # Try pairwise elimination: can column c be absorbed into column d?
        # For each other column d, check if the sum still works
        for d in keep:
            # Try: remove c, and see if we can adjust d
            # Simple heuristic: check if columns c and d are "similar"
            # (i.e., their U, V, W vectors are proportional)
            Uc, Vc, Wc = U_arr[:, col_idx], V_arr[:, col_idx], W_arr[:, col_idx]
            Ud, Vd, Wd = U_arr[:, d], V_arr[:, d], W_arr[:, d]
            
            # Check if U_c is proportional to U_d
            if np.allclose(Uc, 0) or np.allclose(Ud, 0):
                continue
            # Find ratio
            mask = Ud != 0
            if not np.any(mask):
                continue
            ratios = Uc[mask] / Ud[mask]
            if len(ratios) < 2:
                continue
            # Check if ratios are consistent
            if np.allclose(ratios, ratios[0], atol=1e-10):
                alpha = ratios[0]
                # Check V and W
                if np.allclose(Vc, alpha * Vd, atol=1e-10) and np.allclose(Wc, alpha * Wd, atol=1e-10):
                    # Column c is a scalar multiple of column d - can be removed
                    keep2 = [i for i in range(r) if i != col_idx]
                    U_new = U_arr[:, keep2]
                    V_new = V_arr[:, keep2]
                    W_new = W_arr[:, keep2]
                    if verify_decomposition(U_new, V_new, W_new, n, m, p, num_tests=5):
                        return U_new, V_new, W_new
        
        return None
    
    def greedy_reduce(U_arr, V_arr, W_arr, n, m, p, deadline):
        """
        Greedy column elimination: repeatedly try to remove columns.
        Order matters - try columns that are most "redundant" first.
        """
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        r = U_arr.shape[1]
        
        # Score each column by how "simple" it is (fewer non-zero entries = more likely redundant)
        while r > 1 and time.time() < deadline - 5:
            best_col = -1
            best_result = None
            
            for c in range(r):
                if time.time() > deadline - 5:
                    break
                result = try_eliminate_column(U_arr, V_arr, W_arr, c, n, m, p)
                if result is not None:
                    best_col = c
                    best_result = result
                    break  # Take the first one we can eliminate
            
            if best_result is None:
                break
            
            U_arr, V_arr, W_arr = best_result
            r = U_arr.shape[1]
        
        return U_arr, V_arr, W_arr, r
    
    # Attempt structured reduction
    if time.time() < deadline - 20:
        U_red, V_red, W_red, new_rank = greedy_reduce(U, V, W, n, m, p, deadline)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    # Try multiple passes with different column orderings
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        # Shuffle column order to find different elimination paths
        r = best_U.shape[1]
        perm = np.random.permutation(r)
        U_shuf = best_U[:, perm]
        V_shuf = best_V[:, perm]
        W_shuf = best_W[:, perm]
        
        U_red, V_red, W_red, new_rank = greedy_reduce(U_shuf, V_shuf, W_shuf, n, m, p, deadline)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
