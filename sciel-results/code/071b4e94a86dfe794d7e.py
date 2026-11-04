"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Start with the known rank-50 decomposition from Strassen blocks,
    then attempt to reduce rank by finding linear dependencies and structured
    simplifications among the rank-1 terms.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the rank-50 decomposition as starting point
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=object)
    V = np.zeros((m * p, rank_init), dtype=object)
    W = np.zeros((p * n, rank_init), dtype=object)
    
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
    
    # Verify: rank should be 50
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    # Convert to float for numerical linear algebra
    Uf = U.astype(float)
    Vf = V.astype(float)
    Wf = W.astype(float)
    
    def verify_decomposition(U_m, V_m, W_m, n, m, p):
        """Check if the decomposition is exact by testing on random inputs."""
        np.random.seed(42)
        for _ in range(20):
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_expected = A @ B
            C_comp = np.zeros((n, p), dtype=float)
            rank_m = U_m.shape[1]
            for i in range(n):
                for k in range(p):
                    val = 0
                    for j in range(m):
                        for r in range(rank_m):
                            val += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
                    C_comp[i, k] = val
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    # Verify initial decomposition
    if not verify_decomposition(Uf, Vf, Wf, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = Uf.copy(), Vf.copy(), Wf.copy()
    best_rank = rank_init
    
    # Greedy elimination of linearly dependent columns
    def try_reduce_rank(U_arr, V_arr, W_arr, n, m, p, max_iter=100):
        """Attempt to reduce rank by eliminating linearly dependent columns."""
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]
        
        eliminated = set()
        for _ in range(max_iter):
            if time.time() > deadline - 30:
                break
            found_elim = False
            for c in range(current_rank):
                if c in eliminated:
                    continue
                other_cols = [oc for oc in range(current_rank) if oc != c and oc not in eliminated]
                if not other_cols:
                    continue
                
                U_sub = U_arr[:, other_cols].T
                V_sub = V_arr[:, other_cols].T
                W_sub = W_arr[:, other_cols].T
                
                u_target = U_arr[:, c]
                v_target = V_arr[:, c]
                w_target = W_arr[:, c]
                
                stacked = np.vstack([U_sub, V_sub, W_sub]).astype(float)
                target = np.concatenate([u_target, v_target, w_target]).astype(float)
                
                try:
                    coeffs, residuals, rank_s, sv = np.linalg.lstsq(stacked, target, rcond=None)
                    recon = stacked @ coeffs
                    if np.allclose(recon, target, atol=1e-9):
                        eliminated.add(c)
                        found_elim = True
                        break
                except:
                    continue
            
            if not found_elim:
                break
        
        keep_cols = [c for c in range(current_rank) if c not in eliminated]
        new_rank = len(keep_cols)
        if new_rank < current_rank:
            return U_arr[:, keep_cols], V_arr[:, keep_cols], W_arr[:, keep_cols], new_rank
        return U_arr, V_arr, W_arr, current_rank
    
    # Attempt greedy reduction
    if time.time() < deadline - 40:
        U_red, V_red, W_red, new_rank = try_reduce_rank(Uf, Vf, Wf, n, m, p)
        if new_rank < best_rank:
            if verify_decomposition(U_red, V_red, W_red, n, m, p):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = new_rank
    
    # Structured search: try to find rank-1 terms that can replace pairs
    # Look for patterns where two Strassen terms can be replaced by one
    # with appropriate coefficients
    
    def try_structured_reduction(U_arr, V_arr, W_arr, n, m, p):
        """Try to find structural simplifications in the decomposition."""
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]
        
        # For each pair of columns, check if their tensor product can be
        # expressed as a linear combination of fewer terms
        # This is expensive, so we limit the search
        
        # First, try to find columns that are "almost" linearly dependent
        # but with small integer coefficients
        improved = False
        for c1 in range(current_rank):
            if time.time() > deadline - 30:
                break
            for c2 in range(c1 + 1, current_rank):
                if time.time() > deadline - 30:
                    break
                # Check if column c2 can be expressed as a combination
                # of column c1 and others with small integer coefficients
                other_cols = [oc for oc in range(current_rank) if oc != c2]
                if len(other_cols) < 1:
                    continue
                
                U_sub = U_arr[:, other_cols].T
                V_sub = V_arr[:, other_cols].T
                W_sub = W_arr[:, other_cols].T
                
                u_target = U_arr[:, c2]
                v_target = V_arr[:, c2]
                w_target = W_arr[:, c2]
                
                stacked = np.vstack([U_sub, V_sub, W_sub]).astype(float)
                target = np.concatenate([u_target, v_target, w_target]).astype(float)
                
                try:
                    coeffs, residuals, rank_s, sv = np.linalg.lstsq(stacked, target, rcond=None)
                    # Check if coefficients are small integers or half-integers
                    # and if reconstruction is exact
                    recon = stacked @ coeffs
                    if np.allclose(recon, target, atol=1e-9):
                        # Check if most coefficients are zero (sparse)
                        nonzero = np.count_nonzero(np.abs(coeffs) > 1e-9)
                        if nonzero <= 2:  # Can be expressed with at most 2 other columns
                            # Eliminate c2
                            keep = [oc for oc in range(current_rank) if oc != c2]
                            U_arr = U_arr[:, keep]
                            V_arr = V_arr[:, keep]
                            W_arr = W_arr[:, keep]
                            current_rank = len(keep)
                            improved = True
                            break
                except:
                    continue
            if improved:
                break
        
        return U_arr, V_arr, W_arr, current_rank
    
    if time.time() < deadline - 30:
        U_red, V_red, W_red, new_rank = try_structured_reduction(best_U, best_V, best_W, n, m, p)
        if new_rank < best_rank:
            if verify_decomposition(U_red, V_red, W_red, n, m, p):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = new_rank
    
    # Final verification and return
    if not verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = Uf, Vf, Wf
        best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
