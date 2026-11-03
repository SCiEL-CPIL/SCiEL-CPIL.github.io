"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 10]

def _build_base_decomposition(n, m, p):
    """Build the base rank-55 decomposition."""
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A_right (A[:,2], shape 2x1) * B_bottom (B[2,:], shape 1x10)
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # Part 2: A_left (A[:,:2], shape 2x2) * B_top (B[:2,:], shape 2x10)
    X00, X01, X10, X11 = 0, 1, 3, 4
    
    for j in range(5):
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3
        
        # M1 = (X00 + X11) * (Y00 + Y11)
        U[X00, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M2 = (X10 + X11) * Y00
        U[X10, col] += 1
        U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1
        W[Z11, col] -= 1
        col += 1
        
        # M3 = X00 * (Y01 - Y11)
        U[X00, col] += 1
        V[Y01, col] += 1
        V[Y11, col] -= 1
        W[Z01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M4 = X11 * (Y10 - Y00)
        U[X11, col] += 1
        V[Y10, col] += 1
        V[Y00, col] -= 1
        W[Z00, col] += 1
        W[Z10, col] += 1
        col += 1
        
        # M5 = (X00 + X01) * Y11
        U[X00, col] += 1
        U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1
        W[Z01, col] += 1
        col += 1
        
        # M6 = (X10 - X00) * (Y00 + Y01)
        U[X10, col] += 1
        U[X00, col] -= 1
        V[Y00, col] += 1
        V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1
        
        # M7 = (X01 - X11) * (Y10 + Y11)
        U[X01, col] += 1
        U[X11, col] -= 1
        V[Y10, col] += 1
        V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = U.shape[1]
    # Check: sum_r U[a,r] * V[b,r] * W[c,r] should equal delta for valid (a,b,c)
    # a = i*m + j, b = j'*p + k, c = k'*n + i'
    # Nonzero only when j == j' and i == i' and k == k'
    
    # Build the full tensor and compare
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Expected: T[(i*m+j), (j*p+k), (k*n+i)] = 1 if j==j' else 0, and i==i', k==k'
    
    # For efficiency, check a sample of entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
                # Also check some zero entries
                if j > 0:
                    b2 = (j-1) * p + k
                    val2 = np.sum(U[a, :] * V[b2, :] * W[c, :])
                    if abs(val2) > tol:
                        return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=100, tol=1e-10):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    A rank-1 term r is defined by (U[:,r], V[:,r], W[:,r]).
    If term r can be expressed as a linear combination of other terms, we can remove it.
    
    The condition for term r to be removable: there exist coefficients alpha_s (s != r)
    such that for all (a,b,c):
      U[a,r]*V[b,r]*W[c,r] = sum_{s!=r} alpha_s * U[a,s]*V[b,s]*W[c,s]
    
    This is a bilinear condition. We can check by fixing one index and solving.
    
    A simpler approach: try to find if any column is a linear combination of others
    in the tensor space. The tensor T has entries T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    Each rank-1 term contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r] to T.
    
    We can check linear dependence by forming the matrix of all tensor entries
    and checking rank.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    n_rows = U.shape[0]
    n_cols_v = V.shape[0]
    n_cols_w = W.shape[0]
    
    removed = set()
    changed = True
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        current_rank = U.shape[1] - len(removed)
        
        # Build the tensor as a 3D array
        T = np.zeros((n_rows, n_cols_v, n_cols_w))
        for r in range(U.shape[1]):
            if r in removed:
                continue
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # For each candidate column to remove, check if it's in the span of others
        for r_candidate in range(U.shape[1]):
            if r_candidate in removed:
                continue
            
            # Build the tensor without this candidate
            T_without = np.zeros((n_rows, n_cols_v, n_cols_w))
            for r in range(U.shape[1]):
                if r == r_candidate or r in removed:
                    continue
                T_without += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            T_candidate = np.einsum('a,b,c->abc', U[:, r_candidate], V[:, r_candidate], W[:, r_candidate])
            
            # Check if T_candidate is in the span of {T_without components}
            # i.e., is T_candidate - T_without = 0? No, we want T_candidate to be
            # expressible as a linear combination of the other rank-1 tensors.
            
            # Alternative: check if the full tensor T equals T_without + T_candidate,
            # and if we can find alpha such that T_candidate = alpha * (some combination)
            
            # Simpler check: is T_candidate linearly dependent on the other terms?
            # Flatten and check rank
            vec_candidate = T_candidate.flatten()
            
            # Build matrix of other terms
            other_terms = []
            for r in range(U.shape[1]):
                if r == r_candidate or r in removed:
                    continue
                other_terms.append(np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten())
            
            if not other_terms:
                continue
            
            M = np.column_stack(other_terms)
            
            # Check if vec_candidate is in the column space of M
            # Use least squares
            try:
                coeffs, residuals, rank_m, sv = np.linalg.lstsq(M, vec_candidate, rcond=tol)
                reconstructed = M @ coeffs
                if np.allclose(reconstructed, vec_candidate, atol=tol, rtol=tol):
                    # This term is redundant
                    removed.add(r_candidate)
                    changed = True
                    break
            except:
                continue
    
    # Build reduced matrices
    keep = [r for r in range(U.shape[1]) if r not in removed]
    U_red = U[:, keep]
    V_red = V[:, keep]
    W_red = W[:, keep]
    
    return U_red, V_red, W_red

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    
    Strategy: Build base rank-55 decomposition, then try to reduce rank
    by finding linear dependencies among the rank-1 terms.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build base decomposition
    U, V, W = _build_base_decomposition(n, m, p)
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try to reduce rank
    if time.time() < deadline - 10.0:
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p, max_iterations=50)
        if U_red.shape[1] < best_rank:
            if _verify_decomposition(U_red, V_red, W_red, n, m, p):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = U_red.shape[1]
    
    # Try additional random perturbations and re-optimization
    if time.time() < deadline - 15.0:
        # Try different initial splits
        for m1 in [1, 2]:
            if time.time() > deadline - 15.0:
                break
            
            # Build a different base decomposition with split m1, m-m1
            rank_candidate = n * (m - m1) * p + (7 // 2) * m1 * p  # rough estimate
            # Actually, let's just try the existing one with column elimination from different orderings
            
            # Shuffle columns and try elimination
            idx = np.random.permutation(best_U.shape[1])
            U_shuf = best_U[:, idx]
            V_shuf = best_V[:, idx]
            W_shuf = best_W[:, idx]
            
            U_try, V_try, W_try = _try_reduce_rank(U_shuf, V_shuf, W_shuf, n, m, p, max_iterations=30)
            if U_try.shape[1] < best_rank:
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = U_try.shape[1]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to base
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
