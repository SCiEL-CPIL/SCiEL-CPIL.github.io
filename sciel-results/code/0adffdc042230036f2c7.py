import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _build_strassen_rank39():
    """Build the rank-39 Strassen decomposition for 2x2 * 2x11."""
    n, m, p = 2, 2, 11
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    r = 0

    def add_strassen_block(k0, k1):
        nonlocal r
        v_e, v_f = k0, k1
        v_g, v_h = 11 + k0, 11 + k1
        w_c00, w_c10 = 2 * k0, 2 * k0 + 1
        w_c01, w_c11 = 2 * k1, 2 * k1 + 1

        U[0, r] = 1; U[3, r] = 1
        V[v_e, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1; W[w_c11, r] = 1
        r += 1

        U[2, r] = 1; U[3, r] = 1
        V[v_e, r] = 1
        W[w_c10, r] = 1; W[w_c11, r] = -1
        r += 1

        U[0, r] = 1
        V[v_f, r] = 1; V[v_h, r] = -1
        W[w_c01, r] = 1; W[w_c11, r] = 1
        r += 1

        U[3, r] = 1
        V[v_g, r] = 1; V[v_e, r] = -1
        W[w_c00, r] = 1; W[w_c10, r] = 1
        r += 1

        U[0, r] = 1; U[1, r] = 1
        V[v_h, r] = 1
        W[w_c00, r] = -1; W[w_c01, r] = 1
        r += 1

        U[2, r] = 1; U[0, r] = -1
        V[v_e, r] = 1; V[v_f, r] = 1
        W[w_c11, r] = 1
        r += 1

        U[1, r] = 1; U[3, r] = -1
        V[v_g, r] = 1; V[v_h, r] = 1
        W[w_c00, r] = 1
        r += 1

    for block in range(5):
        add_strassen_block(block * 2, block * 2 + 1)

    # Last k=10 (odd) handled with 4 additional terms
    k2 = 10
    U[0, r] = 1; V[k2, r] = 1; W[2*k2, r] = 1; r += 1
    U[1, r] = 1; V[11+k2, r] = 1; W[2*k2, r] = 1; r += 1
    U[2, r] = 1; V[k2, r] = 1; W[2*k2+1, r] = 1; r += 1
    U[3, r] = 1; V[11+k2, r] = 1; W[2*k2+1, r] = 1; r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"
    return U, V, W

def _exact_rational_nullspace_check(U, V, W, T_flat):
    """
    Using exact Fraction arithmetic, determine which subset of the rank-39 terms
    is necessary. We check if the tensor T can be expressed as a linear combination
    of a subset of the rank-1 tensors u_r ⊗ v_r ⊗ w_r.
    
    For a given subset S, we need: T = sum_{r in S} alpha_r * (u_r ⊗ v_r ⊗ w_r)
    for some scalars alpha_r. This is a linear system in the alpha_r's.
    
    We use a greedy approach: start with all 39 terms, try removing one at a time.
    """
    n, m, p = 2, 2, 11
    rank = U.shape[1]
    
    # Convert to exact fractions
    U_frac = [[Fraction(U[i, r]) for r in range(rank)] for i in range(n*m)]
    V_frac = [[Fraction(V[i, r]) for r in range(rank)] for i in range(m*p)]
    W_frac = [[Fraction(W[i, r]) for r in range(rank)] for i in range(p*n)]
    
    # Build the target tensor as exact fractions
    T_frac = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_frac[(i*m+j, j*p+k, k*n+i)] = Fraction(1)
    
    def check_subset(indices):
        """Check if T is in the span of {u_r ⊗ v_r ⊗ w_r : r in indices}."""
        if not indices:
            return False
        
        # Build the linear system: for each nonzero (a,b,c) in T,
        # T[a,b,c] = sum_{r in indices} alpha_r * U[a,r] * V[b,r] * W[c,r]
        
        # Collect all (a,b,c) positions where T is nonzero
        positions = list(T_frac.keys())
        
        # Build coefficient matrix A (len(positions) x len(indices))
        # and rhs vector
        num_pos = len(positions)
        num_vars = len(indices)
        
        if num_pos == 0:
            return True
        
        # Build matrix using exact arithmetic
        # A[i][j] = U[a_i, idx_j] * V[b_i, idx_j] * W[c_i, idx_j]
        # rhs[i] = T[a_i, b_i, c_i]
        
        A = [[Fraction(0)] * num_vars for _ in range(num_pos)]
        rhs = [Fraction(0)] * num_pos
        
        for i, (a, b, c) in enumerate(positions):
            rhs[i] = T_frac[(a, b, c)]
            for j, r in enumerate(indices):
                A[i][j] = U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
        
        # Solve using Gaussian elimination over Q
        # Augmented matrix
        aug = [row[:] + [rhs[i]] for i, row in enumerate(A)]
        
        rank_aug = 0
        for col in range(num_vars):
            # Find pivot
            pivot = -1
            for row in range(rank_aug, num_pos):
                if aug[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            aug[rank_aug], aug[pivot] = aug[pivot], aug[rank_aug]
            
            # Eliminate
            pivot_val = aug[rank_aug][col]
            for row in range(rank_aug + 1, num_pos):
                factor = aug[row][col] / pivot_val
                if factor != 0:
                    for k in range(col, num_vars + 1):
                        aug[row][k] -= factor * aug[rank_aug][k]
            rank_aug += 1
        
        # Check consistency: no row of form [0, 0, ..., 0 | nonzero]
        for row in range(rank_aug, num_pos):
            if aug[row][num_vars] != 0:
                return False
        return True
    
    return check_subset

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s budget (150s limit - margin)

    # Build the rank-39 Strassen decomposition
    U, V, W = _build_strassen_rank39()
    rank = 39
    
    # Verify the base construction
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    base_err = np.linalg.norm(T_recon.flatten() - T_flat)
    
    if base_err > 1e-10:
        # Should not happen, but fallback
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook))
        V_sb = np.zeros((m*p, rank_schoolbook))
        W_sb = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Greedy removal using exact rational arithmetic
    best_indices = list(range(rank))
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Convert to fractions for exact computation
    U_frac = [[Fraction(U[i, r]) for r in range(rank)] for i in range(n*m)]
    V_frac = [[Fraction(V[i, r]) for r in range(rank)] for i in range(m*p)]
    W_frac = [[Fraction(W[i, r]) for r in range(rank)] for i in range(p*n)]
    
    # Build target tensor positions
    T_positions = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_positions[(i*m+j, j*p+k, k*n+i)] = Fraction(1)
    
    def check_subset(indices):
        """Check if T is in the span of the given rank-1 tensors (exact)."""
        if not indices:
            return len(T_positions) == 0
        
        positions = list(T_positions.keys())
        num_pos = len(positions)
        num_vars = len(indices)
        
        if num_pos == 0:
            return True
        
        # Build augmented matrix
        aug = []
        for (a, b, c) in positions:
            row = []
            for r in indices:
                row.append(U_frac[a][r] * V_frac[b][r] * W_frac[c][r])
            row.append(T_positions[(a, b, c)])
            aug.append(row)
        
        # Gaussian elimination over Q
        rank_aug = 0
        for col in range(num_vars):
            pivot = -1
            for row in range(rank_aug, num_pos):
                if aug[row][col] != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            aug[rank_aug], aug[pivot] = aug[pivot], aug[rank_aug]
            
            pivot_val = aug[rank_aug][col]
            for row in range(rank_aug + 1, num_pos):
                if aug[row][col] != 0:
                    factor = aug[row][col] / pivot_val
                    for k in range(col, num_vars + 1):
                        aug[row][k] -= factor * aug[rank_aug][k]
            rank_aug += 1
        
        for row in range(rank_aug, num_pos):
            if aug[row][num_vars] != 0:
                return False
        return True
    
    # Greedy: try removing each term one at a time, keep if it works
    current_indices = list(range(rank))
    
    for idx in range(rank):
        if time.time() > deadline - 5:
            break
        
        # Try removing current_indices[idx]
        test_indices = [x for x in current_indices if x != idx]
        
        if check_subset(test_indices):
            current_indices = test_indices
    
    best_rank = len(current_indices)
    best_U = U[:, current_indices]
    best_V = V[:, current_indices]
    best_W = W[:, current_indices]
    
    # Try a second pass of greedy removal (in case order matters)
    if time.time() < deadline - 10:
        current_indices2 = list(current_indices)
        for idx in range(len(current_indices2)):
            if time.time() > deadline - 5:
                break
            test_indices = [x for x in current_indices2 if x != idx]
            if check_subset(test_indices):
                current_indices2 = test_indices
        
        if len(current_indices2) < best_rank:
            best_rank = len(current_indices2)
            best_U = U[:, current_indices2]
            best_V = V[:, current_indices2]
            best_W = W[:, current_indices2]
    
    # Final verification
    T_final = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    final_err = np.linalg.norm(T_final.flatten() - T_flat)
    
    if final_err > 1e-8:
        # Fallback to original rank-39
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
