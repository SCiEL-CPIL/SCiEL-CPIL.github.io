"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    Attempts to find a lower rank decomposition via search.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
    def idx_v(j, k):
        return j * p + k
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        U_reshaped = U.reshape((n, m, R))
        V_reshaped = V.reshape((m, p, R))
        W_reshaped = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    # Build the rank-26 baseline
    def build_rank26():
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        # Strassen 7 terms
        terms = [
            # (U_entries, V_entries, W_entries)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for r, (u_terms, v_terms, w_terms) in enumerate(terms):
            for (i, j, val) in u_terms:
                U[idx_u(i, j), r] = val
            for (j, k, val) in v_terms:
                V[idx_v(j, k), r] = val
            for (k, i, val) in w_terms:
                W[idx_w(k, i), r] = val
        
        # Standard 19 terms
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        
        return U, V, W, R
    
    U, V, W, R = build_rank26()
    best_U, best_V, best_W, best_R = U.copy(), V.copy(), W.copy(), R
    
    if not verify(U, V, W, R):
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a rank-24 or rank-25 decomposition
    # Strategy: try to find non-trivial combinations that eliminate terms
    # We'll search over possible coefficient patterns for the border terms
    
    # The 19 border terms have indices:
    border_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                border_indices.append((i, j, k))
    
    # Try to find if we can reduce rank by finding linear dependencies
    # in the border sub-tensor. We'll use a randomized search approach.
    
    # Attempt 1: Try to find a valid rank-25 by merging two border terms
    # into one non-trivial term
    if time.time() < deadline:
        # Try all pairs of border terms to see if they can be merged
        # A merge is possible if there exists a rank-1 tensor that covers
        # both positions with appropriate coefficients
        for a_idx in range(len(border_indices)):
            if time.time() > deadline - 10:
                break
            for b_idx in range(a_idx + 1, len(border_indices)):
                (i1, j1, k1) = border_indices[a_idx]
                (i2, j2, k2) = border_indices[b_idx]
                
                # For a single rank-1 term to cover both (i1,j1,k1) and (i2,j2,k2),
                # we need U[i1,j1]*V[j1,k1]*W[k1,i1] = 1 and
                # U[i2,j2]*V[j2,k2]*W[k2,i2] = 1
                # with all other positions being 0 or contributing to already-covered positions
                
                # This is complex; let's try a different approach
                # Just check if we can find a rank-24 solution by random search
                pass
    
    # Attempt 2: Randomized search for lower rank
    # Try to find a valid decomposition with R' < 26
    # by randomly assigning sparse patterns and checking validity
    
    if time.time() < deadline:
        np.random.seed(42)
        for target_R in [25, 24, 23]:
            if time.time() > deadline - 15:
                break
            
            # Try multiple random initializations
            for attempt in range(50):
                if time.time() > deadline - 15:
                    break
                
                U_try = np.zeros((n * m, target_R))
                V_try = np.zeros((m * p, target_R))
                W_try = np.zeros((p * n, target_R))
                
                # Initialize with the Strassen 7 terms
                terms = [
                    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                    ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)]),
                    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
                    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
                    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]),
                    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
                ]
                
                for r, (u_terms, v_terms, w_terms) in enumerate(terms):
                    if r >= target_R:
                        break
                    for (i, j, val) in u_terms:
                        U_try[idx_u(i, j), r] = val
                    for (j, k, val) in v_terms:
                        V_try[idx_v(j, k), r] = val
                    for (k, i, val) in w_terms:
                        W_try[idx_w(k, i), r] = val
                
                # Fill remaining terms with random sparse patterns for border
                remaining = target_R - 7
                for r in range(7, target_R):
                    # Choose random support for U, V, W
                    n_u = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    n_v = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    n_w = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                    
                    u_positions = np.random.choice(n * m, size=n_u, replace=False)
                    v_positions = np.random.choice(m * p, size=n_v, replace=False)
                    w_positions = np.random.choice(p * n, size=n_w, replace=False)
                    
                    for pos in u_positions:
                        U_try[pos, r] = np.random.choice([-1, 1])
                    for pos in v_positions:
                        V_try[pos, r] = np.random.choice([-1, 1])
                    for pos in w_positions:
                        W_try[pos, r] = np.random.choice([-1, 1])
                
                if verify(U_try, V_try, W_try, target_R):
                    best_U, best_V, best_W, best_R = U_try.copy(), V_try.copy(), W_try.copy(), target_R
                    break
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_R):
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
