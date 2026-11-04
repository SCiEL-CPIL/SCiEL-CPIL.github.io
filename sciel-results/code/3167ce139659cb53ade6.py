"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _tensor_from_matrices(U, V, W):
    """Compute the tensor from factor matrices."""
    n_a, rank = U.shape
    n_b = V.shape[0]
    n_c = W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def _try_rank16_search(start_time, deadline):
    """Try to find a rank-16 decomposition by perturbing the rank-17 solution."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    T = _build_tensor(n, m, p)
    
    # Build the 17-term solution as a matrix of 17 vectors in R^6 x R^9 x R^6
    # Flatten each term into a vector of length 6*9*6=324
    # We want to find 16 vectors that sum to T
    
    # Start from the 17-term solution, try dropping each term and solving
    # for adjustments to the remaining 16 terms
    
    # First, build the 17-term solution
    terms_17 = [
        # Part 1: Strassen
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        # Part 2: Direct right column
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        # Part 3: Direct bottom row
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    U17 = np.zeros((n_a, 17))
    V17 = np.zeros((n_b, 17))
    W17 = np.zeros((n_c, 17))
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms_17):
        for a, val in zip(u_idx, u_val):
            U17[a, r] = val
        for b, val in zip(v_idx, v_val):
            V17[b, r] = val
        for c, val in zip(w_idx, w_val):
            W17[c, r] = val
    
    # Try dropping each term r0 and see if we can adjust the other 16
    # to compensate. This is a nonlinear problem, so use iterative refinement.
    
    rng = np.random.default_rng(42)
    best_rank = 17
    
    for r0 in range(17):
        if time.time() > deadline - 15:
            break
        # Keep all terms except r0
        keep = [r for r in range(17) if r != r0]
        U0 = U17[:, keep].copy()
        V0 = V17[:, keep].copy()
        W0 = W17[:, keep].copy()
        
        # The residual we need to fill: T - sum of kept terms
        T_current = _tensor_from_matrices(U0, V0, W0)
        residual = T - T_current
        
        # Try to absorb residual into the 16 terms via gradient-free search
        # Use a simple iterative approach: for each iteration, pick a term
        # and a coordinate, adjust it to reduce the residual
        
        success = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            U_try = U0.copy()
            V_try = V0.copy()
            W_try = W0.copy()
            
            # Add small random perturbations
            if attempt > 0:
                scale = 0.1 / (1 + attempt * 0.5)
                U_try += rng.normal(0, scale, U0.shape)
                V_try += rng.normal(0, scale, V0.shape)
                W_try += rng.normal(0, scale, W0.shape)
            
            # Check if this works (it won't exactly, but let's see the error)
            T_try = _tensor_from_matrices(U_try, V_try, W_try)
            err = np.max(np.abs(T_try - T))
            
            if err < 1e-9:
                success = True
                break
        
        if success:
            best_rank = 16
            return {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # First try to find rank-16
    result_16 = _try_rank16_search(start_time, deadline)
    if result_16 is not None:
        return result_16
    
    # Fall back to rank-17 solution
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = []
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [2], [1]))
    terms.append(([2], [1], [8], [1], [4], [1]))
    terms.append(([5], [1], [6], [1], [1], [1]))
    terms.append(([5], [1], [7], [1], [3], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
            
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
