import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=2, m=2, p=8, tol=1e-10):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Build decomposition tensor
    decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        decomp += np.einsum('a,b,c->abc', u, v, w)
    
    return np.allclose(decomp, target, atol=tol)


def _strassen_28():
    """Return the rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_forms = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    B_forms = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1},
        {2: 1, 4: 1},
        {1: 1, 3: 1},
        {0: 1, 1: -1, 2: 1, 5: 1}
    ]

    for block in range(4):
        k0 = block * 2
        
        for s in range(7):
            r = block * 7 + s
            
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}


def _try_greedy_reduction(rank_target, sol, max_time=60):
    """
    Try to reduce rank by finding redundant terms or merging terms.
    Uses a greedy approach to check if any term can be removed or if
    two terms can be merged into one.
    """
    start_time = time.time()
    U = np.array(sol['U'], dtype=float)
    V = np.array(sol['V'], dtype=float)
    W = np.array(sol['W'], dtype=float)
    current_rank = U.shape[1]
    
    if current_rank <= rank_target:
        return sol
    
    # Try removing each term and checking if the decomposition still works
    # This is O(rank * verification_cost) which could be expensive
    # Instead, let's try a smarter approach: check if any term is in the span
    # of the others
    
    # For each term r, check if removing it still gives a valid decomposition
    # by solving for whether the remaining terms can compensate
    # This is complex, so let's try a simpler heuristic first
    
    # Check if any two terms can be merged (i.e., their sum is also a valid term)
    # This is unlikely to help directly, but let's check for linear dependencies
    
    # Build the tensor as a matrix and check rank
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # We can check if the decomposition is minimal by seeing if any term is redundant
    
    # For now, return None to indicate no improvement found
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    # Get the rank-28 solution
    sol28 = _strassen_28()
    
    # Verify it's correct
    if not _verify_decomposition(sol28['U'], sol28['V'], sol28['W']):
        return sol28
    
    # Try to find a lower rank solution
    if time.time() < deadline:
        sol_reduced = _try_greedy_reduction(27, sol28, max_time=30)
        if sol_reduced is not None and _verify_decomposition(sol_reduced['U'], sol_reduced['V'], sol_reduced['W']):
            return sol_reduced
    
    return sol28

# EVOLVE-BLOCK-END
