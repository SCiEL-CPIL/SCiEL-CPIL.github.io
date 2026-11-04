import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140.0 # Safety margin for 150s limit

    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(U, V, W):
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        return U_h, V_h, W_h

    def to_lists(U, V, W, rank):
        U_list = [[float(U[i*m+j, c]) for c in range(rank)] for i in range(n) for j in range(m)]
        V_list = [[float(V[j*p+k, c]) for c in range(rank)] for j in range(m) for k in range(p)]
        W_list = [[float(W[k*n+i, c]) for c in range(rank)] for k in range(p) for i in range(n)]
        return U_list, V_list, W_list

    # Strategy: Exploit the small dimension n=2.
    # The tensor is T[i,j,k] = 1.
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    # For a fixed rank R, we can try to solve for U, V, W.
    # This is a system of polynomial equations.
    # Let's use a structured search.
    # Since n=2, W has 2p rows. i=0 and i=1.
    # W[k,0,c] and W[k,1,c].
    
    # Let's try a randomized approach with a specific structure.
    # We can fix U and V to be simple (e.g., sparse or from a small set) and solve for W?
    # Or fix W to be simple and solve for U and V?
    
    # Actually, a better approach for small ranks:
    # Use the fact that the tensor is all 1s.
    # If we can find a decomposition of rank R, it's a valid one.
    # Let's try ranks from 14 to 55.
    # For each rank, use random restarts with a smarter initialization or solver.
    
    # The previous ALS approach might be slow or stuck.
    # Let's try a direct search for small ranks using integer/half-integer constraints?
    # No, that's a huge search space.
    
    # Let's stick to ALS but with better convergence criteria and more restarts,
    # and crucially, try to snap to half-integers at the end.
    # The key insight from the prompt is "integer/half-integer entries".
    # This suggests the solution might be rational with small denominators.
    
    # Let's implement a more robust solver.
    # We can use scipy.optimize.least_squares to minimize the error sum (sum_c ... - 1)^2.
    # But the variables are U, V, W. Total variables = (nm + mp + pn) * R.
    # For R=14, n=2, m=4, p=7:
    # nm=8, mp=28, pn=14. Total = 50 * 14 = 700 variables.
    # This is a large optimization problem.
    
    # Alternative:
    # Since T[i,j,k] = 1, maybe we can construct a solution explicitly?
    # For 2x4x7, is there a known low rank decomposition?
    # The rank of matrix multiplication tensor M_{n,m,p} is known to be nmp for small sizes, but can be lower.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x4x7, the rank is likely lower than 56.
    
    # Let's try to find a rank-30 solution as suggested in the directive.
    # Or just search for any rank < 56.
    
    # Let's use a simple random search for small ranks, but with a twist:
    # We can fix one of the matrices to be an identity-like or sparse matrix to reduce degrees of freedom.
    # For example, if we set U to be a permutation matrix (or scaled), then the problem simplifies.
    # But U has 8 rows. If R >= 8, we can have U as a submatrix of Identity.
    
    # Let's try this:
    # Set U to be a random matrix with entries in {0, 0.5, 1}.
    # Then solve for V and W?
    # This is still hard.
    
    # Let's go back to ALS but make it faster and more robust.
    # The main issue with the previous ALS was that it might converge to a local minimum with non-integer values.
    # We will add a step to snap to half-integers and verify.
    # If the snapped solution works, we return it.
    
    # We will also try a wider range of seeds and ranks.
    
    def try_decompose(rank, max_iter=300, seed=0):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale down to avoid overflow?
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        def solve_U(V, W):
            new_U = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        M[k, :] = V[v_idx, :] * W[w_idx, :]
                    try:
                        u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        new_U[i * m + j, :] = u
                    except:
                        return None
            return new_U

        def solve_V(U, W):
            new_V = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        M[i, :] = U[u_idx, :] * W[w_idx, :]
                    try:
                        v, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        new_V[j * p + k, :] = v
                    except:
                        return None
            return new_V

        def solve_W(U, V):
            new_W = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        M[j, :] = U[u_idx, :] * V[v_idx, :]
                    try:
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        new_W[k * n + i, :] = w
                    except:
                        return None
            return new_W

        for it in range(max_iter):
            U_new = solve_U(V, W)
            if U_new is None: return None
            V_new = solve_V(U_new, W)
            if V_new is None: return None
            W_new = solve_W(U_new, V_new)
            if W_new is None: return None
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c]
                        error += (val - 1.0)**2
            if error < 1e-6:
                # Snap to half-integers
                U_h, V_h, W_h = snap_to_half_integers(U_new, V_new, W_new)
                if verify(U_h, V_h, W_h, rank):
                    return U_h, V_h, W_h
            
            U, V, W = U_new, V_new, W_new
            
            if time.time() - start_time > time_limit - 5:
                return None

        return None

    # Try ranks from 14 to 55
    # We prioritize ranks that are more likely to be the true minimal rank.
    # For 2x4x7, the rank is likely around 30-40.
    # Let's try 20 to 55 first, then 14 to 19.
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Order of ranks to try
    ranks_to_try = list(range(20, 56)) + list(range(14, 20))
    
    for r in ranks_to_try:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try 10 random restarts for this rank
        found = False
        for seed in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            res = try_decompose(r, max_iter=200, seed=seed)
            if res is not None:
                U, V, W = res
                U_list, V_list, W_list = to_lists(U, V, W, r)
                return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
