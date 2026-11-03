import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Build target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for a in range(dim_u):
        i_a = a // m
        j_a = a % m
        for b in range(dim_v):
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(dim_w):
                k_c = c // n
                i_c = c % n
                if i_a == i_c and k_b == k_c:
                    T[a, b, c] = 1.0

    def build_strassen_init(R):
        """
        Build an initial rank-R decomposition based on Strassen-like patterns.
        For 2x3 * 3x5, we use a structured ansatz.
        """
        U = np.zeros((dim_u, R))
        V = np.zeros((dim_v, R))
        W = np.zeros((dim_w, R))
        
        # Index mapping:
        # U row: a = i*m + j, where i in [0,n), j in [0,m)
        # V row: b = j*p + k, where j in [0,m), k in [0,p)
        # W row: c = k*n + i, where k in [0,p), i in [0,n)
        
        # Strassen-like: group by (i, j, k) triples and use sums/differences
        # For n=2, m=3, p=5, we need to cover all 2*3*5=30 terms
        # With rank 27, we can merge 3 pairs
        
        # Initialize with identity-like structure for all 30 terms
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= R:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # For R < 30, we need to merge some terms
        # Use Strassen-style linear combinations
        if R < 30:
            # Try to merge terms that share structure
            # For example, merge (i,j,k) and (i',j,k) if possible
            # This is heuristic; we'll refine later
            pass
            
        return U, V, W

    def verify_decomposition(U, V, W, T, tol=1e-9):
        """Verify that U, V, W exactly decompose T."""
        R = U.shape[1]
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True

    def round_to_halfint(arr, tol=1e-8):
        """Round array to nearest half-integer."""
        doubled = arr * 2
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=tol * 2):
            return rounded / 2.0
        return None

    def solve_for_W_given_UV(U, V, T, R):
        """
        Given U and V, solve for W using least squares for each c independently.
        Returns W if it exists and is close to half-integers, else None.
        """
        W = np.zeros((dim_w, R))
        for c in range(dim_w):
            T_c = T[:, :, c].flatten()
            # Build M: (dim_u * dim_v) x R
            M = np.zeros((dim_u * dim_v, R))
            for a in range(dim_u):
                for b in range(dim_v):
                    eq_idx = a * dim_v + b
                    M[eq_idx, :] = U[a, :] * V[b, :]
            
            try:
                W_c, residuals, rank, s = np.linalg.lstsq(M, T_c, rcond=None)
            except np.linalg.LinAlgError:
                return None
            
            W[c, :] = W_c
        
        # Check if W is close to half-integers
        W_rounded = round_to_halfint(W)
        if W_rounded is None:
            return None
        
        return W_rounded

    def refine_UV_fixed_W(U, V, W, T, R, max_iters=100):
        """
        Iteratively refine U and V given W using alternating least squares
        with hard thresholding to half-integers.
        """
        U_curr = U.copy()
        V_curr = V.copy()
        
        for it in range(max_iters):
            # Update U: for each a, solve V[b,:] * W[c,:] pattern
            # Actually, the constraint is: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # For fixed V and W, this is linear in U[a,:].
            # But it's coupled across a,b,c.
            
            # Use gradient descent on U and V with projection
            # Simpler: alternating least squares per row
            
            # Update U rows
            for a in range(dim_u):
                # Equations involving U[a, :]:
                # For each (b, c): sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                # Let M_a be (dim_v * dim_w) x R
                M_a = np.zeros((dim_v * dim_w, R))
                rhs_a = np.zeros(dim_v * dim_w)
                idx = 0
                for b in range(dim_v):
                    for c in range(dim_w):
                        M_a[idx, :] = V[b, :] * W[c, :]
                        rhs_a[idx] = T[a, b, c]
                        idx += 1
                
                try:
                    U_a, _, _, _ = np.linalg.lstsq(M_a, rhs_a, rcond=None)
                except np.linalg.LinAlgError:
                    U_a = U_curr[a, :]
                
                # Project to half-integers
                U_a_rounded = round_to_halfint(U_a)
                if U_a_rounded is not None:
                    U_curr[a, :] = U_a_rounded
            
            # Update V rows
            for b in range(dim_v):
                M_b = np.zeros((dim_u * dim_w, R))
                rhs_b = np.zeros(dim_u * dim_w)
                idx = 0
                for a in range(dim_u):
                    for c in range(dim_w):
                        M_b[idx, :] = U_curr[a, :] * W[c, :]
                        rhs_b[idx] = T[a, b, c]
                        idx += 1
                
                try:
                    V_b, _, _, _ = np.linalg.lstsq(M_b, rhs_b, rcond=None)
                except np.linalg.LinAlgError:
                    V_b = V_curr[b, :]
                
                V_b_rounded = round_to_halfint(V_b)
                if V_b_rounded is not None:
                    V_curr[b, :] = V_b_rounded
            
            # Check convergence
            if verify_decomposition(U_curr, V_curr, W, T):
                return U_curr, V_curr
        
        return None, None

    def try_rank(R, num_trials=50):
        """Try to find a valid rank-R decomposition."""
        for trial in range(num_trials):
            # Initialize
            U, V, W = build_strassen_init(R)
            
            # Add random perturbation for diversity
            if trial > 0:
                noise_scale = 0.1 * trial
                U = U + np.random.randn(dim_u, R) * noise_scale
                V = V + np.random.randn(dim_v, R) * noise_scale
                W = W + np.random.randn(dim_w, R) * noise_scale
            
            # Project to half-integers
            U_proj = round_to_halfint(U)
            V_proj = round_to_halfint(V)
            W_proj = round_to_halfint(W)
            
            if U_proj is None or V_proj is None or W_proj is None:
                continue
            
            U_curr = U_proj
            V_curr = V_proj
            W_curr = W_proj
            
            # Check if already valid
            if verify_decomposition(U_curr, V_curr, W_curr, T):
                return U_curr, V_curr, W_curr
            
            # Refine using alternating least squares
            U_ref, V_ref = refine_UV_fixed_W(U_curr, V_curr, W_curr, T, R, max_iters=200)
            if U_ref is not None and V_ref is not None:
                return U_ref, V_ref, W_curr
            
            # Try solving for W given U, V
            W_solved = solve_for_W_given_UV(U_curr, V_curr, T, R)
            if W_solved is not None:
                if verify_decomposition(U_curr, V_curr, W_solved, T):
                    return U_curr, V_curr, W_solved
            
            # Try random restarts with different patterns
            if trial % 10 == 9:
                # Try pure random initialization
                U_rand = np.random.choice([-1, 0, 1], size=(dim_u, R))
                V_rand = np.random.choice([-1, 0, 1], size=(dim_v, R))
                W_rand = np.random.choice([-1, 0, 1], size=(dim_w, R))
                
                U_rand_proj = round_to_halfint(U_rand.astype(float))
                V_rand_proj = round_to_halfint(V_rand.astype(float))
                W_rand_proj = round_to_halfint(W_rand.astype(float))
                
                if U_rand_proj is not None and V_rand_proj is not None and W_rand_proj is not None:
                    if verify_decomposition(U_rand_proj, V_rand_proj, W_rand_proj, T):
                        return U_rand_proj, V_rand_proj, W_rand_proj
                    
                    W_solved_rand = solve_for_W_given_UV(U_rand_proj, V_rand_proj, T, R)
                    if W_solved_rand is not None:
                        if verify_decomposition(U_rand_proj, V_rand_proj, W_solved_rand, T):
                            return U_rand_proj, V_rand_proj, W_solved_rand
        
        return None, None, None

    # Try ranks from 27 down to 25
    for R in [27, 26, 25]:
        U, V, W = try_rank(R, num_trials=100)
        if U is not None:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}
