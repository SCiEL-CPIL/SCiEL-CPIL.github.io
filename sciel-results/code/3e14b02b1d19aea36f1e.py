import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def check_solution(U, V, W, R):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if U, V, W form a valid decomposition
        # Condition:
        # For all i_out, k_out:
        #   For all j:
        #     Sum_r U[i_out*m+j, r] V[j*p+k_out, r] W[k_out*n+i_out, r] == 1
        #   For all i_in, k_in:
        #     If (i_in, k_in) != (i_out, k_out):
        #       For all j:
        #         Sum_r U[i_in*m+j, r] V[j*p+k_in, r] W[k_out*n+i_out, r] == 0
        
        # Note: The sum over r is the dot product of the vectors U[:, r], V[:, r], W[:, r]
        # But we sum over r of the product of scalars.
        
        for i_out in range(n):
            for k_out in range(p):
                # Check diagonal terms (should be 1)
                for j in range(m):
                    s = 0.0
                    for r in range(R):
                        s += U[i_out * m + j, r] * V[j * p + k_out, r] * W[k_out * n + i_out, r]
                    if not np.isclose(s, 1.0):
                        return False
                
                # Check off-diagonal terms (should be 0)
                for i_in in range(n):
                    for k_in in range(p):
                        if i_in == i_out and k_in == k_out:
                            continue
                        for j in range(m):
                            s = 0.0
                            for r in range(R):
                                s += U[i_in * m + j, r] * V[j * p + k_in, r] * W[k_out * n + i_out, r]
                            if not np.isclose(s, 0.0):
                                return False
        return True

    def trivial_solution():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Search for minimal rank
    # Start from a lower bound. For 3x4x4, rank is likely > 12.
    # Try ranks 17 to 48.
    for R in range(17, 49):
        # Try 20 random initializations
        found = False
        for _ in range(20):
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Simple refinement: Project to integers and check
            # This is a very crude search.
            # To improve, we could do ALS, but it's complex.
            # Let's try rounding.
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if check_solution(U_int, V_int, W_int, R):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
            # Try half-integers?
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            if check_solution(U_half, V_half, W_half, R):
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
    # Fallback
    return trivial_solution()
# EVOLVE-BLOCK-END
