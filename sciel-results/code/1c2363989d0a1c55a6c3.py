"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from fractions import Fraction
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :]) * W[w_idx, :]
                    # Actually, the bilinear form is sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_find_decomposition(rank, max_attempts=5000, coeff_range=4):
        """Try to find a rank-decomposition with integer/half-integer coefficients."""
        np.random.seed(42 + rank)
        
        for attempt in range(max_attempts):
            # Generate random coefficients with small values
            U = np.random.randint(-coeff_range, coeff_range + 1, (n*m, rank)).astype(float)
            V = np.random.randint(-coeff_range, coeff_range + 1, (m*p, rank)).astype(float)
            W = np.random.randint(-coeff_range, coeff_range + 1, (p*n, rank)).astype(float)
            
            # Scale to get half-integers sometimes
            if attempt % 3 == 0:
                U = U / 2.0
            if attempt % 3 == 1:
                V = V / 2.0
            if attempt % 3 == 2:
                W = W / 2.0
            
            if verify(U.tolist(), V.tolist(), W.tolist()):
                # Convert to fractions for exact representation
                U_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in U]
                V_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in V]
                W_frac = [[Fraction(int(x * 2)) / 2 for x in row] for row in W]
                return [U_frac, V_frac, W_frac]
        
        return None
    
    # Try ranks from lower bound to schoolbook
    # Lower bound: at least max(n, m, p) * min(n, m, p) / something
    # For M_{2,3,3}, let's try from 10 to 17
    for rank in range(10, 18):
        result = try_find_decomposition(rank, max_attempts=2000)
        if result is not None:
            U_frac, V_frac, W_frac = result
            # Convert Fraction to float for return (evaluator should handle)
            U_out = [[float(x) for x in row] for row in U_frac]
            V_out = [[float(x) for x in row] for row in V_frac]
            W_out = [[float(x) for x in row] for row in W_frac]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
