"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
    
    The tensor T_{i,j,k} corresponds to the (i,j,k) entry of the multiplication tensor.
    We seek U (shape n*m x r), V (shape m*p x r), W (shape p*n x r) such that
    for all i, j, k: T_{i,j,k} = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    We use a randomized approach with integer/half-integer constraints by scaling by 2.
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition is exact."""
        T_check = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_check[i,j,k] = np.sum(
                        U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    )
        return np.allclose(T_check, 1.0, atol=1e-10)
    
    def clean_matrix(mat):
        """Clean up values to be integers or half-integers."""
        mat_clean = []
        for row in mat:
            row_clean = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    row_clean.append(int(round(val)))
                else:
                    row_clean.append(val)
            mat_clean.append(row_clean)
        return mat_clean
    
    def try_rank(r, num_trials=30, seed=42):
        """Try to find a decomposition of rank r."""
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            # Initialize with random small integers
            U_init = rng.randint(-1, 2, size=(nm, r)).astype(float)
            V_init = rng.randint(-1, 2, size=(mp, r)).astype(float)
            W_init = rng.randint(-1, 2, size=(pn, r)).astype(float)
            
            # Define the residual function
            def residual(params):
                U = params[:nm*r].reshape(nm, r)
                V = params[nm*r:nm*r+mp*r].reshape(mp, r)
                W = params[nm*r+mp*r:].reshape(pn, r)
                
                T_approx = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_approx[i,j,k] = np.sum(
                                U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                            )
                
                return (T_approx - 1.0).flatten()
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use least squares to solve
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=2000, ftol=1e-12)
                if np.linalg.norm(result.fun) < 1e-8:
                    U_sol = result.x[:nm*r].reshape(nm, r)
                    V_sol = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                    W_sol = result.x[nm*r+mp*r:].reshape(pn, r)
                    
                    # Check if solution is close to half-integers
                    U_int = np.round(2 * U_sol)
                    V_int = np.round(2 * V_sol)
                    W_int = np.round(2 * W_sol)
                    
                    if (np.allclose(U_sol, U_int/2, atol=1e-6) and
                        np.allclose(V_sol, V_int/2, atol=1e-6) and
                        np.allclose(W_sol, W_int/2, atol=1e-6)):
                        
                        U_val = U_int / 2.0
                        V_val = V_int / 2.0
                        W_val = W_int / 2.0
                        
                        if verify_decomposition(U_val, V_val, W_val, r):
                            U_list = clean_matrix(U_val.tolist())
                            V_list = clean_matrix(V_val.tolist())
                            W_list = clean_matrix(W_val.tolist())
                            return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                continue
        
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    for r in range(24, 48):
        result = try_rank(r, num_trials=15, seed=42 + r)
        if result is not None:
            return result
    
    # If we couldn't find a better decomposition, return schoolbook
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
# EVOLVE-BLOCK-END
