"""Matrix multiplication 2x3x3 tensor decomposition.
Attempts to find a rank-16 exact decomposition using a structured ansatz
based on Strassen's rank-7 for 2x2x2 and low-rank corrections.
"""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 3]


def construct():
    """
    Returns a dict with keys 'U', 'V', 'W' whose entries form an exact
    rank-r tensor decomposition of the 2x3x3 matrix multiplication tensor.
    
    U: shape (6, r)  rows indexed by (i, j) with i in [0,2), j in [0,3)
    V: shape (9, r)  rows indexed by (j, k) with j in [0,3), k in [0,3)
    W: shape (6, r)  rows indexed by (k, i) with k in [0,3), i in [0,2)
    
    The tensor T[i, j, k] = A[i,j] * B[j,k] is decomposed as:
    T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where n=2, m=3, p=3.
    
    So T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
    """
    n, m, p = 2, 3, 3
    
    # Try to construct a rank-16 decomposition
    # Start with a rank-17 Strassen-based construction and try to reduce
    
    # For 2x3x3, we can think of it as:
    # A is 2x3, B is 3x3
    # Split A into two 2x2 blocks and one 2x1 column: A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Split B into three 2x3 rows... actually let's think differently.
    
    # Better approach: use the known rank-17 decomposition and try to find linear dependencies
    
    # Let's use a systematic search approach with a structured ansatz.
    
    # The multiplication tensor for 2x3x3 has 2*3*3 = 18 entries.
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    
    # We want: T[i,j,k] = sum_{r=0}^{R-1} U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
    
    # Let's try R = 16 first, then R = 17, etc.
    
    # Strategy: Use a bilinear approach. For each pair (i, k), the tensor entry 
    # T[i,j,k] as a function of j is linear in B's row j. 
    
    # Actually, let's use a different approach: enumerate all 18 tensor entries and
    # try to find a low-rank decomposition.
    
    # For rank R, we need:
    # For all (i, j, k): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This gives 18 equations. Each equation is multilinear in the variables.
    
    # Let's try a constructive approach based on Strassen.
    
    # Strassen for 2x2x2 uses rank 7:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For 2x3x3, we can extend this. Let's partition:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    
    # C = [[c00, c01, c02],
    #      [c10, c11, c12]]
    
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # Let's try to find a rank-16 decomposition by solving a system of equations.
    
    # Given the complexity, let's use a known construction or search.
    
    # For now, let's return a valid rank-18 schoolbook decomposition as fallback,
    # but try to find something better.
    
    # Actually, let's implement a search that tries to find rank-16 or rank-17.
    
    # Given time constraints, let's use a deterministic construction.
    
    # Known result: The tensor rank of 2x3x3 matrix multiplication is at most 17.
    # Let's construct a rank-17 decomposition.
    
    # We'll use a construction based on the 2x2x2 Strassen decomposition extended to 2x3x3.
    
    # Partition A as [A_left (2x2) | A_right (2x1)]
    # Partition B into three rows, each of size 1x3.
    
    # Actually, let's use a different partitioning:
    # Think of the 2x3x3 multiplication as involving 2 inputs to A, 3 middle indices, 3 outputs.
    
    # Let's try to use the following approach:
    # For each of the 3 middle indices j, we have a 2x1 vector A[:,j] and a 1x3 vector B[j,:].
    # C = sum_j A[:,j] * B[j,:]
    
    # This is a sum of 3 rank-1 outer products (in the bilinear sense).
    # Each term A[:,j] * B[j,:] contributes to all 6 entries of C.
    
    # So the trivial decomposition has rank 3*6 = 18 (each of the 3 middle indices 
    # requires 6 products for the 6 entries of C, but that's not quite right).
    
    # Actually, the schoolbook decomposition has rank 18: for each (i,j,k) there's one term.
    
    # Let's try to find a rank-17 or lower decomposition.
    
    # Given the difficulty of finding an exact low-rank decomposition analytically,
    # let's use a numerical optimization approach.
    
    # Start with rank 16 and try to solve.
    
    R = 16
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Initialize with random values
    rng = np.random.default_rng(42)
    U = rng.standard_normal((n*m, R))
    V = rng.standard_normal((m*p, R))
    W = rng.standard_normal((p*n, R))
    
    # Define the residual function
    def compute_residual(U, V, W):
        """Compute the residual of the tensor decomposition."""
        residual = 0.0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = 1
                    pred = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                    err = abs(pred - 1.0)
                    residual += err ** 2
                    max_err = max(max_err, err)
        return residual, max_err
    
    # Use gradient-free optimization (Nelder-Mead) to minimize the residual
    from scipy.optimize import minimize
    
    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_params(params):
        u_size = n*m*R
        v_size = m*p*R
        u = params[:u_size].reshape(n*m, R)
        v = params[u_size:u_size+v_size].reshape(m*p, R)
        w = params[u_size+v_size:].reshape(p*n, R)
        return u, v, w
    
    def objective(params):
        u, v, w = unflatten_params(params)
        res, _ = compute_residual(u, v, w)
        return res
    
    # Initial parameters
    x0 = flatten_params(U, V, W)
    
    # Try Nelder-Mead
    start_time = time.time()
    timeout = 120  # seconds
    
    best_params = x0
    best_obj = objective(x0)
    
    # Run optimization
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
    
    if result.fun < best_obj:
        best_obj = result.fun
        best_params = result.x
    
    elapsed = time.time() - start_time
    print(f"Optimization took {elapsed:.2f}s, best objective: {best_obj:.2e}")
    
    if best_obj < 1e-10:
        # Found an exact (or nearly exact) decomposition
        U_best, V_best, W_best = unflatten_params(best_params)
        
        # Try to round to rational numbers with small denominators
        def rationalize(mat):
            """Try to round matrix entries to simple fractions."""
            rounded = np.zeros_like(mat)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    # Try to find a fraction with small denominator
                    best_frac = val
                    best_err = abs(val - best_frac)
                    for denom in range(1, 11):
                        num = round(val * denom)
                        frac = num / denom
                        err = abs(val - frac)
                        if err < best_err:
                            best_err = err
                            best_frac = frac
                    rounded[i, j] = best_frac
            return rounded
        
        U_rounded = rationalize(U_best)
        V_rounded = rationalize(V_best)
        W_rounded = rationalize(W_best)
        
        # Verify the rounded version
        res, max_err = compute_residual(U_rounded, V_rounded, W_rounded)
        if max_err < 1e-8:
            # Convert to lists
            U_list = [[float(U_rounded[i, j]) for j in range(R)] for i in range(n*m)]
            V_list = [[float(V_rounded[i, j]) for j in range(R)] for i in range(m*p)]
            W_list = [[float(W_rounded[i, j]) for j in range(R)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook rank-18 decomposition
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
